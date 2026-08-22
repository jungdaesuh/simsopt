"""A100 Landau banana OMP contract sweep. Best-of-contract is this box's bar.

Frozen thread set {4,8,12,14,16,20,24,32}. Not a nested speed claim and
not the 5090 OMP=16 result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    F3_B37_BANANA_OMP_CONTRACT_THREADS,
    evaluate_f3_b37_banana_omp_sweep,
    write_strict_json,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "receipts" / "evidence"
PUBLICATION = (
    "A100 Landau OMP-pinned banana contract sweep "
    "{4,8,12,14,16,20,24,32}. Best-of-contract is this host's native "
    "bar. Not a nested speed claim and not F3 7.70x."
)
OUT_JSON = EVIDENCE / "nested_ls_reduced_a100_banana_omp_20260822.json"
OUT_LOG = EVIDENCE / "nested_ls_reduced_a100_banana_omp_20260822.log"

# Dual-socket EPYC 7452 banana is ~400 s/child; 16 children need >3600 s.
probe = evaluate_f3_b37_banana_omp_sweep(
    threads=F3_B37_BANANA_OMP_CONTRACT_THREADS,
    wall_seconds=14400.0,
)
print(
    "banana_omp_a100 fail",
    probe.fail_closed_reason,
    "rows",
    len(probe.rows),
    flush=True,
)
best_threads: int | None = None
best_inner: float | None = None
for row in probe.rows:
    print("row", row, flush=True)
    if not bool(row.get("success")):
        continue
    inner = float(row["inner_solver_seconds"])
    threads = int(row["omp_num_threads"])
    if best_inner is None or inner < best_inner:
        best_inner = inner
        best_threads = threads
payload = {
    "claim_boundary": {
        "cap_2048_attempted": False,
        "comparable_operators": False,
        "explicit_inverse_m_production": False,
        "f3_sealed": True,
        "host": "landau",
        "inherits_f3_7_70x": False,
        "inner_and_process_wall": True,
        "interleaved_repeats": True,
        "nested_speed_claim": False,
        "omp_pinned": True,
        "omp_swept": True,
    },
    "command": (
        "JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "
        "$LANDAU_PY benchmarks/nested_ls_a100_banana_omp.py"
    ),
    "date": datetime.now(timezone.utc).date().isoformat(),
    "driver": "benchmarks.nested_ls_a100_banana_omp",
    "execution_log": str(OUT_LOG.relative_to(REPO)),
    "best_omp_num_threads": best_threads,
    "best_inner_solver_seconds": best_inner,
    "probe": probe.as_payload(),
    "publication": PUBLICATION,
    "schema": "nested-ls-reduced-a100-banana-omp.v1",
    "threads": list(F3_B37_BANANA_OMP_CONTRACT_THREADS),
    "written_by_pytest": False,
}
write_strict_json(OUT_JSON, payload)
print("wrote", OUT_JSON, flush=True)
print("best_omp", best_threads, "best_inner", best_inner, flush=True)
ok = probe.fail_closed_reason is None and best_threads is not None
print("ok", ok, flush=True)
if not ok:
    raise SystemExit(f"a100 banana omp failed: {probe.fail_closed_reason}")
