"""Fill the OMP {20,24} hole between 16 and 32. Writes under evidence/."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    F3_B37_BANANA_OMP_GAP_THREADS,
    evaluate_f3_b37_banana_omp_sweep,
    write_strict_json,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "receipts" / "evidence"
PUBLICATION = (
    "OMP-pinned banana gap fill at 20 and 24 threads. Completes the "
    "Gate-6 native sweep set on a 32-core box. Not a nested speed claim."
)
OUT_JSON = EVIDENCE / "nested_ls_reduced_gpu_banana_omp_gap_20260821.json"
OUT_LOG = EVIDENCE / "nested_ls_reduced_gpu_banana_omp_gap_20260821.log"

probe = evaluate_f3_b37_banana_omp_sweep(threads=F3_B37_BANANA_OMP_GAP_THREADS)
print(
    "banana_omp_gap fail",
    probe.fail_closed_reason,
    "rows",
    len(probe.rows),
    flush=True,
)
for row in probe.rows:
    print("row", row, flush=True)
payload = {
    "claim_boundary": {
        "cap_2048_attempted": False,
        "comparable_operators": False,
        "explicit_inverse_m_production": False,
        "f3_sealed": True,
        "inherits_f3_7_70x": False,
        "gap_fill_20_24": True,
        "interleaved_repeats": True,
        "nested_speed_claim": False,
        "omp_pinned": True,
        "omp_swept": True,
    },
    "command": (
        "JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 "
        ".venv-qn-gpu/bin/python benchmarks/nested_ls_banana_omp_gap.py"
    ),
    "date": datetime.now(timezone.utc).date().isoformat(),
    "driver": "benchmarks.nested_ls_banana_omp_gap",
    "execution_log": str(OUT_LOG.relative_to(REPO)),
    "probe": probe.as_payload(),
    "publication": PUBLICATION,
    "schema": "nested-ls-reduced-gpu-banana-omp-gap.v1",
    "threads": list(F3_B37_BANANA_OMP_GAP_THREADS),
    "written_by_pytest": False,
}
write_strict_json(OUT_JSON, payload)
print("wrote", OUT_JSON, flush=True)
ok = probe.fail_closed_reason is None
print("ok", ok, flush=True)
if not ok:
    raise SystemExit(f"banana omp gap failed: {probe.fail_closed_reason}")
