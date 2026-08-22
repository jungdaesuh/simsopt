"""Repo-path GPU canaries for nested-LS banana OMP and warm chunk sweeps.

Writes JSON under docs/receipts/evidence/. Not a nested speed claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import jax

print("backend", jax.default_backend(), flush=True)
print("devices", [str(d) for d in jax.devices()], flush=True)
if jax.default_backend() != "gpu":
    raise SystemExit(f"expected gpu, got {jax.default_backend()!r}")

from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE,
    F3_B37_BANANA_OMP_THREADS,
    evaluate_f3_b37_banana_omp_sweep,
    evaluate_f3_b37_chunk_warm_probe,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    write_strict_json,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "receipts" / "evidence"
BANANA_PUBLICATION = (
    "OMP-pinned interleaved native banana run_code sweep. "
    "Not a nested speed claim and not F3 7.70x."
)
WARM_PUBLICATION = (
    "Warm in-process dense-assemble repeats at the LU endpoint. "
    "Cold first-touch discarded. Not a default switch and not a nested speed claim."
)
BANANA_JSON = EVIDENCE / "nested_ls_reduced_gpu_banana_omp_20260821.json"
WARM_JSON = EVIDENCE / "nested_ls_reduced_gpu_chunk_warm_20260821.json"
BANANA_LOG = EVIDENCE / "nested_ls_reduced_gpu_banana_omp_20260821.log"
WARM_LOG = EVIDENCE / "nested_ls_reduced_gpu_chunk_warm_20260821.log"

coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
_native, jax_boozer, _target = load_archived_nested_ls_pair(
    coil_coordinates=coils,
    surface_coordinates=surface,
)
del _native, _target

banana = evaluate_f3_b37_banana_omp_sweep()
print(
    "banana_omp fail",
    banana.fail_closed_reason,
    "rows",
    len(banana.rows),
    flush=True,
)
banana_payload = {
    "claim_boundary": {
        "cap_2048_attempted": False,
        "comparable_operators": False,
        "explicit_inverse_m_production": False,
        "f3_sealed": True,
        "inherits_f3_7_70x": False,
        "nested_speed_claim": False,
        "omp_pinned": True,
        "omp_swept": True,
        "interleaved_repeats": True,
    },
    "command": (
        "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 "
        ".venv-qn-gpu/bin/python benchmarks/nested_ls_f3_b37_gpu_canaries.py"
    ),
    "date": datetime.now(timezone.utc).date().isoformat(),
    "driver": "benchmarks.nested_ls_f3_b37_gpu_canaries",
    "execution_log": str(BANANA_LOG.relative_to(REPO)),
    "probe": banana.as_payload(),
    "publication": BANANA_PUBLICATION,
    "schema": "nested-ls-reduced-gpu-banana-omp.v1",
    "threads": list(F3_B37_BANANA_OMP_THREADS),
    "written_by_pytest": False,
}
write_strict_json(BANANA_JSON, banana_payload)
print("wrote", BANANA_JSON, flush=True)

warm = evaluate_f3_b37_chunk_warm_probe(jax_boozer)
print("chunk_warm fail", warm.fail_closed_reason, "rows", len(warm.rows), flush=True)
warm_payload = {
    "claim_boundary": {
        "cap_2048_attempted": False,
        "cold_sweep": False,
        "explicit_inverse_m_production": False,
        "f3_sealed": True,
        "inherits_f3_7_70x": False,
        "nested_speed_claim": False,
        "production_chunk_default_unchanged": True,
        "warm_repeated": True,
    },
    "command": (
        "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 "
        ".venv-qn-gpu/bin/python benchmarks/nested_ls_f3_b37_gpu_canaries.py"
    ),
    "date": datetime.now(timezone.utc).date().isoformat(),
    "driver": "benchmarks.nested_ls_f3_b37_gpu_canaries",
    "execution_log": str(WARM_LOG.relative_to(REPO)),
    "probe": warm.as_payload(),
    "publication": WARM_PUBLICATION,
    "schema": "nested-ls-reduced-gpu-chunk-warm.v1",
    "written_by_pytest": False,
}
write_strict_json(WARM_JSON, warm_payload)
print("wrote", WARM_JSON, flush=True)
print(
    "banana_ok",
    banana.fail_closed_reason is None,
    "warm_ok",
    warm.fail_closed_reason is None,
    flush=True,
)
