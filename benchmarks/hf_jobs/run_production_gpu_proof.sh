#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-60}"
GPU_DETERMINISM_XLA_FLAG="--xla_gpu_exclude_nondeterministic_ops=true"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RESULTS_DIR=""
EQUILIBRIA_DIR=""
PLASMA_SURF_FILENAME="wout_nfp22ginsburg_000_014417_iota15.nc"
STAGE2_BS_PATH=""
STAGE2_PLATFORM="cuda"
SINGLE_STAGE_PLATFORM="cuda"
STAGE2_NPHI="255"
STAGE2_NTHETA="64"
STAGE2_MAXITER="20"
STAGE2_OPTIMIZER_BACKEND="ondevice"
GEOMETRY_REL_TOL=""
SINGLE_STAGE_NPHI="255"
SINGLE_STAGE_NTHETA="64"
SINGLE_STAGE_MPOL="8"
SINGLE_STAGE_NTOR="6"
SINGLE_STAGE_MAXITER="300"
SINGLE_STAGE_OPTIMIZER_BACKEND="ondevice"
SINGLE_STAGE_BOOZER_OPTIMIZER_BACKEND=""
SINGLE_STAGE_WARM_START_RUN_DIR=""
SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC=""
SINGLE_STAGE_BENCHMARK_MODE="0"
SINGLE_STAGE_DISABLE_TARGET_LANE_SUCCESS_FILTER="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --equilibria-dir) EQUILIBRIA_DIR="$2"; shift 2 ;;
    --plasma-surf-filename) PLASMA_SURF_FILENAME="$2"; shift 2 ;;
    --stage2-bs-path) STAGE2_BS_PATH="$2"; shift 2 ;;
    --stage2-platform) STAGE2_PLATFORM="$2"; shift 2 ;;
    --single-stage-platform) SINGLE_STAGE_PLATFORM="$2"; shift 2 ;;
    --stage2-nphi) STAGE2_NPHI="$2"; shift 2 ;;
    --stage2-ntheta) STAGE2_NTHETA="$2"; shift 2 ;;
    --stage2-maxiter) STAGE2_MAXITER="$2"; shift 2 ;;
    --stage2-optimizer-backend) STAGE2_OPTIMIZER_BACKEND="$2"; shift 2 ;;
    --geometry-rel-tol) GEOMETRY_REL_TOL="$2"; shift 2 ;;
    --single-stage-nphi) SINGLE_STAGE_NPHI="$2"; shift 2 ;;
    --single-stage-ntheta) SINGLE_STAGE_NTHETA="$2"; shift 2 ;;
    --single-stage-mpol) SINGLE_STAGE_MPOL="$2"; shift 2 ;;
    --single-stage-ntor) SINGLE_STAGE_NTOR="$2"; shift 2 ;;
    --single-stage-maxiter) SINGLE_STAGE_MAXITER="$2"; shift 2 ;;
    --single-stage-optimizer-backend) SINGLE_STAGE_OPTIMIZER_BACKEND="$2"; shift 2 ;;
    --single-stage-boozer-optimizer-backend) SINGLE_STAGE_BOOZER_OPTIMIZER_BACKEND="$2"; shift 2 ;;
    --single-stage-warm-start-run-dir) SINGLE_STAGE_WARM_START_RUN_DIR="$2"; shift 2 ;;
    --single-stage-jax-runtime-seed-spec) SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC="$2"; shift 2 ;;
    --single-stage-benchmark-mode) SINGLE_STAGE_BENCHMARK_MODE="1"; shift ;;
    --single-stage-disable-target-lane-success-filter) SINGLE_STAGE_DISABLE_TARGET_LANE_SUCCESS_FILTER="1"; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RESULTS_DIR}" || -z "${EQUILIBRIA_DIR}" || -z "${STAGE2_BS_PATH}" ]]; then
  echo "Missing required arguments: --results-dir, --equilibria-dir, --stage2-bs-path" >&2
  exit 1
fi
if [[ -z "${SINGLE_STAGE_WARM_START_RUN_DIR}" && -z "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC}" ]]; then
  echo "Production single-stage GPU proof requires --single-stage-warm-start-run-dir or --single-stage-jax-runtime-seed-spec" >&2
  exit 1
fi
if [[ "${SIMSOPT_FAKE_GPU:-}" != "1" && ( "${STAGE2_PLATFORM}" != "cuda" || "${SINGLE_STAGE_PLATFORM}" != "cuda" ) ]]; then
  echo "Production GPU proof requires --stage2-platform cuda and --single-stage-platform cuda" >&2
  exit 1
fi
if [[ "${STAGE2_PLATFORM}" == "cuda" || "${SINGLE_STAGE_PLATFORM}" == "cuda" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} ${GPU_DETERMINISM_XLA_FLAG}"
fi

GEOMETRY_REL_TOL_ARG="${GEOMETRY_REL_TOL}"
if [[ -z "${GEOMETRY_REL_TOL_ARG}" ]]; then
  GEOMETRY_REL_TOL_ARG="__NONE__"
fi
if ! mapfile -t STAGE2_RUNG_NAMES < <(
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${STAGE2_MAXITER}" "${GEOMETRY_REL_TOL_ARG}" <<'PY'
from pathlib import Path
import sys

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root))

from benchmarks.validation_ladder_contract import build_stage2_hf_plan

maxiter = int(sys.argv[2])
raw_geometry_rel_tol = sys.argv[3]
geometry_rel_tol = None if raw_geometry_rel_tol == "__NONE__" else float(raw_geometry_rel_tol)
try:
    plan = build_stage2_hf_plan(maxiter, geometry_rel_tol)
except ValueError as exc:
    raise SystemExit(str(exc))
for rung_name in plan["stage2_rungs"]:
    print(rung_name)
PY
); then
  exit 1
fi

export HF_HUB_DISABLE_TELEMETRY=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${REPO_ROOT}/.artifacts/jax_compilation_cache/hf-production-proof}"
export SIMSOPT_JAX_CUDA_LIBRARY_MODE="${SIMSOPT_JAX_CUDA_LIBRARY_MODE:-bundled}"
# Keep JAX from reserving most VRAM before the proof kernels allocate.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${RESULTS_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

OVERALL_RC=0
declare -a EXPECTED_PROBES=(
  "${STAGE2_RUNG_NAMES[@]}"
  "single_stage_cold"
  "single_stage_warm"
  "boozer_well_conditioned_adjoint"
  "reduction_cancellation_stress"
)

emit_payload_summary() {
  local name="$1"
  local output_json="$2"
  "${PYTHON_BIN}" - "${name}" "${output_json}" <<'PY'
import json
import sys
from pathlib import Path

name = sys.argv[1]
payload_path = Path(sys.argv[2])
try:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    print(
        f"[summary] {name} corrupt payload: {payload_path} "
        f"({exc.__class__.__name__}: {exc})"
    )
    raise SystemExit(1)
print({"passed": payload.get("passed"), "failures": payload.get("failures")})
PY
}

sample_gpu_peak() {
  local output_file="$1"
  local stop_file="$2"
  local peak_mb=0
  local used_mb=""
  while [[ ! -f "${stop_file}" ]]; do
    used_mb="$(
      nvidia-smi \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 \
        | tr -d '[:space:]'
    )"
    if [[ "${used_mb}" =~ ^[0-9]+$ ]] && (( used_mb > peak_mb )); then
      peak_mb="${used_mb}"
    fi
    sleep 1
  done
  if (( peak_mb > 0 )); then
    printf '%s\n' "${peak_mb}" > "${output_file}"
  fi
}

annotate_probe_payload() {
  local output_json="$1"
  local peak_file="$2"
  shift 2
  if [[ ! -f "${output_json}" ]]; then
    return
  fi
  "${PYTHON_BIN}" - "${output_json}" "${peak_file}" "$@" <<'PY'
import json
import sys
from pathlib import Path

payload_path = Path(sys.argv[1])
peak_path = Path(sys.argv[2])
command_argv = sys.argv[3:]
payload = json.loads(payload_path.read_text(encoding="utf-8"))
provenance = payload.get("provenance")
if isinstance(provenance, dict):
    provenance.setdefault("command_argv", command_argv)
    if peak_path.exists():
        raw_peak = peak_path.read_text(encoding="utf-8").strip()
        if raw_peak:
            sampled_peak_mb = float(raw_peak)
            current_peak = provenance.get("peak_gpu_memory_mb")
            provenance["peak_gpu_memory_mb"] = (
                sampled_peak_mb
                if current_peak is None
                else max(float(current_peak), sampled_peak_mb)
            )
payload_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_probe() {
  local name="$1"
  local output_json="$2"
  shift 2

  echo "=== ${name} ==="
  local start_ts
  start_ts="$(date +%s)"
  local peak_file="${output_json}.peak_gpu_memory_mb"
  local stop_file="${output_json}.gpu_sampler_stop"
  local sampler_pid=""
  rm -f "${peak_file}" "${stop_file}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    sample_gpu_peak "${peak_file}" "${stop_file}" &
    sampler_pid="$!"
  fi
  "$@" &
  local pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    echo "[heartbeat] ${name} $(date -u +%FT%TZ)"
    sleep "${HEARTBEAT_INTERVAL_S}"
  done
  local rc=0
  if wait "${pid}"; then
    rc=0
  else
    rc=$?
  fi
  if [[ -n "${sampler_pid}" ]]; then
    touch "${stop_file}"
    wait "${sampler_pid}" || true
  fi
  local end_ts
  end_ts="$(date +%s)"
  echo "[result] ${name} rc=${rc} wall_s=$((end_ts-start_ts))"
  if [[ -f "${output_json}" ]]; then
    annotate_probe_payload "${output_json}" "${peak_file}" "$@"
    cat "${output_json}"
    if ! emit_payload_summary "${name}" "${output_json}"; then
      rc=1
    fi
  else
    echo "[summary] ${name} missing payload: ${output_json}"
    if [[ "${rc}" -eq 0 ]]; then
      rc=1
    fi
  fi
  return "${rc}"
}

run_cuda_canary() {
  local mode="$1"
  local force_ptx_jit="$2"
  local disable_ptx_jit="$3"
  local output_json="$4"
  local canary_cache_dir="${RESULTS_DIR}/cuda_canary_cache_${mode}"
  local cuda_cache_dir="${RESULTS_DIR}/cuda_driver_cache_${mode}"

  echo "=== cuda_canary_${mode} ==="
  mkdir -p "${canary_cache_dir}" "${cuda_cache_dir}"
  CUDA_FORCE_PTX_JIT="${force_ptx_jit}" \
    CUDA_DISABLE_PTX_JIT="${disable_ptx_jit}" \
    CUDA_CACHE_DISABLE=1 \
    CUDA_CACHE_PATH="${cuda_cache_dir}" \
    JAX_COMPILATION_CACHE_DIR="${canary_cache_dir}" \
    "${PYTHON_BIN}" - "${mode}" "${output_json}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


class CanaryPayload(TypedDict):
    mode: str
    backend: str
    devices: List[str]
    cuda_force_ptx_jit: Optional[str]
    cuda_disable_ptx_jit: Optional[str]
    cuda_cache_disable: Optional[str]
    cuda_cache_path: Optional[str]
    jax_compilation_cache_dir: Optional[str]
    value: float


mode: str = sys.argv[1]
output_json: Path = Path(sys.argv[2])
backend: str = str(jax.default_backend()).lower()
devices: List[str] = [str(device) for device in jax.devices()]
if backend not in {"gpu", "cuda"}:
    raise SystemExit(
        f"CUDA canary {mode} expected GPU backend, got {backend!r} on {devices}"
    )


@jax.jit
def canary_kernel(x: jax.Array) -> jax.Array:
    return jnp.sum(jnp.sin(x) * jnp.cos(x) + x * x)


x: jax.Array = jnp.arange(1.0, 1025.0, dtype=jnp.float64).reshape((32, 32))
value: jax.Array = canary_kernel(x)
value.block_until_ready()
payload: CanaryPayload = {
    "mode": mode,
    "backend": backend,
    "devices": devices,
    "cuda_force_ptx_jit": os.environ.get("CUDA_FORCE_PTX_JIT"),
    "cuda_disable_ptx_jit": os.environ.get("CUDA_DISABLE_PTX_JIT"),
    "cuda_cache_disable": os.environ.get("CUDA_CACHE_DISABLE"),
    "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
    "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
    "value": float(value),
}
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
}

if [[ "${SIMSOPT_FAKE_GPU:-}" != "1" && ( "${STAGE2_PLATFORM}" == "cuda" || "${SINGLE_STAGE_PLATFORM}" == "cuda" ) ]]; then
  run_cuda_canary ptx 1 0 "${RESULTS_DIR}/cuda_canary_ptx.json"
  run_cuda_canary cubin 0 1 "${RESULTS_DIR}/cuda_canary_cubin.json"
fi

run_probe stage2_cold "${RESULTS_DIR}/stage2_cold.json" \
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
    --platform "${STAGE2_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --nphi "${STAGE2_NPHI}" \
    --ntheta "${STAGE2_NTHETA}" \
    --maxiter "${STAGE2_MAXITER}" \
    --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
    --output-json "${RESULTS_DIR}/stage2_cold.json" || OVERALL_RC=1

run_probe stage2_warm "${RESULTS_DIR}/stage2_warm.json" \
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
    --platform "${STAGE2_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --nphi "${STAGE2_NPHI}" \
    --ntheta "${STAGE2_NTHETA}" \
    --maxiter "${STAGE2_MAXITER}" \
    --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
    --output-json "${RESULTS_DIR}/stage2_warm.json" || OVERALL_RC=1

if [[ " ${STAGE2_RUNG_NAMES[*]} " == *" stage2_warm_repro "* ]]; then
  run_probe stage2_warm_repro "${RESULTS_DIR}/stage2_warm_repro.json" \
    "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
      --platform "${STAGE2_PLATFORM}" \
      --equilibria-dir "${EQUILIBRIA_DIR}" \
      --nphi "${STAGE2_NPHI}" \
      --ntheta "${STAGE2_NTHETA}" \
      --maxiter "${STAGE2_MAXITER}" \
      --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
      --geometry-rel-tol "${GEOMETRY_REL_TOL}" \
      --output-json "${RESULTS_DIR}/stage2_warm_repro.json" || OVERALL_RC=1
fi

single_stage_probe_args=(
  --equilibria-dir "${EQUILIBRIA_DIR}"
  --plasma-surf-filename "${PLASMA_SURF_FILENAME}"
  --stage2-bs-path "${STAGE2_BS_PATH}"
  --nphi "${SINGLE_STAGE_NPHI}"
  --ntheta "${SINGLE_STAGE_NTHETA}"
  --mpol "${SINGLE_STAGE_MPOL}"
  --ntor "${SINGLE_STAGE_NTOR}"
  --optimizer-backend "${SINGLE_STAGE_OPTIMIZER_BACKEND}"
  --maxiter "${SINGLE_STAGE_MAXITER}"
)
if [[ -n "${SINGLE_STAGE_BOOZER_OPTIMIZER_BACKEND}" ]]; then
  single_stage_probe_args+=(
    --boozer-optimizer-backend "${SINGLE_STAGE_BOOZER_OPTIMIZER_BACKEND}"
  )
fi
if [[ -n "${SINGLE_STAGE_WARM_START_RUN_DIR}" ]]; then
  single_stage_probe_args+=(
    --warm-start-run-dir "${SINGLE_STAGE_WARM_START_RUN_DIR}"
  )
fi
if [[ -n "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC}" ]]; then
  single_stage_probe_args+=(
    --jax-runtime-seed-spec "${SINGLE_STAGE_JAX_RUNTIME_SEED_SPEC}"
  )
fi
if [[ "${SINGLE_STAGE_BENCHMARK_MODE}" == "1" ]]; then
  single_stage_probe_args+=(--benchmark-mode)
fi
if [[ "${SINGLE_STAGE_DISABLE_TARGET_LANE_SUCCESS_FILTER}" == "1" ]]; then
  single_stage_probe_args+=(--disable-target-lane-success-filter)
fi

run_probe single_stage_cold "${RESULTS_DIR}/single_stage_cold.json" \
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/single_stage_init_parity.py" \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    "${single_stage_probe_args[@]}" \
    --case-artifacts-dir "${RESULTS_DIR}/artifacts/single_stage_cold" \
    --output-json "${RESULTS_DIR}/single_stage_cold.json" || OVERALL_RC=1

run_probe single_stage_warm "${RESULTS_DIR}/single_stage_warm.json" \
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/single_stage_init_parity.py" \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    "${single_stage_probe_args[@]}" \
    --case-artifacts-dir "${RESULTS_DIR}/artifacts/single_stage_warm" \
    --output-json "${RESULTS_DIR}/single_stage_warm.json" || OVERALL_RC=1

run_probe boozer_well_conditioned_adjoint \
  "${RESULTS_DIR}/boozer_well_conditioned_adjoint.json" \
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/hf_jobs/cuda_pytest_probe.py" \
    --name boozer_well_conditioned_adjoint \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    --output-json "${RESULTS_DIR}/boozer_well_conditioned_adjoint.json" \
    -- \
    -q \
    tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_exact_well_conditioned_operator_adjoint_cpu_gpu_same_state_parity || OVERALL_RC=1

run_probe reduction_cancellation_stress \
  "${RESULTS_DIR}/reduction_cancellation_stress.json" \
  "${PYTHON_BIN}" "${REPO_ROOT}/benchmarks/hf_jobs/cuda_pytest_probe.py" \
    --name reduction_cancellation_stress \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    --output-json "${RESULTS_DIR}/reduction_cancellation_stress.json" \
    -- \
    -q \
    tests/core/test_reductions.py::test_pairwise_and_compensated_reductions_match_cpu_gpu_on_cancellation_stress || OVERALL_RC=1

"${PYTHON_BIN}" - "${RESULTS_DIR}" "${STAGE2_PLATFORM}" "${SINGLE_STAGE_PLATFORM}" "${REPO_ROOT}" "${EXPECTED_PROBES[@]}" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
stage2_platform = sys.argv[2]
single_stage_platform = sys.argv[3]
repo_root = Path(sys.argv[4])
sys.path.insert(0, str(repo_root))

from benchmarks.validation_ladder_contract import (
    gpu_proof_parity_contract,
    parity_ladder_tolerances,
)

expected = sys.argv[5:]
fake_mode = os.environ.get("SIMSOPT_FAKE_GPU") == "1"
summary = {}
PYTEST_PROBE_LANES = {
    "boozer_well_conditioned_adjoint": "exact_well_conditioned_adjoint",
    "reduction_cancellation_stress": "reduction_cpu_gpu",
}


def _requested_platform_for_probe(probe_name):
    if probe_name.startswith("stage2_"):
        return stage2_platform
    if probe_name.startswith("single_stage_"):
        return single_stage_platform
    if probe_name in PYTEST_PROBE_LANES:
        return single_stage_platform
    return "auto"


def _backend_is_cuda(backend):
    return str(backend).lower() in {"cuda", "gpu"}


def _probe_kind(probe_name):
    if probe_name.startswith("stage2_"):
        return "stage2"
    if probe_name.startswith("single_stage_"):
        return "single_stage"
    return probe_name


def _validate_pytest_probe_parity(probe_name, payload):
    validation_failures = []
    proof_parity = payload.get("proof_parity")
    if not isinstance(proof_parity, dict):
        validation_failures.append("missing proof_parity object")
        return validation_failures
    lane = PYTEST_PROBE_LANES[probe_name]
    if proof_parity.get("lane") != lane:
        validation_failures.append(
            f"proof_parity.lane={proof_parity.get('lane')!r} "
            f"does not match expected lane {lane!r}"
        )
    contract = parity_ladder_tolerances(lane)
    for key, expected_value in contract.items():
        if proof_parity.get(key) != expected_value:
            validation_failures.append(
                f"proof_parity.{key}={proof_parity.get(key)!r} "
                f"does not match ladder contract {expected_value!r}"
            )
    return validation_failures


def _relative_error(actual, reference):
    denominator = abs(float(reference)) + 1e-30
    return abs(float(actual) - float(reference)) / denominator


def _numeric_field(mapping, key, validation_failures):
    if key not in mapping:
        validation_failures.append(f"missing proof_parity.{key}")
        return None
    value = mapping[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        validation_failures.append(f"non-numeric proof_parity.{key}")
        return None
    return float(value)


def _validate_proof_parity(probe_name, payload):
    if probe_name in PYTEST_PROBE_LANES:
        return _validate_pytest_probe_parity(probe_name, payload)

    validation_failures = []
    bundle_provenance = payload.get("bundle_provenance")
    if not isinstance(bundle_provenance, dict):
        validation_failures.append("missing bundle_provenance object")
    elif "fake" not in bundle_provenance:
        validation_failures.append("missing bundle_provenance.fake discriminator")

    proof_parity = payload.get("proof_parity")
    if not isinstance(proof_parity, dict):
        validation_failures.append("missing proof_parity object")
        return validation_failures

    maxiter = None
    provenance = payload.get("provenance")
    if isinstance(provenance, dict) and probe_name.startswith("stage2_"):
        maxiter = provenance.get("maxiter")
    contract = gpu_proof_parity_contract(
        _probe_kind(probe_name),
        maxiter=None if maxiter is None else int(maxiter),
    )
    for key in (
        "value_lane",
        "value_contract_key",
        "gradient_lane",
        "gradient_contract_key",
    ):
        if proof_parity.get(key) != contract[key]:
            validation_failures.append(
                f"proof_parity.{key}={proof_parity.get(key)!r} "
                f"does not match ladder contract {contract[key]!r}"
            )

    cpu_value = _numeric_field(proof_parity, "cpu_oracle_value", validation_failures)
    gpu_value = _numeric_field(proof_parity, "gpu_value", validation_failures)
    value_rel_diff = _numeric_field(
        proof_parity,
        "value_rel_diff",
        validation_failures,
    )
    value_rtol = _numeric_field(proof_parity, "value_rtol", validation_failures)
    gradient_rtol = _numeric_field(
        proof_parity,
        "gradient_rtol",
        validation_failures,
    )
    if value_rtol is not None and value_rtol > float(contract["value_rtol"]):
        validation_failures.append(
            "proof_parity.value_rtol exceeds ladder contract "
            f"{float(contract['value_rtol']):.2e}"
        )
    if gradient_rtol is not None and gradient_rtol > float(contract["gradient_rtol"]):
        validation_failures.append(
            "proof_parity.gradient_rtol exceeds ladder contract "
            f"{float(contract['gradient_rtol']):.2e}"
        )
    if (
        cpu_value is not None
        and gpu_value is not None
        and value_rel_diff is not None
        and value_rtol is not None
    ):
        computed_value_rel_diff = _relative_error(gpu_value, cpu_value)
        if not math.isclose(
            value_rel_diff,
            computed_value_rel_diff,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            validation_failures.append(
                "proof_parity.value_rel_diff does not match CPU/GPU values "
                f"({value_rel_diff:.2e} != {computed_value_rel_diff:.2e})"
            )
        if computed_value_rel_diff > value_rtol:
            validation_failures.append(
                "proof_parity CPU/GPU value rel_diff "
                f"{computed_value_rel_diff:.2e} exceeds value_rtol {value_rtol:.2e}"
            )
    if _probe_kind(probe_name) == "stage2":
        gradient_rel_diff = _numeric_field(
            proof_parity,
            "gradient_rel_diff",
            validation_failures,
        )
        if (
            gradient_rel_diff is not None
            and gradient_rtol is not None
            and gradient_rel_diff > gradient_rtol
        ):
            validation_failures.append(
                "proof_parity.gradient_rel_diff "
                f"{gradient_rel_diff:.2e} exceeds gradient_rtol "
                f"{gradient_rtol:.2e}"
            )
    return validation_failures


def _validate_payload(probe_name, payload):
    validation_failures = []
    provenance = payload.get("provenance")
    bundle_provenance = payload.get("bundle_provenance") or {}
    fake_runner = bool(bundle_provenance.get("fake"))
    if fake_runner:
        if not fake_mode:
            validation_failures.append(
                "fake proof runner payload requires SIMSOPT_FAKE_GPU=1"
            )
        validation_failures.extend(_validate_proof_parity(probe_name, payload))
        return validation_failures

    if not isinstance(provenance, dict):
        validation_failures.append("missing provenance object")
        return validation_failures
    validation_failures.extend(_validate_proof_parity(probe_name, payload))
    if "backend" not in provenance:
        validation_failures.append("missing provenance.backend")
    if not provenance.get("devices"):
        validation_failures.append("missing provenance.devices")
    for key in ("cuda_force_ptx_jit", "cuda_disable_ptx_jit", "xla_flags"):
        if key not in provenance:
            validation_failures.append(f"missing provenance.{key}")
    for key in ("repo_sha", "git_status_short", "worktree_dirty", "command_argv"):
        if key not in provenance:
            validation_failures.append(f"missing provenance.{key}")
    for key in ("peak_rss_mb", "peak_gpu_memory_mb"):
        if key not in provenance:
            validation_failures.append(f"missing provenance.{key}")

    requested_platform = _requested_platform_for_probe(probe_name)
    if requested_platform == "cuda" and not _backend_is_cuda(provenance.get("backend")):
        validation_failures.append(
            "requested CUDA proof initialized backend "
            f"{provenance.get('backend')!r}"
        )
    if requested_platform == "cuda":
        for key in ("cuda_runtime_version", "cuda_driver_version", "nvidia_smi_gpus"):
            if not provenance.get(key):
                validation_failures.append(f"missing provenance.{key}")
        if provenance.get("x64_enabled") is not True:
            validation_failures.append("missing provenance.x64_enabled=True")
    return validation_failures


for probe_name in expected:
    path = results_dir / f"{probe_name}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary[path.name] = {
                "passed": False,
                "elapsed_s": None,
                "failures": [f"corrupt payload: {exc}"],
                "missing_payload": False,
                "corrupt_payload": True,
            }
            continue
        validation_failures = _validate_payload(probe_name, payload)
        failures = list(payload.get("failures") or [])
        failures.extend(validation_failures)
        summary[path.name] = {
            "passed": bool(payload.get("passed")) and not validation_failures,
            "elapsed_s": payload.get("elapsed_s"),
            "failures": failures,
            "missing_payload": False,
            "corrupt_payload": False,
            "provenance": payload.get("provenance"),
            "bundle_provenance": payload.get("bundle_provenance"),
            "comparison": payload.get("comparison"),
            "proof_parity": payload.get("proof_parity"),
        }
        continue
    summary[path.name] = {
        "passed": False,
        "elapsed_s": None,
        "failures": ["missing payload"],
        "missing_payload": True,
        "corrupt_payload": False,
    }
print(json.dumps(summary, indent=2))
if any(not item["passed"] for item in summary.values()):
    raise SystemExit(1)
PY

exit "${OVERALL_RC}"
