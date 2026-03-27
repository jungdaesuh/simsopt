#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
GEOMETRY_REL_TOL="5e-6"
SINGLE_STAGE_NPHI="255"
SINGLE_STAGE_NTHETA="64"
SINGLE_STAGE_MPOL="8"
SINGLE_STAGE_NTOR="6"
SINGLE_STAGE_MAXITER="300"
SINGLE_STAGE_OPTIMIZER_BACKEND="ondevice"
SINGLE_STAGE_BOOZER_OPTIMIZER_BACKEND="scipy"

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

export HF_HUB_DISABLE_TELEMETRY=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-compilation-cache}"
mkdir -p "${RESULTS_DIR}" "${JAX_COMPILATION_CACHE_DIR}"
unset LD_LIBRARY_PATH

run_probe() {
  local name="$1"
  local output_json="$2"
  shift 2

  echo "=== ${name} ==="
  local start_ts
  start_ts="$(date +%s)"
  "$@" &
  local pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    echo "[heartbeat] ${name} $(date -u +%FT%TZ)"
    sleep 60
  done
  wait "${pid}"
  local rc=$?
  local end_ts
  end_ts="$(date +%s)"
  echo "[result] ${name} rc=${rc} wall_s=$((end_ts-start_ts))"
  cat "${output_json}"
  python - "${output_json}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print({"passed": payload.get("passed"), "failures": payload.get("failures")})
PY
  return "${rc}"
}

run_probe stage2_cold "${RESULTS_DIR}/stage2_cold.json" \
  python "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
    --platform "${STAGE2_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --nphi "${STAGE2_NPHI}" \
    --ntheta "${STAGE2_NTHETA}" \
    --maxiter "${STAGE2_MAXITER}" \
    --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
    --output-json "${RESULTS_DIR}/stage2_cold.json"

run_probe stage2_warm "${RESULTS_DIR}/stage2_warm.json" \
  python "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
    --platform "${STAGE2_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --nphi "${STAGE2_NPHI}" \
    --ntheta "${STAGE2_NTHETA}" \
    --maxiter "${STAGE2_MAXITER}" \
    --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
    --output-json "${RESULTS_DIR}/stage2_warm.json"

run_probe stage2_warm_geometry_gate "${RESULTS_DIR}/stage2_warm_geometry_gate.json" \
  python "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
    --platform "${STAGE2_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --nphi "${STAGE2_NPHI}" \
    --ntheta "${STAGE2_NTHETA}" \
    --maxiter "${STAGE2_MAXITER}" \
    --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
    --geometry-rel-tol "${GEOMETRY_REL_TOL}" \
    --output-json "${RESULTS_DIR}/stage2_warm_geometry_gate.json"

run_probe single_stage_cold "${RESULTS_DIR}/single_stage_cold.json" \
  python "${REPO_ROOT}/benchmarks/single_stage_init_parity.py" \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --plasma-surf-filename "${PLASMA_SURF_FILENAME}" \
    --stage2-bs-path "${STAGE2_BS_PATH}" \
    --nphi "${SINGLE_STAGE_NPHI}" \
    --ntheta "${SINGLE_STAGE_NTHETA}" \
    --mpol "${SINGLE_STAGE_MPOL}" \
    --ntor "${SINGLE_STAGE_NTOR}" \
    --optimizer-backend "${SINGLE_STAGE_OPTIMIZER_BACKEND}" \
    --boozer-optimizer-backend "${SINGLE_STAGE_BOOZER_OPTIMIZER_BACKEND}" \
    --maxiter "${SINGLE_STAGE_MAXITER}" \
    --output-json "${RESULTS_DIR}/single_stage_cold.json"

run_probe single_stage_warm "${RESULTS_DIR}/single_stage_warm.json" \
  python "${REPO_ROOT}/benchmarks/single_stage_init_parity.py" \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --plasma-surf-filename "${PLASMA_SURF_FILENAME}" \
    --stage2-bs-path "${STAGE2_BS_PATH}" \
    --nphi "${SINGLE_STAGE_NPHI}" \
    --ntheta "${SINGLE_STAGE_NTHETA}" \
    --mpol "${SINGLE_STAGE_MPOL}" \
    --ntor "${SINGLE_STAGE_NTOR}" \
    --optimizer-backend "${SINGLE_STAGE_OPTIMIZER_BACKEND}" \
    --boozer-optimizer-backend "${SINGLE_STAGE_BOOZER_OPTIMIZER_BACKEND}" \
    --maxiter "${SINGLE_STAGE_MAXITER}" \
    --output-json "${RESULTS_DIR}/single_stage_warm.json"

python - "${RESULTS_DIR}" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
summary = {}
for path in sorted(results_dir.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary[path.name] = {
        "passed": payload.get("passed"),
        "elapsed_s": payload.get("elapsed_s"),
        "failures": payload.get("failures"),
    }
print(json.dumps(summary, indent=2))
PY
