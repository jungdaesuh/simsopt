#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HEARTBEAT_INTERVAL_S="${HEARTBEAT_INTERVAL_S:-60}"

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

GEOMETRY_REL_TOL_ARG="${GEOMETRY_REL_TOL}"
if [[ -z "${GEOMETRY_REL_TOL_ARG}" ]]; then
  GEOMETRY_REL_TOL_ARG="__NONE__"
fi
if ! mapfile -t STAGE2_RUNG_NAMES < <(
  python - "${REPO_ROOT}" "${STAGE2_MAXITER}" "${GEOMETRY_REL_TOL_ARG}" <<'PY'
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
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-compilation-cache}"
mkdir -p "${RESULTS_DIR}" "${JAX_COMPILATION_CACHE_DIR}"
unset LD_LIBRARY_PATH

OVERALL_RC=0
declare -a EXPECTED_PROBES=("${STAGE2_RUNG_NAMES[@]}" "single_stage_cold" "single_stage_warm")

emit_payload_summary() {
  local name="$1"
  local output_json="$2"
  python - "${name}" "${output_json}" <<'PY'
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
    sleep "${HEARTBEAT_INTERVAL_S}"
  done
  local rc=0
  if wait "${pid}"; then
    rc=0
  else
    rc=$?
  fi
  local end_ts
  end_ts="$(date +%s)"
  echo "[result] ${name} rc=${rc} wall_s=$((end_ts-start_ts))"
  if [[ -f "${output_json}" ]]; then
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

run_probe stage2_cold "${RESULTS_DIR}/stage2_cold.json" \
  python "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
    --platform "${STAGE2_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --nphi "${STAGE2_NPHI}" \
    --ntheta "${STAGE2_NTHETA}" \
    --maxiter "${STAGE2_MAXITER}" \
    --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
    --output-json "${RESULTS_DIR}/stage2_cold.json" || OVERALL_RC=1

run_probe stage2_warm "${RESULTS_DIR}/stage2_warm.json" \
  python "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
    --platform "${STAGE2_PLATFORM}" \
    --equilibria-dir "${EQUILIBRIA_DIR}" \
    --nphi "${STAGE2_NPHI}" \
    --ntheta "${STAGE2_NTHETA}" \
    --maxiter "${STAGE2_MAXITER}" \
    --optimizer-backend "${STAGE2_OPTIMIZER_BACKEND}" \
    --output-json "${RESULTS_DIR}/stage2_warm.json" || OVERALL_RC=1

if [[ " ${STAGE2_RUNG_NAMES[*]} " == *" stage2_warm_repro "* ]]; then
  run_probe stage2_warm_repro "${RESULTS_DIR}/stage2_warm_repro.json" \
    python "${REPO_ROOT}/benchmarks/stage2_e2e_comparison.py" \
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

run_probe single_stage_cold "${RESULTS_DIR}/single_stage_cold.json" \
  python "${REPO_ROOT}/benchmarks/single_stage_init_parity.py" \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    "${single_stage_probe_args[@]}" \
    --output-json "${RESULTS_DIR}/single_stage_cold.json" || OVERALL_RC=1

run_probe single_stage_warm "${RESULTS_DIR}/single_stage_warm.json" \
  python "${REPO_ROOT}/benchmarks/single_stage_init_parity.py" \
    --platform "${SINGLE_STAGE_PLATFORM}" \
    "${single_stage_probe_args[@]}" \
    --output-json "${RESULTS_DIR}/single_stage_warm.json" || OVERALL_RC=1

python - "${RESULTS_DIR}" "${EXPECTED_PROBES[@]}" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
expected = sys.argv[2:]
summary = {}
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
        summary[path.name] = {
            "passed": payload.get("passed"),
            "elapsed_s": payload.get("elapsed_s"),
            "failures": payload.get("failures"),
            "missing_payload": False,
            "corrupt_payload": False,
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
PY

exit "${OVERALL_RC}"
