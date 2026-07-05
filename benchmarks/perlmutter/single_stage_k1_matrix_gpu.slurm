#!/usr/bin/env bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J ss-k1-matrix

set -euo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch; do not run it on a login node}"
: "${REPO_ROOT:?set REPO_ROOT to the clean-local checkout}"
: "${RUN_ROOT:?set RUN_ROOT to a writable scratch run directory}"

JOB_ROOT="${RUN_ROOT}/${SLURM_JOB_ID}"
ENV_ROOT="${JOB_ROOT}/venv"
LOG_DIR="${JOB_ROOT}/logs"
ARTIFACT_DIR="${JOB_ROOT}/artifacts"
FIXTURE_STAGE2_BS="${FIXTURE_STAGE2_BS:-benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json}"
JAX_GPU_WHEEL_SPEC="${JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.10.0}"
MATRIX_MAXITER="${MATRIX_MAXITER:-1}"
MATRIX_OUTER_MAXLS="${MATRIX_OUTER_MAXLS:-4}"
MATRIX_INITIAL_STEP_SCALE="${MATRIX_INITIAL_STEP_SCALE:-0.35}"
MATRIX_BOOZER_BFGS_MAXITER="${MATRIX_BOOZER_BFGS_MAXITER:-1500}"
MATRIX_TRIAL_BOOZER_BFGS_MAXITER="${MATRIX_TRIAL_BOOZER_BFGS_MAXITER:-}"
MATRIX_TRIAL_BOOZER_BFGS_TOL="${MATRIX_TRIAL_BOOZER_BFGS_TOL:-}"
MATRIX_NEWTON_MAXITER="${MATRIX_NEWTON_MAXITER:-50}"
MATRIX_TRIAL_NEWTON_POLISH_POLICY="${MATRIX_TRIAL_NEWTON_POLISH_POLICY:-run}"
MATRIX_LSMR_NEWTON_STAB="${MATRIX_LSMR_NEWTON_STAB:-1e-10}"
# Reference-leg-only override for the child's traceable Newton linear solver. The
# harness reads this from the environment (not a CLI flag) and applies it to the
# jax-CPU reference leg alone; the jax-GPU target leg keeps its own selector. Empty
# means unset. Accepted codes are resolved by the child (operator_gmres, dense_lu,
# hybrid_final_dense_lu, hybrid_final_dense_ir), which fails loud on an unknown value.
export MATRIX_REFERENCE_NEWTON_LINEAR_SOLVER="${MATRIX_REFERENCE_NEWTON_LINEAR_SOLVER:-}"
MATRIX_MPOL="${MATRIX_MPOL:-10}"
MATRIX_NTOR="${MATRIX_NTOR:-10}"
MATRIX_NPHI="${MATRIX_NPHI:-64}"
MATRIX_NTHETA="${MATRIX_NTHETA:-32}"
MATRIX_JAX_CACHE_DIR="${MATRIX_JAX_CACHE_DIR:-${RUN_ROOT}/jax_compilation_cache}"
MATRIX_WARM_START_RUN_DIR="${MATRIX_WARM_START_RUN_DIR:-}"
MATRIX_IOTA_TARGET="${MATRIX_IOTA_TARGET:-}"
MATRIX_VOL_TARGET="${MATRIX_VOL_TARGET:-}"
MATRIX_CASES="${MATRIX_CASES:-all}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-7200}"
TARGET_CASE_TIMEOUT_SECONDS="${TARGET_CASE_TIMEOUT_SECONDS:-7200}"
MATRIX_DISABLE_SUCCESS_FILTER="${MATRIX_DISABLE_SUCCESS_FILTER:-0}"
MATRIX_ASSERT_K1_PROGRESS="${MATRIX_ASSERT_K1_PROGRESS:-0}"
MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE="${MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE:-0}"
MATRIX_TRACE_K1_SUBTIMERS="${MATRIX_TRACE_K1_SUBTIMERS:-0}"
MATRIX_REQUIRE_TRACE_REUSE="${MATRIX_REQUIRE_TRACE_REUSE:-0}"
MATRIX_REQUIRE_FINAL_SYNC_REUSE="${MATRIX_REQUIRE_FINAL_SYNC_REUSE:-0}"

repo_root_real="$(realpath "${REPO_ROOT}")"
run_root_real="$(realpath -m "${RUN_ROOT}")"
case "${run_root_real}/" in
  "${repo_root_real}/"*)
    echo "RUN_ROOT must be outside REPO_ROOT for clean-source signoff" >&2
    echo "REPO_ROOT=${repo_root_real}" >&2
    echo "RUN_ROOT=${run_root_real}" >&2
    exit 1
    ;;
esac

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}" "${MATRIX_JAX_CACHE_DIR}"
exec > >(tee "${LOG_DIR}/driver_stdout.log")
exec 2> >(tee "${LOG_DIR}/driver_stderr.log" >&2)

cd "${REPO_ROOT}"
module load python
python3 -m venv "${ENV_ROOT}"
. "${ENV_ROOT}/bin/activate"

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT:-1.9.4.dev0}"
python -m pip install --no-compile --upgrade pip setuptools wheel
python -m pip install --no-compile --upgrade "${JAX_GPU_WHEEL_SPEC}" etils
python -m pip install --no-compile -e ".[JAX_GPU,test,ALGS]" "shapely>=2.1,<3" "numba>=0.64,<0.66"
python -m pip freeze > "${JOB_ROOT}/pip_freeze.txt"

git rev-parse HEAD > "${JOB_ROOT}/repo_sha.txt"
git status --short --branch > "${JOB_ROOT}/git_status_short_branch.txt"
git status --short > "${JOB_ROOT}/git_status_short.txt"
git diff --no-ext-diff HEAD > "${JOB_ROOT}/dirty.patch"
git diff --stat HEAD > "${JOB_ROOT}/dirty-diff-stat.txt"

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD=disallow
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export SIMSOPT_JAX_PLATFORM=cuda
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export JAX_COMPILATION_CACHE_DIR="${MATRIX_JAX_CACHE_DIR}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
unset LD_LIBRARY_PATH

python - <<'PY' > "${JOB_ROOT}/runtime_versions.json"
import importlib.metadata as metadata
import json
import os

import jax
import jaxlib

payload = {
    "env": {
        key: os.environ.get(key)
        for key in (
            "JAX_COMPILATION_CACHE_DIR",
            "JAX_ENABLE_X64",
            "JAX_PLATFORMS",
            "SIMSOPT_BACKEND_MODE",
            "SIMSOPT_BACKEND_STRICT",
            "SIMSOPT_JAX_TRANSFER_GUARD",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
        )
    },
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "jax_cuda12_plugin": metadata.version("jax-cuda12-plugin"),
    "jax_cuda12_pjrt": metadata.version("jax-cuda12-pjrt"),
    "default_backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
    "x64": bool(jax.config.read("jax_enable_x64")),
}
if payload["default_backend"] not in {"cuda", "gpu"}:
    raise SystemExit(f"expected cuda/gpu backend, got {payload['default_backend']}")
if not payload["x64"]:
    raise SystemExit("expected JAX x64 mode")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

write_case_manifest() {
  local case_label="$1"
  local chunk_batch="$2"
  local limited_memory="$3"
  local newton_policy="$4"
  local adjoint_solver="$5"
  local newton_stab="$6"
  python - "$case_label" "$chunk_batch" "$limited_memory" "$newton_policy" "$adjoint_solver" "$newton_stab" "${MATRIX_TRIAL_NEWTON_POLISH_POLICY}" <<'PY'
import json
import os
import sys

(
    label,
    chunk,
    limited_memory,
    newton_policy,
    adjoint_solver,
    newton_stab,
    trial_newton_policy,
) = sys.argv[1:]
payload = {
    "label": label,
    "SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE": int(chunk),
    "boozer_limited_memory": limited_memory == "1",
    "target_lane_boozer_newton_polish_policy": newton_policy,
    "target_lane_trial_boozer_newton_polish_policy": trial_newton_policy,
    "SIMSOPT_ADJOINT_LINEAR_SOLVER": adjoint_solver,
    "target_lane_boozer_newton_stab": None if newton_stab == "" else float(newton_stab),
    "MATRIX_WARM_START_RUN_DIR": os.environ.get("MATRIX_WARM_START_RUN_DIR") or None,
    "MATRIX_IOTA_TARGET": os.environ.get("MATRIX_IOTA_TARGET") or None,
    "MATRIX_VOL_TARGET": os.environ.get("MATRIX_VOL_TARGET") or None,
    "MATRIX_TRIAL_BOOZER_BFGS_TOL": (
        os.environ.get("MATRIX_TRIAL_BOOZER_BFGS_TOL") or None
    ),
    "MATRIX_TRIAL_BOOZER_BFGS_MAXITER": (
        os.environ.get("MATRIX_TRIAL_BOOZER_BFGS_MAXITER") or None
    ),
    "MATRIX_ASSERT_K1_PROGRESS": os.environ.get("MATRIX_ASSERT_K1_PROGRESS") or None,
    "MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE": (
        os.environ.get("MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE") or None
    ),
    "MATRIX_TRACE_K1_SUBTIMERS": os.environ.get("MATRIX_TRACE_K1_SUBTIMERS") or None,
    "MATRIX_REQUIRE_TRACE_REUSE": os.environ.get("MATRIX_REQUIRE_TRACE_REUSE") or None,
    "MATRIX_REQUIRE_FINAL_SYNC_REUSE": (
        os.environ.get("MATRIX_REQUIRE_FINAL_SYNC_REUSE") or None
    ),
}

print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

assert_case_progress() {
  local case_label="$1"
  local case_dir="${ARTIFACT_DIR}/${case_label}"
  if [[ "${MATRIX_ASSERT_K1_PROGRESS}" != "1" ]]; then
    return 0
  fi
  local progress_json
  local progress_jsons=()
  while IFS= read -r progress_json; do
    # K1 forward events are decomposed-lane telemetry. Native lanes
    # (cpu_outputs, seed_outputs) legitimately record none, so requiring
    # them there would fail healthy fixture/native-reference runs.
    case "${progress_json}" in
      */reference_outputs/*|*/target_outputs/*)
        progress_jsons+=("${progress_json}")
        ;;
    esac
  done < <(find "${case_dir}/cases" -name outer_optimizer_progress.json -type f | sort)
  if [[ "${#progress_jsons[@]}" == "0" ]]; then
    echo "Missing decomposed-lane outer_optimizer_progress.json for ${case_label}" >&2
    return 1
  fi
  local expect_skip="false"
  if [[ "${MATRIX_TRIAL_NEWTON_POLISH_POLICY}" == "skip" ]]; then
    expect_skip="true"
  fi
  local assertion_dir="${case_dir}/k1_progress_assertions"
  mkdir -p "${assertion_dir}"
  local assertion_status=0
  local progress_label
  local assertion_args
  for progress_json in "${progress_jsons[@]}"; do
    progress_label="$(
      python - "${case_dir}" "${progress_json}" <<'PY'
import pathlib
import re
import sys

case_dir = pathlib.Path(sys.argv[1])
progress_json = pathlib.Path(sys.argv[2])
relative_parts = progress_json.relative_to(case_dir).parts
label = relative_parts[1] if len(relative_parts) >= 4 and relative_parts[0] == "cases" else progress_json.parent.name
print(re.sub(r"[^A-Za-z0-9_.-]+", "_", label))
PY
    )"
    assertion_args=(
      --progress-json "${progress_json}"
      --require-k1-forward-events
      --expect-newton-skip "${expect_skip}"
      --output-json "${assertion_dir}/${progress_label}.json"
    )
    if [[ "${MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE}" == "1" ]]; then
      assertion_args+=(--require-k1-per-objective)
    fi
    if [[ "${MATRIX_REQUIRE_TRACE_REUSE}" == "1" ]]; then
      assertion_args+=(--require-trace-forward-reuse)
    fi
    if [[ "${MATRIX_REQUIRE_FINAL_SYNC_REUSE}" == "1" ]]; then
      assertion_args+=(--require-final-sync-reuse)
    fi
    if ! python benchmarks/single_stage_k1_progress_assertions.py "${assertion_args[@]}"; then
      assertion_status=1
    fi
  done
  return "${assertion_status}"
}

run_case() {
  local case_label="$1"
  local chunk_batch="$2"
  local limited_memory="$3"
  local newton_policy="$4"
  local adjoint_solver="$5"
  local newton_stab="$6"
  local case_dir="${ARTIFACT_DIR}/${case_label}"
  mkdir -p "${case_dir}"
  write_case_manifest \
    "${case_label}" \
    "${chunk_batch}" \
    "${limited_memory}" \
    "${newton_policy}" \
    "${adjoint_solver}" \
    "${newton_stab}" > "${case_dir}/case_manifest.json"

  export SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE="${chunk_batch}"
  export SIMSOPT_ADJOINT_LINEAR_SOLVER="${adjoint_solver}"
  local limited_memory_args=()
  if [[ "${limited_memory}" == "1" ]]; then
    limited_memory_args=(--boozer-limited-memory)
  else
    limited_memory_args=(--no-boozer-limited-memory)
  fi
  local success_filter_args=()
  if [[ "${MATRIX_DISABLE_SUCCESS_FILTER}" == "1" ]]; then
    success_filter_args=(--disable-target-lane-success-filter)
  fi
  local trace_args=()
  if [[ "${MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE}" == "1" ]]; then
    trace_args=(--record-objective-evaluation-trace)
  fi
  local subtimer_args=()
  if [[ "${MATRIX_TRACE_K1_SUBTIMERS}" == "1" ]]; then
    subtimer_args=(--trace-target-lane-k1-subtimers)
  fi
  local newton_stab_args=()
  if [[ -n "${newton_stab}" ]]; then
    newton_stab_args=(--target-lane-boozer-newton-stab "${newton_stab}")
  fi
  local trial_bfgs_args=()
  if [[ -n "${MATRIX_TRIAL_BOOZER_BFGS_TOL}" ]]; then
    trial_bfgs_args+=(
      --target-lane-trial-boozer-bfgs-tol
      "${MATRIX_TRIAL_BOOZER_BFGS_TOL}"
    )
  fi
  if [[ -n "${MATRIX_TRIAL_BOOZER_BFGS_MAXITER}" ]]; then
    trial_bfgs_args+=(
      --target-lane-trial-boozer-bfgs-maxiter
      "${MATRIX_TRIAL_BOOZER_BFGS_MAXITER}"
    )
  fi
  local warm_start_args=()
  if [[ -n "${MATRIX_WARM_START_RUN_DIR}" ]]; then
    warm_start_args=(--warm-start-run-dir "${MATRIX_WARM_START_RUN_DIR}")
  fi
  local target_args=()
  if [[ -n "${MATRIX_VOL_TARGET}" ]]; then
    target_args+=(--vol-target "${MATRIX_VOL_TARGET}")
  fi
  if [[ -n "${MATRIX_IOTA_TARGET}" ]]; then
    target_args+=(--iota-target "${MATRIX_IOTA_TARGET}")
  fi

  echo "=== ${case_label} ==="
  nvidia-smi > "${case_dir}/nvidia_smi_before.txt" || true
  set +e
  /usr/bin/time -v \
    srun -n 1 -c "${SLURM_CPUS_PER_TASK:-32}" --gpus-per-task=1 \
      python benchmarks/single_stage_init_parity.py \
        --platform cuda \
        --optimizer-backend scipy-jax-decomposed \
        --boozer-optimizer-backend ondevice \
        "${limited_memory_args[@]}" \
        --record-jax-compile-diagnostics \
        "${subtimer_args[@]}" \
        "${trace_args[@]}" \
        --benchmark-mode \
        --minimal-artifacts \
        "${success_filter_args[@]}" \
        "${warm_start_args[@]}" \
        "${target_args[@]}" \
        --reference-optimizer-method lbfgs \
        --maxiter "${MATRIX_MAXITER}" \
        --initial-step-scale "${MATRIX_INITIAL_STEP_SCALE}" \
        --outer-maxls "${MATRIX_OUTER_MAXLS}" \
        --target-lane-boozer-bfgs-maxiter "${MATRIX_BOOZER_BFGS_MAXITER}" \
        "${trial_bfgs_args[@]}" \
        --target-lane-boozer-newton-polish-policy "${newton_policy}" \
        --target-lane-trial-boozer-newton-polish-policy "${MATRIX_TRIAL_NEWTON_POLISH_POLICY}" \
        --target-lane-boozer-newton-maxiter "${MATRIX_NEWTON_MAXITER}" \
        "${newton_stab_args[@]}" \
        --mpol "${MATRIX_MPOL}" \
        --ntor "${MATRIX_NTOR}" \
        --nphi "${MATRIX_NPHI}" \
        --ntheta "${MATRIX_NTHETA}" \
        --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
        --case-artifacts-dir "${case_dir}/cases" \
        --single-stage-case-timeout-seconds "${CASE_TIMEOUT_SECONDS}" \
        --single-stage-target-case-timeout-seconds "${TARGET_CASE_TIMEOUT_SECONDS}" \
        --output-json "${case_dir}/summary.json" \
    > "${case_dir}/stdout.log" \
    2> "${case_dir}/stderr_time.log"
  local status="$?"
  set -e
  echo "${status}" > "${case_dir}/exit_status.txt"
  nvidia-smi > "${case_dir}/nvidia_smi_after.txt" || true
  if [[ "${status}" == "0" ]]; then
    set +e
    assert_case_progress "${case_label}" \
      > "${case_dir}/k1_progress_assertions_stdout.log" \
      2> "${case_dir}/k1_progress_assertions_stderr.log"
    local assertion_status="$?"
    set -e
    echo "${assertion_status}" > "${case_dir}/k1_progress_assertions_exit_status.txt"
    if [[ "${assertion_status}" != "0" ]]; then
      status="${assertion_status}"
      echo "${status}" > "${case_dir}/exit_status.txt"
    fi
  fi
  return 0
}

case_enabled() {
  local case_label="$1"
  local selected_case
  if [[ -z "${MATRIX_CASES}" || "${MATRIX_CASES}" == "all" ]]; then
    return 0
  fi
  for selected_case in ${MATRIX_CASES//,/ }; do
    if [[ "${selected_case}" == "${case_label}" ]]; then
      return 0
    fi
  done
  return 1
}

maybe_run_case() {
  local case_label="$1"
  if case_enabled "${case_label}"; then
    run_case "$@"
  else
    echo "=== ${case_label} skipped by MATRIX_CASES=${MATRIX_CASES} ==="
    mkdir -p "${ARTIFACT_DIR}/${case_label}"
    printf 'MATRIX_CASES=%s\n' "${MATRIX_CASES}" > "${ARTIFACT_DIR}/${case_label}/skip_reason.txt"
  fi
}

nvidia-smi > "${JOB_ROOT}/nvidia_smi_start.txt" || true

maybe_run_case baseline_dense_run_chunk8 8 0 run dense ""
maybe_run_case lbfgs_run_chunk8 8 1 run dense ""
maybe_run_case dense_skip_chunk8 8 0 skip dense ""
maybe_run_case dense_run_lsmrj_chunk8 8 0 run lsmr_j "${MATRIX_LSMR_NEWTON_STAB}"
maybe_run_case dense_run_chunk16 16 0 run dense ""

nvidia-smi > "${JOB_ROOT}/nvidia_smi_end.txt" || true

python - <<'PY' "${ARTIFACT_DIR}" > "${ARTIFACT_DIR}/matrix_exit_summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {}
for path in sorted(root.glob("*/exit_status.txt")):
    payload[path.parent.name] = int(path.read_text().strip())
print(json.dumps(payload, indent=2, sort_keys=True))
PY

cat "${ARTIFACT_DIR}/matrix_exit_summary.json"
python -m benchmarks.single_stage_k1_matrix_report \
  --artifact-dir "${ARTIFACT_DIR}" \
  --output-json "${ARTIFACT_DIR}/k1_matrix_report.json"
cat "${ARTIFACT_DIR}/k1_matrix_report.json"
python - <<'PY' "${ARTIFACT_DIR}/matrix_exit_summary.json"
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not summary:
    raise SystemExit("MATRIX_CASES did not select any runnable cases")
failed = {label: status for label, status in summary.items() if status != 0}
if failed:
    print("matrix cases failed:", json.dumps(failed, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)
PY
