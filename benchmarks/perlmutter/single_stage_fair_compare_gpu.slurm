#!/usr/bin/env bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J ss-fair-compare

# Fair same-node CPU-vs-GPU comparison for the scipy-jax-fullgraph (51-dim) lane.
#
# One build on a single GPU node, then the identical fullgraph mpol10 warm-started
# case is evaluated back-to-back under --platform cpu and --platform cuda. Because
# both JAX lanes -- and the co-produced native cpp/CPU reference -- share the exact
# same host, thread count, SHA, seed, and donor warm-start, the only difference is
# the device. So per-iteration throughput and the harness's same-candidate replay
# parity (scipy-jax-fullgraph is replay-capable) are directly comparable, unlike
# the earlier cross-pod numbers where a throttled CPU slice inflated host wall ~50x.
#
# Reference handling: each invocation co-produces the cpp reference itself; there
# is no separate reference run. See the Fair-comparison protocol in
# docs/scipy_jax_11_51_matrix_implementation_plan.md.
#
# NOTE: JAX compile diagnostics are not enabled here -- the harness does not expose
# enable_compile_diagnostics as a CLI flag (single_stage_init_parity.py keeps it
# default-False), so compile-vs-steady-state separation currently comes from the
# harness timing fields. Wiring a real compile-diagnostics flag is a follow-up.

set -euo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch; do not run it on a login node}"
: "${REPO_ROOT:?set REPO_ROOT to the clean source checkout}"
: "${RUN_ROOT:?set RUN_ROOT to a writable scratch run directory}"
: "${FAIR_WARM_START_RUN_DIR:?set FAIR_WARM_START_RUN_DIR to the continuation donor warm-start dir (mpol10 requires it)}"

JOB_ROOT="${RUN_ROOT}/${SLURM_JOB_ID}"
ENV_ROOT="${JOB_ROOT}/venv"
LOG_DIR="${JOB_ROOT}/logs"
ARTIFACT_DIR="${JOB_ROOT}/artifacts"
FIXTURE_STAGE2_BS="benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json"
JAX_GPU_WHEEL_SPEC="${JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.10.0}"
# mpol10 production resolution + budgets (match the matrix mpol10 tier).
FAIR_MPOL="${FAIR_MPOL:-10}"
FAIR_NTOR="${FAIR_NTOR:-10}"
FAIR_NPHI="${FAIR_NPHI:-64}"
FAIR_NTHETA="${FAIR_NTHETA:-32}"
FAIR_MAXITER="${FAIR_MAXITER:-1500}"
FAIR_BOOZER_BFGS_MAXITER="${FAIR_BOOZER_BFGS_MAXITER:-1500}"
FAIR_NEWTON_MAXITER="${FAIR_NEWTON_MAXITER:-50}"
FAIR_NEWTON_POLISH_POLICY="${FAIR_NEWTON_POLISH_POLICY:-run}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-36000}"
TARGET_CASE_TIMEOUT_SECONDS="${TARGET_CASE_TIMEOUT_SECONDS:-36000}"

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

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"
exec > >(tee "${LOG_DIR}/driver_stdout.log")
exec 2> >(tee "${LOG_DIR}/driver_stderr.log" >&2)

cd "${REPO_ROOT}"
module load python
python3 -m venv "${ENV_ROOT}"
. "${ENV_ROOT}/bin/activate"

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT:-1.9.4.dev0}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "${JAX_GPU_WHEEL_SPEC}" etils
python -m pip install -e ".[JAX_GPU,test,ALGS]" "shapely>=2.1,<3" "numba>=0.64,<0.66"
python -m pip freeze > "${JOB_ROOT}/pip_freeze.txt"

git rev-parse HEAD > "${JOB_ROOT}/repo_sha.txt"
git status --short --branch > "${JOB_ROOT}/git_status_short_branch.txt"
git diff --no-ext-diff HEAD > "${JOB_ROOT}/dirty.patch"
if [[ -s "${JOB_ROOT}/dirty.patch" ]]; then
  echo "tracked worktree must be clean for fair-comparison signoff" >&2
  exit 1
fi
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "${SOURCE_STATUS}" ]]; then
  echo "non-clean source state would invalidate fair-comparison signoff" >&2
  printf '%s\n' "${SOURCE_STATUS}" >&2
  exit 1
fi

# Common env shared by both lanes so ONLY the device differs.
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD=disallow
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

python - <<'PY' > "${JOB_ROOT}/simsoptpp_curve_smoke.txt"
from simsoptpp import Curve

print(Curve)
PY

run_lane() {
  # $1 = platform (cpu|cuda). Everything else is identical across lanes.
  local platform="$1"
  local step_dir="${ARTIFACT_DIR}/single_stage_fair_${platform}"
  mkdir -p "${step_dir}"
  local gpu_monitor_pid=""
  local srun_gpu=()

  if [[ "${platform}" == "cuda" ]]; then
    export SIMSOPT_BACKEND_MODE=jax_gpu_parity
    export SIMSOPT_JAX_PLATFORM=cuda
    export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
    export JAX_PLATFORMS=cuda,cpu
    export XLA_FLAGS="--xla_gpu_exclude_nondeterministic_ops=true"
    unset CUDA_VISIBLE_DEVICES
    srun_gpu=(--gpus-per-task=1)
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv -l 5 > "${step_dir}/gpu_memory.csv" &
    gpu_monitor_pid="$!"
  else
    export SIMSOPT_BACKEND_MODE=jax_cpu_parity
    export SIMSOPT_JAX_PLATFORM=cpu
    export JAX_PLATFORMS=cpu
    export CUDA_VISIBLE_DEVICES=""
    unset SIMSOPT_JAX_CUDA_LIBRARY_MODE
    unset XLA_FLAGS
  fi

  set +e
  /usr/bin/time -v \
    srun -n 1 -c "${SLURM_CPUS_PER_TASK:-32}" "${srun_gpu[@]}" \
      python benchmarks/single_stage_init_parity.py \
        --platform "${platform}" \
        --optimizer-backend scipy-jax-fullgraph \
        --boozer-optimizer-backend ondevice \
        --reference-optimizer-method lbfgs \
        --maxiter "${FAIR_MAXITER}" \
        --target-lane-boozer-bfgs-maxiter "${FAIR_BOOZER_BFGS_MAXITER}" \
        --mpol "${FAIR_MPOL}" --ntor "${FAIR_NTOR}" \
        --nphi "${FAIR_NPHI}" --ntheta "${FAIR_NTHETA}" \
        --warm-start-run-dir "${FAIR_WARM_START_RUN_DIR}" \
        --target-lane-boozer-newton-polish-policy "${FAIR_NEWTON_POLISH_POLICY}" \
        --target-lane-boozer-newton-maxiter "${FAIR_NEWTON_MAXITER}" \
        --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
        --case-artifacts-dir "${step_dir}/cases" \
        --single-stage-case-timeout-seconds "${CASE_TIMEOUT_SECONDS}" \
        --single-stage-target-case-timeout-seconds "${TARGET_CASE_TIMEOUT_SECONDS}" \
        --output-json "${step_dir}/single_stage_fair_${platform}.json" \
    > "${step_dir}/stdout.log" \
    2> "${step_dir}/stderr_time.log"
  local status="$?"
  set -e
  echo "${status}" > "${step_dir}/exit_status.txt"

  if [[ -n "${gpu_monitor_pid}" ]]; then
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true
  fi
  echo "lane ${platform}: exit=${status}"
  return 0
}

nvidia-smi > "${JOB_ROOT}/nvidia_smi_start.txt" || true

# CPU lane first (clean host measurement), then the CUDA lane on the same node.
run_lane cpu
run_lane cuda

nvidia-smi > "${JOB_ROOT}/nvidia_smi_end.txt" || true

cpu_rc="$(cat "${ARTIFACT_DIR}/single_stage_fair_cpu/exit_status.txt" 2>/dev/null || echo missing)"
cuda_rc="$(cat "${ARTIFACT_DIR}/single_stage_fair_cuda/exit_status.txt" 2>/dev/null || echo missing)"
echo "fair-compare done: cpu_rc=${cpu_rc} cuda_rc=${cuda_rc}"

# Fail the job if either lane failed, so a half-broken comparison is not green.
[[ "${cpu_rc}" == "0" && "${cuda_rc}" == "0" ]]
