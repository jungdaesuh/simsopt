#!/usr/bin/env bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J banana-e2e-cpu-gpu

set -euo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch; do not run it on a login node}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to a writable scratch results directory}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
JOB_RESULTS="${RESULTS_ROOT}/${SLURM_JOB_ID}"
ENV_ROOT="${ENV_ROOT:-${JOB_RESULTS}/venv}"
JAX_GPU_WHEEL_SPEC="${JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.10.0}"
JAX_CACHE_WHEEL_SPEC="${JAX_CACHE_WHEEL_SPEC:-etils}"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT:-1.9.4.dev0}"
LOG_DIR="${JOB_RESULTS}/logs"
ARTIFACT_DIR="${JOB_RESULTS}/artifacts"
BENCHMARK_TIMING_CLASSIFICATION="${BENCHMARK_TIMING_CLASSIFICATION:-mixed-parity-reference}"
BENCHMARK_SUPPORTS_PERFORMANCE_HEADLINE="${BENCHMARK_SUPPORTS_PERFORMANCE_HEADLINE:-0}"
BENCHMARK_HEADLINE_TIMING_CLASSIFICATION="${BENCHMARK_HEADLINE_TIMING_CLASSIFICATION:-}"
FIXTURE_DIR="benchmarks/fixtures/single_stage_seed_iota15"
FIXTURE_STAGE2_BS="${FIXTURE_DIR}/biot_savart_opt.json"

repo_root_real="$(realpath "${REPO_ROOT}")"
results_root_real="$(realpath -m "${RESULTS_ROOT}")"
case "${results_root_real}/" in
  "${repo_root_real}/"*)
    echo "RESULTS_ROOT must be outside REPO_ROOT for clean-source signoff" >&2
    echo "REPO_ROOT=${repo_root_real}" >&2
    echo "RESULTS_ROOT=${results_root_real}" >&2
    exit 1
    ;;
esac

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"
printf '%s\n' "${BENCHMARK_TIMING_CLASSIFICATION}" > "${JOB_RESULTS}/timing_classification.txt"
printf '{"timing_classification":"%s","supports_performance_headline":%s,"headline_timing_classification":"%s"}\n' \
  "${BENCHMARK_TIMING_CLASSIFICATION}" \
  "${BENCHMARK_SUPPORTS_PERFORMANCE_HEADLINE}" \
  "${BENCHMARK_HEADLINE_TIMING_CLASSIFICATION}" \
  > "${JOB_RESULTS}/benchmark_timing_label.json"
exec > >(tee "${LOG_DIR}/driver_stdout.log")
exec 2> >(tee "${LOG_DIR}/driver_stderr.log" >&2)
cd "${REPO_ROOT}"

module load python

python3 -m venv "${ENV_ROOT}"
. "${ENV_ROOT}/bin/activate"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "${JAX_GPU_WHEEL_SPEC}" "${JAX_CACHE_WHEEL_SPEC}"
python -m pip install -e ".[JAX_GPU,test,ALGS]" "shapely>=2.1,<3" "numba>=0.64,<0.66"
python -m pip freeze > "${JOB_RESULTS}/pip_freeze.txt"

git rev-parse HEAD > "${JOB_RESULTS}/repo_sha.txt"
git status --short > "${JOB_RESULTS}/git_status_short.txt"
git status --short --branch > "${JOB_RESULTS}/git_status_short_branch.txt"
git diff --no-ext-diff HEAD > "${JOB_RESULTS}/dirty.patch"
git diff --stat HEAD > "${JOB_RESULTS}/dirty-diff-stat.txt"
if [[ -s "${JOB_RESULTS}/dirty.patch" ]]; then
  echo "tracked worktree must be clean for benchmark signoff" >&2
  cat "${JOB_RESULTS}/dirty-diff-stat.txt" >&2
  exit 1
fi
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "${SOURCE_STATUS}" ]]; then
  echo "non-clean source state would invalidate benchmark signoff" >&2
  printf '%s\n' "${SOURCE_STATUS}" >&2
  exit 1
fi

export SLURM_CPU_BIND="cores"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD=disallow
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export SIMSOPT_JAX_PLATFORM=cuda
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

python - <<'PY' > "${JOB_RESULTS}/simsoptpp_curve_smoke.txt"
from simsoptpp import Curve

print(Curve)
PY

nvidia-smi > "${JOB_RESULTS}/nvidia_smi_start.txt"

python - <<'PY' > "${JOB_RESULTS}/runtime_versions.json"
import importlib.metadata as metadata
import json
import os

import jax
import jaxlib

payload = {
    "env": {
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "SIMSOPT_BACKEND_MODE": os.environ.get("SIMSOPT_BACKEND_MODE"),
        "SIMSOPT_BACKEND_STRICT": os.environ.get("SIMSOPT_BACKEND_STRICT"),
        "SIMSOPT_JAX_TRANSFER_GUARD": os.environ.get(
            "SIMSOPT_JAX_TRANSFER_GUARD"
        ),
        "SIMSOPT_JAX_CUDA_LIBRARY_MODE": os.environ.get(
            "SIMSOPT_JAX_CUDA_LIBRARY_MODE"
        ),
        "SIMSOPT_JAX_PLATFORM": os.environ.get("SIMSOPT_JAX_PLATFORM"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
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
    raise SystemExit(
        f"expected default JAX backend cuda/gpu, got {payload['default_backend']}"
    )
if (payload["jax"], payload["jaxlib"]) != ("0.10.0", "0.10.0"):
    raise SystemExit(
        f"expected JAX/JAXLIB 0.10.0, got {payload['jax']}/{payload['jaxlib']}"
    )
if not payload["x64"]:
    raise SystemExit("expected JAX x64 mode to be enabled")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

run_step() {
  local name="$1"
  local backend_mode="$2"
  shift 2
  local step_dir="${ARTIFACT_DIR}/${name}"
  mkdir -p "${step_dir}"

  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv \
    -l 5 > "${step_dir}/gpu_memory.csv" &
  local monitor_pid="$!"

  set +e
  SIMSOPT_BACKEND_MODE="${backend_mode}" \
    /usr/bin/time -v \
    srun -n 1 -c 32 --gpus-per-task=1 "$@" \
    > "${step_dir}/stdout.log" \
    2> "${step_dir}/stderr_time.log"
  local status="$?"
  set -e

  kill "${monitor_pid}" 2>/dev/null
  wait "${monitor_pid}" 2>/dev/null || true

  echo "${status}" > "${step_dir}/exit_status.txt"
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
}

run_step stage2_cpu \
  jax_cpu_parity \
  python benchmarks/stage2_e2e_comparison.py \
    --platform cpu \
    --output-json "${ARTIFACT_DIR}/stage2_cpu.json"

run_step stage2_cuda \
  jax_gpu_parity \
  python benchmarks/stage2_e2e_comparison.py \
    --platform cuda \
    --output-json "${ARTIFACT_DIR}/stage2_cuda.json"

run_step single_stage_cpu \
  jax_cpu_parity \
  python benchmarks/single_stage_init_parity.py \
    --platform cpu \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --case-artifacts-dir "${ARTIFACT_DIR}/single_stage_cpu_cases" \
    --output-json "${ARTIFACT_DIR}/single_stage_cpu.json"

run_step single_stage_cuda \
  jax_gpu_parity \
  python benchmarks/single_stage_init_parity.py \
    --platform cuda \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --case-artifacts-dir "${ARTIFACT_DIR}/single_stage_cuda_cases" \
    --output-json "${ARTIFACT_DIR}/single_stage_cuda.json"

run_step tier5_cuda \
  jax_gpu_parity \
  python benchmarks/tier5_performance_characterization.py \
    --platform cuda \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --output-json "${ARTIFACT_DIR}/tier5_cuda.json" \
    --phase gpu

nvidia-smi > "${JOB_RESULTS}/nvidia_smi_end.txt"
