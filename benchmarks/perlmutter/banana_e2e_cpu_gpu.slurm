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
JAX_GPU_WHEEL_SPEC="${JAX_GPU_WHEEL_SPEC:-jax[cuda12]}"
LOG_DIR="${JOB_RESULTS}/logs"
ARTIFACT_DIR="${JOB_RESULTS}/artifacts"
FIXTURE_DIR="benchmarks/fixtures/single_stage_seed_iota15"
FIXTURE_STAGE2_BS="${FIXTURE_DIR}/biot_savart_opt.json"
FIXTURE_RUNTIME_SPEC="${FIXTURE_DIR}/single_stage_jax_runtime_spec.json"

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"
exec > >(tee "${LOG_DIR}/driver_stdout.log")
exec 2> >(tee "${LOG_DIR}/driver_stderr.log" >&2)
cd "${REPO_ROOT}"

module load python

python3 -m venv "${ENV_ROOT}"
. "${ENV_ROOT}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "${JAX_GPU_WHEEL_SPEC}"
python -m pip install -e ".[deploy_gpu]"
python -m pip freeze > "${JOB_RESULTS}/pip_freeze.txt"

git rev-parse HEAD > "${JOB_RESULTS}/repo_sha.txt"
git status --short > "${JOB_RESULTS}/git_status_short.txt"

export SLURM_CPU_BIND="cores"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export SIMSOPT_JAX_PLATFORM=cuda
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

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
if not payload["x64"]:
    raise SystemExit("expected JAX x64 mode to be enabled")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

run_step() {
  local name="$1"
  shift
  local step_dir="${ARTIFACT_DIR}/${name}"
  mkdir -p "${step_dir}"

  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv \
    -l 5 > "${step_dir}/gpu_memory.csv" &
  local monitor_pid="$!"

  set +e
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
  python benchmarks/stage2_e2e_comparison.py \
    --platform cpu \
    --output-json "${ARTIFACT_DIR}/stage2_cpu.json"

run_step stage2_cuda \
  python benchmarks/stage2_e2e_comparison.py \
    --platform cuda \
    --output-json "${ARTIFACT_DIR}/stage2_cuda.json"

run_step single_stage_cpu \
  python benchmarks/single_stage_init_parity.py \
    --platform cpu \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --jax-runtime-seed-spec "${FIXTURE_RUNTIME_SPEC}" \
    --case-artifacts-dir "${ARTIFACT_DIR}/single_stage_cpu_cases" \
    --output-json "${ARTIFACT_DIR}/single_stage_cpu.json"

run_step single_stage_cuda \
  python benchmarks/single_stage_init_parity.py \
    --platform cuda \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --jax-runtime-seed-spec "${FIXTURE_RUNTIME_SPEC}" \
    --case-artifacts-dir "${ARTIFACT_DIR}/single_stage_cuda_cases" \
    --output-json "${ARTIFACT_DIR}/single_stage_cuda.json"

run_step tier5_cuda \
  python benchmarks/tier5_performance_characterization.py \
    --platform cuda \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --output-json "${ARTIFACT_DIR}/tier5_cuda.json" \
    --phase gpu

nvidia-smi > "${JOB_RESULTS}/nvidia_smi_end.txt"
