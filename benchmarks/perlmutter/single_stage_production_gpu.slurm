#!/usr/bin/env bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J prod-ss-gpu

set -euo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch; do not run it on a login node}"
: "${REPO_ROOT:?set REPO_ROOT to the clean source checkout}"
: "${RUN_ROOT:?set RUN_ROOT to a writable scratch run directory}"

JOB_ROOT="${RUN_ROOT}/${SLURM_JOB_ID}"
ENV_ROOT="${JOB_ROOT}/venv"
LOG_DIR="${JOB_ROOT}/logs"
ARTIFACT_DIR="${JOB_ROOT}/artifacts"
FIXTURE_STAGE2_BS="benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json"
JAX_GPU_WHEEL_SPEC="${JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.10.0}"
PROD_MAXITER="${PROD_MAXITER:-1500}"
# Use ${VAR-default} (not :-) so an explicitly exported empty value skips
# the inner Boozer BFGS cap flag and the child falls back to its default.
PROD_BOOZER_BFGS_MAXITER="${PROD_BOOZER_BFGS_MAXITER-1500}"
PROD_NEWTON_MAXITER="${PROD_NEWTON_MAXITER:-50}"
# Default to the decomposed lane (host outer L-BFGS-B + split K1 forward /
# K2 value+adjoint): campaign-productionized as the lane that compiles and
# passes at mpol10, where plain scipy-jax re-fuses K1+K2 into the heavier
# kernel. Export PROD_OPTIMIZER_BACKEND=scipy-jax or scipy-jax-fullgraph to
# exercise the fused lanes instead.
PROD_OPTIMIZER_BACKEND="${PROD_OPTIMIZER_BACKEND:-scipy-jax-decomposed}"
PROD_BOOZER_LS_ALGORITHM="${PROD_BOOZER_LS_ALGORITHM:-}"
PROD_BOOZER_OPTIMIZER_BACKEND="${PROD_BOOZER_OPTIMIZER_BACKEND:-}"
PROD_VARIANT="${PROD_OPTIMIZER_BACKEND}${PROD_BOOZER_LS_ALGORITHM:+-${PROD_BOOZER_LS_ALGORITHM}}"
PROD_MPOL="${PROD_MPOL:-}"
PROD_NTOR="${PROD_NTOR:-}"
PROD_NPHI="${PROD_NPHI:-}"
PROD_NTHETA="${PROD_NTHETA:-}"
PROD_WARM_START_RUN_DIR="${PROD_WARM_START_RUN_DIR:-}"
PROD_NEWTON_POLISH_POLICY="${PROD_NEWTON_POLISH_POLICY:-run}"
# Persistent JAX/XLA compilation cache. Keep this OUTSIDE ${JOB_ROOT} so the one
# slow cold XLA:GPU compile (~73 min on the RunPod cu1290 image; diagnosed
# "once-slow" in docs/jax_scipy_jax_gpu_compile_diagnostic_next.md) is paid once
# and reused across runs (different ${SLURM_JOB_ID}). ${RUN_ROOT} is already a
# writable scratch dir outside the repo, so it persists between jobs.
PROD_JAX_CACHE_DIR="${PROD_JAX_CACHE_DIR:-${RUN_ROOT}/jax_compilation_cache}"
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
git status --short > "${JOB_ROOT}/git_status_short.txt"
git status --short --branch > "${JOB_ROOT}/git_status_short_branch.txt"
git diff --no-ext-diff HEAD > "${JOB_ROOT}/dirty.patch"
git diff --stat HEAD > "${JOB_ROOT}/dirty-diff-stat.txt"
if [[ -s "${JOB_ROOT}/dirty.patch" ]]; then
  echo "tracked worktree must be clean for production benchmark signoff" >&2
  cat "${JOB_ROOT}/dirty-diff-stat.txt" >&2
  exit 1
fi
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "${SOURCE_STATUS}" ]]; then
  echo "non-clean source state would invalidate production benchmark signoff" >&2
  printf '%s\n' "${SOURCE_STATUS}" >&2
  exit 1
fi

export SLURM_CPU_BIND="cores"
# Pin host BLAS/OpenMP threads to the allocated cores so the co-produced cpp
# reference uses a fixed thread count (fair comparison), and to avoid the
# uncapped-thread oversubscription that inflated host-side wall ~50x on RunPod
# GPU pods (cpp reference 44s -> 2299s across otherwise-identical runs).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD=disallow
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export SIMSOPT_JAX_PLATFORM=cuda
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
# Point the JAX persistent compilation cache at scratch. runtime.py defaults it to
# ~/.cache/simsopt-jax-xla (runtime.py:864), which has tight quota on Perlmutter;
# apply_jax_runtime_config reads JAX_COMPILATION_CACHE_DIR (runtime.py:1437) and,
# when set, uses the persistent cache plus the SAFE narrow GPU autotune cache
# (jax_persistent_cache_enable_xla_caches="xla_gpu_per_fusion_autotune_cache_dir",
# runtime.py:2394-2397). We deliberately set ONLY the cache dir and do NOT set
# JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES: a broader mode (e.g. "all") can force
# nvlink through the container CUDA toolkit vs the wheel-bundled NVIDIA libs
# (runtime.py:2391-2393; cf. the RunPod cu1290 nvlink block).
mkdir -p "${PROD_JAX_CACHE_DIR}"
export JAX_COMPILATION_CACHE_DIR="${PROD_JAX_CACHE_DIR}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

python - <<'PY' > "${JOB_ROOT}/simsoptpp_curve_smoke.txt"
from simsoptpp import Curve

print(Curve)
PY

nvidia-smi > "${JOB_ROOT}/nvidia_smi_start.txt"

python - <<'PY' > "${JOB_ROOT}/runtime_versions.json"
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

echo "${PROD_VARIANT}" > "${JOB_ROOT}/production_variant.txt"
boozer_bfgs_args=()
if [[ -n "${PROD_BOOZER_BFGS_MAXITER}" ]]; then
  boozer_bfgs_args=(--target-lane-boozer-bfgs-maxiter "${PROD_BOOZER_BFGS_MAXITER}")
fi
boozer_backend_args=()
if [[ -n "${PROD_BOOZER_OPTIMIZER_BACKEND}" ]]; then
  # Force the inner Boozer LS solver backend (ondevice for the
  # scipy-jax-fullgraph lane under jax_*_parity; see build_single_stage_matrix.py).
  boozer_backend_args=(--boozer-optimizer-backend "${PROD_BOOZER_OPTIMIZER_BACKEND}")
fi
resolution_args=()
[[ -n "${PROD_MPOL}" ]] && resolution_args+=(--mpol "${PROD_MPOL}")
[[ -n "${PROD_NTOR}" ]] && resolution_args+=(--ntor "${PROD_NTOR}")
[[ -n "${PROD_NPHI}" ]] && resolution_args+=(--nphi "${PROD_NPHI}")
[[ -n "${PROD_NTHETA}" ]] && resolution_args+=(--ntheta "${PROD_NTHETA}")
if [[ -n "${PROD_WARM_START_RUN_DIR}" ]]; then
  resolution_args+=(--warm-start-run-dir "${PROD_WARM_START_RUN_DIR}")
fi
if [[ -n "${PROD_BOOZER_LS_ALGORITHM}" ]]; then
  # The example child reads this env var as its --boozer-least-squares-algorithm
  # default; the parity wrapper's subprocess env inherits os.environ.
  export BOOZER_LEAST_SQUARES_ALGORITHM="${PROD_BOOZER_LS_ALGORITHM}"
fi

STEP_DIR="${ARTIFACT_DIR}/single_stage_cuda_production_${PROD_VARIANT}"
mkdir -p "${STEP_DIR}"

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv \
  -l 5 > "${STEP_DIR}/gpu_memory.csv" &
monitor_pid="$!"

set +e
/usr/bin/time -v \
  srun -n 1 -c 32 --gpus-per-task=1 python benchmarks/single_stage_init_parity.py \
    --platform cuda \
    --optimizer-backend "${PROD_OPTIMIZER_BACKEND}" \
    "${boozer_backend_args[@]}" \
    --reference-optimizer-method lbfgs \
    --maxiter "${PROD_MAXITER}" \
    "${boozer_bfgs_args[@]}" \
    "${resolution_args[@]}" \
    --target-lane-boozer-newton-polish-policy "${PROD_NEWTON_POLISH_POLICY}" \
    --target-lane-boozer-newton-maxiter "${PROD_NEWTON_MAXITER}" \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --case-artifacts-dir "${ARTIFACT_DIR}/single_stage_cuda_production_${PROD_VARIANT}_cases" \
    --single-stage-case-timeout-seconds "${CASE_TIMEOUT_SECONDS}" \
    --single-stage-target-case-timeout-seconds "${TARGET_CASE_TIMEOUT_SECONDS}" \
    --output-json "${ARTIFACT_DIR}/single_stage_cuda_production_${PROD_VARIANT}.json" \
  > "${STEP_DIR}/stdout.log" \
  2> "${STEP_DIR}/stderr_time.log"
status="$?"
set -e

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
echo "${status}" > "${STEP_DIR}/exit_status.txt"

nvidia-smi > "${JOB_ROOT}/nvidia_smi_end.txt" || true
exit "${status}"
