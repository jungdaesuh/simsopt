#!/usr/bin/env bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -J prod-ss-cpu

set -euo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch; do not run it on a login node}"
: "${REPO_ROOT:?set REPO_ROOT to the clean source checkout}"
: "${RUN_ROOT:?set RUN_ROOT to a writable scratch run directory}"

JOB_ROOT="${RUN_ROOT}/${SLURM_JOB_ID}"
ENV_ROOT="${JOB_ROOT}/venv"
LOG_DIR="${JOB_ROOT}/logs"
ARTIFACT_DIR="${JOB_ROOT}/artifacts"
FIXTURE_STAGE2_BS="benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json"
PROD_MAXITER="${PROD_MAXITER:-1500}"
# Use ${VAR-default} (not :-) so an explicitly exported empty value skips
# the inner Boozer BFGS cap flag and the child falls back to its default.
PROD_BOOZER_BFGS_MAXITER="${PROD_BOOZER_BFGS_MAXITER-1500}"
PROD_NEWTON_MAXITER="${PROD_NEWTON_MAXITER:-50}"
# Default to the decomposed lane (host outer L-BFGS-B + split K1 forward /
# K2 value+adjoint). Export PROD_OPTIMIZER_BACKEND=scipy-jax only for the
# deprecated legacy reduced lane; use scipy-jax-fullgraph for fused parity.
PROD_OPTIMIZER_BACKEND="${PROD_OPTIMIZER_BACKEND:-scipy-jax-decomposed}"
PROD_BOOZER_LS_ALGORITHM="${PROD_BOOZER_LS_ALGORITHM:-}"
PROD_BOOZER_OPTIMIZER_BACKEND="${PROD_BOOZER_OPTIMIZER_BACKEND:-}"
PROD_VARIANT="${PROD_OPTIMIZER_BACKEND}${PROD_BOOZER_LS_ALGORITHM:+-${PROD_BOOZER_LS_ALGORITHM}}"
PROD_TIMING_CLASSIFICATION="${PROD_TIMING_CLASSIFICATION:-mixed-parity-reference}"
PROD_SUPPORTS_PERFORMANCE_HEADLINE="${PROD_SUPPORTS_PERFORMANCE_HEADLINE:-0}"
PROD_HEADLINE_TIMING_CLASSIFICATION="${PROD_HEADLINE_TIMING_CLASSIFICATION:-}"
PROD_MPOL="${PROD_MPOL:-}"
PROD_NTOR="${PROD_NTOR:-}"
PROD_NPHI="${PROD_NPHI:-}"
PROD_NTHETA="${PROD_NTHETA:-}"
PROD_WARM_START_RUN_DIR="${PROD_WARM_START_RUN_DIR:-}"
PROD_NEWTON_POLISH_POLICY="${PROD_NEWTON_POLISH_POLICY:-run}"
PROD_TRIAL_NEWTON_POLISH_POLICY="${PROD_TRIAL_NEWTON_POLISH_POLICY:-skip}"
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
python -m pip install --upgrade "jax==0.10.0" "jaxlib==0.10.0" etils
python -m pip install -e ".[JAX,test,ALGS]" "shapely>=2.1,<3" "numba>=0.64,<0.66"
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
# Pin host BLAS/OpenMP threads to the allocated cores so the cpp reference and
# the JAX-CPU host work use the same thread count (fair comparison), and to avoid
# the uncapped-thread oversubscription that inflated host-side wall ~50x on
# RunPod GPU pods (cpp reference 44s -> 2299s across otherwise-identical runs).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export SIMSOPT_BACKEND_MODE=jax_cpu_parity
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD=disallow
export SIMSOPT_JAX_PLATFORM=cpu
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH

python - <<'PY' > "${JOB_ROOT}/simsoptpp_curve_smoke.txt"
from simsoptpp import Curve

print(Curve)
PY

python - <<'PY' > "${JOB_ROOT}/runtime_versions.json"
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
        "SIMSOPT_JAX_PLATFORM": os.environ.get("SIMSOPT_JAX_PLATFORM"),
        "SIMSOPT_JAX_TRANSFER_GUARD": os.environ.get(
            "SIMSOPT_JAX_TRANSFER_GUARD"
        ),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
    },
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "default_backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
    "x64": bool(jax.config.read("jax_enable_x64")),
}
if payload["default_backend"] != "cpu":
    raise SystemExit(
        f"expected default JAX backend cpu, got {payload['default_backend']}"
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
printf '%s\n' "${PROD_TIMING_CLASSIFICATION}" > "${JOB_ROOT}/timing_classification.txt"
printf '{"timing_classification":"%s","supports_performance_headline":%s,"headline_timing_classification":"%s"}\n' \
  "${PROD_TIMING_CLASSIFICATION}" \
  "${PROD_SUPPORTS_PERFORMANCE_HEADLINE}" \
  "${PROD_HEADLINE_TIMING_CLASSIFICATION}" \
  > "${JOB_ROOT}/benchmark_timing_label.json"
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

STEP_DIR="${ARTIFACT_DIR}/single_stage_cpu_production_${PROD_VARIANT}"
mkdir -p "${STEP_DIR}"
set +e
/usr/bin/time -v \
  srun -n 1 -c 32 python benchmarks/single_stage_init_parity.py \
    --platform cpu \
    --optimizer-backend "${PROD_OPTIMIZER_BACKEND}" \
    "${boozer_backend_args[@]}" \
    --reference-optimizer-method lbfgs \
    --maxiter "${PROD_MAXITER}" \
    "${boozer_bfgs_args[@]}" \
    "${resolution_args[@]}" \
    --target-lane-boozer-newton-polish-policy "${PROD_NEWTON_POLISH_POLICY}" \
    --target-lane-trial-boozer-newton-polish-policy "${PROD_TRIAL_NEWTON_POLISH_POLICY}" \
    --target-lane-boozer-newton-maxiter "${PROD_NEWTON_MAXITER}" \
    --stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --case-artifacts-dir "${ARTIFACT_DIR}/single_stage_cpu_production_${PROD_VARIANT}_cases" \
    --single-stage-case-timeout-seconds "${CASE_TIMEOUT_SECONDS}" \
    --single-stage-target-case-timeout-seconds "${TARGET_CASE_TIMEOUT_SECONDS}" \
    --output-json "${ARTIFACT_DIR}/single_stage_cpu_production_${PROD_VARIANT}.json" \
  > "${STEP_DIR}/stdout.log" \
  2> "${STEP_DIR}/stderr_time.log"
status="$?"
set -e
echo "${status}" > "${STEP_DIR}/exit_status.txt"
exit "${status}"
