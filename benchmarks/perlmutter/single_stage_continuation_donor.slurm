#!/usr/bin/env bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -J ss-cont-donor

set -euo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch; do not run it on a login node}"
: "${REPO_ROOT:?set REPO_ROOT to the clean source checkout}"
: "${RUN_ROOT:?set RUN_ROOT to a writable scratch run directory}"

JOB_ROOT="${RUN_ROOT}/${SLURM_JOB_ID}"
ENV_ROOT="${JOB_ROOT}/venv"
LOG_DIR="${JOB_ROOT}/logs"
DONOR_ROOT="${JOB_ROOT}/continuation_outputs"
FIXTURE_STAGE2_BS="benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json"
DONOR_MPOL="${DONOR_MPOL:-10}"
DONOR_NTOR="${DONOR_NTOR:-10}"
DONOR_NPHI="${DONOR_NPHI:-64}"
DONOR_NTHETA="${DONOR_NTHETA:-32}"
DONOR_MAXITER="${DONOR_MAXITER:-300}"
# Extra continuation rungs for a gentler ladder (2,4,6,8,10 by default).
DONOR_INTERMEDIATE_RUNGS="${DONOR_INTERMEDIATE_RUNGS:-8}"
DONOR_STAGE_TIMEOUT_SECONDS="${DONOR_STAGE_TIMEOUT_SECONDS:-21600}"

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

mkdir -p "${LOG_DIR}" "${DONOR_ROOT}"
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
git status --short --branch > "${JOB_ROOT}/git_status_short_branch.txt"
git diff --no-ext-diff HEAD > "${JOB_ROOT}/dirty.patch"
if [[ -s "${JOB_ROOT}/dirty.patch" ]]; then
  echo "tracked worktree must be clean for donor-build signoff" >&2
  exit 1
fi
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "${SOURCE_STATUS}" ]]; then
  echo "non-clean source state would invalidate donor-build signoff" >&2
  printf '%s\n' "${SOURCE_STATUS}" >&2
  exit 1
fi

export SLURM_CPU_BIND="cores"
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

STEP_DIR="${JOB_ROOT}/continuation_step"
mkdir -p "${STEP_DIR}"
set +e
/usr/bin/time -v \
  srun -n 1 -c 32 python examples/single_stage_optimization/SINGLE_STAGE/run_single_stage_continuation.py \
    --mpol "${DONOR_MPOL}" \
    --ntor "${DONOR_NTOR}" \
    --nphi "${DONOR_NPHI}" \
    --ntheta "${DONOR_NTHETA}" \
    --maxiter "${DONOR_MAXITER}" \
    --intermediate-rungs "${DONOR_INTERMEDIATE_RUNGS}" \
    --initial-stage2-bs-path "${FIXTURE_STAGE2_BS}" \
    --stage-timeout-seconds "${DONOR_STAGE_TIMEOUT_SECONDS}" \
    --output-root "${DONOR_ROOT}" \
  > "${STEP_DIR}/stdout.log" \
  2> "${STEP_DIR}/stderr_time.log"
status="$?"
set -e
echo "${status}" > "${STEP_DIR}/exit_status.txt"
ls -la "${DONOR_ROOT}" > "${JOB_ROOT}/donor_root_listing.txt" 2>/dev/null || true
exit "${status}"
