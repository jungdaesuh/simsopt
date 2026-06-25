#!/usr/bin/env bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J clean-stale-cuda

set -euo pipefail

: "${SLURM_JOB_ID:?submit this script with sbatch; do not run it on a login node}"
: "${REPO_ROOT:?set REPO_ROOT to the clean source checkout}"
: "${RUN_ROOT:?set RUN_ROOT to a writable scratch run directory}"

JOB_ROOT="${RUN_ROOT}/${SLURM_JOB_ID}"
ENV_ROOT="${JOB_ROOT}/venv"
RESULTS_DIR="${JOB_ROOT}/results"
LOG_DIR="${JOB_ROOT}/logs"

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

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
exec > >(tee "${LOG_DIR}/driver_stdout.log")
exec 2> >(tee "${LOG_DIR}/driver_stderr.log" >&2)

cd "${REPO_ROOT}"
module load python
python3 -m venv "${ENV_ROOT}"
. "${ENV_ROOT}/bin/activate"

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT:-1.9.4.dev0}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "jax[cuda12]==0.10.0" etils
python -m pip install -e ".[JAX_GPU,test,ALGS]" "shapely>=2.1,<3" "numba>=0.64,<0.66"
python -m pip freeze > "${JOB_ROOT}/pip_freeze.txt"

git rev-parse HEAD > "${JOB_ROOT}/repo_sha.txt"
git status --short > "${JOB_ROOT}/git_status_short.txt"
git status --short --branch > "${JOB_ROOT}/git_status_short_branch.txt"
git diff --no-ext-diff HEAD > "${JOB_ROOT}/dirty.patch"
git diff --stat HEAD > "${JOB_ROOT}/dirty-diff-stat.txt"
if [[ -s "${JOB_ROOT}/dirty.patch" ]]; then
  echo "tracked worktree must be clean for CUDA signoff" >&2
  cat "${JOB_ROOT}/dirty-diff-stat.txt" >&2
  exit 1
fi
SOURCE_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "${SOURCE_STATUS}" ]]; then
  echo "non-clean source state would invalidate CUDA signoff" >&2
  printf '%s\n' "${SOURCE_STATUS}" >&2
  exit 1
fi

nvidia-smi > "${JOB_ROOT}/nvidia_smi_start.txt"

python - <<'PY' > "${JOB_ROOT}/runtime_versions.json"
import importlib.metadata as metadata
import json

import jax
import jaxlib

payload = {
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
print(json.dumps(payload, indent=2, sort_keys=True))
PY

set +e
/usr/bin/time -v \
  srun -n 1 -c 32 --gpus-per-task=1 python scripts/jax_gpu_failed_stale_tests_signoff.py \
    --repo "${REPO_ROOT}" \
    --python-bin "${ENV_ROOT}/bin/python" \
    --results-dir "${RESULTS_DIR}" \
  > "${JOB_ROOT}/signoff_stdout.log" \
  2> "${JOB_ROOT}/signoff_stderr_time.log"
status="$?"
set -e
echo "${status}" > "${JOB_ROOT}/signoff_exit_status.txt"

nvidia-smi > "${JOB_ROOT}/nvidia_smi_end.txt" || true
exit "${status}"
