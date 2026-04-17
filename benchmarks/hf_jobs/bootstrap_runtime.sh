#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_MODE="${SIMSOPT_HF_JOB_BOOTSTRAP_MODE:-auto}"
EXPECTED_JAX_VERSION="${SIMSOPT_HF_JOB_EXPECTED_JAX_VERSION:-0.9.2}"
JAX_GPU_WHEEL_SPEC="${SIMSOPT_HF_JOB_JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.9.2}"
CUDA_LIBRARY_MODE="${SIMSOPT_JAX_CUDA_LIBRARY_MODE:-bundled}"
APT_RETRY_ATTEMPTS="${SIMSOPT_HF_JOB_APT_RETRY_ATTEMPTS:-3}"
VENV_DIR="/opt/venv"
BOOTSTRAP_SOURCED=0

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  BOOTSTRAP_SOURCED=1
fi

activate_runtime_env() {
  export PATH="${VENV_DIR}/bin:${PATH}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export SIMSOPT_JAX_CUDA_LIBRARY_MODE="${CUDA_LIBRARY_MODE}"
}

bootstrap_finish() {
  local status="${1:-0}"
  if [[ "${BOOTSTRAP_SOURCED}" -eq 1 ]]; then
    return "${status}"
  fi
  exit "${status}"
}

verify_runtime_versions() {
  python - "${EXPECTED_JAX_VERSION}" <<'PY'
import json
import sys

import jax
import jaxlib
import numpy
import scipy

expected = sys.argv[1]
payload = {
    "jax": jax.__version__,
    "jaxlib": jaxlib.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
}
print(json.dumps(payload, sort_keys=True))
if jax.__version__ != expected or jaxlib.__version__ != expected:
    raise SystemExit(
        f"Expected JAX/JAXLIB {expected}, got "
        f"jax={jax.__version__} jaxlib={jaxlib.__version__}"
    )
PY
}


run_apt_with_retry() {
  local -a cmd=("$@")
  local attempt=1
  local max_attempts
  max_attempts="${APT_RETRY_ATTEMPTS}"
  while true; do
    if "${cmd[@]}"; then
      return 0
    fi
    if [[ "${attempt}" -ge "${max_attempts}" ]]; then
      return 1
    fi
    sleep "$((attempt * 5))"
    attempt="$((attempt + 1))"
  done
}


normalize_ubuntu_apt_sources_to_https() {
  local source_file
  for source_file in \
    /etc/apt/sources.list \
    /etc/apt/sources.list.d/*.list \
    /etc/apt/sources.list.d/*.sources; do
    [[ -f "${source_file}" ]] || continue
    sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' \
      "${source_file}"
  done
}

if [[ -x "${VENV_DIR}/bin/python" && "${BOOTSTRAP_MODE}" != "always" ]]; then
  activate_runtime_env
  verify_runtime_versions
  bootstrap_finish 0
fi

if [[ "${BOOTSTRAP_MODE}" == "never" ]]; then
  echo "Requested bootstrap mode 'never' but ${VENV_DIR}/bin/python is unavailable." >&2
  bootstrap_finish 1
fi

export DEBIAN_FRONTEND=noninteractive
normalize_ubuntu_apt_sources_to_https
run_apt_with_retry apt-get -o Acquire::Retries=3 update
run_apt_with_retry apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
  build-essential \
  gfortran \
  git \
  libboost-all-dev \
  libfftw3-dev \
  libhdf5-dev \
  libhdf5-serial-dev \
  liblapack-dev \
  libnetcdf-dev \
  libnetcdff-dev \
  libopenblas-dev \
  libopenmpi-dev \
  openmpi-bin \
  python3-venv
rm -rf /var/lib/apt/lists/*

rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
activate_runtime_env

# Proof infra keeps the exact JAX wheel pinned here; project deps stay in pyproject.
python -m pip install --upgrade pip setuptools wheel
python -m pip install cmake ninja "pybind11<3" scikit-build-core "setuptools-scm>=8.0"
python -m pip install "${JAX_GPU_WHEEL_SPEC}"
python -m pip install -e ".[deploy_gpu]"
verify_runtime_versions
bootstrap_finish 0
