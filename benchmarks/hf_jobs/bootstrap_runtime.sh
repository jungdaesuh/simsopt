#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_MODE="${SIMSOPT_HF_JOB_BOOTSTRAP_MODE:-auto}"
EXPECTED_JAX_VERSION="${SIMSOPT_HF_JOB_EXPECTED_JAX_VERSION:-0.9.2}"
JAX_GPU_WHEEL_SPEC="${SIMSOPT_HF_JOB_JAX_GPU_WHEEL_SPEC:-jax[cuda12]==0.9.2}"
VENV_DIR="/opt/venv"
BOOTSTRAP_SOURCED=0

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  BOOTSTRAP_SOURCED=1
fi

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

if [[ -x "${VENV_DIR}/bin/python" && "${BOOTSTRAP_MODE}" != "always" ]]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
  export VIRTUAL_ENV="${VENV_DIR}"
  verify_runtime_versions
  bootstrap_finish 0
fi

if [[ "${BOOTSTRAP_MODE}" == "never" ]]; then
  echo "Requested bootstrap mode 'never' but ${VENV_DIR}/bin/python is unavailable." >&2
  bootstrap_finish 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
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
export PATH="${VENV_DIR}/bin:${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"

# Proof infra keeps the exact JAX wheel pinned here; project deps stay in pyproject.
python -m pip install --upgrade pip setuptools wheel
python -m pip install cmake ninja "pybind11<3" scikit-build-core "setuptools-scm>=8.0"
python -m pip install "${JAX_GPU_WHEEL_SPEC}"
python -m pip install -e ".[deploy_gpu]"
verify_runtime_versions
bootstrap_finish 0
