#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_MODE="${SIMSOPT_HF_JOB_BOOTSTRAP_MODE:-auto}"
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

if [[ -x "${VENV_DIR}/bin/python" && "${BOOTSTRAP_MODE}" != "always" ]]; then
  export PATH="${VENV_DIR}/bin:${PATH}"
  export VIRTUAL_ENV="${VENV_DIR}"
  python -c 'import jax, numpy, scipy; print({"jax": jax.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__})'
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

# All Python dependency versions are defined in pyproject.toml (SSOT).
# This only installs build prerequisites then delegates to the extras.
python -m pip install --upgrade pip setuptools wheel
python -m pip install cmake ninja "pybind11<3" scikit-build-core "setuptools-scm>=8.0"
python -m pip install -e ".[deploy_gpu]"
bootstrap_finish 0
