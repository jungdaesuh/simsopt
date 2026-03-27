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

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  "numpy>=2.0" \
  cmake \
  scikit-build-core \
  ninja \
  "setuptools-scm>=8.0" \
  "scipy>=1.13" \
  pytest \
  sympy \
  f90nml \
  pyevtk \
  matplotlib \
  shapely \
  numba \
  "ground==9.0.0" \
  "bentley_ottmann==8.0.0" \
  ruamel.yaml \
  monty \
  Deprecated \
  "pybind11<3"
python -m pip install "jax[cuda12]==0.9.2"
bootstrap_finish 0
