#!/usr/bin/env bash
#
# Create a fully-contained conda env at ./env with all build + test deps.
# Re-runs are idempotent: conda install is a no-op on packages that already
# satisfy the spec.
#
# Usage:
#     scripts/dev/env-create.sh                       # primary env at ./env
#     ENV_PREFIX=./env-asan scripts/dev/env-create.sh # variant env
#
# Requires: miniforge / mambaforge / miniconda installed and on PATH.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_PREFIX="${ENV_PREFIX:-./env}"

if command -v mamba >/dev/null 2>&1; then
    CONDA="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA="conda"
else
    echo "error: neither mamba nor conda found on PATH" >&2
    echo "install miniforge: https://github.com/conda-forge/miniforge" >&2
    exit 1
fi

echo "==> Creating conda env at $ENV_PREFIX using $CONDA"
"$CONDA" create --prefix "$ENV_PREFIX" -c conda-forge -y \
    python=3.11 \
    cmake ninja pybind11 \
    scikit-build-core setuptools_scm \
    compilers llvm-openmp \
    boost \
    openmpi mpi4py \
    numpy scipy matplotlib \
    pytest pytest-xdist hypothesis \
    asv

echo ""
echo "==> Installing Python deps into $ENV_PREFIX"
"$ENV_PREFIX/bin/pip" install --upgrade pip
"$ENV_PREFIX/bin/pip" install -r requirements.txt

# Critical: the editable install compiles simsoptpp via cmake, which
# consults CONDA_PREFIX, CMAKE_PREFIX_PATH, and PATH to resolve boost,
# pybind11, openmp, etc. Running ./env/bin/pip alone is not enough —
# the child cmake process sees the shell's ambient CONDA_PREFIX (often
# miniforge base) and finds brew's headers instead of the env's.
# Explicitly activate the env for the install.
ABS_PREFIX="$(cd "$ENV_PREFIX" && pwd)"
echo ""
echo "==> Compiling simsoptpp with CONDA_PREFIX=$ABS_PREFIX"
env -i HOME="$HOME" USER="$USER" SHELL="$SHELL" \
    CONDA_PREFIX="$ABS_PREFIX" \
    PATH="$ABS_PREFIX/bin:/usr/bin:/bin" \
    CMAKE_PREFIX_PATH="$ABS_PREFIX" \
    "$ABS_PREFIX/bin/pip" install -e . --no-build-isolation

echo ""
echo "==> Env ready."
echo "    Activate:        conda activate $ENV_PREFIX"
echo "    Or run directly: $ENV_PREFIX/bin/python -c 'import simsopt'"
echo ""
echo "    Next: scripts/dev/build.sh <variant>"
