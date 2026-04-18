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
"$ENV_PREFIX/bin/pip" install -e . --no-build-isolation

echo ""
echo "==> Env ready."
echo "    Activate:        conda activate $ENV_PREFIX"
echo "    Or run directly: $ENV_PREFIX/bin/python -c 'import simsopt'"
echo ""
echo "    Next: scripts/dev/build.sh <variant>"
