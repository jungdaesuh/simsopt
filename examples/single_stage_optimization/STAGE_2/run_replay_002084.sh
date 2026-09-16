#!/usr/bin/env bash
# One-shot replay of the Stage-2 root coil seed_penalty_002084_tight (see REPLAY_002084.md).
# Creates a Python 3.11 venv next to the repo, installs runtime deps, uses the prebuilt
# simsoptpp extension under build/ (or builds one with `pip install -e .` if none is found),
# and runs banana_coil_solver.py from this directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
VENV="${REPLAY_VENV:-$REPO/.venv-replay}"
THREADS="${OMP_NUM_THREADS:-4}"
WOUT="$HERE/../equilibria/wout_nfp5ginsburg_000_002084_iota20.nc"

[ -f "$WOUT" ] || { echo "missing equilibrium: $WOUT (it is gitignored; obtain it separately)" >&2; exit 1; }

DEPS=("numpy>=2.4,<3" scipy "jax[cpu]" jaxlib Deprecated monty ruamel.yaml sympy f90nml pyevtk matplotlib shapely numba)
if command -v uv >/dev/null 2>&1; then
    [ -x "$VENV/bin/python" ] || uv venv -p 3.11 "$VENV"
    pip_install() { uv pip install -p "$VENV/bin/python" "$@"; }
else
    [ -x "$VENV/bin/python" ] || python3.11 -m venv "$VENV"
    pip_install() { "$VENV/bin/python" -m pip install "$@"; }
fi
pip_install "${DEPS[@]}"
PY="$VENV/bin/python"

# Prebuilt extension for this interpreter, if present.
TAG="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"   # e.g. cpython-311-x86_64-linux-gnu
SO="$(ls "$REPO"/build/*/simsoptpp."$TAG".so 2>/dev/null | head -1 || true)"
if [ -n "$SO" ]; then
    export PYTHONPATH="$REPO/src:$(dirname "$SO")${PYTHONPATH:+:$PYTHONPATH}"
else
    # build/ is gitignored, so a fresh clone lands here. NOTE: this tree does not compile with GCC 12-15 or
    # Apple Clang 21 (xt::pyarray "operator*=" is ambiguous between xtensor 0.21 and pybind11; see
    # hbt-compare/reports/SUMMARY.md). Ship the prebuilt build/<tag>/simsoptpp*.so with the repo instead,
    # or apply simsopt-surrogate's xtensor_compat refactor to src/simsoptpp. The attempt below is kept for
    # toolchains that still accept the tree.
    echo "no prebuilt simsoptpp for $TAG under $REPO/build; attempting to compile simsoptpp with ${CXX:-c++} (known to FAIL on GCC 12-15 and Apple Clang 21)" >&2
    command -v cmake >/dev/null 2>&1 || { echo "cmake not found" >&2; exit 1; }
    [ -f "$REPO/thirdparty/xtensor/CMakeLists.txt" ] || { echo "thirdparty submodules missing: run 'git -C $REPO submodule update --init --recursive'" >&2; exit 1; }
    pip_install --no-deps -e "$REPO"
fi

"$PY" -c 'import simsoptpp, simsopt; print("simsoptpp:", simsoptpp.__file__); print("simsopt:", simsopt.__file__)'

cd "$HERE"
export OMP_NUM_THREADS="$THREADS" JAX_PLATFORMS=cpu MPLBACKEND=Agg
exec "$PY" banana_coil_solver.py
