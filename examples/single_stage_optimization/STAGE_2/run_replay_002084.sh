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

if [ ! -x "$VENV/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv -p 3.11 "$VENV"
        uv pip install -p "$VENV/bin/python" "numpy<2.3" scipy "jax[cpu]" jaxlib Deprecated monty ruamel.yaml sympy f90nml pyevtk matplotlib shapely numba
    else
        python3.11 -m venv "$VENV"
        "$VENV/bin/pip" install "numpy<2.3" scipy "jax[cpu]" jaxlib Deprecated monty ruamel.yaml sympy f90nml pyevtk matplotlib shapely numba
    fi
fi
PY="$VENV/bin/python"

# Prebuilt extension for this interpreter, if present.
TAG="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"   # e.g. cpython-311-x86_64-linux-gnu
SO="$(ls "$REPO"/build/*/simsoptpp."$TAG".so 2>/dev/null | head -1 || true)"
if [ -n "$SO" ]; then
    export PYTHONPATH="$REPO/src:$(dirname "$SO")${PYTHONPATH:+:$PYTHONPATH}"
else
    echo "no prebuilt simsoptpp for $TAG under $REPO/build; building with pip install -e ." >&2
    "$PY" -m pip install --no-deps -e "$REPO"
fi

"$PY" -c 'import simsoptpp, simsopt; print("simsoptpp:", simsoptpp.__file__); print("simsopt:", simsopt.__file__)'

cd "$HERE"
export OMP_NUM_THREADS="$THREADS" JAX_PLATFORMS=cpu MPLBACKEND=Agg
exec "$PY" banana_coil_solver.py
