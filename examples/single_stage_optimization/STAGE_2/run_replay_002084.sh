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

# jax/jaxlib are pinned for reproducibility of the environment, NOT because the version decides
# the outcome. The Lp(4) curvature penalty reaches 1e13 on L-BFGS-B trial steps around evaluation
# 16-26, and whether the line search recovers from that spike (300 iterations, ~1.85 m coil) or
# collapses (8-50 iterations, ~0.5-0.6 m coil) is decided by last-bit floating-point noise.
# Measured 2026-09-16: macOS arm64 0.10.0 long / 0.10.2 stall (one run each); Linux x86_64
# 0.10.0 stall at 1 and 4 threads, long at 8 threads; 0.10.2 long at 1 and 4 threads. Hence the
# outcome check + thread-count retry below.
DEPS=("numpy>=2.4,<3" scipy "jax[cpu]==0.10.0" "jaxlib==0.10.0" Deprecated monty ruamel.yaml sympy f90nml pyevtk matplotlib shapely numba)
if command -v uv >/dev/null 2>&1; then
    [ -x "$VENV/bin/python" ] || uv venv -p 3.11 "$VENV"
    pip_install() { uv pip install -p "$VENV/bin/python" "$@"; }
else
    [ -x "$VENV/bin/python" ] || python3.11 -m venv "$VENV"
    pip_install() { "$VENV/bin/python" -m pip install "$@"; }
fi
pip_install "${DEPS[@]}"
PY="$VENV/bin/python"

# setuptools_scm writes src/simsopt/_version.py at build time and .gitignore excludes it, so a fresh
# clone has no such file and `import simsopt` fails at src/simsopt/__init__.py. The prebuilt-extension
# path below puts $REPO/src on PYTHONPATH without ever installing simsopt, so generate it here.
if [ ! -f "$REPO/src/simsopt/_version.py" ]; then
    pip_install setuptools_scm
    "$PY" - "$REPO" <<'PYEOF'
import sys
from setuptools_scm import get_version
repo = sys.argv[1]
version = get_version(root=repo)
with open(f"{repo}/src/simsopt/_version.py", "w") as handle:
    handle.write(f"version = {version!r}\n__version__ = {version!r}\n")
print(f"generated src/simsopt/_version.py: {version}")
PYEOF
fi

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
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
OUT_DIR="outputs-wout_nfp5ginsburg_000_002084_iota20.nc/R0=0.915-s=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.210-Order=2"

# Outcome check against the April root's own results.json: the root ran to the 300-iteration cap at
# 1.844 m. A stalled replay ends in <100 iterations at ~0.5-0.6 m. Anything else is a genuinely
# different result and is reported as such.
check_outcome() {
    "$PY" - "$HERE/$OUT_DIR/results.json" "$HERE/replay_002084_root_results.json" <<'PYEOF'
import json, sys
got, root = (json.load(open(p)) for p in sys.argv[1:3])
it, L = got["iterations"], got["COIL_LENGTH"]
L0 = root["COIL_LENGTH"]
if it >= 250 and abs(L - L0) / L0 < 0.05:
    print(f"outcome: reproduces the root basin ({it} iterations, {L:.4f} m vs root {L0:.4f} m)"); sys.exit(0)
if it < 100 and L < 1.0:
    print(f"outcome: STALLED ({it} iterations, {L:.4f} m) - line-search knife edge, retrying"); sys.exit(2)
print(f"outcome: DIFFERENT ({it} iterations, {L:.4f} m vs root {L0:.4f} m) - not a stall, not the root basin"); sys.exit(3)
PYEOF
}

# Thread count perturbs the last bits (OpenMP reduction order); on a serial build every attempt is
# bit-identical, so a stall there is reported rather than retried.
ATTEMPT_THREADS=("$THREADS" 8 1 2 3)
for t in "${ATTEMPT_THREADS[@]}"; do
    echo "=== replay attempt with OMP_NUM_THREADS=$t" >&2
    OMP_NUM_THREADS="$t" "$PY" banana_coil_solver.py
    check_outcome && exit 0
    rc=$?
    [ "$rc" -eq 2 ] || exit "$rc"
done
echo "all attempts stalled; on a serial (non-OpenMP) build the attempts are identical - see REPLAY_002084.md" >&2
exit 2
