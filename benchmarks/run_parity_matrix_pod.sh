#!/usr/bin/env bash
# Reproducible launcher for the single-stage objective parity matrix on the A40 pod.
#
# Encapsulates the exact per-platform environment (so the cpu and cuda lanes are
# launched identically every time) for
# ``benchmarks.single_stage_objective_parity_matrix``.
#
# Usage:
#   run_parity_matrix_pod.sh <cpu|cuda> <out.json> <lane-label> [extra matrix args...]
# e.g.
#   run_parity_matrix_pod.sh cpu  /tmp/matrix_cpu.json jax-cpu  --nphi 64 --ntheta 24 --seeds <dir> ...
#   run_parity_matrix_pod.sh cuda /tmp/matrix_gpu.json jax-gpu  --nphi 64 --ntheta 24 --seeds <dir> ...
#
# The cuda lane requires ptxas 12.8 (jax 0.10 cubins); this script warns if the
# active ptxas is not 12.8 (restore from ptxas.bundled129.bak if a reinstall
# reverted it).  Run it under setsid for a disconnect-proof detached run.
set -euo pipefail

PLATFORM="${1:?platform: cpu|cuda}"
OUT="${2:?output json path}"
LANE="${3:?jax lane label}"
shift 3

REPO="${SIMSOPT_MATRIX_REPO:-/workspace/simsopt_lbfgs_fullcpu_a40_20260619T2155Z}"
PY="${SIMSOPT_MATRIX_PY:-/workspace/venvs/simsopt-lbfgs-a40-py312/bin/python}"
SP="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
cd "$REPO"

# float64 on both lanes (the objective is evaluated in double precision); fixed
# repo-provenance env so the example/runtime does not refuse to run.
COMMON=(SIMSOPT_REPO_SHA=x SIMSOPT_GIT_STATUS_SHORT=x JAX_ENABLE_X64=1 \
        PYTHONPATH="$REPO/src:$REPO")

if [ "$PLATFORM" = cuda ]; then
  PTX="$("$SP"/nvidia/cuda_nvcc/bin/ptxas --version | grep -o 'release [0-9.]*' | head -1)"
  echo "[launcher] ptxas $PTX"
  case "$PTX" in *12.8*) ;; *) echo "[launcher] WARNING: ptxas is not 12.8 ($PTX)";; esac
  export LD_LIBRARY_PATH="$(ls -d "$SP"/nvidia/*/lib | paste -sd: -)"
  exec env -u OPTIMIZER_BACKEND "${COMMON[@]}" \
    JAX_PLATFORMS=cuda,cpu \
    SIMSOPT_BACKEND_MODE=jax_gpu_parity \
    SIMSOPT_JAX_PLATFORM=cuda \
    SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled \
    XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true \
    JAX_COMPILATION_CACHE_DIR=/workspace/jaxcache_cuda \
    "$PY" -m benchmarks.single_stage_objective_parity_matrix \
      --out "$OUT" --lane-label "$LANE" "$@"
else
  exec env -u OPTIMIZER_BACKEND "${COMMON[@]}" \
    JAX_PLATFORMS=cpu \
    JAX_COMPILATION_CACHE_DIR=/workspace/jaxcache_cpu \
    "$PY" -m benchmarks.single_stage_objective_parity_matrix \
      --out "$OUT" --lane-label "$LANE" "$@"
fi
