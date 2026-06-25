#!/usr/bin/env bash
# Focused CUDA gate for the BiotSavart JAX performance/memory plan.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUESTED_REPO="${REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

if [ ! -d "$REQUESTED_REPO" ]; then
  echo "ERROR: REPO does not exist: $REQUESTED_REPO" >&2
  exit 1
fi

if ! REPO_TOPLEVEL="$(git -C "$REQUESTED_REPO" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "ERROR: REPO is not a git checkout: $REQUESTED_REPO" >&2
  exit 1
fi

REPO="$REPO_TOPLEVEL"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/.artifacts/biotsavart_gpu_gate}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required for the BiotSavart CUDA gate." >&2
  exit 1
fi

cd "$REPO"
mkdir -p "$RESULTS_DIR"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export SIMSOPT_JAX_PLATFORM="${SIMSOPT_JAX_PLATFORM:-cuda}"
export SIMSOPT_JAX_CUDA_LIBRARY_MODE="${SIMSOPT_JAX_CUDA_LIBRARY_MODE:-bundled}"

case " ${XLA_FLAGS:-} " in
  *" --xla_gpu_exclude_nondeterministic_ops=true "*)
    ;;
  *)
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
    ;;
esac

HEAD_SHA="$(git rev-parse HEAD)"
{
  echo "repo=$REPO"
  echo "head=$HEAD_SHA"
  echo "python=$PYTHON_BIN"
  echo "results_dir=$RESULTS_DIR"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$RESULTS_DIR/biotsavart_gpu_gate_env.txt"
nvidia-smi | tee "$RESULTS_DIR/nvidia-smi.txt"

"$PYTHON_BIN" - <<'PY' | tee "$RESULTS_DIR/jax_cuda_smoke.txt"
import sys

import jax
import simsoptpp

assert sys.version_info[:2] == (3, 11), sys.version
assert hasattr(simsoptpp, "Curve")
backend = str(jax.default_backend()).lower()
print("python", sys.executable)
print("python_version", sys.version.split()[0])
print("jax", jax.__version__)
print("backend", backend)
print("devices", [(device.platform, str(device)) for device in jax.devices()])
print("simsoptpp", simsoptpp.__file__)
if backend not in {"cuda", "gpu"}:
    raise SystemExit(f"expected CUDA/GPU backend, got {backend!r}")
PY

"$PYTHON_BIN" -m pytest -q \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppCoilCurrentParity \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedSelfConsistency \
  tests/field/test_biotsavart_jax.py::TestBiotSavartJaxAnalytical \
  tests/field/test_biotsavart_jax_cpu_ordered.py \
  2>&1 | tee "$RESULTS_DIR/pytest_biotsavart_focused.txt"

for backend_mode in jax_gpu_parity jax_gpu_fast; do
  export SIMSOPT_BACKEND_MODE="$backend_mode"
  "$PYTHON_BIN" benchmarks/biotsavart_quadrature_chunking_probe.py \
    --platform cuda \
    --ncoils 32 \
    --nquad 128 \
    --npoints 64 \
    --coil-chunk-size 16 \
    --block-sizes 0 32 64 128 \
    --kernels B,dB,B_and_dB \
    --repeat 3 \
    --warmup 1 \
    --output-json "$RESULTS_DIR/biotsavart_quadrature_chunking_${backend_mode}.json"
done

"$PYTHON_BIN" - "$RESULTS_DIR" <<'PY' | tee "$RESULTS_DIR/biotsavart_gpu_gate_summary.txt"
import json
from pathlib import Path
import sys

results_dir = Path(sys.argv[1])
for backend_mode in ("jax_gpu_parity", "jax_gpu_fast"):
    path = results_dir / f"biotsavart_quadrature_chunking_{backend_mode}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    backend = str(payload.get("backend", "")).lower()
    if backend not in {"cuda", "gpu"}:
        raise SystemExit(f"{path.name} did not run on CUDA/GPU: {backend!r}")
    if payload.get("block_sizes") != [0, 32, 64, 128]:
        raise SystemExit(f"{path.name} has unexpected block sizes")
    if not payload.get("results"):
        raise SystemExit(f"{path.name} has no results")
    print(f"{backend_mode}: backend={backend}, kernels={payload.get('kernels')}")
print(f"BiotSavart CUDA gate artifacts: {results_dir}")
PY
