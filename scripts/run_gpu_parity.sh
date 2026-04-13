#!/usr/bin/env bash
# GPU vs CPU parity test runner for GCP VMs
# Usage: PLATFORM=cuda|cpu [REPO=/path/to/repo] bash run_gpu_parity.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PLATFORM="${PLATFORM:-cpu}"
REPO="${REPO:-${PARITY_REPO:-$DEFAULT_REPO}}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/jax-parity/bin/python}"
GPU_LOCK_ENABLED="${GPU_LOCK_ENABLED:-1}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/simsopt-jax-gpu-parity.lock}"

if [ ! -d "$REPO" ]; then
  echo "ERROR: REPO does not exist: $REPO" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if ! REPO_TOPLEVEL="$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "ERROR: REPO is not a git checkout: $REPO" >&2
  exit 1
fi

REPO="$REPO_TOPLEVEL"
cd "$REPO"

cleanup_gpu_lock() {
  if [ "${_GPU_LOCK_HELD:-0}" -eq 1 ] && [ -d "$GPU_LOCK_DIR" ]; then
    rm -rf "$GPU_LOCK_DIR"
  fi
}

acquire_gpu_lock() {
  if [ "$PLATFORM" != "cuda" ] || [ "$GPU_LOCK_ENABLED" = "0" ]; then
    return
  fi

  if mkdir "$GPU_LOCK_DIR" 2>/dev/null; then
    :
  else
    existing_pid=""
    existing_repo=""
    existing_started=""
    if [ -f "$GPU_LOCK_DIR/pid" ]; then
      existing_pid="$(cat "$GPU_LOCK_DIR/pid")"
    fi
    if [ -f "$GPU_LOCK_DIR/repo" ]; then
      existing_repo="$(cat "$GPU_LOCK_DIR/repo")"
    fi
    if [ -f "$GPU_LOCK_DIR/started_at" ]; then
      existing_started="$(cat "$GPU_LOCK_DIR/started_at")"
    fi

    if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "ERROR: Another GPU parity run is active." >&2
      echo "  pid: $existing_pid" >&2
      echo "  repo: ${existing_repo:-unknown}" >&2
      echo "  started_at: ${existing_started:-unknown}" >&2
      echo "Set GPU_LOCK_ENABLED=0 only if you intentionally want overlapping GPU jobs." >&2
      exit 1
    fi

    echo "WARN: Removing stale GPU parity lock at $GPU_LOCK_DIR" >&2
    rm -rf "$GPU_LOCK_DIR"
    if ! mkdir "$GPU_LOCK_DIR"; then
      echo "ERROR: Failed to acquire GPU parity lock at $GPU_LOCK_DIR" >&2
      exit 1
    fi
  fi

  _GPU_LOCK_HELD=1
  printf '%s\n' "$$" > "$GPU_LOCK_DIR/pid"
  printf '%s\n' "$REPO" > "$GPU_LOCK_DIR/repo"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$GPU_LOCK_DIR/started_at"
  trap cleanup_gpu_lock EXIT INT TERM
}

acquire_gpu_lock

export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1

GIT_HEAD="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short --untracked-files=no)"

echo "============================================"
echo "  GPU vs CPU Parity Test Suite"
echo "  Platform: $PLATFORM"
echo "  Repo: $REPO"
echo "  Git HEAD: $GIT_HEAD"
if [ -n "$GIT_STATUS" ]; then
  echo "  Git status: DIRTY"
else
  echo "  Git status: CLEAN"
fi
if [ "$PLATFORM" = "cuda" ] && [ "$GPU_LOCK_ENABLED" != "0" ]; then
  echo "  GPU lock: $GPU_LOCK_DIR"
fi
echo "  Python: $("$PYTHON_BIN" --version)"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"

if [ -n "$GIT_STATUS" ]; then
  echo ""
  echo "Tracked worktree modifications:"
  printf '%s\n' "$GIT_STATUS"
fi

# --- Phase 1: Import smoke ---
echo ""
echo "=== Phase 1: Import Smoke ==="
"$PYTHON_BIN" -m pytest tests/test_jax_import_smoke.py -v --tb=short 2>&1 || true

# --- Phase 2: Pure JAX unit tests ---
echo ""
echo "=== Phase 2: Pure JAX Unit Tests ==="
PARITY_EXPR="cpu_parity"
if [ "$PLATFORM" = "cuda" ]; then
  PARITY_EXPR="gpu_parity"
fi

"$PYTHON_BIN" -m pytest \
  tests/field/test_biotsavart_jax.py \
  tests/geo/test_surface_fourier_jax.py \
  tests/objectives/test_integral_bdotn_jax.py \
  -v --tb=short 2>&1 || true

"$PYTHON_BIN" -m pytest \
  tests/geo/test_boozer_residual_jax.py \
  -k "$PARITY_EXPR" \
  -v --tb=short 2>&1 || true

# --- Phase 3: Biot-Savart parity tests ---
echo ""
echo "=== Phase 3: Biot-Savart Parity ==="
"$PYTHON_BIN" -m pytest tests/field/test_biotsavart_jax_parity.py -v --tb=short 2>&1 || true

# --- Phase 4: Boozer derivatives ---
echo ""
echo "=== Phase 4: Boozer Derivatives ==="
"$PYTHON_BIN" -m pytest tests/geo/test_boozer_derivatives_jax.py -v --tb=short 2>&1 || true

# --- Phase 5: Boozer surface solver ---
echo ""
echo "=== Phase 5: Boozer Surface Solver ==="
"$PYTHON_BIN" -m pytest tests/geo/test_boozersurface_jax.py -m "not private_optimizer_runtime" -v --tb=short 2>&1 || true

# --- Phase 6: GPU reproducibility contract (GPU only) ---
if [ "$PLATFORM" = "cuda" ]; then
  echo ""
  echo "=== Phase 6: GPU Reproducibility Contract ==="
  "$PYTHON_BIN" scripts/jax_ci_contract.py --platform cuda 2>&1 || true
fi

# --- Phase 7: JAX native path integration ---
echo ""
echo "=== Phase 7: JAX Native Path Integration ==="
"$PYTHON_BIN" -m pytest tests/integration/test_jax_native_path.py -v --tb=short 2>&1 || true

# --- Phase 8: Quick device sanity ---
echo ""
echo "=== Phase 8: Device Sanity Check ==="
"$PYTHON_BIN" -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
print(f'Default backend: {jax.default_backend()}')
print(f'Float64 enabled: {jax.config.x64_enabled}')

# Quick reduction order check
vals = np.logspace(12, -12, 1000, dtype=np.float64)
cpu_sum = float(jax.jit(lambda x: jnp.sum(x), backend='cpu')(jnp.array(vals)))
device_sum = float(jax.jit(lambda x: jnp.sum(x))(jnp.array(vals)))
rel_err = abs(device_sum - cpu_sum) / (abs(cpu_sum) + 1e-30)
print(f'Reduction order rel error: {rel_err:.2e}')
print(f'CPU sum: {cpu_sum}')
print(f'Device sum: {device_sum}')
print('PASS' if rel_err < 1e-10 else 'FAIL')
"

echo ""
echo "============================================"
echo "  Parity test suite complete."
echo "============================================"
