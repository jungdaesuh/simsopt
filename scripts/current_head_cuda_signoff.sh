#!/usr/bin/env bash
# Fail-closed current-head CUDA signoff for docs/jax_port_review_remediation_plan.md.
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
RESULTS_DIR="${RESULTS_DIR:-$REPO/.artifacts/current_head_cuda_signoff}"
cd "$REPO"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO:$REPO/src"
export JAX_ENABLE_X64=True
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD=disallow

HEAD_SHA="$(git rev-parse HEAD)"
TRACKED_STATUS="$(git status --short --untracked-files=no)"
if [ -n "$TRACKED_STATUS" ]; then
  echo "ERROR: tracked worktree must be clean for CUDA signoff." >&2
  printf '%s\n' "$TRACKED_STATUS" >&2
  exit 1
fi

UNTRACKED_STATUS="$(git status --short --untracked-files=normal | sed -n 's/^?? //p')"
UNTRACKED_RELEASE_STATUS=""
while IFS= read -r untracked_path; do
  if [ -z "$untracked_path" ]; then
    continue
  fi
  case "$untracked_path" in
    .antigravitycli/* | .conda/* | .artifacts/* | analysis/* | runs/*)
      ;;
    *)
      UNTRACKED_RELEASE_STATUS="${UNTRACKED_RELEASE_STATUS}${untracked_path}"$'\n'
      ;;
  esac
done <<< "$UNTRACKED_STATUS"
if [ -n "$UNTRACKED_RELEASE_STATUS" ]; then
  echo "ERROR: non-artifact untracked paths would invalidate current-head CUDA signoff." >&2
  printf '%s' "$UNTRACKED_RELEASE_STATUS" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required for CUDA signoff." >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
printf '%s\n' "$HEAD_SHA" > "$RESULTS_DIR/repo-head.txt"
nvidia-smi > "$RESULTS_DIR/nvidia-smi.txt"

"$PYTHON_BIN" - <<'PY' | tee "$RESULTS_DIR/jax-device-probe.txt"
import jax

print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", [(device.platform, str(device)) for device in jax.devices()])
if str(jax.default_backend()).lower() not in {"cuda", "gpu"}:
    raise SystemExit("expected a CUDA/GPU JAX backend for signoff")
PY

"$PYTHON_BIN" - <<'PY' | tee "$RESULTS_DIR/transfer-guard-probe.txt"
import numpy as np

import jax
import jax.numpy as jnp

probe_value = jnp.asarray([1.0])
with jax.transfer_guard("disallow"):
    transfer_blocked = False
    try:
        np.asarray(probe_value)
    except RuntimeError:
        transfer_blocked = True

if not transfer_blocked:
    raise SystemExit("expected transfer_guard('disallow') to block implicit transfers")
print("transfer_guard disallow probe blocked implicit transfer")
PY

"$PYTHON_BIN" benchmarks/single_stage_init_parity.py \
  --platform cuda \
  --output-json "$RESULTS_DIR/single_stage_init_cuda_scipy_jax.json" \
  --case-artifacts-dir "$RESULTS_DIR/single_stage_init_cuda_scipy_jax_cases" \
  --single-stage-case-timeout-seconds 7200 \
  --single-stage-target-case-timeout-seconds 7200

"$PYTHON_BIN" benchmarks/single_stage_init_parity.py \
  --platform cuda \
  --optimizer-backend ondevice \
  --benchmark-mode \
  --output-json "$RESULTS_DIR/single_stage_init_cuda_ondevice.json" \
  --case-artifacts-dir "$RESULTS_DIR/single_stage_init_cuda_ondevice_cases" \
  --single-stage-case-timeout-seconds 7200 \
  --single-stage-target-case-timeout-seconds 7200

"$PYTHON_BIN" - "$RESULTS_DIR" "$HEAD_SHA" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
expected_repo_sha = sys.argv[2]
for name in (
    "single_stage_init_cuda_scipy_jax.json",
    "single_stage_init_cuda_ondevice.json",
):
    payload = json.loads((results_dir / name).read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise SystemExit(f"{name} did not pass: {payload.get('failures')}")
    provenance = payload["provenance"]
    if provenance.get("repo_sha") != expected_repo_sha:
        raise SystemExit(f"{name} has stale repo_sha={provenance.get('repo_sha')!r}")
    backend = str(provenance.get("backend", "")).lower()
    if backend not in {"cuda", "gpu"}:
        raise SystemExit(f"{name} did not run on CUDA/GPU backend: {backend!r}")
    if provenance.get("transfer_guard") != "disallow":
        raise SystemExit(f"{name} did not record transfer_guard=disallow")

print(f"CUDA signoff packet passed: {results_dir}")
PY
