#!/usr/bin/env bash
# Replay an autoresearch single-stage seed in the simsopt-jax single-stage lane.
#
# WHY THIS EXISTS
# ---------------
# `scripts/select_replayable_seeds.py` mines an autoresearch results jsonl for
# converged, hardware-feasible, format-ready seeds and emits a replay manifest.
# This launcher consumes one such seed (a single-stage artifact dir holding
# biot_savart_opt.json + surf_opt.json + results.json) and re-runs it here, on
# CPU or CUDA, through the chosen optimizer backend, recording compile
# diagnostics. It is the bridge from "good seed picked" to "run it in this repo".
#
# The seed dir serves a DUAL role: it is both the `--stage2-bs-path` source
# (biot_savart_opt.json) and the `--warm-start-run-dir` donor (surf_opt.json +
# results.json). The JAX runtime seed spec is reconstructed on the fly at the
# requested resolution, so no separate seed-compile step is needed.
#
# PHYSICS CAVEAT
# --------------
# simsopt-jax single-stage is a VACUUM path: it does not read finite Boozer I /
# net plasma current. A finite-current surrogate seed (BOOZER_I != 0) is
# format-replayable but is re-projected onto the vacuum Boozer surface at init,
# so the surrogate's converged metrics are NOT a byte oracle for this run. Use
# such seeds for CPU-vs-GPU parity (does this lane agree across devices), not as
# a physics ground truth. The selector flags this as PARITY-INPUT-ONLY.
#
# DURABILITY
# ----------
# Multi-hour remote runs launched in an SSH foreground die on disconnect
# (SIGHUP). Always launch under tmux/nohup on a remote box:
#   tmux new-session -d -s replay 'bash scripts/replay_surrogate_seed.sh'
#
# USAGE
# -----
#   SEED_DIR=/path/to/outputs-.../mpol=8-ntor=6-XXXX \
#   EQUILIBRIUM_PATH=/path/to/wout_....nc \
#   PLATFORM=cpu OPTIMIZER_BACKEND=scipy-jax \
#   MPOL=8 NTOR=6 NPHI=127 NTHETA=32 IOTA_TARGET=0.16 MAXITER=3 \
#   EXTRA_ARGS="--cc-weight 200 --res-weight 2000 --iotas-weight 200" \
#   bash scripts/replay_surrogate_seed.sh
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PLATFORM="${PLATFORM:-cpu}"
OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-scipy-jax}"
MPOL="${MPOL:-8}"
NTOR="${NTOR:-6}"
NPHI="${NPHI:-127}"
NTHETA="${NTHETA:-32}"
MAXITER="${MAXITER:-3}"
VOL_TARGET="${VOL_TARGET:-0.1}"
IOTA_TARGET="${IOTA_TARGET:-0.16}"
NUM_TF_COILS="${NUM_TF_COILS:-20}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "${SEED_DIR:-}" ]; then
  echo "ERROR: SEED_DIR (single-stage artifact dir with biot_savart_opt.json + surf_opt.json + results.json) is required." >&2
  exit 1
fi
if [ -z "${EQUILIBRIUM_PATH:-}" ]; then
  echo "ERROR: EQUILIBRIUM_PATH (target wout_*.nc) is required." >&2
  exit 1
fi
for required in "$SEED_DIR/biot_savart_opt.json" "$SEED_DIR/surf_opt.json" "$SEED_DIR/results.json" "$EQUILIBRIUM_PATH"; do
  if [ ! -e "$required" ]; then
    echo "ERROR: missing required input: $required" >&2
    exit 1
  fi
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/.artifacts/seed_replay/replay_m${MPOL}n${NTOR}_${PLATFORM}_${OPTIMIZER_BACKEND}_${TS}}"
mkdir -p "$OUTPUT_ROOT/out"

# The JAX lane needs an immutable runtime seed spec compiled from the seed dir's
# surf_opt.json + results.json + biot_savart_opt.json. We write it to a local
# cache keyed by (seed, resolution) so the external autoresearch dir is never
# mutated and repeat runs reuse the conversion.
SEED_SPEC_CACHE="${SEED_SPEC_CACHE:-$REPO/.artifacts/seed_specs}"
SPEC_KEY="$(basename "$(dirname "$SEED_DIR")")__$(basename "$SEED_DIR")__m${MPOL}n${NTOR}_p${NPHI}_t${NTHETA}"
SPEC_PATH="$SEED_SPEC_CACHE/${SPEC_KEY}.json"
mkdir -p "$SEED_SPEC_CACHE"

if REPO_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"; then
  GIT_STATUS_SHORT="$(git -C "$REPO" status --short --untracked-files=no 2>/dev/null)"
else
  REPO_SHA="${SIMSOPT_REPO_SHA:-unknown}"
  GIT_STATUS_SHORT="${SIMSOPT_GIT_STATUS_SHORT:-}"
fi

# Import-order contract: simsopt MUST import before jax (GPU memory env resolves
# at simsopt import). Launch through the bootstrap_local_simsopt() -c wrapper.
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export SIMSOPT_REPO_SHA="$REPO_SHA"
export SIMSOPT_GIT_STATUS_SHORT="$GIT_STATUS_SHORT"
export SIMSOPT_BACKEND=jax          # --backend jax (CLI) alone does NOT set the runtime config.
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD="${SIMSOPT_JAX_TRANSFER_GUARD:-disallow}"
export JAX_ENABLE_X64=1
export SIMSOPT_JAX_PLATFORM="$PLATFORM"
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM="$PLATFORM"
if [ "$PLATFORM" = "cuda" ]; then
  export JAX_PLATFORMS=cuda,cpu
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export SIMSOPT_JAX_GPU_PREALLOCATE="${SIMSOPT_JAX_GPU_PREALLOCATE:-false}"
  export SIMSOPT_JAX_GPU_MEM_FRACTION="${SIMSOPT_JAX_GPU_MEM_FRACTION:-0.90}"
  export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_exclude_nondeterministic_ops=true}"
else
  export JAX_PLATFORM_NAME=cpu
fi

cd "$REPO"

echo "============================================"
echo "  Surrogate-seed replay"
echo "  Platform:    $PLATFORM   Optimizer: $OPTIMIZER_BACKEND"
echo "  Resolution:  mpol=$MPOL ntor=$NTOR nphi=$NPHI ntheta=$NTHETA maxiter=$MAXITER"
echo "  Repo SHA:    $REPO_SHA${GIT_STATUS_SHORT:+ (DIRTY)}"
echo "  Seed dir:    $SEED_DIR"
echo "  Equilibrium: $EQUILIBRIUM_PATH"
echo "  Output:      $OUTPUT_ROOT"
echo "  Extra args:  ${EXTRA_ARGS:-<none>}"
echo "  Timeout:     ${TIMEOUT_SECONDS}s"
echo "============================================"

MON=""
if [ "$PLATFORM" = "cuda" ] && command -v nvidia-smi >/dev/null 2>&1; then
  ( while true; do
      nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits \
        >> "$OUTPUT_ROOT/nvidia-smi.csv" 2>/dev/null || true
      sleep 5
    done ) &
  MON=$!
  trap 'kill $MON 2>/dev/null || true' EXIT INT TERM
fi

BOOTSTRAP='import runpy, sys; repo_root, script_path, *script_args = sys.argv[1:]; sys.path.insert(0, repo_root); sys.path.insert(0, repo_root + "/src"); from benchmarks.validation_ladder_common import bootstrap_local_simsopt; bootstrap_local_simsopt(); sys.argv = [script_path, *script_args]; runpy.run_path(script_path, run_name="__main__")'

# Peak-RSS capture needs GNU time (`-v` + `-o`). Linux /usr/bin/time is GNU;
# macOS ships BSD time (no -v/-o) so prefer Homebrew `gtime`, else skip the
# wrapper (the example still records elapsed in results.json).
TIME_BIN=""
if command -v gtime >/dev/null 2>&1; then
  TIME_BIN="gtime -v -o $OUTPUT_ROOT/time.stderr"
elif /usr/bin/time -v true >/dev/null 2>&1; then
  TIME_BIN="/usr/bin/time -v -o $OUTPUT_ROOT/time.stderr"
fi

# STEP 1 -- compile the runtime seed spec from the seed dir (idempotent; cached).
if [ -f "$SPEC_PATH" ]; then
  echo "[step1/convert] reusing cached seed spec: $SPEC_PATH"
else
  echo "[step1/convert] compiling seed spec -> $SPEC_PATH"
  "$PYTHON_BIN" -c "$BOOTSTRAP" \
    "$REPO" \
    "$REPO/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py" \
    --backend jax \
    --compile-jax-runtime-seed-spec \
    --warm-start-run-dir "$SEED_DIR" \
    --jax-runtime-seed-spec "$SPEC_PATH" \
    --equilibrium-path "$EQUILIBRIUM_PATH" \
    --stage2-bs-path "$SEED_DIR/biot_savart_opt.json" \
    --nphi "$NPHI" --ntheta "$NTHETA" --mpol "$MPOL" --ntor "$NTOR" \
    --num-tf-coils "$NUM_TF_COILS" \
    --output-root "$OUTPUT_ROOT/out" \
    > "$OUTPUT_ROOT/convert.stdout.log" 2> "$OUTPUT_ROOT/convert.stderr.log"
  CONVERT_RC=$?
  echo "CONVERT_RC=$CONVERT_RC" >> "$OUTPUT_ROOT/exit_code.txt"
  if [ "$CONVERT_RC" -ne 0 ] || [ ! -f "$SPEC_PATH" ]; then
    echo "ERROR: seed-spec conversion failed (RC=$CONVERT_RC); see $OUTPUT_ROOT/convert.stderr.log" >&2
    tail -20 "$OUTPUT_ROOT/convert.stderr.log" >&2 2>/dev/null
    exit 1
  fi
fi

# STEP 2 -- run the single-stage optimization from the converted spec.
echo "[step2/run] launching $OPTIMIZER_BACKEND on $PLATFORM (maxiter=$MAXITER)"
# shellcheck disable=SC2086
$TIME_BIN timeout --preserve-status "${TIMEOUT_SECONDS}s" "$PYTHON_BIN" -c "$BOOTSTRAP" \
  "$REPO" \
  "$REPO/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py" \
  --backend jax --optimizer-backend "$OPTIMIZER_BACKEND" \
  --equilibrium-path "$EQUILIBRIUM_PATH" \
  --stage2-bs-path "$SEED_DIR/biot_savart_opt.json" \
  --warm-start-run-dir "$SEED_DIR" \
  --jax-runtime-seed-spec "$SPEC_PATH" \
  --nphi "$NPHI" --ntheta "$NTHETA" --mpol "$MPOL" --ntor "$NTOR" \
  --vol-target "$VOL_TARGET" --iota-target "$IOTA_TARGET" --num-tf-coils "$NUM_TF_COILS" \
  --maxiter "$MAXITER" \
  --record-jax-compile-diagnostics \
  --output-root "$OUTPUT_ROOT/out" \
  $EXTRA_ARGS \
  > "$OUTPUT_ROOT/stdout.log" 2> "$OUTPUT_ROOT/stderr.log"
EXIT_CODE=$?

echo "REPLAY_EXIT=$EXIT_CODE" >> "$OUTPUT_ROOT/exit_code.txt"
echo "Exit code: $EXIT_CODE"
echo "Logs: $OUTPUT_ROOT/{stdout.log,stderr.log,time.stderr,nvidia-smi.csv}"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
exit "$EXIT_CODE"
