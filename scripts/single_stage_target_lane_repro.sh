#!/usr/bin/env bash
# Target-lane-only single-stage reproducer for the `lbfgs-ondevice` outer optimizer.
#
# WHY THIS EXISTS
# ---------------
# `benchmarks/single_stage_init_parity.py` is the signoff *parity* wrapper. For
# every rung it runs FOUR lanes serially (seed -> C++ CPU reference -> JAX
# same-candidate replay -> JAX target). At m04 the C++ reference alone is ~76 min
# and runs at 0% GPU utilization on the rented card, so using the full parity
# wrapper as a *debug loop* for the JAX target lane re-pays ~1.5 h of non-target
# work before the lane under test even starts.
#
# This script runs ONLY the JAX target lane (the production `lbfgs-ondevice`
# outer optimizer) against a precomputed warm-start donor. It is the smallest
# reproducer for the target-lane compile/stall blocker. It is a DEBUG tool, not a
# signoff path: parity/oracle comparison still belongs to single_stage_init_parity.py.
#
# DURABILITY
# ----------
# Multi-hour runs launched in an SSH foreground die on disconnect (SIGHUP). Always
# launch this under tmux/nohup on a remote box, e.g.:
#   tmux new-session -d -s repro 'bash scripts/single_stage_target_lane_repro.sh'
#
# USAGE
# -----
#   REPO=/path/to/simsopt-jax \
#   PYTHON_BIN=/path/to/venv/bin/python \
#   DONOR=/path/to/prior_run/case_artifacts/seed_outputs/mpol=4-ntor=4-XXXX \
#   PLATFORM=cuda MPOL=4 NTOR=4 NPHI=63 NTHETA=32 MAXITER=1 \
#   bash scripts/single_stage_target_lane_repro.sh
#
# The DONOR must be a prior single-stage run directory containing surf_opt.json +
# results.json (a maxiter=0 seed_outputs dir or any completed donor works).
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PLATFORM="${PLATFORM:-cuda}"
OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-ondevice}"
MPOL="${MPOL:-4}"
NTOR="${NTOR:-4}"
NPHI="${NPHI:-63}"
NTHETA="${NTHETA:-32}"
MAXITER="${MAXITER:-1}"
VOL_TARGET="${VOL_TARGET:-0.1}"
IOTA_TARGET="${IOTA_TARGET:-0.15}"
NUM_TF_COILS="${NUM_TF_COILS:-20}"
OUTER_MAXLS="${OUTER_MAXLS:-8}"
PLASMA_SURF="${PLASMA_SURF:-wout_nfp22ginsburg_000_014417_iota15.nc}"
STAGE2_BS_PATH="${STAGE2_BS_PATH:-$REPO/benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json}"
EQUILIBRIA_DIR="${EQUILIBRIA_DIR:-$REPO/examples/single_stage_optimization/equilibria}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"

if [ -z "${DONOR:-}" ]; then
  echo "ERROR: DONOR (warm-start run dir with surf_opt.json + results.json) is required." >&2
  exit 1
fi
for required in "$DONOR/surf_opt.json" "$DONOR/results.json" "$STAGE2_BS_PATH" "$EQUILIBRIA_DIR/$PLASMA_SURF"; do
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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/.artifacts/target_lane_repro/repro_m${MPOL}n${NTOR}_${PLATFORM}_${TS}}"
mkdir -p "$OUTPUT_ROOT/out"

# Provenance: prefer git, fall back to the env contract the deploy uses on .git-less copies.
if REPO_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"; then
  GIT_STATUS_SHORT="$(git -C "$REPO" status --short --untracked-files=no 2>/dev/null)"
else
  REPO_SHA="${SIMSOPT_REPO_SHA:-unknown}"
  GIT_STATUS_SHORT="${SIMSOPT_GIT_STATUS_SHORT:-}"
fi

# Import-order contract: simsopt MUST be imported (to resolve GPU memory env)
# before jax. The example imports jax before its own bootstrap, so we launch it
# through the same bootstrap_local_simsopt() `-c` wrapper the parity harness uses.
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export SIMSOPT_REPO_SHA="$REPO_SHA"
export SIMSOPT_GIT_STATUS_SHORT="$GIT_STATUS_SHORT"
export SIMSOPT_BACKEND=jax          # NOTE: --backend jax (CLI) alone does NOT set the runtime config.
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
fi

cd "$REPO"

echo "============================================"
echo "  Single-stage target-lane reproducer"
echo "  Platform:    $PLATFORM   Optimizer: $OPTIMIZER_BACKEND"
echo "  Resolution:  mpol=$MPOL ntor=$NTOR nphi=$NPHI ntheta=$NTHETA maxiter=$MAXITER"
echo "  Repo SHA:    $REPO_SHA${GIT_STATUS_SHORT:+ (DIRTY)}"
echo "  Donor:       $DONOR"
echo "  Output:      $OUTPUT_ROOT"
echo "  Timeout:     ${TIMEOUT_SECONDS}s"
echo "============================================"

# Background GPU monitor (CUDA only).
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

TIME_BIN=""
if command -v /usr/bin/time >/dev/null 2>&1; then
  TIME_BIN="/usr/bin/time -v -o $OUTPUT_ROOT/time.stderr"
fi

# shellcheck disable=SC2086
$TIME_BIN timeout --preserve-status "${TIMEOUT_SECONDS}s" "$PYTHON_BIN" -c "$BOOTSTRAP" \
  "$REPO" \
  "$REPO/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py" \
  --backend jax --optimizer-backend "$OPTIMIZER_BACKEND" \
  --plasma-surf-filename "$PLASMA_SURF" \
  --stage2-bs-path "$STAGE2_BS_PATH" \
  --nphi "$NPHI" --ntheta "$NTHETA" --mpol "$MPOL" --ntor "$NTOR" \
  --vol-target "$VOL_TARGET" --iota-target "$IOTA_TARGET" --num-tf-coils "$NUM_TF_COILS" \
  --initial-step-scale 1.0 --initial-step-maxiter 0 --outer-maxls "$OUTER_MAXLS" \
  --warm-start-run-dir "$DONOR" \
  --maxiter "$MAXITER" \
  --equilibria-dir "$EQUILIBRIA_DIR" \
  --output-root "$OUTPUT_ROOT/out" \
  --benchmark-mode --minimal-artifacts \
  > "$OUTPUT_ROOT/stdout.log" 2> "$OUTPUT_ROOT/stderr.log"
EXIT_CODE=$?

echo "REPRO_EXIT=$EXIT_CODE" > "$OUTPUT_ROOT/exit_code.txt"
echo "Exit code: $EXIT_CODE"
echo "Logs: $OUTPUT_ROOT/{stdout.log,stderr.log,time.stderr,nvidia-smi.csv}"
exit "$EXIT_CODE"
