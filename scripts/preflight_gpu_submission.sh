#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.conda/jax-0.10.0/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/simsopt-jax-gpu-preflight}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$REPO"

export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export SIMSOPT_BACKEND_MODE=jax_cpu_parity
export SIMSOPT_BACKEND_STRICT=1
export SIMSOPT_JAX_TRANSFER_GUARD=disallow
export SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cpu

NON_BANANA_JSON="$RESULTS_DIR/non_banana_full_strict_preflight.json"

echo "GPU submission preflight"
echo "  repo:        $REPO"
echo "  python:      $PYTHON_BIN"
echo "  results_dir: $RESULTS_DIR"
echo "  transfer:    $SIMSOPT_JAX_TRANSFER_GUARD"

"$PYTHON_BIN" - <<'PY'
import jax
import jaxlib

print(f"  jax:         {jax.__version__}")
print(f"  jaxlib:      {jaxlib.__version__}")
print(f"  backend:     {jax.default_backend()}")
print(f"  devices:     {jax.devices()}")
print(f"  x64:         {jax.config.x64_enabled}")

if jax.default_backend() != "cpu":
    raise SystemExit(f"expected CPU backend, got {jax.default_backend()!r}")
if jax.config.x64_enabled is not True:
    raise SystemExit("expected JAX x64 to be enabled")
PY

"$PYTHON_BIN" benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
  --lanes cpu_cpp,jax_cpu \
  --output-json "$NON_BANANA_JSON"

"$PYTHON_BIN" - "$NON_BANANA_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

fixtures = payload["fixtures"]
failed = [
    (fixture["fixture_id"], fixture.get("verdict"), fixture.get("error"))
    for fixture in fixtures
    if not fixture.get("passed", False)
]
counts = {}
for fixture in fixtures:
    counts[fixture.get("verdict")] = counts.get(fixture.get("verdict"), 0) + 1

print(f"non-banana fixtures: {len(fixtures)}")
print(f"non-banana verdicts: {counts}")
if failed:
    for fixture_id, verdict, error in failed:
        print(f"FAILED {fixture_id}: {verdict}: {error}", file=sys.stderr)
    raise SystemExit(1)
PY

"$PYTHON_BIN" -m pytest -q \
  tests/test_host_boundary.py \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_relax_and_split_jax_runs_eager_under_strict_transfer_guard \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_gpmo_baseline_jax_runs_eager_under_strict_transfer_guard \
  tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_gpmo_arbvec_backtracking_jax_runs_eager_under_strict_transfer_guard \
  tests/solve/test_wireframe_optimization_jax_item31.py::test_regularized_constrained_least_squares_runs_eager_under_transfer_guard \
  tests/solve/test_wireframe_optimization_jax_item31.py::test_rcls_wireframe_jax_runs_eager_under_transfer_guard

echo "GPU submission preflight passed."
