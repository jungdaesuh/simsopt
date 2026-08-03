#!/bin/sh
# Six-round AB/BA fused-vs-Optax qualification on the landau A100 under
# deliberate host contention: 48 busy CPU workers (of 64 cores) run for
# the whole sweep. Round 0 is the discard; rounds 1-5 are retained.
# Burner runs in its OWN process group (setsid) and is killed by group
# id on exit, so worker reparenting cannot leak load (the 2026-08-03
# leak: killing the parent first orphaned 48 workers past pkill -P).
set -u
ROOT=$HOME/qn-offhost-replay/checkout
OUT=$HOME/qn-a100/artifacts/coil47-a100-contended3-63638856f
PYBIN=$HOME/qn-a100/venv/bin/python
COMPAT=$HOME/qn-a100/compat-12-6/usr/local/cuda-12.6/compat

setsid "$PYBIN" - <<'PYEOF' &
import multiprocessing, time

def burn():
    x = 1.0001
    while True:
        x = x * x % 1.7 + 1.0001

if __name__ == "__main__":
    workers = [multiprocessing.Process(target=burn, daemon=True) for _ in range(48)]
    for w in workers:
        w.start()
    time.sleep(3600)
PYEOF
LOAD_PID=$!
trap "kill -9 -- -$LOAD_PID 2>/dev/null" EXIT INT TERM
sleep 5
uptime

cd "$ROOT" || exit 1
for round in 0 1 2 3 4 5; do
  if [ $((round % 2)) -eq 0 ]; then providers="custom,optax"; else providers="optax,custom"; fi
  dest="$OUT/round-$round"
  mkdir -p "$dest"
  MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  SIMSOPT_BACKEND_MODE=jax_gpu_fast SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  LD_LIBRARY_PATH="$COMPAT" \
  CUDA_VISIBLE_DEVICES=GPU-250014ca-8cb3-bdcd-ad1d-2f6f64529b8d PYTHONPATH=src:. \
  "$PYBIN" benchmarks/custom_quasi_newton_runtime.py \
    --device gpu --intent fast --providers "$providers" --cases coil47 \
    --method lbfgs --maxiter 20 \
    --output "$dest" > "$dest/run.log" 2>&1
  echo "round=$round providers=$providers exit=$?"
done
uptime
echo "A100 CONTENDED SWEEP DONE"
