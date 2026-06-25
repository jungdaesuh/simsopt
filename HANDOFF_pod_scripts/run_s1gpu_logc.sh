#!/usr/bin/env bash
set -x
source /workspace/run_env.sh
# nvlink-safe: use the jax[cuda12] wheel's bundled CUDA 12.9 userspace (matches the
# committed Perlmutter launchers); avoids the container toolkit's 12.9-vs-12.4 nvlink fatal.
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cuda
export JAX_LOG_COMPILES=1
/usr/bin/time -v python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax --mpol 8 --ntor 8 --nphi 255 --ntheta 64 \
  --plasma-surf-filename wout_s01_1f082f_opt.nc \
  --stage2-bs-path /workspace/seeds/s1_m8_iota046/biot_savart_opt.json \
  --vol-target 0.05 --iota-target 0.0459 --num-tf-coils 20 \
  --warm-start-run-dir /workspace/seeds/s1_m8_iota046 \
  --jax-runtime-seed-spec /workspace/s1_spec.json \
  --initial-step-scale 1.0 --initial-step-maxiter 0 --outer-maxls 8 --init-only \
  --optimizer-backend scipy-jax \
  --output-root /workspace/bench/s1logc/out \
  --equilibria-dir examples/single_stage_optimization/equilibria
echo "S1LOGC_EXIT=$?"
