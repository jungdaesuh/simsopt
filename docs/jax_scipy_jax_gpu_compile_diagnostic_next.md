# scipy-jax GPU compile-time residual — diagnosis and decision

Referenced by `docs/scipy_jax_11_51_matrix_implementation_plan.md:16,195` and
Phase 3 of `docs/single_stage_compile_blowup_fix_implementation_plan.md`.

## Question

The host-driven `scipy-jax` single-stage lane fixed the 422 GiB `ondevice`
compile OOM, but a compile-*time* residual remained: the lane was reported
~73 min on a RunPod cu1290 GPU image vs ~11 min on Perlmutter A100. Two
hypotheses:

- **(once-slow)** the lane compiles its kernels once (cold) and the single cold
  compile is slow on a particular GPU image, or
- **(recompile-per-eval)** the per-outer-step evaluation recompiles (a
  cache-token / shape-stability bug), so cost grows with the iteration budget.

Only (recompile-per-eval) is a code defect in the lane. The classification is
**backend-independent**: JAX `jit` recompiles are driven by Python-level
cache-token / shape logic, not hardware, so it reproduces on CPU.

## Method (CPU, no GPU required)

`benchmarks/single_stage_init_parity.py` self-compiles the runtime seed from a
CPU reference run, then runs the `scipy-jax` JAX lane with
`--record-jax-compile-diagnostics` (`jax_log_compiles` +
`jax_explain_cache_misses`). Run at two outer budgets and compare compile counts:

```
PYTHONPATH=src JAX_ENABLE_X64=1 \
  <jax-0.10.0 python> benchmarks/single_stage_init_parity.py \
  --platform cpu --mpol 2 --ntor 2 --nphi 31 --ntheta 16 --maxiter <N> \
  --optimizer-backend scipy-jax --record-jax-compile-diagnostics \
  --case-artifacts-dir <dir> --output-json <out.json>
```

(Local env: `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.10.0`.)
Read `compile_diagnostics.{cache_miss_count, compile_event_count,
cache_miss_sites}` from the output JSON.

## Result (2026-06-15, CPU, mpol2)

| run | outer iterations done | cache_miss_count | compile_event_count |
|-----|-----------------------|------------------|---------------------|
| `--maxiter 3` | 3 | 127 | 178 |
| `--maxiter 6` | 5 | 127 | 178 |

Compile counts — and the full per-site `cache_miss_sites` distribution — are
**byte-identical** across the two runs, even though they executed a **different
number of outer steps (3 vs 5)**. All 127 cache-miss sites are bounded
inner-solver / objective kernels (`geo/optimizers/private/_bfgs.py`,
`_line_search.py`, `optimizers/optimizer.py`,
`simsopt_jax_adapters/geo/surface_objectives*.py`, `core/biotsavart.py`).

Both runs exited `1` on **final-metric parity thresholds** (iota/volume/field
differences too large at a toy maxiter), not on any crash; the failures shrank
from maxiter 3 → 6 (iota `5.96e-04` → `3.90e-06`), confirming the optimizer did
real, differing work. The JAX lane completed and recorded diagnostics in both.
Peak host RSS stayed ~0.16 GiB (this lane is not the 422 GiB monolith).

This corroborates the bundle-level proof in
`tests/geo/test_boozersurface_jax.py::test_penalty_value_and_grad_bundle_reuses_compiled_executable`
(`_cache_size() == 1`, no growth on `x`-change).

## Decision

**Classification: once-slow. recompile-per-eval is REFUTED.** The outer loop
compiles its kernels exactly once and reuses them across all subsequent steps;
the iteration budget does not add compiles.

Consequences:

- **No lane code fix is warranted** for the GPU residual. There is no
  cache-token / shape-instability bug to repair (the static-vs-dynamic concern
  on the baked geometry closure, `boozer_surface.py:4761-4770`, is not triggering
  recompiles).
- The ~73-min RunPod figure is the **one cold compile** being slow on that GPU
  image, not work that scales with the optimizer. This is a platform/image
  magnitude issue: XLA:GPU autotuning, cubin / persistent-compilation cache, and
  container CUDA-toolkit vs wheel-bundled libraries (cf.
  `project_scipy_jax_gpu_compile_bound`).

## Remaining (GPU-only, magnitude — not a code defect)

- **DONE** — persistent compilation cache wired into
  `benchmarks/perlmutter/single_stage_production_gpu.slurm`:
  `JAX_COMPILATION_CACHE_DIR=${RUN_ROOT}/jax_compilation_cache` (outside
  `${JOB_ROOT}`, persists across jobs; relocated off the `$HOME` default which
  has tight Perlmutter quota). Uses runtime.py's SAFE narrow autotune mode
  (`xla_gpu_per_fusion_autotune_cache_dir`); no broader, nvlink-triggering cache
  modes. The persistent cache helps *repeat* compiles, not the first.
  **RunPod path (the tracked CUDA launcher):**
  `benchmarks/prepare_cuda_gpu_lowres_tests.py` `_cuda_env()` writes the CUDA
  shell-runner env. It was setting `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="all"`
  — the broad mode `runtime.py:2391-2397` warns forces nvlink through the
  container CUDA toolkit (the cu1290 block). **Fixed to the narrow
  `xla_gpu_per_fusion_autotune_cache_dir`** to match runtime.py and stay
  nvlink-safe. Tradeoff: the narrow mode caches only per-fusion autotune
  results (not the full XLA compile), so the cold-compile speedup is partial but
  safe. For cross-pod reuse, point the generator's `output_dir` at the persistent
  RunPod **network volume** (`/workspace/...`) so
  `output_dir/jax_compilation_cache/<label>` survives pod restarts; otherwise
  each ephemeral pod recompiles cold.
- If the first cold GPU compile itself must be cut, evaluate an XLA:GPU
  compile-time preset / reduced autotuning on that image — analogous to the CPU
  `--xla_cpu_opt_preset=FAST_COMPILE`. (CPU A/B at mpol2: ~18–19% less compile
  *work* (`user` CPU time), but no wall/RSS win at toy resolution; production
  effect unmeasured.)
- Measuring the absolute cold-compile wall and the cache-hit speedup at
  **production resolution** on the GPU image is the only step that still needs a
  GPU; the recompile-vs-once question is closed on CPU.
