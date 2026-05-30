# NEXT GPU TASK — diagnose scipy-jax single-stage compile cost

**Run this FIRST on the next GPU session, before anything else.**

## Finding (2026-05-29/30, Runpod A100 `cu1290`)

The `scipy-jax` lane — Plan A's intended **compile-once** vehicle (host SciPy
outer + one jitted `value_and_grad`) — is **compile-bound** on this box:

| Run | Lane | wall | GPU util during | result |
|---|---|---|---|---|
| single-stage (m04, maxiter=8) | `scipy-jax` (`lbfgs-scipy-jax`) | **1:12:50** | ~80% then idle | completed: J=1.124, success |
| single-stage (m04, maxiter=5) | `scipy-jax` | killed >23 min | 82% | in iteration phase |
| Stage 2 (banana, maxiter=15) | `scipy-jax` (default for `--backend jax`) | killed >18 min | **0% (CPU-bound compile)** | still compiling |

Both used `scipy-jax` (single-stage explicit `--optimizer-backend scipy-jax`;
Stage 2 via the `--backend jax` default). Compile is **CPU-bound** (100%+ CPU,
GPU near-idle). Reference: prior **Perlmutter** A100 single-stage GPU ≈ **11 min**
(see memory `project_perlmutter_run_timings`) — so this `cu1290` box is ~6× slower.

## The undiagnosed question

Is the 70-min cost **(A) one pathologically-slow compile** (compile-once works,
but the single XLA compile is huge/slow on this box) **or (B) recompile-per-eval**
(the `make_traceable` runtime-bundle cache token contract not landing → effectively
fullgraph behavior)? These have **opposite fixes** (box/XLA config vs a code bug).
The runs recorded `--record-jax-compile-diagnostics` but the compile count was
**not extracted** before the pod was stopped. This is exactly Plan A's Step-1
diagnostic ("compiles-per-eval for the `scipy-jax` lane — never measured").

## The diagnostic (decisive, cheap)

Use a tree that includes the committed gate fix `f287bde96` (so single-stage
reaches the inner solve, not the old `make_traceable` block). Run the **same**
scipy-jax single-stage probe at **two** maxiters and compare compile counts:

```bash
# repo root on the GPU box; PYTHONPATH set to the tree; seed = an m04 biot_savart_opt.json
for MI in 3 6; do
  SIMSOPT_BACKEND_MODE=jax_gpu_fast SIMSOPT_JAX_PLATFORM=cuda \
  JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_LOG_COMPILES=1 \
  env -u LD_LIBRARY_PATH python \
    examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
    --backend jax --optimizer-backend scipy-jax \
    --stage2-bs-path <SEED>/biot_savart_opt.json \
    --mpol 4 --ntor 4 --nphi 63 --ntheta 32 \
    --iota-target 0.15 --vol-target 0.10 \
    --maxiter $MI --record-jax-compile-diagnostics \
    --output-root /root/compile_probe_mi$MI 2> /root/compile_probe_mi$MI.stderr
done
# then read, for each run:
#   - jax_compile_diagnostics.json  -> compile-event count
#   - nfev / njev from results.json
#   - grep -c 'Compiling ' /root/compile_probe_mi*.stderr   (JAX_LOG_COMPILES belt-and-suspenders)
```

## Decision tree

- **Compile count ≈ constant across maxiter 3 vs 6** (and small relative to nfev)
  → **Case A: compile-once works.** The cost is one slow XLA compile on the
  `cu1290` box. Fix is environment-side: profile the XLA compile, try a newer
  jaxlib/CUDA image, or run on a faster-compiling host (Perlmutter ~11 min).
  NOT a code bug.
- **Compile count grows ~linearly with maxiter / ≈ nfev** → **Case B / Plan A
  Case III: recompile-per-eval.** The `make_traceable_objective_runtime_bundle`
  cache key is invalidating each outer step. Code fix — audit the cache token
  contract (`_traceable_solve_state_token`, `_coil_dof_state_token`, coil-layout
  signature; see CLAUDE.md "Traceable runtime bundle cache contract"). This is
  the real Plan A target.

## Why it matters

This single read decides whether "GPU instead of CPU" for the full example
workflows is **a box/image problem** (swap hosts) or **a code problem** (land
Plan A's compile-once cache fix). Until then: GPU kernels + Stage-2 field math
are validated and usable; the full `scipy-jax` example optimization loops are
compile-cost-limited on this box and are **not** a drop-in CPU replacement.
