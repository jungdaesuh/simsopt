# Using the JAX Backend

This document is the user-facing contract for the Columbia `simsopt-jax`
backend lane as of 2026-04-01.

It is intentionally narrower than the architecture notes in
[`gpu_jax_pro.md`](/Users/suhjungdae/code/columbia/simsopt-jax/gpu_jax_pro.md).
Use this file for:

- backend selection
- parity expectations
- strict-mode expectations
- reporting terminology
- current supported workflow lanes

## Quick start

Programmatic selection is the primary interface:

```python
import simsopt.config

cfg = simsopt.config.set_backend("jax_cpu_parity")
policy = simsopt.config.get_backend_policy()

print(cfg.mode)
print(policy.jax_platform)
```

Current public modes:

- `native_cpu`
- `jax_cpu_parity`
- `jax_gpu_parity`
- `jax_gpu_fast`

Default rollout lane:

- `native_cpu` remains the default indefinitely.
- Treat JAX lanes as opt-in validation and acceleration lanes until the
  production-scale benchmark and ship-proof gates are closed.

## Mode guide

### `native_cpu`

- backend: CPU reference path
- role: oracle / baseline
- expected use: correctness comparison, non-JAX workflows

### `jax_cpu_parity`

- backend: JAX on CPU
- role: algorithmic parity oracle for the JAX backend
- contract:
  - x64 required
  - stable chunking policy
  - `transfer_guard="log"` by default

### `jax_gpu_parity`

- backend: JAX on GPU
- role: device-parity lane
- contract:
  - x64 required
  - parity-oriented chunking policy
  - `transfer_guard="log"` by default

### `jax_gpu_fast`

- backend: JAX on GPU
- role: performance-oriented target lane
- contract:
  - may use more aggressive chunking
  - not the primary parity oracle

## Strict mode

Strict mode is for catching unsupported compatibility routes early:

```python
import simsopt.config

simsopt.config.set_backend("jax_gpu_parity", strict=True)
```

Use `strict=True` to fail immediately instead of silently using known forbidden
compatibility seams.

## Reporting contract

Parity reporting is split into three categories. Do not mix them in one bucket.

### Algorithmic parity

Compare:

- `native_cpu` vs `jax_cpu_parity`

Meaning:

- same formulas
- same quadrature and chunking intent
- same dtype policy
- same reference objective/gradient contracts

Typical artifacts:

- objective value parity
- gradient or adjoint parity
- finite-difference checks
- kernel-level invariant checks

### Device parity

Compare:

- `jax_cpu_parity` vs `jax_gpu_parity`

Meaning:

- same JAX algorithm on different devices
- cross-device agreement under parity mode

Typical artifacts:

- CPU JAX vs GPU JAX value comparisons
- CPU JAX vs GPU JAX gradient comparisons
- reduced-fixture GPU regression smoke tests

### Physics parity

Compare:

- final workflow outcomes, not just local numerics

Meaning:

- optimization-level or solver-level agreement
- invariants and final objective quality still hold

Typical artifacts:

- final solver outcome quality
- final objective / residual agreement
- physics invariants on representative fixtures

## Current supported lanes

The currently strongest JAX-backed lanes are:

- Stage 2 fixed-surface target/objective paths
- single-stage traceable target lane
- grouped Biot-Savart forward and derivative validation paths
- Boozer and single-stage objective slices already routed through immutable
  specs where documented in the roadmap docs

Single-stage traceable target-lane contract:

- `make_traceable_objective()` returns a pure JAX scalar callable for traced /
  ondevice optimizer paths.
- `make_traceable_objective_value_and_grad()` returns the fused pure JAX
  `(value, grad)` callable for ondevice L-BFGS.
- `make_traceable_objective_runtime_bundle()` exposes those pure-JAX callables
  plus `host_objective` and `host_value_and_grad` wrappers for Python-float /
  NumPy consumers that need an explicit host boundary.
- The runtime bundle is cached against deterministic signatures of the solved
  baseline state, objective kwargs, and coil/runtime specs. Rebuild it after
  changing those inputs; do not mutate captured objects and expect an existing
  bundle to retarget itself.
- The single-stage accepted-step path now uses that runtime bundle as the
  primary reporting source, caches one accepted-step summary in array-native run
  state, and reuses it at final reporting time when the final DOFs still match
  the accepted state.
- When available, solved-state snapshots should come from the explicit
  `BoozerSurfaceJAX.get_solved_runtime_state()` contract rather than by
  reconstructing solved geometry from mutable host wrappers.
- On the supported on-device LS lane, adjoint consumers should call
  `BoozerSurfaceJAX.get_adjoint_runtime_state()` and use its solve callbacks.
  Dense `PLU` artifacts may still exist for parity/debug purposes, but they are
  no longer the supported JAX runtime contract.
- JAX runtime states now report `linear_solve_backend="operator"` and
  `dense_linear_solve_factors_available`; `linear_solve_factors` is not used by
  the JAX adjoint runtime.

The current CPU reference lane remains the oracle for broad workflow trust.
Public acceptance still centers on the `native_cpu` / `scipy` oracle lane.
When `backend="jax"` is active, the supported optimizer lane is `ondevice`;
the retained `scipy` adapter stays available only on the CPU/reference path.

Exact Boozer note:

- The exact Newton solve keeps the loop matrix-free with JAX JVP + GMRES.
- The final dense Jacobian and optional public `PLU` metadata remains size-limited by
  `BoozerSurfaceJAX(..., options={"max_dense_jacobian_bytes": ...})` on the
  public `run_code()` / CPU-compatible result lane.
- `run_code_traceable()` disables exact dense-Jacobian finalization entirely so
  the target runtime lane stays operator-only even for small exact fixtures.
- Exact JAX adjoints do not use those dense factors. They solve forward and
  transposed systems through the Jacobian operator callbacks, including traceable
  warm-start prediction.
- Newton-polish runtime steps use the Hessian operator only. Dense Hessian
  materialization is limited to the explicit final public metadata path when
  requested.
- On the public exact result lane, if dense finalization would exceed that byte
  ceiling, the solve skips dense metadata materialization instead of attempting
  a multi-GB allocation.
- Public-lane ceiling hits are reported explicitly as
  `failure_category="scaling_limit"` at
  `failure_stage="dense_jacobian_finalization"`, with
  `jacobian_materialized=False`,
  `dense_jacobian_shape`, `dense_jacobian_bytes`, and
  `max_dense_jacobian_bytes` included in the result dict.
- Treat that outcome as a predictable exact-mode size limit, not as random
  Newton instability or a physics-correctness failure.

Adjoint and warm-start linear solve note:

- JAX adjoint and warm-start solves are operator-backed. Exact JAX has no dense
  PLU shortcut or substitute path.
- Batched exact adjoints intentionally call the operator solve once per RHS
  column; current standard-wrapper batch width is small, and exact mode is not
  the production hot path for the wrapper trio.
- If a traceable forward solve succeeds but the adjoint operator solve fails,
  the forward value remains the primal value and the gradient is non-finite.
  Do not replace that with a direct-gradient or failure-penalty substitute.

## Copy-paste workflow examples

All commands below assume:

```bash
cd /Users/suhjungdae/code/columbia/simsopt-jax
```

### Stage 2 on the default CPU reference lane

```bash
SIMSOPT_BACKEND_MODE=native_cpu \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend cpu \
  --optimizer-backend scipy \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --probe-only \
  --skip-postprocess
```

Use this first when you want:

- the default rollout lane
- the broadest trusted oracle behavior
- a cheap contract check before any JAX run

### Stage 2 on the JAX CPU parity lane

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --optimizer-backend ondevice \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --probe-only \
  --skip-postprocess
```

Use this to validate the JAX Stage 2 path on CPU before moving to GPU.
It does not exercise the retained SciPy oracle lane.

Stage 2 live constraint-method note:

- The active CLI/runtime path is penalty only.
- The older ALM helper code is intentionally parked off the live path while its
  closeout semantics are unfinished.
- Results serialization still preserves the parked ALM fields as optional
  metadata so historical payload readers do not break when those attributes are
  absent.

### Single-stage on the default CPU reference lane

```bash
SIMSOPT_BACKEND_MODE=native_cpu \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend cpu \
  --optimizer-backend scipy \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --init-only
```

This is the safest single-stage initialization proof lane.

### Single-stage on the JAX parity lane

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend ondevice \
  --boozer-optimizer-backend ondevice \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --init-only
```

This is the required single-stage JAX execution lane. Do not treat it as a
replacement for the public CPU/reference `scipy` oracle lane:

```bash
SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled \
SIMSOPT_JAX_PLATFORM=cuda \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
env -u LD_LIBRARY_PATH \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend ondevice \
  --boozer-optimizer-backend ondevice \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --init-only
```

Do not treat the last command as the first proof lane. Keep `native_cpu` and
`jax_cpu_parity` ahead of it.

### Single-stage JAX runtime seed spec

`single_stage_jax_runtime_spec.json` is the first-class startup artifact for
production JAX single-stage runs. It freezes the seed surface, coil spec,
coil dofs, Boozer initialization scalars, hardware constants, and Stage 2 seed
metadata into an immutable JSON payload consumed on the target lane.

Compile a new spec from a warm-start run:

```bash
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --warm-start-run-dir /path/to/single_stage/warm_start_run \
  --compile-jax-runtime-seed-spec \
  --jax-runtime-seed-spec /path/to/single_stage_jax_runtime_spec.json
```

Use the compiled spec on JAX CPU or JAX GPU:

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend ondevice \
  --boozer-optimizer-backend ondevice \
  --jax-runtime-seed-spec /path/to/single_stage_jax_runtime_spec.json \
  --init-only
```

For small copy-paste checks, the checked-in fixture is:

```text
benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json
```

## Recommended usage pattern

### 1. Validate on CPU parity first

```python
import simsopt.config

simsopt.config.set_backend("jax_cpu_parity", strict=True)
```

Use this to validate:

- algorithmic parity
- finite-difference checks
- objective/gradient consistency

### 2. Move to GPU parity

```python
import simsopt.config

simsopt.config.set_backend("jax_gpu_parity", strict=True)
```

Use this to validate:

- device parity
- reduced-fixture GPU smoke coverage

### 3. Move to the fast lane only after parity is green

```python
import simsopt.config

simsopt.config.set_backend("jax_gpu_fast", strict=True)
```

Use this for:

- profiling
- warm-run timing
- throughput experiments

Do not use `jax_gpu_fast` as the first proof lane.

## Runtime inspection

Use the runtime helpers to record the current backend contract:

```python
import simsopt.config

policy = simsopt.config.get_backend_policy()

print(policy.mode)
print(policy.backend)
print(policy.jax_platform)
print(policy.requires_x64)
print(policy.transfer_guard)
print(policy.debug_nans)
print(policy.chunk_policy)
print(policy.tolerance_tier)
print(policy.provenance_label)
```

For GPU parity modes, treat the policy fields
`gpu_reduction_order_max_ulp`, `gpu_reduction_order_rel_tol`,
`gpu_reproducibility_seed`, `gpu_reproducibility_sample_size`, and
`tolerance_ratchet_factor` as reporting/acceptance metadata. They document the
expected tolerance budget and diagnostic defaults. For CUDA parity lanes, the
runtime validates that a deterministic XLA GPU flag was configured before JAX
initialization, but these policy fields do not directly force kernel execution
behavior by themselves.

## Benchmark and reporting contract

The benchmark/productization SSOT now includes:

- manifest:
  - [`/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/manifests/stable_hardware_weekly_tier5.json`](/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/manifests/stable_hardware_weekly_tier5.json)
- standardized markdown report template:
  - [`/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/reports/STANDARD_REPORT_TEMPLATE.md`](/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/reports/STANDARD_REPORT_TEMPLATE.md)
- report renderer:
  - [`/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/render_benchmark_report.py`](/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/render_benchmark_report.py)
- scheduled reporting workflow:
  - [`/Users/suhjungdae/code/columbia/simsopt-jax/.github/workflows/jax_benchmark_reporting.yml`](/Users/suhjungdae/code/columbia/simsopt-jax/.github/workflows/jax_benchmark_reporting.yml)
- legacy compatibility wrapper:
  - [`/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/gpu_benchmark.py`](/Users/suhjungdae/code/columbia/simsopt-jax/benchmarks/gpu_benchmark.py)

The scheduled benchmark workflow assumes a dedicated self-hosted runner for the
stable benchmark lane. If that runner is not provisioned, use
`workflow_dispatch` manually after preparing the benchmark environment.

Treat the checked-in Tier 2 / Tier 3 probes, the PR CUDA smoke job, and the
stable-hardware weekly workflow as the authoritative evidence path. The legacy
`gpu_benchmark.py` wrapper is for manual convenience and delegates to that
checked-in ladder rather than defining an independent source of truth.

## Performance expectations

Keep the timing claims narrow and honest:

- cold compile time:
  - first JAX calls can be dominated by compilation and cache setup
  - do not compare cold JAX against warm CPU and call that steady-state speed
- warm time:
  - use repeated or in-process reruns to measure steady-state timing
  - prefer the Tier 5 trusted-fixture report for workflow-level timing claims
- tracing:
  - fieldline, Cartesian guiding-centre, Boozer guiding-centre, and full-orbit
    JAX routes use immutable specs and local device batches before converting
    the fixed-shape result buffers back to SIMSOPT's list-of-arrays API
  - fieldline, Cartesian guiding-centre, and full-orbit routes use `vmap`;
    Boozer guiding-centre preserves the upstream singleton `modB/G` setup for
    each initial point, then runs the orbit integration through a device-side
    `lax.map` batch so the existing comm split/gather parity test remains
    stable
  - this removes the per-trajectory integration-result synchronization boundary
    on each MPI rank for fieldline, Cartesian guiding-centre, and full-orbit
    tracing; Boozer still has a per-particle host scalar setup boundary before
    the device-side integration batch
  - trajectory/event/status handling remains a parity and diagnostic surface
    rather than a smooth optimizer objective
  - any CUDA speedup claim for tracing still needs the normal GPU performance
    gate on real A100/H100-class hardware
- parity mode:
  - `jax_cpu_parity` and `jax_gpu_parity` favor x64 stability and explicit
    reporting over maximum throughput
  - Biot-Savart parity lanes now use a fixed pairwise tree for quadrature-axis
    sums instead of relying on a raw `jnp.sum(..., axis=1)` reduction order
  - the stabilized `integral_BdotN` kernel now uses the same fixed-tree idea
    for the global `"normalized"` denominator sum
  - shared reduction helpers now codify the escalation ladder used by the
    parity-sensitive kernels:
    - Tier 1: fixed pairwise/tree reductions for hot vector and scalar sites
      that already showed reduction-order sensitivity
    - Tier 2: compensated scalar summation is available for especially unstable
      scalar objectives, but is not promoted by default
    - Tier 3: selected kernels expose `reduction_mode="strict_oracle"` to opt
      into the compensated scalar path during parity investigations
    - Tier 4: ExBLAS/ReproBLAS/OzBLAS-style exact reproducibility remains a
      deferred research path, not a default runtime mode
  - expect a smaller but still real throughput cost from the extra
    pad/reshape/add stages and temporary arrays versus a flat reduction; keep
    that stronger arithmetic in the parity-sensitive lanes unless benchmark
    data justifies promoting it to `jax_gpu_fast`
  - the final scalar residual contraction in `integral_BdotN` intentionally
    stays on the validated baseline by default, but
    `reduction_mode="strict_oracle"` now promotes it to compensated summation
    when parity investigations need a stricter scalar oracle
  - the Boozer residual scalar stays on pairwise accumulation by default; the
    same `strict_oracle` mode upgrades only its final scalar contraction rather
    than changing the rest of the optimizer arithmetic
  - gate any Tier 4 exact-reproducibility work behind demonstrated need: both
    throughput cost and implementation complexity rise sharply relative to the
    Tier 1–3 ladder
- fast mode:
  - `jax_gpu_fast` is the throughput-oriented lane after parity is green
  - it is not the default oracle and should not replace `native_cpu`
- memory:
  - grouped-field and lower-level chunking reduce pressure, but memory claims
    must still be tied to the exact fixture, hardware, and compilation state
  - track both host RSS and GPU memory in benchmark artifacts when available

## Current caveats

- `native_cpu` is still the default and the broadest trusted lane.
- Routine GPU regression CI is still intentionally minimal.
- Not every legacy object family is fully routed through immutable specs yet:
  - spec-backed launch paths currently cover the grouped-coil Biot-Savart
    lane, scalar currents, field evaluation points, fixed-surface flux,
    `SurfaceRZFourier`, `SurfaceXYZFourier`, unclamped
    `SurfaceXYZTensorFourier`, and curve specs for `CurveXYZFourier`,
    `CurveRZFourier`, `CurvePlanarFourier`, `CurveHelical`,
    `CurveCWSFourierRZ`, `CurvePerturbed`, `CurveFilament`, and the fieldline,
    Cartesian guiding-centre, Boozer guiding-centre, and full-orbit tracing
    routes
  - remaining upstream-visible families such as `SurfaceGarabedian`,
    `SurfaceHenneberg`, `SurfaceRZPseudospectral`, clamped
    `SurfaceXYZTensorFourier`, analytic/interpolated magnetic fields,
    wireframe/permanent-magnet fields, and broad objective wrappers remain
    native-CPU/reference territory unless a dedicated immutable spec and parity
    test exists
  - unsupported live-graph conversion is a strict JAX boundary; use
    `native_cpu` for those families until their spec contracts are implemented
- Some broader workflow families remain planned rather than fully implemented.

## What this file does not claim

This file does not claim:

- universal full-GPU completion
- bitwise-identical cross-device results
- full PM / wireframe / greedy coverage
- that every legacy mutable compatibility path is already removed

Use the roadmap docs for the broader status:

- [`/Users/suhjungdae/code/columbia/analysis/jax_backend_master_updates_2026-03-31.md`](/Users/suhjungdae/code/columbia/analysis/jax_backend_master_updates_2026-03-31.md)
- [`/Users/suhjungdae/code/columbia/analysis/jax_combined_backend_10_10_plan.md`](/Users/suhjungdae/code/columbia/analysis/jax_combined_backend_10_10_plan.md)
- [`/Users/suhjungdae/code/columbia/simsopt-jax/gpu_jax_pro.md`](/Users/suhjungdae/code/columbia/simsopt-jax/gpu_jax_pro.md)
