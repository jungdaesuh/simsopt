# jax-mps Custom Metal Boozer Solve Implementation Plan

## Purpose

Define the implementation path for turning the current SIMSOPT MPS custom-call prototype into a real performance path: a jax-mps backend custom operation that keeps the Boozer solve/value-gradient work on the Apple GPU and removes the MLX-composed dense-Jacobian/CPU-solve bottleneck.

This file is an execution plan, not a performance claim. Current measurements still show no proven end-to-end MPS-over-CPU win for the single-stage run.

## Goals

- Replace wrapper-level routing work with backend-owned Metal/MLX-extension kernels for the hot Boozer residual, linearized residual, adjoint, and value-gradient path.
- Remove the current full-solve implementation's dense Jacobian materialization and CPU triangular solve from the MPS target path.
- Reduce host synchronization and PJRT execute count materially versus the current MLX-composed `full_solve` custom call.
- Preserve SIMSOPT's fail-loud contract: no silent fallback from custom MPS to CPU/JAX when the explicit experimental path is requested.
- Prove parity before performance: same-candidate value/gradient replay must pass before using optimizer timing as evidence.

## Non-Goals

- Do not build a standalone replacement for jax-mps or a new JAX backend.
- Do not optimize the host SciPy L-BFGS-B outer loop; it remains CPU-hosted in both CPU and MPS lanes.
- Do not broaden SIMSOPT problem support beyond the current MPS Boozer contract before the narrow path is correct and measured.
- Do not use Python-side MLX kernels as the production path for the fused solve.
- Do not claim production-resolution speedups until the path runs at the intended resolution and passes the same acceptance gates as CPU/CUDA evidence.

## Current Context

- Confirmed: `simsopt-jax` owns the public contract and routing in `src/simsopt/jax_core/mps_boozer_kernel_contract.py`. The current target name is `mps.simsopt_boozer_value_grad`, and schema version 4 supports `mps_solver_mode="full_solve"`.
- Confirmed: `jax-mps` owns the backend implementation in `/Users/suhjungdae/code/opensource/jax-mps/src/pjrt_plugin/ops/simsopt_custom_call.cc`.
- Confirmed: the current `full_solve` code is not a direct Metal solve. It still calls MLX AD and dense linear algebra:
  - `BoozerFullStateValueGrad` uses `mlx::core::vjp`.
  - `BoozerFullStateDenseJacobian` loops over state columns and calls `mlx::core::jvp` once per column.
  - `SolveDensePositiveDefiniteSystem` calls `mlx::core::linalg::solve_triangular(..., mlx::core::Device::cpu)`.
  - `BoozerFullStateDenseLevenbergMarquardtStep` forms a dense Jacobian, normal matrix, and right-hand side.
  - `BoozerFullStateImplicitCoilGradient` again uses MLX `vjp` for stationarity pullback.
- Confirmed: the latest saved maxiter=1 rerun cleared the old crash but did not pass end-to-end optimizer acceptance. It was a residual-only fixture run (`experimental_mps_boozer_residual_only_fixture=true`), so its CPU outer optimizer at about 3.57 s, MPS custom at about 40.64 s, and MPS float32 reference at about 78.13 s are bootstrap-regression numbers, not full single-stage acceptance numbers.
- Confirmed: the saved rerun artifact stores these fields under nested JSON paths, not as top-level keys:
  - `/provenance/experimental_mps_boozer_residual_only_fixture = true`
  - `/timings/cpu_outer_optimizer_s = 3.5726914170081727`
  - `/timings/jax_outer_optimizer_s = 40.63711470802082`
  - `/timings/mps_float32_reference_outer_optimizer_s = 78.12672462494811`
  - `/timings/jax_while_profile_scalar_condition_reads = 924`
  - `/timings/jax_while_profile_while_record_count = 95`
  - `/timings/jax_while_profile_loop_trips = 829`
  - `/jax_mps_while_profiles/jax/pjrt_execute_record_count = 645597`
  - `/jax_mps_while_profiles/jax/loop_trips = 829`
  - `/jax_mps_while_profiles/jax/scalar_condition_reads = 924`
  - `/passed = false`, with failures showing that the JAX single-stage outer-loop probe and the MPS custom-kernel float32 reference did not accept an optimizer step.
- Confirmed: the same rerun still recorded very high dispatch/synchronization pressure: about 645,597 PJRT execute records, 95 while records, 829 loop trips, and 924 scalar condition reads.
- Confirmed: current MPS custom work shows a useful subpath signal versus stock MPS, but not an end-to-end MPS-over-CPU win.
- Confirmed from official docs:
  - JAX FFI/custom calls support external compiled kernels, and FFI calls are opaque to automatic differentiation unless paired with explicit derivative rules. For this project, the analogous implementation boundary is the existing `stablehlo.custom_call` target handled by `jax-mps`, not a new Python-level JAX FFI wrapper. Source: <https://docs.jax.dev/en/latest/ffi.html>
  - JAX asynchronous dispatch can hide host/kernel latency when work stays queued instead of forcing host values. Source: <https://docs.jax.dev/en/latest/async_dispatch.html>
  - MLX lazy evaluation records compute graphs and only executes them when `eval()` is performed; the local profile's repeated scalar-condition reads make those forced evaluations part of the measured bottleneck. Source: <https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html>
  - MLX supports custom Metal kernels through Python and C++ APIs, and its extension docs provide CMake support such as `mlx_build_metallib()` for compiled Metal libraries. Those facilities still need deliberate integration into the `jax-mps` PJRT plugin; Python `fast.metal_kernel` alone is not the production path for this plan. Sources: <https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html>, <https://ml-explore.github.io/mlx/build/html/dev/extensions.html>
- Confirmed: `/Users/suhjungdae/code/opensource/jax-mps/CMakeLists.txt` currently enables C/C++ and links Metal/MPS frameworks, but there are no `.metal` or `.mm` SIMSOPT sources yet.

## Rationale

The current custom-call boundary is in the right place, but the implementation under that boundary is still composed from many MLX operations, MLX AD calls, and at least one CPU linear solve. That cannot deliver the intended MPS win because it preserves the same problem shape that made stock jax-mps slow: many small launches, forced evaluation pressure, host-visible loop progress, and device-to-CPU detours.

The best next path is not more wrapper routing in `simsopt-jax`. The wrapper already can request the custom op and fail loudly. The real work is in `jax-mps`: replace the MLX-composed solver with direct backend kernels and matrix-free linear algebra so each objective+gradient evaluation does fewer, larger, GPU-resident operations.

The plan is staged because a single monolithic fused solve is too large to debug in one step. The first useful milestone is proving that a direct Metal or MLX C++ extension kernel can be built, loaded, dispatched from the PJRT custom call, and compared against the existing oracle. After that, replace the hot algebra in dependency order: residual, Jacobian-vector product, transpose-Jacobian-vector product, reductions/norms, matrix-free LM/CG, and finally the full value-gradient solve.

## Assumptions

- The near-term target remains Apple Silicon MPS with float32 inputs, matching current jax-mps/MLX constraints.
- The narrow supported SIMSOPT case remains one or two coil groups, stellsym `SurfaceXYZTensorFourier`, volume-label/zero constraint, `optimize_G=True`, and bounded small-resolution smoke cases before production-scale claims.
- `jax-mps` can either add Objective-C++/Metal source support directly or host the needed kernels through MLX C++ extension APIs without introducing Python runtime dependencies.
- The current CPU timing and MPS timing artifacts are representative enough to define regression thresholds, but final performance claims require fresh reruns after each backend milestone.
- The correct production shape is matrix-free for the inner linear solve; dense Jacobian materialization is a stepping stone to delete, not a performance endpoint.

## Implementation Plan

1. Establish the native Metal integration boundary in `jax-mps`
   - [ ] Decide and document the backend integration mechanism: direct Objective-C++/Metal sources in `jax-mps`, or MLX C++ extension kernels linked into the PJRT plugin.
   - [ ] If using direct Metal, update `/Users/suhjungdae/code/opensource/jax-mps/CMakeLists.txt` to enable Objective-C++/Metal sources, compile/load a SIMSOPT metallib, and keep existing MLX linkage intact.
   - [ ] If using MLX C++ extension kernels, add a minimal compiled extension path inside the plugin build and verify it has no Python-side runtime dependency.
   - [ ] Reuse or extend the existing `mps.simsopt_custom_call_smoke` target for the first one-kernel native-dispatch proof; do not route a build/dispatch smoke through `mps.simsopt_boozer_value_grad` until the actual solve math is present.
   - [ ] Add a `jax-mps` test that proves the smoke kernel executes on the MPS backend and fails loudly when the local plugin library is missing the target.

2. Split the SIMSOPT custom-call implementation into testable backend components
   - [ ] Extract the SIMSOPT custom-call payload parsing and shape validation from `src/pjrt_plugin/ops/simsopt_custom_call.cc` into smaller helpers without changing behavior.
   - [ ] Keep the current MLX-composed path as an oracle-only fallback behind an internal test switch, not as the production experimental route.
   - [ ] Add explicit instrumentation counters for custom-kernel launches, device-to-host scalar reads, CPU-device linear solves, and PJRT execute records when profiling is enabled.
   - [ ] Extend `tests/test_simsopt_custom_call.py` to assert which backend path ran, so tests cannot pass accidentally through the old MLX-composed full solve.

3. Implement direct residual and objective kernels
   - [ ] Port the Boozer residual evaluation used by the full-state value/gradient path (`BoozerFullStateScaledResidual`, built on `BoozerResidualVector`, with the residual norm from `BoozerFullStateUnscaledResidualNorm`) to Metal/extension kernels for the current schema-4 payload.
   - [ ] Include reductions for residual norm, objective value, gradient norm diagnostics, and finite checks on device.
   - [ ] Compare direct-kernel residual/objective outputs against the existing NumPy/JAX/MLX oracle for fixed small fixtures.
   - [ ] Replace the current MLX residual/objective calls in the production custom path after parity passes.

4. Implement matrix-free linearized operators
   - [ ] Implement a Metal/extension Jacobian-vector product `Jv` for the full state variables used by the Boozer solve.
   - [ ] Implement a Metal/extension transpose-Jacobian-vector product `J^T v` for adjoint and normal-equation operations.
   - [ ] Implement the coil pullback required for `coil_gradient` without calling MLX `vjp` in the production custom path.
   - [ ] Add finite-difference and oracle-vector tests that compare `Jv` and `J^T v` against the existing dense-Jacobian fixture within float32 tolerances.

5. Replace dense LM step with device-side matrix-free LM/CG
   - [ ] Remove production dependence on `BoozerFullStateDenseJacobian` for `mps_solver_mode="full_solve"`.
   - [ ] Remove production dependence on `SolveDensePositiveDefiniteSystem` and its `mlx::core::Device::cpu` triangular solves.
   - [ ] Implement a bounded device-side CG or LM-CG loop using the direct `Jv`, `J^T v`, damping, and reductions.
   - [ ] Keep loop trip counts bounded by the existing contract fields and return explicit `converged`, `finite`, `newton_iteration_count`, and `gmres_iteration_count` status.
   - [ ] Add tests for convergence, nonconvergence, NaN/Inf fail-loud behavior, and bounded-iteration reporting.

6. Fuse the full solve/value-gradient path
   - [ ] Wire residual, LM/CG, adjoint, and coil-gradient kernels into the `mps.simsopt_boozer_value_grad` full-solve target.
   - [ ] Ensure the production path returns `value`, `coil_gradient`, `final_x_inner`, `residual_norm`, `gradient_norm`, iteration counts, `converged`, and `finite` without host-side reconstruction.
   - [ ] Preserve schema-version checks and unsupported-case errors in `simsopt-jax`.
   - [ ] Keep status masking in `simsopt-jax`: optimizer-facing value/gradient must be NaN unless the custom solve is both finite and converged.

7. Tighten `simsopt-jax` routing only after backend parity
   - [ ] Update `src/simsopt/jax_core/mps_boozer_kernel_contract.py` only for schema changes required by the direct backend implementation.
   - [ ] Keep the explicit experimental flag as the only way to select this path.
   - [ ] Add acceptance tests that prove unsupported payloads fail loudly and supported payloads use the direct backend path.
   - [ ] Avoid adding wrapper retries, fallback routing, or alternate objective pathways as a substitute for backend kernel work.

8. Measure and iterate from maxiter=1 to maxiter=3
   - [ ] First rerun maxiter=1 with profiling and same-candidate replay enabled.
   - [ ] Only after maxiter=1 passes parity, rerun maxiter=3 and compare CPU, stock MPS float32 reference, and custom MPS.
   - [ ] Record compact JSON artifacts only; do not keep large raw logs unless a failure requires them.
   - [ ] Promote performance claims only when the artifact passes acceptance gates and includes timing plus profiling counters.

## Validation Plan

- [ ] Record source and disk preflight before running builds or benchmarks:

  ```bash
  git -C /Users/suhjungdae/code/columbia/simsopt-jax status --short
  git -C /Users/suhjungdae/code/opensource/jax-mps status --short
  df -h / /Users/suhjungdae /tmp /Users/suhjungdae/.cache/uv
  du -sh ~/.cache/uv /tmp/jax-mps-deps-build /Users/suhjungdae/.local/jax-mps-deps .artifacts runs 2>/dev/null
  ```

  Treat dirty worktree measurements as dirty-tree artifacts: record the exact diff or commit provenance for both repos before comparing timings. Near-full local disk is a validation blocker because `uv`, CMake, profiling, and benchmark artifact writes all need free space. `~/.cache/uv` and `/tmp/jax-mps-deps-build` are regenerable cache/build-scratch paths; `/Users/suhjungdae/.local/jax-mps-deps` is an installed dependency prefix and should not be deleted unless the rebuild cost is acceptable.

- [ ] In `/Users/suhjungdae/code/opensource/jax-mps`, build the plugin:

  ```bash
  cmake --build build/cp313-cp313-macosx_26_0_arm64 --target pjrt_plugin_mps
  ```

- [ ] In `/Users/suhjungdae/code/opensource/jax-mps`, run the custom-call tests against the rebuilt local plugin:

  ```bash
  JAX_MPS_LIBRARY_PATH=/Users/suhjungdae/code/opensource/jax-mps/build/cp313-cp313-macosx_26_0_arm64/lib/libpjrt_plugin_mps.dylib \
  uv run pytest tests/test_simsopt_custom_call.py -q
  ```

- [ ] In `/Users/suhjungdae/code/opensource/jax-mps`, run source checks:

  ```bash
  uv run python -m ruff check tests/test_simsopt_custom_call.py
  uv run python -m ruff format --check tests/test_simsopt_custom_call.py
  git diff --check -- src/pjrt_plugin/ops/simsopt_custom_call.cc tests/test_simsopt_custom_call.py CMakeLists.txt
  ```

- [ ] In `/Users/suhjungdae/code/columbia/simsopt-jax`, run contract tests:

  ```bash
  .conda/jax/bin/python -m pytest tests/jax_core/test_mps_boozer_kernel_contract.py -q
  ```

- [ ] In `/Users/suhjungdae/code/columbia/simsopt-jax`, run focused single-stage MPS custom-kernel tests:

  ```bash
  .conda/jax/bin/python -m pytest tests/geo/test_single_stage_example.py -q \
    -k 'experimental_mps_boozer_custom_kernel or trace_forward_result or target_lane_trace_wrapper or accepted_step_sync_uses_custom or accepted_step_sync_reuses_custom'
  ```

- [ ] In `/Users/suhjungdae/code/columbia/simsopt-jax`, run syntax and diff checks:

  ```bash
  .conda/jax/bin/python -m py_compile \
    examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
    src/simsopt/jax_core/mps_boozer_kernel_contract.py \
    tests/geo/test_single_stage_example.py \
    tests/jax_core/test_mps_boozer_kernel_contract.py

  git diff --check -- \
    examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
    src/simsopt/jax_core/mps_boozer_kernel_contract.py \
    tests/geo/test_single_stage_example.py \
    tests/jax_core/test_mps_boozer_kernel_contract.py
  ```

- [ ] Run the residual-only maxiter=1 bootstrap artifact with compact profiling. This is only a native-kernel regression and profiler gate; it must not be used as the final full single-stage performance claim:

  ```bash
  JAX_MPS_LIBRARY_PATH=/Users/suhjungdae/code/opensource/jax-mps/build/cp313-cp313-macosx_26_0_arm64/lib/libpjrt_plugin_mps.dylib \
  .conda/jax-mps/bin/python benchmarks/single_stage_init_parity.py \
    --platform auto \
    --output-json .artifacts/mps_custom_metal_solve/single_stage_maxiter1.json \
    --case-artifacts-dir .artifacts/mps_custom_metal_solve/cases \
    --optimizer-backend scipy-jax \
    --maxiter 1 \
    --mpol 2 \
    --ntor 2 \
    --benchmark-mode \
    --single-stage-case-timeout-seconds 900 \
    --single-stage-target-case-timeout-seconds 900 \
    --experimental-mps-boozer-custom-kernel \
    --experimental-mps-boozer-residual-only-fixture \
    --record-objective-evaluation-trace
  ```

- [ ] Parse the residual-only maxiter=1 JSON and require:
  - `/provenance/experimental_mps_boozer_residual_only_fixture` is `true`,
  - `/passed` is `true`; if it is `false`, copy `/failures` into the validation note and do not make a performance claim,
  - `/timings/cpu_outer_optimizer_s`, `/timings/jax_outer_optimizer_s`, and `/timings/mps_float32_reference_outer_optimizer_s` are present,
  - `/timings/jax_while_profile_scalar_condition_reads`, `/timings/jax_while_profile_while_record_count`, and `/timings/jax_while_profile_loop_trips` are present,
  - `/jax_mps_while_profiles/jax/pjrt_execute_record_count`, `/jax_mps_while_profiles/jax/loop_trips`, and `/jax_mps_while_profiles/jax/scalar_condition_reads` are present,
  - same-candidate replay passes,
  - the target optimizer accepts a step; if it does not, record the artifact as blocked and do not make a performance claim,
  - custom MPS outer optimizer improves materially over the current residual-only about-40.64 s maxiter=1 custom baseline,
  - PJRT execute records drop materially from the current residual-only about-645,597 count,
  - no production path reports CPU linear solves.
- [ ] Run the full single-stage maxiter=1 artifact without `--experimental-mps-boozer-residual-only-fixture`:

  ```bash
  JAX_MPS_LIBRARY_PATH=/Users/suhjungdae/code/opensource/jax-mps/build/cp313-cp313-macosx_26_0_arm64/lib/libpjrt_plugin_mps.dylib \
  .conda/jax-mps/bin/python benchmarks/single_stage_init_parity.py \
    --platform auto \
    --output-json .artifacts/mps_custom_metal_solve/full_single_stage_maxiter1.json \
    --case-artifacts-dir .artifacts/mps_custom_metal_solve/full_cases \
    --optimizer-backend scipy-jax \
    --maxiter 1 \
    --mpol 2 \
    --ntor 2 \
    --benchmark-mode \
    --single-stage-case-timeout-seconds 900 \
    --single-stage-target-case-timeout-seconds 900 \
    --experimental-mps-boozer-custom-kernel \
    --record-objective-evaluation-trace
  ```

- [ ] Parse the full single-stage maxiter=1 JSON and require:
  - same-candidate replay passes under the full objective,
  - target optimizer accepts a step,
  - custom MPS is compared against CPU and stock MPS float32 reference in the same artifact,
  - PJRT execute records and scalar condition reads are reported,
  - no production path reports CPU linear solves.
- [ ] After the full maxiter=1 artifact passes, rerun the full command with `--maxiter 3` and compare CPU, stock MPS float32 reference, and custom MPS.

## Risks and Mitigations

- Risk: direct Metal build integration in `jax-mps` is more invasive than expected.
  Mitigation: prove the one-kernel smoke path first and keep it isolated from SIMSOPT math until build/load/dispatch is stable.

- Risk: MLX C++ extension APIs are easier to build but still force evaluation patterns that preserve the bottleneck.
  Mitigation: require launch/path counters and device-resident status before accepting an MLX-extension implementation as the production path.

- Risk: matrix-free LM/CG differs numerically from the dense-Jacobian oracle and changes optimizer behavior.
  Mitigation: validate residual, `Jv`, `J^T v`, adjoint, and coil-gradient independently before enabling full optimizer runs.

- Risk: float32 tolerance hides real gradient defects.
  Mitigation: use same-candidate replay, finite-difference probes, and independent oracle-vector tests; do not rely only on optimizer convergence.

- Risk: performance improves versus stock MPS but remains slower than CPU.
  Mitigation: treat that as partial progress only. Completion requires reducing dispatch/host-sync counters and showing a credible path toward CPU parity or better.

- Risk: near-full local disk or large profiling logs block `uv` hooks, CMake builds, or benchmark artifact writes before validation completes.
  Mitigation: run the disk preflight first, keep structured JSON summaries, cap raw logs, clear only regenerable cache/build-scratch paths when needed, and record any pruned paths in the validation note.

## Completion Criteria

- [ ] `jax-mps` contains a native Metal or compiled-extension SIMSOPT solve path with no Python MLX sidecar.
- [ ] Production `mps_solver_mode="full_solve"` no longer calls MLX dense-column `jvp`, MLX `vjp`, or CPU triangular solve for the hot solve/value-gradient path.
- [ ] `simsopt-jax` contract tests and focused single-stage routing tests pass against the rebuilt local plugin.
- [ ] residual-only maxiter=1 same-candidate replay passes with explicit custom-backend evidence and materially lower dispatch counters than the current residual-only baseline.
- [ ] full single-stage maxiter=1 same-candidate replay passes, accepts a target optimizer step, and reports no CPU linear solve in the custom path.
- [ ] full single-stage maxiter=3 custom MPS artifact passes acceptance gates and is compared against CPU and stock MPS float32 reference.
- [ ] Final performance artifacts record source provenance for both repos, including whether each timing came from a clean or dirty worktree.
- [ ] Documentation distinguishes proven performance from expected performance and records any remaining CPU-over-MPS gap honestly.

## Open Questions

- Should the production implementation use direct `.metal`/Objective-C++ sources in `jax-mps`, or MLX C++ extension kernels linked into the PJRT plugin?
- Does the current `MlxClient::metal_device()` pointer provide enough access for direct command queue/library management, or should SIMSOPT kernels use MLX allocation/stream ownership exclusively?
- What is the smallest matrix-free state/operator set that preserves the current Boozer solve's optimizer behavior under float32 tolerances?
- Can `J^T v` and coil pullback be hand-coded directly with acceptable maintenance cost, or should an intermediate generated-kernel path be used?
- What performance threshold is sufficient to continue: stock-MPS improvement only, CPU parity at maxiter=1, or CPU win at maxiter=3?
