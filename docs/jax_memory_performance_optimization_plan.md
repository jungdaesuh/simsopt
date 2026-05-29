# JAX Memory and Performance Optimization Plan

## Purpose

This document turns the current static audit of SIMSOPT-JAX memory and
performance hotspots into an executable implementation plan. It is scoped to
reducing avoidable JAX/CUDA memory pressure, compile pressure, and diagnostic
overhead while preserving SIMSOPT public behavior and C++/CPU/JAX parity.

## Goals

- Reduce production memory footprint in permanent-magnet optimization paths by
  avoiding full history materialization when callers only need final state or a
  compact trace.
- Reduce JAX intermediate allocation in candidate-cost and dense-derivative
  paths without changing tie-break, parity, or public result contracts.
- Keep single-stage CUDA production runs on compile-bounded optimizer routes
  and make monolithic `lbfgs-ondevice` usage explicitly diagnostic or
  evidence-gated.
- Add measurement gates that report JAX compile time, XLA memory analysis,
  device memory snapshots, CUDA profiler artifacts, and parity metrics before
  and after each optimization.
- Preserve SIMSOPT compatibility surfaces, especially public history-returning
  PM APIs, Optimizable-style adapters, and BiotSavart compatibility kwargs.

## Non-Goals

- No tolerance relaxation, parity-oracle deletion, or CPU/C++ behavior drift.
- No rewrite of SIMSOPT's public `Optimizable` model.
- No removal of `lbfgs-ondevice`; this plan only prevents treating it as the
  default production path for large CUDA single-stage runs without evidence.
- No speculative custom CUDA/Pallas kernels before profiling proves that
  existing JAX/XLA kernels and chunking cannot meet the target.
- No cleanup of unrelated dirty worktree changes.

## Current Context

- Current audit basis: `HEAD` short SHA `5bcd9061c`. The working tree is dirty
  with many unrelated edits; execute this plan in narrow commits and stage only
  intended files.
- `src/simsopt/jax_core/pm_optimization.py` result dataclasses carry
  `x_history` in every GPMO family, for example `GPMOBaselineResult` at
  `:368` and `GPMOMultiResult` at `:424`.
- No-record GPMO paths still reconstruct full post-step histories:
  `_baseline_x_history` at `pm_optimization.py:779`,
  `_arbvec_x_history` at `:1065`, and `_multi_x_history` at `:2314`.
- Public PM wrappers duplicate normalized history into physical `m_history`,
  for example `GPMO_baseline_jax` at
  `src/simsopt/solve/permanent_magnet_optimization_jax.py:421`, with analogous
  work at `:469`, `:520`, `:577`, and `:640`.
- ArbVec candidate scoring materializes `contributions` via
  `_gpmo_arbvec_contributions` at `pm_optimization.py:951` and concatenates
  plus/minus candidates at `:984`; multi-neighbour scoring concatenates at
  `:2261`.
- `mwpgp_solve` recomputes `A_arr @ x_flat` every scan step to emit diagnostic
  residual history at `pm_optimization.py:3387`.
- Surface scalar Hessian helpers use dense `jax.hessian`, with area/volume RZ
  guards at `src/simsopt/jax_core/surface_rzfourier.py:1325`, but minor,
  major, and aspect Hessians at `:1337`, `:1345`, and `:1353` lack the same
  guard. Generic surface objectives host-materialize dense Hessians through
  `src/simsopt/geo/surfaceobjectives_jax.py:786`.
- Boozer has matrix-free HVP/JVP/VJP helpers
  (`src/simsopt/geo/boozer_residual_jax.py:815`, `:890`, `:914`), but the dense
  composed Jacobian still builds identity bases and vmaps them at `:937`, and
  `src/simsopt/solve/jax/_dispatch.py:867` can materialize dense Jacobian and
  Hessian arrays when requested.
- Single-stage example routing already defaults GPU/CUDA JAX runs to
  `scipy-jax` and CPU JAX runs to `scipy-jax-fullgraph`; the code comments at
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:8007`
  state that legacy `ondevice` compiled the outer loop and objective as one
  XLA graph and did not finish compiling at production resolution.
- BiotSavart VJP grouping stacks grouped coil data in
  `src/simsopt/field/biotsavart_jax_backend.py:1852` and runs grouped VJP at
  `:1969`; this is a lower-priority optimization until profiling proves stack
  rebuilds dominate.

## Requirements Analysis

- **Correctness requirement:** Every optimization must preserve C++/CPU/JAX
  parity and existing tie-break semantics. PM GPMO plus/minus reductions must
  keep the existing `+` before `-` order where tests or C++ parity depend on it.
- **Measurement requirement:** No implementation task is complete on intuition.
  The JAX docs require asynchronous dispatch to be accounted for with
  `.block_until_ready()` during timing, and this plan requires before/after
  metrics for compile time, wall time, peak host RSS, XLA memory estimates, and
  device memory snapshots.
- **CUDA requirement:** CUDA-facing claims require a CUDA run, not CPU-only
  inference. Use NVIDIA profiling guidance to distinguish CPU compilation,
  host-device transfer, kernel execution, memory bandwidth, and occupancy
  limits. CPU-only local tests can prove correctness but not CUDA performance.
- **JAX memory requirement:** JAX GPU memory behavior is affected by preallocation
  and fragmentation policy; CUDA validation must record the relevant
  `XLA_PYTHON_CLIENT_*` variables and should compare steady-state memory under
  controlled settings.
- **Host/device purity requirement:** Use JAX transfer guard smoke tests for
  production target lanes. Host materialization is allowed at documented
  reporting boundaries, not inside compiled hot loops.
- **SIMSOPT API requirement:** Existing public wrappers that expose `m_history`,
  `x_history`, dense `ddJ`, and compatibility kwargs remain default-compatible.
  Memory-saving behavior must be additive or guarded by a documented API
  evolution step.
- **Design requirement:** Any cross-module API change is Tier 3 under
  `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md`; it needs a caller
  inventory, compatibility test, migration path, rollback path, and explicit
  validation.

## Documentation Sources

- JAX docs:
  [Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html),
  [GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html),
  [Transfer guard](https://docs.jax.dev/en/latest/transfer_guard.html),
  [Benchmarking FAQ](https://docs.jax.dev/en/latest/faq.html#benchmarking-jax-code),
  [Device memory profile](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.device_memory_profile.html),
  and Context7 `/jax-ml/jax` docs fetched for memory/performance topics.
- CUDA/NVIDIA docs:
  [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/),
  [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/),
  [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/),
  and [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/).
- SIMSOPT docs:
  [SIMSOPT documentation](https://simsopt.readthedocs.io/),
  [SIMSOPT field API](https://simsopt.readthedocs.io/stable/simsopt.field.html),
  [SIMSOPT geo API](https://simsopt.readthedocs.io/stable/simsopt.geo.html),
  and [SIMSOPT objectives API](https://simsopt.readthedocs.io/stable/simsopt.objectives.html).

## Rationale

The high-value work is not "make everything more JAX-like"; much of the current
port already uses `lax.scan`, JIT caches, HVP/JVP/VJP helpers, chunking, and
callback-disabled compiled variants. The remaining leverage is in separating
production outputs from diagnostic histories, replacing dense materialization
with operator products where the caller only needs a product, and enforcing
compile-bounded optimizer routes for CUDA production runs.

The plan therefore starts with measurement and compatibility inventory, then
targets the largest memory surfaces first: PM history buffers, PM candidate
arrays, MwPGP diagnostic history, dense derivative finalization, and single-stage
optimizer policy. BiotSavart grouping and pairwise chunking are intentionally
profile-gated because existing code already contains chunking and grouped
kernels.

## Assumptions

- Most production PM callers need final moments, objective/residual summaries,
  and selected dipole/sign traces more often than full `(K, ndipoles, 3)`
  histories.
- Full history and dense Hessian/Jacobian APIs are compatibility surfaces and
  should stay available by default until a caller inventory proves a safer
  migration.
- CUDA validation will run on a GPU host or remote environment; the local machine
  may not have `nvidia-smi` or CUDA devices available.
- Existing `scipy-jax` and `scipy-jax-fullgraph` lanes are intended production
  paths; `lbfgs-ondevice` remains useful for smaller/diagnostic parity work.

## Implementation Plan

1. Measurement and acceptance harness
   - [ ] Add a small benchmark manifest for the optimization work with one PM
     GPMO fixture, one MwPGP fixture, one surface dense-Hessian fixture, one
     Boozer LS/dense-linearization fixture, and one single-stage CUDA fixture.
   - [ ] For each fixture, capture baseline wall time with `.block_until_ready()`,
     compile time, host RSS, XLA `memory_analysis()` where available, and
     `jax.profiler.device_memory_profile()` output when running on CUDA.
   - [ ] Add a CUDA profiling recipe that records `nvidia-smi` sampling,
     Nsight Systems timeline, and Nsight Compute kernel/memory summaries for the
     PM and single-stage fixtures.
   - [ ] Record runtime policy in the artifact: `jax.__version__`,
     `jaxlib.__version__`, `jax.devices()`, `JAX_ENABLE_X64`,
     `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, backend
     mode, optimizer backend, and transfer guard state.
   - [ ] Define acceptance budgets before editing: maximum allowed parity drift,
     maximum compile-count growth, maximum host RSS, maximum device peak memory,
     and minimum speedup/memory reduction required to keep each optimization.

2. PM output-history memory reduction
   - [ ] Inventory every public and internal caller of `x_history`, `m_history`,
     `residual_history`, `selected_*`, and PM plotting utilities across `src/`,
     `tests/`, `benchmarks/`, `examples/`, and `docs/`.
   - [ ] Design two approaches before implementation:
     `compact result family` that carries final `x`, residual, and selected
     traces only; and `history view/finalizer` that reconstructs full histories
     only at public wrapper boundaries.
   - [ ] Choose the approach with less cross-module leakage. The default public
     `GPMO_*_jax` behavior must still return the existing history fields unless
     a Tier-3 API gate approves a migration.
   - [ ] Add compact core helpers or result records in
     `src/simsopt/jax_core/pm_optimization.py` that avoid allocating
     `(K, ndipoles, 3)` unless full history is explicitly requested.
   - [ ] In `src/simsopt/solve/permanent_magnet_optimization_jax.py`, avoid
     computing physical `m_history` when the compact result path is selected.
   - [ ] Keep full-history reconstruction tests for all GPMO families and add
     compact-result tests proving final moments, residuals, and selected traces
     match full-history execution.

3. PM candidate-cost and MwPGP diagnostic reductions
   - [ ] Replace full plus/minus concatenation in baseline, ArbVec,
     backtracking, and multi candidate-cost reducers with separate reductions
     that preserve current tie order while materializing only chosen
     sign/index/cost.
   - [ ] For ArbVec, benchmark the current full `(M, N, P)` contribution tensor
     against a streamed/chunked contraction over residual dimension `M` and
     vector dimension `P`; keep the streamed path only if it improves measured
     peak memory without unacceptable runtime regression.
   - [ ] Add an explicit production path for `mwpgp_solve` that returns final
     state without per-step residual history, or records residual history at a
     sampled cadence. Keep the existing full diagnostic history path available.
   - [ ] Add tie-break and parity tests against current full candidate vectors
     for each GPMO variant before replacing the implementation.

4. Dense derivative and operator-product cleanup
   - [ ] Extend the RZ Hessian memory guard to minor radius, major radius, and
     aspect ratio helpers in `surface_rzfourier.py`.
   - [ ] Add additive HVP/quadratic-form helpers for surface scalar metrics so
     production callers can avoid dense `ddJ` when they need only products.
   - [ ] Audit `surfaceobjectives_jax.py` wrappers and tests to identify which
     callers genuinely need dense `d2J_by_dsurfacecoefficientsdsurfacecoefficients`
     versus product-only access.
   - [ ] Audit Boozer dense-linearization callers and route production objective
     gradients through existing HVP/JVP/VJP helpers wherever dense Jacobian or
     Hessian arrays are only an implementation detail.
   - [ ] Keep `materialize_dense_linearization` as an explicit diagnostic/small
     problem path with byte guards and tests; do not silently materialize dense
     arrays for reporting-only metadata.

5. Single-stage CUDA optimizer and callback policy
   - [ ] Update any stale docs/tests that still describe `lbfgs-ondevice` as the
     large CUDA production default when current routing intends `scipy-jax` or
     `scipy-jax-fullgraph`.
   - [ ] Add a guardrail test that default JAX GPU single-stage routing resolves
     to `scipy-jax` and default JAX CPU routing resolves to
     `scipy-jax-fullgraph`, while explicit `lbfgs-ondevice` remains available.
   - [ ] Ensure objective-evaluation trace, optimizer-state trace, accepted-step
     callbacks, and progress callbacks stay opt-in and are disabled in production
     benchmark defaults.
   - [ ] Add compile-count assertions for `scipy-jax` value/grad reuse in the
     single-stage CUDA fixture.
   - [ ] Keep a diagnostic `lbfgs-ondevice` fixture with strict timeout and
     memory budget so regressions remain visible without blocking production
     lane optimization.

6. BiotSavart and pairwise profile-gated work
   - [ ] Profile grouped coil VJP setup to determine whether repeated
     `jnp.stack` grouping is material relative to kernel execution and coil
     projection.
   - [ ] If grouping dominates, design a geometry-versioned grouped-array cache
     keyed by existing coil DOF state tokens; prove invalidation on coil geometry
     and current changes before implementation.
   - [ ] Use `benchmarks/pairwise_reduction_memory_analysis.py` to characterize
     current chunk sizes before changing pairwise scans or introducing denser
     `vmap` paths.
   - [ ] Keep current scan/checkpoint pairwise reducers unless CUDA profiling
     shows the memory/runtime tradeoff is wrong for production fixtures.

7. Documentation and rollout
   - [ ] Update `docs/jax_native_round*_performance_todos_*.md` or supersede them
     with links to this plan once implementation starts, so stale performance
     TODOs do not conflict with the new work.
   - [ ] Update `docs/using_jax_backend.md`, `docs/source/jax_acceptance.rst`,
     and single-stage quickstart docs if optimizer-default wording changes.
   - [ ] Record each accepted optimization with before/after metrics,
     implementation commit, validation commands, CUDA hardware, and artifact
     paths.

## Validation Plan

- [ ] Static checks on touched files:
  `ruff format --check <touched files>` and `ruff check <touched files>`.
- [ ] PM correctness:
  `python -m pytest -q tests/jax_core/test_pm_optimization_jax_item25.py tests/solve/test_permanent_magnet_optimization_jax_item28.py tests/solve/test_pm_optimization.py`.
- [ ] Surface/Boozer dense derivative coverage:
  `python -m pytest -q tests/geo/test_surface_rzfourier_jax.py tests/geo/test_surface_objectives_jax.py tests/geo/test_boozer_residual_jax.py tests/geo/test_boozersurface_jax.py`.
- [ ] Single-stage routing and parity smoke:
  `python -m pytest -q tests/integration/test_single_stage_jax.py tests/integration/test_single_stage_jax_cpu_reference.py tests/test_benchmark_helpers.py -k "optimizer_backend or scipy-jax or lbfgs-ondevice or device_memory_profile"`.
- [ ] Transfer-guard smoke on CPU:
  `SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q tests/geo/test_surface_rzfourier_transfer_guard_jax.py tests/subprocess/jax_runtime_cases.py`.
- [ ] XLA memory analysis:
  run `benchmarks/pairwise_reduction_memory_analysis.py` and add equivalent PM,
  Boozer, and surface probes if missing.
- [ ] CUDA validation on GPU host:
  run the selected PM and single-stage fixtures with `nvidia-smi` sampling,
  JAX device memory profile, Nsight Systems, and Nsight Compute. Attach artifact
  paths to the implementation summary.
- [ ] Regression gate:
  compare before/after final objective, residuals, selected GPMO traces,
  dense derivative values when requested, and CPU/JAX parity-ladder metrics.

## Risks and Mitigations

- Risk: Compact PM outputs accidentally change public history semantics.
  Mitigation: Keep full-history public defaults until API gate completion; add
  compact paths additively and test full vs compact final-state equivalence.

- Risk: Plus/minus separate reductions change tie-break behavior.
  Mitigation: Add golden tests comparing selected index/sign/cost against the
  current concatenated vectors for all GPMO variants before replacing code.

- Risk: Removing diagnostic residual history hides convergence regressions.
  Mitigation: Keep diagnostic history path and add sampled trace mode with clear
  artifact metadata.

- Risk: Dense Hessian replacements break callers that expect `Derivative` or
  dense `ddJ` objects.
  Mitigation: Add HVP/product APIs without removing dense APIs; require caller
  inventory before any public behavior change.

- Risk: CUDA results are confounded by JAX allocator preallocation or
  asynchronous dispatch.
  Mitigation: Record allocator environment, use `.block_until_ready()`, include
  device memory profiles, and report both cold compile and warm steady-state.

- Risk: Host callbacks re-enter hot compiled paths through diagnostics.
  Mitigation: Keep callbacks opt-in, assert production defaults disable them,
  and run transfer-guard smoke tests.

- Risk: BiotSavart cache invalidation misses mutable SIMSOPT coil changes.
  Mitigation: Do not implement cache until keying and invalidation are proven
  against coil DOF state token tests.

## Completion Criteria

- [ ] Every accepted optimization has before/after measurements with
  block-until-ready timing, host RSS, XLA memory estimate where available,
  and CUDA memory/profiler artifacts when CUDA-facing.
- [ ] PM compact paths reduce peak memory on the selected fixture while matching
  full-history final state and selected traces.
- [ ] Candidate-cost changes preserve selected candidate identity and objective
  values against the pre-change implementation.
- [ ] Surface/Boozer production paths avoid dense materialization unless the
  caller explicitly requests dense outputs.
- [ ] Single-stage CUDA production route is documented and tested as
  `scipy-jax`/`scipy-jax-fullgraph`, with `lbfgs-ondevice` retained as explicit
  diagnostic or small-run path.
- [ ] All validation commands relevant to touched files pass, or failures are
  documented as pre-existing/environmental with evidence.
- [ ] Docs link to official JAX, CUDA/NVIDIA, and SIMSOPT references used for
  the implementation decisions.

## Open Questions

- Should compact PM outputs be exposed through new `*_compact_jax` helpers, an
  explicit result-policy argument, or only internal benchmark/production lanes?
- Which production caller actually needs full PM `m_history` at large `K`, and
  which only needs plotting or final-state summaries?
- What are the target CUDA hardware and memory budgets for acceptance:
  A100 80GB, L40S, H100, Perlmutter GPU, or another platform?
- Should `lbfgs-ondevice` remain part of the release validation ladder if it is
  not the large CUDA production default?
- Which dense surface/Boozer derivative consumers need dense matrices for API
  compatibility, and which can accept HVP/JVP/VJP product callbacks?
- Is a BiotSavart grouped-array cache worth the invalidation complexity after
  current profiling, or is kernel execution still the dominant cost?
