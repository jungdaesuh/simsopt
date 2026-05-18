# JAX-native round-2 performance TODOs

Date: 2026-05-18

Validation baseline:

- Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
- Head observed during validation: `efd9ef300`
- Tree state: dirty, with unrelated modified and untracked docs already present
- Local JAX runtime observed during validation: `jax==0.10.0`
- Documentation checked: official JAX docs for persistent compilation cache,
  buffer donation, `jax.checkpoint`, checkpoint policies, and
  `jax.scipy.sparse.linalg.gmres`

## Purpose

Turn the round-2 JAX-native audit into an actionable repo-local implementation
checklist. This document preserves the useful findings, corrects claims that do
not match current code or official JAX behavior, and defines acceptance criteria
for each item before it is treated as complete.

This is not a benchmark report. Payoff estimates stay hypotheses until a focused
before/after run records compile time, wall time, memory, or transfer behavior.

## Goal

Reduce fresh-process compile latency, device memory pressure, host-to-device
traffic, and unnecessary single-device serialization on the JAX/ondevice product
lane while preserving the release-grade trust chain:

1. Existing SIMSOPT C++/SciPy behavior remains the reference oracle.
2. JAX CPU matches the reference behavior.
3. JAX GPU matches the reference behavior.
4. JAX CPU and GPU match each other.

## Scope

In scope:

- JAX runtime configuration that affects compile cache reuse.
- Jitted optimizer runners and matrix-free solver internals.
- Boozer interpolation and radial-profile hot paths.
- GPMO ArbVec and PM candidate-cost kernels.
- Biot-Savart, dipole-field, and surface-quadrature JAX kernels.
- Public matrix-free helper surfaces that avoid dense materialization.
- Sharding opportunities that map existing batch or surface axes to devices.

Out of scope:

- Loosening parity tolerances.
- Rewriting the preserved upstream CPU/reference lane.
- Replacing host-only artifact export or plotting paths.
- Treating speculative Pallas/Triton work as approved implementation without a
  profiling gate.

## Global guardrails

- Keep fixes narrow to the named code paths.
- Do not introduce dynamic imports.
- Do not introduce `any` typing in typed code.
- Do not add broad defensive checks, try/except wrappers, or fallback layers.
- Do not hide parity drift by changing tolerances.
- Separate CPU/HLO proof from real CUDA proof.
- Mark an item complete only after code, tests, and benchmark/profiling evidence
  for the claimed effect are present.
- Donation changes must audit caller buffer reuse first. JAX buffer donation can
  invalidate the donated input after the compiled call.
- Persistent compilation-cache changes must be installed before the first JAX
  compile in the process.
- Host callbacks inside JIT are not a persistent-cache win. They can prevent
  cache hits across processes even if runtime cache thresholds are fixed.

## Official-doc corrections to the pasted audit

- Persistent cache writes are gated by both compile-time and entry-size
  thresholds. To cache every entry, use
  `jax_persistent_cache_min_compile_time_secs = 0.0` and
  `jax_persistent_cache_min_entry_size_bytes = -1`. Setting the entry-size
  threshold to `0` is not the strongest "cache everything" setting.
- `jax.scipy.sparse.linalg.gmres` defaults to `solve_method="batched"`.
  The Newton path already opts into `solve_method="incremental"`; LM should
  match that if numerical/memory parity is the goal.
- `jax.checkpoint` does not, by default, keep all `dot_general` residuals.
  The default policy is effectively "nothing saveable." A local
  `jax.ad_checkpoint.print_saved_residuals` probe confirmed that adding
  `dots_with_no_batch_dims_saveable` can save additional dot outputs.
- The pasted N10 sketch references `_boozer_objective_composed`, which does not
  exist in this tree. The composed public objective is
  `boozer_penalty_composed`.
- The pasted N17 wording says the typical dense Jacobian path uses
  `vmap(vjp_fn)`. For the common `n_res > n_dofs` case, this tree currently
  takes the forward-mode `linearize` branch over a dense dof basis.

## Status summary

| ID | Status | Implementation disposition |
| --- | --- | --- |
| N1 | Confirmed | Implement after config-location test |
| N2 | Confirmed | Implement device-spec caching |
| N3 | Confirmed with safety audit | Add donation only where caller reuse is proven absent |
| N4 | Confirmed | Re-key runner caches by callback presence, not callable identity |
| N5 | Confirmed with caveat | Hoist ArbVec contributions across all variants, including bucketed candidate costs |
| N6 | Confirmed with caveat | Drop scan-output history where replay is exact; backtracking needs removal replay |
| N7 | Confirmed | Add fused non-parity tensor contract |
| N8 | Confirmed | Reuse radial columns across scalar siblings |
| N9 | Invalid as written | Do not implement proposed policy change without profiling |
| N10 | Partial | Add composed penalty HVP with correct public objective |
| N11 | Confirmed opportunity | Design and validate surface-axis sharding |
| N12 | Confirmed opportunity | Add seed-batch sharding route |
| N13 | Confirmed | Match Newton transfer-guard scope |
| N14 | Confirmed | Match Newton GMRES incremental solve method |
| N15 | Confirmed | Replace plus/minus materialization with GEMV algebra |
| N16 | Confirmed | Replace static symmetry unroll with vectorized symmetry axis |
| N17 | Partial | Expose JVP/VJP helpers; shard dense basis only when matrix is required |
| N18 | Speculative | Profile first; Pallas/Triton is not approved by this doc |
| N19 | Confirmed low risk | Increase kernel LRU capacities after cache-info probe |
| N20 | Confirmed low priority | Optional diagnostic-path JAX replay |

## Recommended sequencing

### Wave 1: mechanical and low-risk runtime fixes

- [ ] N1: persistent compilation-cache thresholds.
- [ ] N13: LM matrix-free GMRES transfer-guard parity.
- [ ] N14: LM matrix-free GMRES `solve_method="incremental"`.
- [ ] N4: callback-stable solver-runner cache keys.
- [ ] N3: donation after caller-reuse audit.

Coupled implementation notes:

- N13 and N14 touch the same LM matrix-free GMRES call site and should normally
  land together.
- N3 and N4 both audit cached optimizer-runner factories, but N3's donation work
  must still be gated by caller buffer-liveness checks.

### Wave 2: hot-path refactors

- [ ] N2: Boozer interpolant device-spec caching.
- [ ] N5: GPMO ArbVec contribution hoist.
- [ ] N6: GPMO scan-output history reduction.
- [ ] N7: fused spline tensor contraction for non-parity mode.
- [ ] N8: Boozer radial column reuse.
- [ ] N10: composed penalty HVP.
- [ ] N15: PM candidate-cost GEMV formulation.
- [ ] N16: dipole-field symmetry vectorization.

### Wave 3: sharding research and multi-device proof

- [ ] N11: surface-quadrature sharding.
- [ ] N12: seed-batch sharding.
- [ ] N17: dense Jacobian basis sharding and matrix-free helper API.

### Wave 4: future and diagnostics

- [ ] N18: Biot-Savart Pallas/Triton feasibility study.
- [ ] N19: Biot-Savart kernel cache-size bump.
- [ ] N20: wireframe diagnostic replay cleanup.

## TODO details

## N1: persistent compilation cache thresholds

- [ ] Status: confirmed.

### Context

`src/simsopt/backend/runtime.py` configures `jax_compilation_cache_dir`, but the
current tree does not configure the persistent-cache write thresholds. Official
JAX docs say a persistent-cache entry is written only when both the compile-time
threshold and entry-size threshold pass. The default compile-time threshold is
1.0 second, which filters out many small kernels.

Current evidence:

- `src/simsopt/backend/runtime.py:1733-1734` sets
  `jax_compilation_cache_dir`.
- No `jax_persistent_cache_min_compile_time_secs` or
  `jax_persistent_cache_min_entry_size_bytes` setting was found under
  `src/simsopt` or `tests`.
- `src/simsopt/backend/runtime.py:555-560` provides the default JAX cache
  directory.

### Rationale

Many SIMSOPT JAX kernels are small enough to compile quickly but numerous enough
that fresh-process startup still pays tens of seconds in aggregate. The current
runtime setting enables a cache directory but still allows JAX's default
compile-time threshold to skip small entries.

### Implementation

- [ ] Add explicit threshold configuration next to the cache-dir update:

  ```python
  jax.config.update("jax_compilation_cache_dir", config.compilation_cache_dir)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  ```

- [ ] Ensure this runs before the first JAX compile in process startup.
- [ ] Keep the setting scoped to the runtime path that intentionally enables the
  persistent cache.
- [ ] Do not set entry-size threshold to `0` if the goal is "cache every entry";
  official docs use `-1` to disable the size restriction.
- [ ] Add a subprocess smoke test that imports the runtime, enables a JAX mode,
  and asserts the three config values before any compiled workload runs.
- [ ] Add a focused manual verification command that runs one small kernel twice
  in fresh processes and checks that the second run reuses a persistent entry.

### Acceptance criteria

- [ ] Runtime config reports cache dir plus both thresholds in a fresh process.
- [ ] A small sub-second kernel writes a persistent-cache entry.
- [ ] No host-callback benchmark is used as proof for this item.

## N2: cache InterpolatedBoozerFieldJAX device specs

- [ ] Status: confirmed.

### Context

`InterpolatedBoozerFieldJAX` stores regular-grid interpolation specs as host
objects and the scalar evaluation path rebuilds a staged device spec through
`build_regular_grid_interpolant_3d_device_spec(...)`. The cylindrical
`InterpolatedFieldJAX` path already stages device specs once during
construction.

Current evidence:

- `src/simsopt/jax_core/regular_grid_interp.py:288-308` stages arrays in
  `build_regular_grid_interpolant_3d_device_spec`.
- `src/simsopt/jax_core/regular_grid_interp.py:967-969` calls that builder
  inside `evaluate_batch`.
- `src/simsopt/jax_core/interpolated_boozer_field.py:167-189` stores `specs`, not
  device specs.
- `src/simsopt/jax_core/interpolated_boozer_field.py:646-662` scalar evaluators
  call `evaluate_batch`.
- `src/simsopt/field/boozermagneticfield_jax.py:740-747` dispatches scalar
  field calls through the evaluator cache.
- `src/simsopt/jax_core/interpolated_field.py:84-90` pre-stages cylindrical
  device specs.

### Rationale

The field-level scalar cache prevents repeated scalar recomputation after one
`set_points` cycle, but each scalar sibling still can trigger a fresh host to
device staging of the regular-grid cell table before its own cached value is
created. That is expensive on GPU and avoidable because the interpolant spec is
immutable for a frozen field state.

### Implementation

- [ ] Extend the Boozer frozen state or wrapper state to keep device specs next
  to host specs for all eager specs.
- [ ] Add an internal evaluation helper that accepts a pre-staged device spec
  and bypasses `build_regular_grid_interpolant_3d_device_spec(...)`.
- [ ] Preserve lazy host spec construction semantics where present, but cache
  the corresponding device spec exactly once per resolved lazy spec.
- [ ] Keep the immutable host spec as the source of truth; the device spec is a
  staged execution artifact.
- [ ] Update scalar evaluators to use the cached device spec.
- [ ] Add a regression test that monkeypatches or instruments the builder and
  proves multiple scalar siblings after one `set_points` do not rebuild the same
  device spec.
- [ ] Add CPU and GPU, when available, value parity against the pre-change
  evaluation path on a small grid.

### Acceptance criteria

- [ ] Device spec staging count is one per interpolant spec per frozen state.
- [ ] Scalar outputs match the old path to existing tolerances.
- [ ] No per-scalar host-to-device staging remains on the hot path.

## N3: donate solver-runner input buffers

- [ ] Status: confirmed missing optimization, requires caller-reuse audit.

### Context

No actual `donate_argnums` or `donate_argnames` usage was found under
`src/simsopt`. Multiple jitted solver runners allocate and return large pytrees
without donating their input buffers.

Current evidence:

- `src/simsopt/geo/optimizer_jax.py:1391` returns `jax.jit(run_solver)`.
- `src/simsopt/geo/optimizer_jax.py:2300` calls `jax.jit(run_solver)`.
- `src/simsopt/geo/optimizer_jax.py:3714` returns `jax.jit(run_solver)`.
- `src/simsopt/geo/optimizer_jax.py:3997` returns `jax.jit(run_solver)`.
- `src/simsopt/geo/optimizer_jax_private/_bfgs.py:279` builds a jitted
  BFGS runner without donation.
- `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:172` builds a jitted
  L-BFGS runner without donation.

### Rationale

Donation lets XLA reuse input buffers for outputs. Solver states can include
large vectors, residuals, gradients, dense Hessians, and BFGS history. Reducing
peak device memory matters for larger banana surfaces and matrix-free solver
workflows.

### Implementation

- [ ] For each runner, trace the caller and determine whether the donated
  argument is read after the compiled call.
- [ ] Add donation only where the consumed argument is semantically dead after
  the call.
- [ ] Prefer donation on internal solver state or flat initial state passed into
  a one-shot compiled runner.
- [ ] Do not donate public user arrays that callers may reasonably reuse.
- [ ] Add comments only where a call-site liveness decision is non-obvious.
- [ ] Add tests that call the donated runner and compare the solver result to
  the non-donated behavior on a small problem.
- [ ] Add a memory-profile command for at least one Hessian-bearing solver path.

### Acceptance criteria

- [ ] No donated input is reused by SIMSOPT after the compiled call.
- [ ] Tests do not emit buffer-donation warnings or invalid-buffer errors.
- [ ] Peak memory improves or stays flat on the profiled path.

## N4: make solver-runner LRU keys stable under callbacks

- [ ] Status: confirmed.

### Context

The LM and Newton runner factories include Python callback objects in their LRU
cache keys. Fresh closures with identical semantics therefore miss the cache and
can force retracing/recompilation.

Current evidence:

- `src/simsopt/geo/optimizer_jax.py:1255-1267` includes `callback` and
  `progress_callback` in the LM runner factory key.
- `src/simsopt/geo/optimizer_jax.py:3584` includes Newton-polish
  `progress_callback`.
- `src/simsopt/geo/optimizer_jax.py:3848` includes exact-Newton callback
  identity in the cached runner factory.

### Rationale

Callback identity is not part of the numerical compiled program. It should not
force a new compiled runner when only the closure object changes.

### Implementation

- [ ] Replace callback objects in cached-runner static keys with booleans such
  as `callback_enabled` and `progress_callback_enabled`.
- [ ] Route the active callable through a per-call mechanism outside the LRU
  identity.
- [ ] If a thread-local registry is used, document and test the concurrency
  assumption. Do not silently share callback state across concurrent solves.
- [ ] Keep callback-free benchmark paths callback-free; host callbacks are not
  persistent-cache friendly.
- [ ] Add a test that creates two different callback closures with the same
  enabled/disabled shape and proves the runner factory cache is reused.
- [ ] Add a test that callback-disabled and callback-enabled runners remain
  separate compiled shapes when their JAXPR differs.

### Acceptance criteria

- [ ] Fresh callback closures no longer miss the LRU cache.
- [ ] Callback behavior remains correct for progress reporting.
- [ ] Callback-free runners still contain no host callback operations.

## N5: hoist GPMO ArbVec contributions out of scan bodies

- [ ] Status: confirmed with caveat.

### Context

`_gpmo_arbvec_contributions(...)` builds the `(M, N, P)` contribution tensor.
It does not depend on scan-carried state, but some solve variants recompute it
inside scan bodies. The bucketed solve hoists one contribution tensor for its
update step, but its candidate-cost helper still recomputes internally.

Current evidence:

- `src/simsopt/jax_core/pm_optimization.py:801-805` defines
  `_gpmo_arbvec_contributions`.
- `src/simsopt/jax_core/pm_optimization.py:820` candidate-cost helper calls it.
- `src/simsopt/jax_core/pm_optimization.py:891` regular step recomputes it.
- `src/simsopt/jax_core/pm_optimization.py:972-976` regular scan body uses the
  step helper.
- `src/simsopt/jax_core/pm_optimization.py:1057` bucketed solve hoists
  contributions for update.
- `src/simsopt/jax_core/pm_optimization.py:1266` and `1487` backtracking path
  recompute contribution tensors.

### Rationale

The contribution tensor is large and invariant during a solve. Recomputing it
per iteration burns memory bandwidth and compute for no algorithmic benefit.

### Implementation

- [ ] Refactor candidate-cost helpers to accept precomputed contributions.
- [ ] Hoist contributions once before `gpmo_arbvec_solve` scan.
- [ ] Hoist contributions once before `gpmo_arbvec_solve_bucketed` scan,
  including the candidate-cost path.
- [ ] Hoist contributions once before `gpmo_arbvec_backtracking_solve` scan.
- [ ] Preserve masks, tie-breaking, and selected-vector semantics exactly.
- [ ] Add a small deterministic PM fixture comparing selected dipoles, vector
  indices, signs, and objective values before and after the refactor.
- [ ] Add an HLO or instrumentation check that contribution construction is no
  longer inside the scan body.

### Acceptance criteria

- [ ] All ArbVec solve variants compute invariant contributions once per solve.
- [ ] Selected candidates and final `x` match the current implementation on the
  deterministic fixture.
- [ ] Memory allocation for repeated contribution tensors disappears from the
  profiled scan body.

## N6: stop streaming unused GPMO `x_history` through scan output

- [ ] Status: confirmed with backtracking caveat.

### Context

`jax.lax.scan` materializes stacked outputs along the scan axis. The bucketed
and backtracking ArbVec variants stream full `x_history` through scan outputs,
which can persist hundreds of MB on device.

Current evidence:

- `src/simsopt/jax_core/pm_optimization.py:1091` bucketed scan returns `x_new`.
- `src/simsopt/jax_core/pm_optimization.py:1103` bucketed solve unpacks
  `x_history`.
- `src/simsopt/jax_core/pm_optimization.py:1559-1564` backtracking scan returns
  the next state's full `x`.
- `src/simsopt/jax_core/pm_optimization.py:1703-1710` backtracking solve
  unpacks and returns `x_history`.
- `src/simsopt/jax_core/pm_optimization.py:981-986` regular solve already
  reconstructs `x_history` from selected arrays after scan.

### Rationale

If history is only needed as an artifact, streaming every full vector through
scan output is an avoidable memory cost. For AD-through-history work, this also
inflates the saved activation footprint.

### Implementation

- [ ] For bucketed solve, replace scan-output `x_history` with selected
  dipoles, selected vector indices, selected signs, and final state.
- [ ] Reconstruct bucketed `x_history` after scan with the existing
  `_arbvec_x_history` helper if the selected arrays are sufficient.
- [ ] For backtracking solve, do not blindly use `_arbvec_x_history` until the
  removal and dewyrming semantics are represented in replay data. In this code,
  "dewyrming" is the C++-mirrored pass that removes anti-aligned adjacent placed
  dipole pairs and returns them to the available set.
- [ ] Either keep backtracking `x_history` unchanged for the first patch or add
  exact replay metadata for removals/dewyrming and test it.
- [ ] Add tests proving reconstructed histories match current histories for
  regular and bucketed variants.
- [ ] Add a memory-profile run showing scan-output size reduction.

### Acceptance criteria

- [ ] Bucketed solve no longer streams full `x_history` through scan output.
- [ ] Backtracking history is either unchanged or reconstructed with exact
  removal/dewyrming replay.
- [ ] Returned public artifacts remain identical on deterministic fixtures.

## N7: add fused tensor-product spline contraction for non-parity mode

- [ ] Status: confirmed.

### Context

The regular-grid interpolation path performs a tensor-product contraction with
three nested `lax.fori_loop` calls. This preserves CPU-ordered lane behavior,
but the JAX-native product path pays that serial loop structure even when exact
lane-order parity is not required.

Current evidence:

- `src/simsopt/jax_core/regular_grid_interp.py:639-656` implements
  `_cpu_ordered_tensor_contract` with nested loops.
- `src/simsopt/jax_core/regular_grid_interp.py:659` defines the jitted batch
  evaluator.
- `src/simsopt/jax_core/regular_grid_interp.py:778-779` calls the looped
  contract per evaluated point.

### Rationale

The contraction is algebraically an einsum over local spline weights and local
cell values. A fused form exposes more parallelism to XLA, especially on GPU.
The current looped path should remain available for strict parity lanes.

### Implementation

- [ ] Add a static parity-mode switch to the regular-grid interpolant
  evaluation path.
- [ ] Keep `_cpu_ordered_tensor_contract` for strict parity mode.
- [ ] Add a fused implementation equivalent to:

  ```python
  jnp.einsum("i,j,k,ijkv->v", pkx, pky, pkz, local_vals)
  ```

- [ ] Ensure the switch is static so it does not create runtime data-dependent
  branching in jitted code.
- [ ] Consider batching the fused contraction above the per-point body if that
  produces cleaner HLO.
- [ ] Add CPU parity tests proving the strict path is unchanged.
- [ ] Add numerical equivalence tests for the fused path at existing product
  tolerances.
- [ ] Add an HLO or timing note showing the fused path no longer lowers to
  triple scalar loops.

### Acceptance criteria

- [ ] Strict CPU parity mode still uses the old ordered contraction.
- [ ] Non-parity JAX path uses a fused contraction.
- [ ] Existing interpolation values remain within documented tolerances.

## N8: reuse Boozer radial spline columns across scalar siblings

- [ ] Status: confirmed.

### Context

Boozer radial-field scalar evaluators independently evaluate the same radial
columns for the same `s` points and state arrays. Field-level scalar caching
stores final scalar outputs, not shared intermediate radial columns.

Current evidence:

- `src/simsopt/jax_core/boozer_radial_field.py:354-359` defines `_column_at`.
- `src/simsopt/jax_core/boozer_radial_field.py:388-444` and later scalar
  evaluators repeatedly call `_column_at`.
- `src/simsopt/field/boozermagneticfield_jax.py:144`, `188-199`, and
  `238-242` implement per-scalar cache handling and cache invalidation on
  `set_points`.

### Rationale

Many Boozer scalar siblings need the same radial profile columns. Re-evaluating
them once per scalar wastes compile/runtime work on tracing and derivative
bundles.

### Implementation

- [ ] Add `_eval_radial_columns(state, s)` that evaluates each required radial
  profile column once for the current points.
- [ ] Return a compact immutable structure keyed by profile name.
- [ ] Rewrite scalar evaluators to consume the shared radial-column structure.
- [ ] Cache the radial-column bundle once per `set_points` cycle in the
  BoozerRadialInterpolantJAX wrapper cache.
- [ ] Keep the cache invalidation tied to the same point/state invalidation that
  clears scalar outputs today.
- [ ] Add tests that instrument `_column_at` and prove a scalar bundle does not
  re-evaluate the same column repeatedly.
- [ ] Add value/gradient parity tests for a representative scalar bundle.

### Acceptance criteria

- [ ] A scalar bundle evaluates each radial column once per points/state cycle.
- [ ] Boozer scalar values and derivatives match current behavior.
- [ ] Memory overhead is limited to one per-cycle radial-column bundle.

## N9: do not change checkpoint policies as proposed

- [ ] Status: invalid as written.

### Context

The pasted audit says the default `jax.checkpoint` policy keeps all
`dot_general` residuals and proposes explicit checkpoint policies for
Biot-Savart and tracing. Official docs and a local saved-residuals probe do not
support that claim.

Current evidence:

- `src/simsopt/jax_core/biotsavart.py:346-348` uses
  `jax.checkpoint(chunk_kernel)` and comments that the default remat policy is
  retained pending CUDA profiling.
- `src/simsopt/jax_core/tracing.py:289-290` uses `jax.checkpoint(body)`.
- Official JAX docs describe checkpoint policies as optional control over what
  may be saved.
- Local `jax.ad_checkpoint.print_saved_residuals` probing showed default
  checkpoint saved only inputs for a dot example, while
  `dots_with_no_batch_dims_saveable` saved an extra dot output.

### Rationale

Changing checkpoint policies can increase saved residuals, reduce recomputation,
or alter memory/runtime tradeoffs. The pasted proposed tracing policy may save
dot outputs rather than reduce memory. This needs profiling, not a mechanical
patch.

### Implementation

- [ ] Do not apply the pasted N9 policy change.
- [ ] Add a profiling task, not a code patch, if AD-through-trajectory memory is
  still a blocker.
- [ ] Use `jax.ad_checkpoint.print_saved_residuals` on the actual Biot-Savart
  chunk kernel and tracing body.
- [ ] Profile peak memory and wall time on CPU and real CUDA for:
  - default policy,
  - `nothing_saveable`,
  - `dots_with_no_batch_dims_saveable`,
  - any domain-specific policy chosen from measured residuals.
- [ ] Only change policy if measured evidence shows a net product-lane win.

### Acceptance criteria

- [ ] No checkpoint policy change is merged from this audit without profiling.
- [ ] Any future policy change includes saved-residuals evidence and memory
  measurements.

## N10: expose a composed penalty Hessian-vector product

- [ ] Status: partial. Need corrected API sketch.

### Context

The tree has dense Hessian materialization for one Boozer residual path and a
composed penalty objective/gradient path, but no public composed HVP helper.
The pasted sketch references a nonexistent `_boozer_objective_composed`.

Current evidence:

- `src/simsopt/geo/boozer_residual_jax.py:277-316` implements
  `boozer_residual_hessian` with `jax.hessian`.
- `src/simsopt/geo/boozer_residual_jax.py:290-291` says full composed Hessian
  callers should use `jax.hessian(boozer_penalty_composed)`.
- `src/simsopt/geo/boozer_residual_jax.py:626-687` implements
  `boozer_penalty_composed`.
- `src/simsopt/geo/boozer_residual_jax.py:690-703` implements
  `boozer_penalty_grad_composed`.

### Rationale

A public HVP helper lets Newton/Krylov or trust-region callers avoid dense
Hessian materialization on the composed path. This changes memory complexity
from dense matrix storage to a vector product for matrix-free methods.

### Implementation

- [ ] Add a public helper with a name tied to the actual objective, for example
  `boozer_penalty_hvp_composed(x, v, **kwargs)`.
- [ ] Implement through forward-over-reverse AD:

  ```python
  grad_fn = jax.grad(lambda y: boozer_penalty_composed(y, **kwargs))
  return jax.jvp(grad_fn, (x,), (v,))[1]
  ```

- [ ] Decide whether a residual-level JVP/VJP helper is also needed; do not name
  a penalty HVP as a residual HVP.
- [ ] Export the helper consistently with the surrounding public API.
- [ ] Add a small dense-Hessian comparison test:
  `hvp == jax.hessian(objective)(x) @ v`.
- [ ] Add one matrix-free optimizer integration test if an existing caller can
  consume the helper without broad refactor.

### Acceptance criteria

- [ ] Public composed HVP exists and uses `boozer_penalty_composed`.
- [ ] Small-problem HVP matches dense Hessian-vector multiplication.
- [ ] No dense Hessian is materialized inside the HVP helper.

## N11: shard surface quadrature in `integral_BdotN`

- [ ] Status: confirmed opportunity, requires multi-device design.

### Context

Biot-Savart can shard upstream over points, but the fixed-surface flux integral
path gathers `B` and performs residual/reduction work on one device.

Current evidence:

- `src/simsopt/jax_core/integral_bdotn.py:119-167` computes residual and
  reduction with no sharding.
- `src/simsopt/jax_core/integral_bdotn.py:181-224` exposes the fixed-surface
  integral.
- `src/simsopt/jax_core/objectives_flux.py:62-69` reshapes `B` and calls the
  fixed-surface integral.
- `src/simsopt/objectives/fluxobjective_jax.py:315-321` computes
  `biot_savart_B(...)` then calls the flux integral.

### Rationale

Dense surfaces already have a natural point/quadrature axis. Keeping reduction
local to one device underuses multi-GPU hardware and can force unnecessary
gathering.

### Implementation

- [ ] Define the surface-axis sharding contract in the same style as existing
  point-axis sharding utilities.
- [ ] Add a `shard_map` implementation that computes local residual terms and
  uses `psum` for scalar reductions.
- [ ] Preserve the existing single-device execution route as the single-device
  policy, not as a silent fallback from a failed multi-device path.
- [ ] Add forced multi-CPU-device tests for shape and numerical equivalence.
- [ ] Add real GPU validation before claiming product-lane speedup.

### Acceptance criteria

- [ ] Multi-device path returns the same scalar integral as single-device path.
- [ ] Surface-axis partitions are explicit and documented.
- [ ] Real multi-GPU run shows reduced gather/reduction bottleneck.

## N12: shard seed-batch multi-restart scoring

- [ ] Status: confirmed opportunity.

### Context

The batched surface-objective value-and-gradient pipeline uses serial
`lax.map(...)` over candidate restart seeds.

Current evidence:

- `src/simsopt/geo/surfaceobjectives_jax.py:5069-5077` returns
  `lax.map(compiled_value_and_grad_for, coil_dofs_batch)`.

### Rationale

Multi-restart scoring is embarrassingly parallel. A seed-batch sharding policy
can map one or more restart seeds per device and reduce wall time on multi-GPU
runs.

### Implementation

- [ ] Add a seed-batch sharding config next to trajectory-batch sharding config.
- [ ] Implement a `shard_map` route for seed-batch value-and-gradient scoring.
- [ ] Keep the serial route as the explicit single-device policy.
- [ ] Ensure returned batch ordering is identical to the input seed ordering.
- [ ] Add forced multi-CPU-device tests for shape, value, gradient, and ordering.
- [ ] Add a real multi-GPU restart scoring benchmark before claiming linear
  scaling.

### Acceptance criteria

- [ ] Seed-batch sharding produces identical ordered results.
- [ ] Single-device serial behavior is unchanged.
- [ ] Multi-device benchmark records scaling evidence.

## N13: add transfer-guard scope to LM matrix-free GMRES

- [ ] Status: confirmed.

### Context

The Newton matrix-free GMRES sibling scopes `jax.transfer_guard("allow")`
around the solver call. The LM matrix-free path does not.

Current evidence:

- `src/simsopt/geo/optimizer_jax.py:1427-1434` calls GMRES in the LM
  matrix-free body without a transfer-guard scope.
- `src/simsopt/geo/optimizer_jax.py:2932-2951` Newton's
  `_run_operator_gmres` wraps GMRES with `jax.transfer_guard("allow")`.

### Rationale

Strict transfer-guard lanes should behave consistently across LM and Newton.
The missing scope can trip strict CUDA/parity lanes even when the same GMRES
operation is allowed in the Newton path.

### Implementation

- [ ] Refactor LM to reuse the Newton helper if signatures can be aligned
  cleanly.
- [ ] Otherwise add the same transfer-guard scope directly around the LM GMRES
  call.
- [ ] Keep the guard as narrow as the Newton sibling's scope.
- [ ] Add a strict transfer-guard regression test for the LM matrix-free path.

### Acceptance criteria

- [ ] LM matrix-free GMRES no longer fails strict transfer-guard lanes for the
  known allowed solver transfer.
- [ ] Newton behavior is unchanged.

## N14: use incremental GMRES in LM matrix-free path

- [ ] Status: confirmed.

### Context

Official JAX docs and local signature probing show GMRES defaults to
`solve_method="batched"`. The Newton path explicitly uses
`solve_method="incremental"`, but LM does not.

Current evidence:

- `src/simsopt/geo/optimizer_jax.py:1427-1434` LM matrix-free GMRES omits
  `solve_method`.
- `src/simsopt/geo/optimizer_jax.py:2940-2951` Newton GMRES passes
  `solve_method="incremental"`.

### Rationale

The two matrix-free solver paths should use the same GMRES method unless there
is an explicit algorithmic reason not to. Incremental GMRES is documented as
more numerically stable, though it can have different performance overhead.

### Implementation

- [ ] Add `solve_method="incremental"` to the LM matrix-free GMRES call.
- [ ] Prefer a shared helper with N13 if that avoids duplicated solver options.
- [ ] Add a regression test comparing LM matrix-free convergence on a small
  deterministic problem before and after the change.
- [ ] Record memory and iteration behavior on at least one representative LM
  matrix-free solve.

### Acceptance criteria

- [ ] LM and Newton matrix-free GMRES use the same solve method.
- [ ] Small deterministic LM solve still converges to the same accepted result.

## N15: compute PM candidate costs with GEMV algebra

- [ ] Status: confirmed.

### Context

The PM candidate-cost path materializes plus and minus residual tensors for
all candidates. Algebraically, candidate costs can be computed from a residual
dot product and column norms.

Current evidence:

- `src/simsopt/jax_core/pm_optimization.py:666-667` materializes plus/minus
  candidate residuals.
- `src/simsopt/jax_core/pm_optimization.py:1788-1799` materializes selected
  plus/minus tensors in the multi-neighbor path.

### Rationale

For each candidate column `a_j` and residual `r`:

```text
||r +/- a_j||^2 = ||r||^2 +/- 2 r dot a_j + ||a_j||^2
```

The `||r||^2` term is common to all candidates and can be omitted from argmin
selection. This removes full `(M, 3N)` plus/minus intermediates.

### Implementation

- [ ] Precompute `col_sq = jnp.sum(A_arr * A_arr, axis=0)` once per solve.
- [ ] Compute `dot = A_arr.T @ residual` per step.
- [ ] Build plus/minus candidate costs from `+/- 2 * dot + col_sq` and existing
  penalties/masks.
- [ ] Preserve current tie-breaking order. If the current path concatenates plus
  before minus, keep that ordering.
- [ ] Apply the same algebra to the multi-neighbor selected-candidate path.
- [ ] Add deterministic tests for candidate index, sign, and final cost.
- [ ] Add a memory-profile check confirming plus/minus full tensors disappear.

### Acceptance criteria

- [ ] Candidate choices match current implementation on deterministic fixtures.
- [ ] Full plus/minus residual tensors are not materialized in the candidate
  cost hot path.
- [ ] Multi-neighbor behavior remains identical.

## N16: vectorize dipole-field symmetry copies

- [ ] Status: confirmed.

### Context

`_dipole_field_Bn_jit` statically unrolls symmetry copies with Python loops over
stellarator symmetry and field periods.

Current evidence:

- `src/simsopt/jax_core/dipole_field.py:471-506` loops over
  `range(stellsym + 1)` and `range(nfp)`.

### Rationale

Static unrolling bakes `2 * nfp` copies of the same computation into HLO. A
vectorized symmetry axis can keep compile size less dependent on `nfp` and
opens a natural sharding/batching axis.

### Implementation

- [ ] Build arrays for symmetry signs, field-period angles, and any rotation
  parameters.
- [ ] Implement `per_symmetry` for one symmetry/field-period copy.
- [ ] Use `jax.vmap(per_symmetry)` across the symmetry axis and reduce the
  contributions.
- [ ] Preserve coordinate transforms and normal projections exactly.
- [ ] Add tests comparing current looped behavior against the vectorized path
  for `nfp=1` and `nfp>1`, with and without stellarator symmetry.
- [ ] Add compile-time comparison for a W7-X-like `nfp=5` fixture.

### Acceptance criteria

- [ ] Vectorized result matches looped result to current tolerances.
- [ ] Compile size/time no longer scales linearly with static loop copies in the
  tested fixture.

## N17: avoid dense Jacobian materialization unless callers need it

- [ ] Status: partial. Dense materialization is real; pasted branch description
  needs correction.

### Context

`boozer_residual_jacobian_composed` materializes a dense residual Jacobian. For
the typical `n_res > n_dofs` case, the current tree uses forward-mode
`linearize` over a dense dof basis. For `n_res < n_dofs`, it uses VJP over a
residual basis.

Current evidence:

- `src/simsopt/geo/boozer_residual_jax.py:755-797` implements the dense
  composed residual Jacobian.

### Rationale

Some callers need only JVPs or VJPs. Forcing dense matrix construction costs
memory and can serialize work on one device. Exposing matrix-free helpers lets
callers choose the right contract.

### Implementation

- [ ] Add public or internal helpers for composed residual JVP and VJP:
  - `boozer_residual_jvp_composed(x, v, **kwargs)`
  - `boozer_residual_vjp_composed(x, cotangent, **kwargs)`
- [ ] Keep dense `boozer_residual_jacobian_composed` for callers that explicitly
  require the full matrix.
- [ ] For dense materialization, evaluate whether the basis sweep can be sharded
  over the smaller basis axis.
- [ ] Add tests showing JVP/VJP helpers match multiplication by the dense
  Jacobian on a small fixture.
- [ ] Add a caller audit to replace dense Jacobian use where only products are
  consumed.

### Acceptance criteria

- [ ] Matrix-free JVP/VJP helpers exist and match dense products.
- [ ] Dense Jacobian callers remain available and unchanged.
- [ ] Any sharded dense sweep has multi-device equivalence proof.

## N18: profile Biot-Savart Pallas/Triton feasibility

- [ ] Status: speculative future item.

### Context

The Biot-Savart inner integrand builds several intermediate arrays. A fused
custom kernel might keep more data in registers/shared memory, but that is an
engineering project, not a mechanical cleanup.

Current evidence:

- `src/simsopt/jax_core/biotsavart.py:370-419` contains the inner integrand chain.

### Rationale

Pallas/Triton can be worthwhile only if HBM materialization is the measured
bottleneck and the custom kernel preserves precision, AD behavior, and platform
coverage.

### Implementation

- [ ] Do not start with a Pallas/Triton rewrite.
- [ ] Profile the current compiled kernel on representative CPU and CUDA
  workloads.
- [ ] Identify exact intermediates that dominate memory traffic.
- [ ] Prototype a tiny custom-kernel proof outside the product path.
- [ ] Compare value, gradient/VJP, memory, and runtime against the current XLA
  implementation.
- [ ] Decide whether the maintenance cost is justified.

### Acceptance criteria

- [ ] Feasibility study produces a yes/no decision record grounded in profile
  data.
- [ ] Any future custom-kernel proposal includes benchmark, parity, and AD proof
  before it becomes an implementation item.

## N19: increase Biot-Savart kernel LRU capacities

- [ ] Status: confirmed low-risk opportunity.

### Context

Biot-Savart kernel builders use small `lru_cache` capacities. Mode-sweep
benchmarks can exceed those capacities and trigger recompilation churn.

Current evidence:

- `src/simsopt/jax_core/biotsavart.py:425` sets `_make_kernel` cache size to 32.
- `src/simsopt/jax_core/biotsavart.py:525` sets `_make_B_vjp_kernel` cache size to
  16.

### Rationale

Increasing cache capacities is a low-risk way to avoid Python-level kernel
builder churn during mode sweeps. It does not change numerical behavior.

### Implementation

- [ ] Add a small cache-info probe that reproduces the mode-sweep cache pressure.
- [ ] Increase `_make_kernel` capacity to 256 if the probe shows eviction.
- [ ] Increase `_make_B_vjp_kernel` capacity to 64 if the probe shows eviction.
- [ ] Keep the constants local and documented by the measured sweep shape.
- [ ] Add a lightweight test or script that records expected cache capacity.

### Acceptance criteria

- [ ] Mode-sweep probe no longer evicts hot kernel entries.
- [ ] No numerical behavior changes.

## N20: keep wireframe trajectory reconstruction low priority

- [ ] Status: confirmed low priority.

### Context

`get_gsco_iteration_jax` reconstructs post-solve trajectory diagnostics in a
host Python/NumPy loop.

Current evidence:

- `src/simsopt/solve/wireframe_optimization_jax.py:894-911` performs the host
  loop.

### Rationale

This path matters for large diagnostic runs, not for the main hot optimization
lane. It should not displace higher-leverage compile, memory, interpolation, or
solver fixes.

### Implementation

- [ ] Leave unchanged unless diagnostics on very large runs become a blocker.
- [ ] If reopened, identify whether the loop is on the critical path or only
  artifact reconstruction.
- [ ] If critical, implement a JAX scan/replay path with identical emitted
  diagnostics.
- [ ] Add a large-iteration diagnostic benchmark before and after.

### Acceptance criteria

- [ ] No work starts here until a concrete diagnostic bottleneck is recorded.

## Validation checklist for any completed item

- [ ] Code path matches the intended product lane.
- [ ] Existing CPU/reference behavior remains unchanged or is used as oracle.
- [ ] JAX CPU value/gradient behavior is covered where applicable.
- [ ] JAX GPU behavior is covered before claiming GPU payoff.
- [ ] Compile-time, memory, or wall-time payoff is measured for performance
  claims.
- [ ] No unrelated dirty files are staged with the fix.
- [ ] The status summary in this document is updated when the item lands.

## Official documentation references

- JAX persistent compilation cache:
  `https://docs.jax.dev/en/latest/persistent_compilation_cache.html`
- JAX buffer donation:
  `https://docs.jax.dev/en/latest/buffer_donation.html`
- JAX GMRES:
  `https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html`
- JAX checkpoint:
  `https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html`
- JAX checkpoint policy `nothing_saveable`:
  `https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint_policies.nothing_saveable.html`
