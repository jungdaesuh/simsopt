# Banana Optimization Todo List

Scope: `stage 2` and `single-stage banana` optimization, including the direct
SIMSOPT / `simsoptpp` implementations reached by those code paths.

Ordering rule: top-to-bottom by expected return on engineering time. High-impact
quick wins come first. Correctness-sensitive performance issues are promoted.

Low-level conclusion from the deep dive: most `simsoptpp` kernels are already
reasonable. The biggest remaining opportunities are in the human-written glue
above them.

Update `2026-04-16`: follow-up validation found several more concrete hotspots
in the Boozer solve path, `SquaredFlux`, `Derivative` composition, the Python
`BiotSavart` VJP layer, and `MagneticFieldSum`. Those addenda are recorded
below and should be considered live TODOs.

Update `2026-04-16` (second pass): a deeper validation pass against HEAD
confirmed the above and surfaced additional concrete items in the distance
accumulator allocations, the `Curve` Python derivative path, the ALM driver's
defensive-copy discipline, a `boozer_surface_residual` variable-shadowing
correctness trap, and a partial frontier-lane parallelism opportunity. Those
are recorded as addenda H through L. Claims that did not survive validation
(frontier atomic writes, curve kappa/torsion caching, `BiotSavart`
current-change cache invalidation, Pareto early-exit, and a naive drop-in
`lu_factor/lu_solve` swap) are explicitly excluded.

## Priority Legend

| Impact | Meaning |
|---|---|
| Very High | Likely visible wallclock reduction or removes major redundant work |
| High | Meaningful runtime / memory improvement |
| Medium | Useful but secondary |
| Low | Cleanup or speculative micro-optimization |

| Effort | Meaning |
|---|---|
| Small | Localized change, low coordination |
| Medium | Cross-file refactor or cache design |
| Large | Shared-library redesign or behavior-sensitive optimization |

| Quick Win | Meaning |
|---|---|
| Yes | Good first patch candidate |
| No | Needs deeper design / regression coverage |

## Ranked Todos

### 1. Split optimizer hot path from diagnostics

- Impact: Very High
- Effort: Small
- Quick win: Yes
- Why first: the current optimizer path recomputes expensive diagnostics on line-search probes and accepted steps.
- Files:
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:362-375`
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:728-820`
  - `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:116-212`
  - `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:215-281`
  - `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:284-380`
- Primary blame / last-touch:
  - `Jung Dae Suh`, `2026-04-07` to `2026-04-15`
- Todo:
  - Introduce a strict fast-path evaluator that returns only `total` and `grad`.
  - Move `B·n`, `shortest_distance()`, curvature summaries, per-term `J()/dJ()`, and logging into accepted-step callbacks or explicit reporting functions.
  - Keep search-path code free of reporting work.
- Correctness angle:
  - Reduces the risk of search logic drifting from report logic.

### 2. Remove hot-loop exact distance diagnostics from optimizer evaluations

- Impact: Very High
- Effort: Small
- Quick win: Yes
- Why second: `shortest_distance()` uses full sampled distance scans and is being called from optimization code that should not need exact diagnostics every probe.
- Files:
  - `src/simsopt/geo/curveobjectives.py:211-221`
  - `src/simsopt/geo/curveobjectives.py:315-327`
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:371`
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:759-777`
- Primary blame / last-touch:
  - `Florian Wechsung`, `2022-04-04` / `2022-04-11`
  - `Bharat Medasani`, `2023-06-15` / `2023-06-16`
  - `Jung Dae Suh`, `2026-04-07` / `2026-04-15`
- Todo:
  - Stop calling `shortest_distance()` inside optimizer evaluation functions.
  - Use smoothed constraint surrogates for optimization and reserve exact sampled minimum distance for callback/reporting.
  - If an exact metric must be shown frequently, cache it per accepted iterate only.
- Correctness angle:
  - Keeps the exact diagnostic metric from unintentionally acting like part of the objective.

### 3. Reduce oversized L-BFGS-B memory settings

- Impact: High
- Effort: Small
- Quick win: Yes
- Why third: `maxcor=300` is expensive relative to the problem sizes here and is easy to tune down.
- Files:
  - `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1123-1133`
  - `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1165-1169`
  - `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1219-1224`
  - `src/simsopt/geo/boozersurface.py:659-663`
- Primary blame / last-touch:
  - `Jung Dae Suh`, `2026-03-18` / `2026-04-06` / `2026-04-10`
  - `Andrew Giuliani`, `2024-05-03`
- Todo:
  - Benchmark `maxcor` values like `20`, `40`, `60`.
  - Lower the default unless there is measured evidence that `300` materially helps.
  - Record memory and iteration-count tradeoffs in one reproducible benchmark note.
- Correctness angle:
  - None if tuned carefully, but convergence behavior should be regression-checked.

### 4. Stop cloning full ALM history on every callback

- Impact: Medium
- Effort: Small
- Quick win: Yes
- Why fourth: easy memory reduction with little algorithmic risk.
- Files:
  - `examples/single_stage_optimization/alm_utils.py:1225-1233`
- Primary blame / last-touch:
  - `Jung Dae Suh`, `2026-04-10`
- Todo:
  - Emit only the latest entry plus immutable summary state, or pass the existing history object without rebuilding `[dict(entry) for entry in history]` every time.
  - If snapshot immutability is required, snapshot only on persistence boundaries, not every callback.
- Correctness angle:
  - Make callback ownership explicit so mutable/shared history semantics are clear.

### 5. Fix `CurveSurfaceDistance` dependency invalidation

- Impact: Medium
- Effort: Small
- Quick win: Yes
- Why fifth: this is both a correctness fix and a cache-validity fix.
- Files:
  - `src/simsopt/geo/curveobjectives.py:295-304`
  - `src/simsopt/geo/curveobjectives.py:309-361`
- Primary blame / last-touch:
  - `Bharat Medasani`, `2023-06-15` / `2023-06-16`
  - `Florian Wechsung`, `2022-04-11`
- Todo:
  - Change `super().__init__(depends_on=curves)` so the surface is also a dependency.
  - Add a regression test proving candidate caches are invalidated when the surface changes and curves do not.
- Correctness angle:
  - This is the clearest concrete cache-invalidation bug found in the deep dive.

### 6. Collapse the single-stage Boozer-derived objective graph into a shared evaluation cache

- Impact: Very High
- Effort: Medium
- Quick win: No
- Why sixth: `Iotas`, `NonQuasiSymmetricRatio`, and `BoozerResidual` each redo substantial field / surface / adjoint work, and the bundle currently builds separate `BiotSavart(coils)` objects for them.
- Files:
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:2953-3037`
  - `src/simsopt/geo/surfaceobjectives.py:725-835`
  - `src/simsopt/geo/surfaceobjectives.py:925-979`
  - `src/simsopt/geo/surfaceobjectives.py:1030-1123`
- Primary blame / last-touch:
  - Bundle assembly: `Jung Dae Suh`, `2026-04-08` to `2026-04-15`
  - Shared SIMSOPT terms: mainly `Andrew Giuliani`, `2022-05-04` to `2024-05-15`
- Todo:
  - Design one per-surface evaluation bundle that owns the Boozer solve state, field evaluation, and adjoint factors.
  - Have `Iota`, QS ratio, and Boozer residual objective values read from shared cached intermediates.
  - Avoid constructing parallel `BiotSavart` instances unless there is a proven state-isolation requirement.
- Correctness angle:
  - Shared caching must have precise invalidation rules tied to coil DOFs and Boozer surface DOFs.

### 7. Rework stage 2 smoothed minimum-distance surrogates to avoid Python pair-block materialization

- Impact: High
- Effort: Medium
- Quick win: No
- Why seventh: the current implementation materializes `diffs`, `dists`, masks, and selected-entry lists in Python for every evaluation.
- Files:
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:612-668`
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py:671-725`
- Primary blame / last-touch:
  - `Jung Dae Suh`, `2026-04-08` / `2026-04-15`
- Todo:
  - Replace the current pair-block assembly with a fused implementation that does selection and accumulation without storing all intermediate blocks.
  - Reuse candidate pruning from the distance stack where possible.
  - Consider a compiled path if NumPy-only fusion is still memory-heavy.
- Correctness angle:
  - Preserve the sign convention introduced in the `2026-04-15` return fix.

### 8. Gate `BoozerResidualExact` to true final-only usage

- Impact: High
- Effort: Small
- Quick win: Yes
- Why eighth: it upsamples both dimensions by `4x`, so the exact residual surface has `16x` more points than the solved surface.
- Files:
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1282-1312`
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:2978-2981`
- Primary blame / last-touch:
  - Core implementation: `Rithik Banerjee`, `2026-02-03`
  - Banana-specific edits: `Jung Dae Suh`, `2026-03-26` / `2026-04-03` / `2026-04-09`
- Todo:
  - Verify that `BoozerResidualExact` is never used in search loops except where explicitly intended.
  - Add an assertion or explicit mode gate if needed.
  - Consider a staged-resolution path if exact evaluation needs to run more than once at the end.
- Correctness angle:
  - Preserve exact-stage semantics; this is a usage-policy change, not a mathematical rewrite.

### 9. Optimize `CurveCWSFourierCPP`

- Impact: Very High
- Effort: Large
- Quick win: No
- Why ninth: this is the strongest root-level hotspot in the human-written geometry path. It repeatedly calls surface linear evaluators and builds large temporary `einsum` tensors.
- Files:
  - `src/simsopt/geo/curvecwsfourier.py:164-360`
- Primary blame / last-touch:
  - `Antoine Baillod`, `2024-12-05`
- Todo:
  - Profile `gamma`, `dgamma_by_dcoeff`, `gammadash`, `dgammadash_by_dcoeff`, and `dgammadashdash_by_dcoeff`.
  - Cache reusable surface-linear evaluations at the current `(phi, theta)` sample points.
  - Reduce repeated `np.zeros` allocation and large `einsum` tensor construction.
  - Consider moving the hottest derivative assembly into a compiled implementation if caching alone is not enough.
- Correctness angle:
  - Add numerical derivative regression tests before any aggressive rewrite.

### 10. Add a lower-memory `BiotSavart` compute mode

- Impact: Medium to High
- Effort: Large
- Quick win: No
- Why tenth: the current kernel stores per-coil `B_i`, `dB_i`, and `ddB_i` caches, which is memory-heavy but sometimes necessary.
- Files:
  - `src/simsoptpp/magneticfield_biotsavart.cpp:13-87`
  - `src/simsopt/field/biotsavart.py:94-119`
- Primary blame / last-touch:
  - Core compute path: `Florian Wechsung`, `2021-06-11` / `2021-07-14`
  - OpenMP update: `Antoine Baillod`, `2024-04-23`
  - Python `B_vjp`: mostly `Florian Wechsung`, `2021-06-01` / `2021-07-15`
- Todo:
  - Introduce a mode that computes only total `B`, `dB`, `ddB` when coil-current derivative access is not needed.
  - Keep the current per-coil cache path for VJP cases that require `dB_by_dcoilcurrents`.
  - Benchmark memory footprint and cache behavior on the banana workloads specifically.
- Correctness angle:
  - API semantics must stay explicit so callers know whether per-coil derivative caches are available.

### 11. Revisit the JAX distance objective loop structure

- Impact: Medium
- Effort: Medium
- Quick win: No
- Why eleventh: candidate pruning is decent, but `J()` / `dJ()` still loop in Python and invoke JAX per pair.
- Files:
  - `src/simsopt/geo/curveobjectives.py:223-258`
  - `src/simsopt/geo/curveobjectives.py:329-361`
  - `src/simsoptpp/python_distance.cpp:40-169`
- Primary blame / last-touch:
  - Python objective loops: mainly `Florian Wechsung`, `2022-04-04` / `2022-04-11`
  - C++ candidate pruning: `Florian Wechsung`, `2021-06-11` / `2022-04-04`; minor later edits by `Antoine Baillod`
- Todo:
  - Explore batched candidate evaluation to reduce Python overhead.
  - Remove redundant surface `gamma()` fetches and duplicate `gammas = self.surface.gamma().reshape((-1, 3))`.
  - Leave the C++ pruning alone until a profile shows it matters.
- Correctness angle:
  - Preserve candidate completeness guarantees.

### 12. Keep low-level kernel rewrites last

- Impact: Low unless profiling proves otherwise
- Effort: Large
- Quick win: No
- Why last: most remaining pain is still in Python orchestration and duplicated
  evaluation work above the core kernels. Do not spend time rewriting
  `simsoptpp` arithmetic before the higher-level glue issues and validated
  invalidation bugs are fixed.
- Files:
  - `src/simsopt/geo/boozersurface.py:422-533`
  - `src/simsopt/geo/boozersurface.py:608-668`
- Primary blame / last-touch:
  - `Florian Wechsung`, `Matt Landreman`, `Rogerio Jorge`
  - `Andrew Giuliani`, `mishapadidar`, `Jung Dae Suh`
- Todo:
  - Do not spend time rewriting these kernels before the higher-level
    duplication is removed.
  - Re-profile after items 1 through 11; only then decide whether any kernel-level work remains justified.

## 2026-04-16 Addenda

These items were validated after the initial draft. Several of them likely
belong ahead of items 6 through 12 on a pure gain-to-risk basis.

### A. Fix `SquaredFlux` field-point invalidation and stale surface coupling

- Impact: Very High
- Effort: Small
- Quick win: Yes
- Why this matters: `SquaredFlux` sets field evaluation points once in
  `__init__` and only depends on `field`, not `surface`, so a moving surface can
  silently combine stale `B(x_old)` with fresh normals from `surface.normal()`.
- Files:
  - `src/simsopt/objectives/fluxobjective.py:51-106`
- Todo:
  - Make `surface` part of the dependency / invalidation chain.
  - Refresh field points when the surface geometry changes instead of only once
    at construction.
  - Add a regression test that moves the surface while keeping the field object
    fixed.
- Correctness angle:
  - This is a silent wrong-answer bug, not just a speed issue.

### B. Reuse factorization in exact Boozer Newton and adjoint setup

- Impact: High
- Effort: Small to Medium
- Quick win: Yes
- Why this matters: the exact Newton path currently does two fresh
  `np.linalg.solve(J, ...)` factorizations per step and then a separate `lu(J)`
  for the adjoint data.
- Files:
  - `src/simsopt/geo/boozersurface.py:1077-1119`
  - `src/simsopt/objectives/utilities.py:11-29`
- Todo:
  - Reuse one factorization across the Newton step, iterative refinement, and
    stored adjoint solve data.
  - If switching away from `(P, L, U)`, update `forward_backward(...)`
    accordingly instead of treating `lu_factor/lu_solve` as a drop-in swap.
- Correctness angle:
  - Preserve the current iterative-refinement behavior and adjoint solve
    semantics while reducing duplicate linear algebra work.

### C. Make the Boozer LS second-order path lower-memory and remove easy copies

- Impact: Very High
- Effort: Large
- Quick win: No
- Why this matters: the `derivatives=2` LS path explicitly builds large
  second-order tensors and a dense `H`, which is the clearest OOM risk found in
  the review.
- Files:
  - `src/simsopt/geo/boozersurface.py:733-750`
  - `src/simsopt/geo/surfaceobjectives.py:417-545`
- Todo:
  - Replace explicit `d2B_dcdc` / `H` materialization with a matrix-free or
    reduced second-order formulation.
  - While touching this path, remove the unnecessary
    `x.reshape(...).copy()` / branch `.copy()` allocations in
    `boozer_surface_residual(...)`.
- Correctness angle:
  - Add regression coverage for weighted and unweighted paths before changing
    the second-order implementation.

### D. Tighten Boozer-derived objective reuse and first-use safety

- Impact: High
- Effort: Small to Medium
- Quick win: Yes
- Why this matters: `Iotas`, `NonQuasiSymmetricRatio`, and `BoozerResidual`
  still redo avoidable field / surface / adjoint work, and `BoozerResidual`
  duplicates a same-grid surface plus redundant `set_points` / residual passes.
- Files:
  - `src/simsopt/geo/surfaceobjectives.py:725-835`
  - `src/simsopt/geo/surfaceobjectives.py:925-979`
  - `src/simsopt/geo/surfaceobjectives.py:1048-1143`
- Todo:
  - Reuse the solved-surface data directly where the grid is identical instead
    of cloning an independent same-grid surface.
  - Remove redundant `set_points(...)`, `boozer_surface_residual(...)`, and
    `boozer_surface_residual_dB(...)` passes inside `BoozerResidual.compute()`.
  - Guard `self.boozer_surface.res` for first-use or failed-solve cases in
    `Iotas`, `NonQuasiSymmetricRatio`, and `BoozerResidual`.
  - Fold this into item 6's broader shared-cache design if doing a larger
    refactor.
- Correctness angle:
  - This closes a latent initialization / failed-solve bug in addition to
    cutting repeated work.

### E. Eliminate systemic `Derivative` accumulation copies

- Impact: High
- Effort: Medium
- Quick win: Yes
- Why this matters: the expensive `Derivative.__add__` copy pattern is now
  confirmed in generic composition code, not just in one `BiotSavart` call
  site.
- Files:
  - `src/simsopt/_core/derivative.py:114-143`
  - `src/simsopt/_core/optimizable.py:1825-1828`
  - `src/simsopt/objectives/utilities.py:136-143`
  - `src/simsopt/field/biotsavart.py:69-119`
  - `src/simsopt/field/magneticfield.py:268-269`
  - `src/simsopt/geo/accessibility.py:486-515`
  - `src/simsopt/geo/accessibility.py:685-706`
  - `src/simsopt/field/coilset.py:253-355`
- Todo:
  - Add a one-allocation `Derivative.sum(...)` or an equivalent explicit
    in-place accumulator.
  - Switch `OptimizableSum`, `MPIObjective`, `MagneticFieldSum.B_vjp`,
    accessibility objectives, and the Python `BiotSavart` VJP returns off
    Python `sum(...)`.
  - Reuse `BiotSavart` VJP work buffers when the coil structure and point grid
    are unchanged.
- Correctness angle:
  - Keep derivative-key semantics identical while changing only the aggregation
    strategy.

### F. Replace `MagneticFieldSum` temporary stacks and MPI ndarray sums with in-place accumulation

- Impact: Medium
- Effort: Small
- Quick win: Yes
- Why this matters: `MagneticFieldSum` currently builds temporary full-array
  stacks for `B`, `A`, and their derivatives, and `sum_across_comm(...)` reduces
  gathered ndarrays through repeated Python additions.
- Files:
  - `src/simsopt/field/magneticfield.py:250-266`
  - `src/simsopt/objectives/utilities.py:36-49`
- Todo:
  - Replace `np.sum([child_array, ...], axis=0)` with direct accumulation into
    the destination buffer.
  - Replace `sum(comm.allgather(data))` with an in-place or collective reduction
    path that avoids repeated ndarray temporaries.
- Correctness angle:
  - Preserve dtype and empty-collection behavior while removing the extra
    allocations.

### G. Clean up `CurrentPenalty` JAX tracing

- Impact: Low to Medium
- Effort: Small
- Quick win: Yes
- Why this matters: the penalty and its grad are currently built from plain
  lambdas, so the JAX transform setup is more expensive than it needs to be for
  a hot wrapper objective.
- Files:
  - `src/simsopt/field/coilobjective.py:18-30`
- Todo:
  - Prebuild and, if appropriate, `jit` the scalar penalty and gradient once
    instead of constructing `grad(self.J_jax, ...)` through lambdas.
- Correctness angle:
  - Preserve threshold behavior for positive and negative currents.

### H. Stop over-allocating VJP accumulators in curve distance objectives

- Impact: Medium to High
- Effort: Small
- Quick win: Yes
- Why this matters: `CurveCurveDistance.dJ()` and `CurveSurfaceDistance.dJ()`
  each allocate `np.zeros_like(c.gamma())` and `np.zeros_like(c.gammadash())`
  for *every* curve on every call, but only the curves appearing in the
  candidate-pair list actually receive writes. On single-stage banana
  configurations with tens of coils and only a handful of active candidate
  pairs, most of the allocation is dead weight and it happens hundreds of
  times per ALM outer iteration.
- Files:
  - `src/simsopt/geo/curveobjectives.py:243-258`
  - `src/simsopt/geo/curveobjectives.py:347-361`
- Todo:
  - Key the accumulator on the set of curve indices actually touched by
    candidates, either via a dict keyed on `i`/`j` or a reusable persistent
    buffer zeroed only for touched indices.
  - Hoist the accumulator allocation out of `dJ()` when the set of active
    curves is stable across a call (reset via `recompute_bell`).
- Correctness angle:
  - Keep the existing scatter semantics; the final `dgamma_by_dcoeff_vjp` /
    `dgammadash_by_dcoeff_vjp` sum must still produce identical gradients.

### I. Vectorize the Python DOF loop in `Curve.dkappadash_by_dcoeff`

- Impact: Medium
- Effort: Small
- Quick win: Yes
- Why this matters: `dkappadash_by_dcoeff` iterates `for i in range(self.num_dofs())`
  and redoes a full set of cross / inner products per DOF. For a coil with
  several tens of Fourier DOFs, this is pure Python-level iteration over work
  that is broadcastable.
- Files:
  - `src/simsopt/geo/curve.py:357-411`
- Todo:
  - Replace the per-DOF Python loop with a single broadcasted NumPy
    computation over the last axis of `dgamma_dcoeff_` / `d2gamma_dcoeff_` /
    `d3gamma_dcoeff_`.
  - Keep the expression grouped exactly like the current loop body so the
    change is a pure rewrite.
- Correctness angle:
  - Add a numerical regression test against the current output before
    changing anything, since the expression is non-trivial.

### J. Clean up ALM driver defensive-copy discipline

- Impact: Medium
- Effort: Small to Medium
- Quick win: Yes
- Why this matters: the ALM driver performs `np.asarray(..., dtype=float).copy()`
  and similar defensive copies at many inner-loop entry points, and uses
  `deepcopy(fallback_evaluation)` on the sanitization path. Each inner
  iteration touches several of these. When the caller already provides a
  contiguous `float64` array, the `asarray` is a no-op but the `.copy()` is
  not.
- Files:
  - `examples/single_stage_optimization/alm_utils.py:130`
  - `examples/single_stage_optimization/alm_utils.py:450`
  - `examples/single_stage_optimization/alm_utils.py:868-874`
  - `examples/single_stage_optimization/alm_utils.py:1198-1232`
  - `examples/single_stage_optimization/alm_utils.py:1273`
  - `examples/single_stage_optimization/alm_utils.py:1392-1412`
  - `examples/single_stage_optimization/alm_utils.py:1450`
  - `examples/single_stage_optimization/alm_utils.py:1504-1509`
  - `examples/single_stage_optimization/alm_utils.py:1625`
  - `examples/single_stage_optimization/alm_utils.py:1658`
  - `examples/single_stage_optimization/alm_utils.py:1742`
  - `examples/single_stage_optimization/alm_utils.py:1830`
- Todo:
  - Copy only at persistence / checkpoint boundaries (trust-region snapshots,
    outer-iteration state dumps). At API boundaries inside the hot loop,
    prefer `np.ascontiguousarray(..., dtype=float)` without `.copy()`.
  - Replace `deepcopy(fallback_evaluation)` with a shallow dict rebuild that
    copies only the arrays that are actually mutated downstream.
  - Cache `_build_box_bounds` output keyed on `(normalized_trust_radius,
    id(center))` since the trust radius does not change every inner call.
- Correctness angle:
  - Document which fields are owned versus borrowed so future readers know
    which shallow references are safe.

### K. Eliminate `d2B2_dcdc` variable shadowing in `boozer_surface_residual`

- Impact: Low on runtime, Medium on maintainability and correctness risk
- Effort: Small
- Quick win: Yes
- Why this matters: `d2B2_dcdc` is assigned with one semantic in the
  unweighted branch and then re-assigned with a different semantic inside the
  `weight_inv_modB` branch. The reader has to trace carefully to confirm
  nothing downstream uses the pre-shadow value. This is a live correctness
  trap when the second-order path is next edited.
- Files:
  - `src/simsopt/geo/surfaceobjectives.py:493-533`
- Todo:
  - Split into clearly named locals such as `d2B2_dcdc_base` and
    `d2B2_dcdc_weighted`, and thread them explicitly through the downstream
    expressions.
  - While the branch is being touched, collapse the unnecessary `.copy()`
    calls on `rtil`, `drtil_dc`, `drtil_diota`, and `drtil_dG` when the
    returned arrays are immediately consumed.
- Correctness angle:
  - Pair the rename with a regression test that exercises both the weighted
    and unweighted paths at `derivatives=2`.

### L. Enable partial frontier-lane parallelism within warm-start groups

- Impact: Medium to High on wallclock
- Effort: Medium
- Quick win: No
- Why this matters: the critic correctly flagged that naive cross-lane
  parallelism is unsafe because `reuse_latest_certified` makes later lanes
  depend on earlier certified lanes. However, lanes that do not opt into
  `reuse_latest_certified`, and groups of lanes that share a common
  pre-certified base warm start, are independent and can be executed in
  parallel.
- Files:
  - `examples/single_stage_optimization/run_single_stage_frontier_campaign.py:329-344`
  - `examples/single_stage_optimization/run_single_stage_frontier_campaign.py:715-788`
- Todo:
  - Partition `lane_specs` into warm-start-independent groups based on the
    mode and the base path returned by `resolve_frontier_lane_warm_start`.
  - Within each group, execute with `concurrent.futures.ProcessPoolExecutor`
    or `joblib.Parallel`, merging per-lane archive files at group boundaries.
  - Preserve the existing serial semantics when all lanes share a single
    latest-certified chain.
- Correctness angle:
  - Keep archive writes atomic (already using `mkstemp` + `os.replace`), and
    ensure per-lane result files never collide by including the lane id in
    the filename.

## Suggested Patch Order

`2026-04-16` update: treat addenda `A`, `B`, `D`, `E`, and `F` as better
near-term patch candidates than the old long-horizon items 9 through 12. Treat
addendum `C` as the highest-priority memory fix if the Boozer LS path is
showing OOM or severe RSS growth. Addenda `H`, `I`, and `J` are additional
small-effort quick wins that pair naturally with the earlier items. Addendum
`K` should go in with any edit to the Boozer LS second-order path. Addendum
`L` is the highest-wallclock frontier campaign win once the base path is
stabilized.

1. Item 1: split optimizer fast path from diagnostics.
2. Item 2: stop hot-loop `shortest_distance()` usage.
3. Item 3: lower `maxcor` and benchmark.
4. Item 4: stop cloning ALM history.
5. Item 5: fix `CurveSurfaceDistance` dependencies and add regression coverage.
6. Item 8: gate `BoozerResidualExact` usage.
7. Item 7: refactor stage 2 smoothed distance surrogates.
8. Item 6: build shared single-stage Boozer evaluation cache.
9. Item 11: revisit distance objective batching if still hot.
10. Item 9: optimize `CurveCWSFourierCPP`.
11. Item 10: add lower-memory `BiotSavart` mode if still justified.

## Validation Notes For Future Patches

- Always benchmark accepted-step wallclock and total function evaluations separately.
- Track cumulative allocation or peak RSS, not just runtime.
- For correctness-sensitive refactors, add regression tests before optimizing:
  - `CurveSurfaceDistance` cache invalidation
  - `SquaredFlux` field-point invalidation when the surface moves
  - `CurveCWSFourierCPP` derivative consistency
  - shared Boozer cache invalidation
  - exact Boozer Newton / adjoint factor reuse
  - `Derivative` accumulation semantics after switching away from Python `sum(...)`
  - stage 2 smoothed distance sign / gradient conventions
  - curve distance VJP accumulator output equivalence (addendum H)
  - `Curve.dkappadash_by_dcoeff` vectorization equivalence (addendum I)
  - weighted and unweighted `boozer_surface_residual` second-order paths
    (addenda C and K)
  - ALM evaluation-dict ownership after copy-pattern cleanup (addendum J)
  - frontier-lane partitioning correctness against `reuse_latest_certified`
    semantics (addendum L)

## Claims Intentionally Excluded

These were flagged during the deep dive but did not survive validation against
HEAD. They are recorded here so future reviewers do not re-raise them:

- Frontier campaign writes being corrupt-prone. The campaign path already
  uses `mkstemp` + `os.replace` atomic writes in
  `banana_opt/frontier_engine_base.py` and `banana_opt/frontier_campaign_reporting.py`.
- `Curve.kappa()` and `Curve.torsion()` lacking caching. Both are cached via
  `check_the_cache` in `src/simsoptpp/curve.h`.
- `BiotSavart` returning stale fields after coil current changes. The Python
  wrapper's `MagneticField.recompute_bell(...)` path clears the field cache,
  and the C++ side invalidates the per-coil cache on `set_points_cart`.
- Frontier Pareto dominance lacking an early exit. The per-candidate check in
  `banana_opt/frontier_dominance.py` already returns on the first failing
  metric.
- Swapping the exact Boozer Newton to `scipy.linalg.lu_factor` /
  `lu_solve` as a drop-in change. `forward_backward(...)` in
  `src/simsopt/objectives/utilities.py` explicitly consumes `(P, L, U)`; any
  refactor must update that contract as well (captured in addendum B).
