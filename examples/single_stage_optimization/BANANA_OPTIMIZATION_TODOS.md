# Banana Optimization Todo List

Scope: `stage 2` and `single-stage banana` optimization, including the direct
SIMSOPT / `simsoptpp` implementations reached by those code paths.

Ordering rule: top-to-bottom by expected return on engineering time. High-impact
quick wins come first. Correctness-sensitive performance issues are promoted.

Low-level conclusion from the deep dive: most `simsoptpp` kernels are already
reasonable. The biggest remaining opportunities are in the human-written glue
above them.

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
- Why last: `SquaredFlux.J()`, the vectorized Boozer residual path, and the core `BiotSavart` arithmetic already look structurally sound.
- Files:
  - `src/simsopt/objectives/fluxobjective.py:65-106`
  - `src/simsopt/geo/boozersurface.py:422-533`
  - `src/simsopt/geo/boozersurface.py:608-668`
- Primary blame / last-touch:
  - `Florian Wechsung`, `Matt Landreman`, `Rogerio Jorge`
  - `Andrew Giuliani`, `mishapadidar`, `Jung Dae Suh`
- Todo:
  - Do not spend time rewriting these kernels before the higher-level duplication is removed.
  - Re-profile after items 1 through 11; only then decide whether any kernel-level work remains justified.

## Suggested Patch Order

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
  - `CurveCWSFourierCPP` derivative consistency
  - shared Boozer cache invalidation
  - stage 2 smoothed distance sign / gradient conventions
