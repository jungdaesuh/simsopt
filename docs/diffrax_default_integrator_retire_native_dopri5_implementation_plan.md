# Make diffrax the Default JAX Integrator with Native Fallback

> Created: 2026-06-26 · Status: PLAN (not started) · Repo `simsopt-pr-jax-port-clean` @ `pr/jax-port-clean` (HEAD `131392405`)
> Risk tier: **Tier 3+ (public/widely-used internal API + observable-behavior change)** → API-evolution gate applies (see Validation).

## Purpose

Decide and sequence the work to make **diffrax** the default integration engine for *all* JAX arc-length tracing (field-line **and** particle) while keeping the in-repo hand-rolled DOPRI5 (`dopri5_native`) as the validated fallback/parity reference through the first default flip. Full native removal is a later, separately approved post-soak decision, not a prerequisite for this plan. The intended steady state for Phases 0-3 is JAX tracing = **diffrax for arc-length stepping by default** + the **φ-param DESC tracer** for differentiable Poincaré + **native DOPRI5 still selectable**.

This is a large, staged migration, **not** a flag flip. The safe default flip keeps the C++-validated native path available; any deletion phase removes that fallback and is gated on diffrax soaking at full parity across every RHS family first.

## Goals

- `integrator="dopri5_diffrax"` becomes the **default** for `FieldlineTracingSpec` (and the new equivalent on the particle specs), with `integrator="dopri5_native"` retained as the explicit fallback until a separate removal decision.
- diffrax backends exist for **all four RHS families** the native engine owns today, at **C++-oracle parity**: field-line, Cartesian guiding-center (vacuum), full-orbit Lorentz (vacuum), Boozer guiding-center (`vacuum`/`no_k`/`full`).
- The diffrax field-line backend supports the field-line stopping-criteria contract: geometric criteria, `ToroidalTransitStoppingCriterion`, and `IterStoppingCriterion`; toroidal-flux criteria remain Boozer/flux-coordinate criteria, matching the native Cartesian no-op behavior.
- The hand-rolled integrator machinery is **not** deleted in Phases 0-3. Deleting `dopri5_step`, `_dopri5_adaptive_step`, the PI-controller constants, and the `lax.scan` adaptive drivers belongs only to optional Phase 4 after soak; `bracket_root_jax`/`_scan_angle_plane_events` may remain as shared localizers.
- One JAX-tracing dependency story: `diffrax` is required for JAX tracing environments once it is the default. It is currently pinned in the `JAX`/`JAX_GPU` extras, not in the base project dependency list.

## Non-Goals

- **No** change to the C++ tracer (`simsopt.field.tracing` → simsoptpp). It stays the general-domain diagnostic reference and the parity oracle.
- **No** change to the **φ-param DESC tracer** (`tracing_poincare_phi.py`) — it already uses diffrax and is field-line-only; it is orthogonal to this arc-length-integrator migration.
- **No** new physics. Same RHS equations, same accuracy contract — only the stepping engine changes.
- **No** native deletion in the default-flip work. Phase 4 is explicitly optional and post-soak.
- **No** removal of `bracket_root_jax`'s *capability* — sub-step crossing localization is still needed (diffrax events stop at step granularity); it may be retained as a shared helper even after the native driver goes.

## Current Context

Grounded in the repo at HEAD `131392405`:

- **Native engine** `src/simsopt_jax/core/tracing.py` owns four RHS families + drivers:
  - field-line: `fieldline_rhs` (`:756`), `trace_fieldline` (`:1219`), `trace_fieldlines_batched` (`:1657`)
  - Cartesian GC vacuum: `guiding_center_vacuum_rhs` (`:1828`), `trace_guiding_center` (`:1896`), `trace_guiding_centers_batched` (`:2317`)
  - Boozer GC ×3 modes: `guiding_center_vacuum_boozer_rhs` (`:2829`), `guiding_center_no_k_boozer_rhs` (`:2903`), `guiding_center_boozer_rhs` (`:2989`), `trace_guiding_center_boozer` (`:3075`), `trace_guiding_centers_boozer_batched` (`:3601`)
  - full-orbit vacuum: `fullorbit_vacuum_rhs` (`:3802`), `trace_fullorbit` (`:3851`), `trace_fullorbits_batched` (`:4275`)
  - shared machinery: `dopri5_step` (`:785`), `_dopri5_adaptive_step` (`:1037`), PI-controller constants (`:164`), `bracket_root_jax` Illinois localizer (`:906`), `_scan_angle_plane_events` (`:1097`), the fixed-length `lax.scan` drivers.
- **Specs**: `FieldlineTracingSpec` has `integrator: Literal["dopri5_native","dopri5_diffrax"] = "dopri5_native"` (`:421`, in the registered meta_fields `:427`). `GuidingCenterTracingSpec` (`:1753`) and `FullorbitTracingSpec` (`:3723`) have **no integrator field**.
- **diffrax backend** `src/simsopt_jax/core/tracing_diffrax.py` is **field-line only** and supports **only geometric stopping criteria** (MinR/MaxR/MinZ/MaxZ/Levelset). There is no particle diffrax entry point yet; the current `NotImplementedError` path is for unsupported stopping criteria (Iteration/ToroidalTransit/flux) (`:130-139`). Stop-exit terminal state differs from native by ≤1 step (documented divergence).
- **Dependency state**: `diffrax==0.7.2` is pinned in `pyproject.toml` `JAX` and `JAX_GPU` extras (`:81`, `:98`) and is verified in the shared JAX env, but it is not a base `project.dependencies` requirement. Tests currently pin optional-import behavior (`tests/field/test_tracing_diffrax_fieldline.py:323-336`, `tests/field/test_tracing_poincare_phi.py:321-330`).
- **Consumers** (caller inventory): `src/simsopt_jax_adapters/field/tracing.py` — `compute_fieldlines(integrator=…)`, `trace_particles`, `trace_particles_boozer`; plus benchmarks/examples that call these. (Confirm full list in Phase 0.)
- **Hard-won constraint**: the existing field-line diffrax backend documents that diffrax `Event` root-finders broke under vmap in this route — `optimistix.Bisection` raised when lanes disagreed on firing, `VeryChord` diverged — so it uses `root_finder=None` (step-granularity) and re-localizes φ-crossings with the native `dopri5_step` sub-step. Treat particle loss-boundary localization as the same risk class until a particle prototype proves otherwise.

## Rationale

**Why do it.** One vetted, maintained integrator instead of growing the hand-rolled one; diffrax's richer solver suite and — the practical differentiability prize — its adjoint APIs such as **`RecursiveCheckpointAdjoint`** for differentiating through solves. The particle/α-loss memory win is a hypothesis to validate against native `lax.scan` autodiff, not a completed benchmark.

**Why it is expensive (design-it-twice — full removal vs deprecate-and-keep).** The native engine is **C++-oracle-validated** and currently the only path that (a) traces particles, (b) supports iteration/transit/flux criteria, and (c) localizes multi-lane events under vmap. Two end-states:
- **(A) Full removal** — delete native once diffrax reaches parity. Lowest long-term maintenance; **highest risk** (no validated fallback; one bug in a diffrax particle RHS has no in-repo reference).
- **(B) diffrax default, native retained as a deprecated fallback/parity-reference** — flip defaults to diffrax, keep native behind `integrator="dopri5_native"` and the C++-parity tests as the oracle, delete only after a long soak (or never). Slightly more code; **far safer**.

**Recommendation:** execute toward (B) first — diffrax default with native kept as the validated fallback — and treat full deletion (A) as a *separate, later* decision after diffrax has soaked at parity across all families. The phases below reach a flag-flip-reversible default before any deletion.

## Assumptions

- diffrax can express every RHS family in **time** (it can — general ODE solver); the particle EOMs are the same ones already coded in the native RHS functions, reusable verbatim as `ODETerm` vector fields.
- Transit and Boozer flux criteria may be realizable by augmenting the integrated state (unwrapped angle/flux-coordinate state) or by a post-hoc pass over the diffrax step grid. `IterStoppingCriterion` is different: a continuous ODE state is not a real accepted/rejected step counter, so it needs an explicit `max_steps`/status mapping or solver-stat based design. Prototype these criteria separately before claiming full parity.
- `bracket_root_jax` (vmap-safe Illinois localizer) is reusable as a sub-step localizer behind diffrax for precise event/crossing times, so diffrax's `root_finder=None` step-granularity is not a parity blocker.
- C++ parity fixtures exist (or can be built) for each particle family to validate against — the same oracles the native ports were validated with.
- Making `diffrax` required for JAX tracing is acceptable only after dependency acknowledgement. Do not silently promote it into base `project.dependencies`; decide whether the requirement remains scoped to `JAX`/`JAX_GPU` extras or becomes a base install dependency.

## Implementation Plan

### Phase 0 — Decision + design (Tier-3 gate; do before any code)
- [ ] **Confirm end-state (A) full-removal vs (B) diffrax-default-native-fallback** with the user. Default to (B) until diffrax soaks. *Open Question 1.*
- [ ] Full **caller inventory**: `rg` every call to the four `trace_*_batched` drivers, the single-lane `trace_*` drivers, and `integrator=` across `src/`, `benchmarks/`, `examples/`, `tests/`. Record each in the migration matrix.
- [ ] Write the **interface comment** for the unified `integrator` field on the particle specs (≤5 lines). Decide whether the integrator selector is per-spec (one field each) or a single shared policy.
- [ ] Decide the **integrator-state-criteria mechanism**. Prototype transit and flux via augmented-state vs post-hoc-grid designs; prototype iteration separately as `max_steps`/status or solver-stat handling. Pick only mechanisms that preserve the public result contracts. *Open Question 2.*

### Phase 1 — Field-line diffrax to FULL parity
- [ ] Add `ToroidalTransitStoppingCriterion` and `IterStoppingCriterion` support to `tracing_diffrax.py` via the Phase-0 mechanisms (remove the relevant `NotImplementedError` at `:133`). Keep toroidal-flux criteria inactive on Cartesian field-lines, matching `_stopping_criterion_should_stop`.
- [ ] Add sub-step stop-event localization (reuse `bracket_root_jax`) if Phase 0 selects exact stop-state parity. Do not expose a public localizer knob unless the API-evolution gate proves caller ownership; prefer one internal default before the diffrax default flip.
- [ ] Parity test: diffrax field-line == native field-line across the field-line criteria contract (geometric, transit, iteration, and Cartesian-inactive flux no-ops), batched/vmap, to ~1e-8 (and the stop-exit terminal state to tolerance once the sub-step localizer is on).

### Phase 2 — Port the particle RHS families to diffrax (one at a time, parity-gated)
For each of {Cartesian-GC-vacuum, full-orbit-vacuum, Boozer-GC `vacuum`/`no_k`/`full`}:
- [ ] New diffrax driver mirroring `tracing_diffrax.py`: wrap the existing native RHS (`guiding_center_vacuum_rhs`, `fullorbit_vacuum_rhs`, the three Boozer RHS) as an `ODETerm`; reuse the criteria + localizer machinery from Phase 1; return the existing `GuidingCenterTracingResult` or `FullorbitTracingResult` layouts.
- [ ] Add `integrator` field to `GuidingCenterTracingSpec` (`:1753`) and `FullorbitTracingSpec` (`:3723`) + register in their meta_fields; lazy-dispatch in `trace_guiding_center` / `trace_fullorbit` / `trace_guiding_center_boozer` mirroring `trace_fieldline:1263`.
- [ ] **C++-oracle parity test** for that family (gc/fullorbit/boozer), batched + single, before moving to the next family.

### Phase 3 — Flip defaults to diffrax (reversible)
- [ ] Change every spec default `integrator` from `"dopri5_native"` → `"dopri5_diffrax"`.
- [ ] Decide dependency scope before editing packaging. If JAX tracing remains extra-gated, keep `diffrax` in `JAX`/`JAX_GPU` extras and make those extras the required install path for the default JAX tracer. Move `diffrax` to base `project.dependencies` only with explicit dependency acknowledgement. Update import policy only after that decision; until then preserve lazy imports for base/native imports.
- [ ] Run the full tracing suite + benchmarks/examples with diffrax default; fix fallout. Native still selectable via `integrator="dopri5_native"` (end-state B).

### Phase 4 — Retire native (only end-state A, only after soak)
- [ ] Remove the `dopri5_native` branch + the `integrator` Literal option; delete `dopri5_step`, `_dopri5_adaptive_step`, PI-controller constants, the `lax.scan` adaptive drivers, and the native single/batched trace bodies — **keeping** `bracket_root_jax`/`_scan_angle_plane_events` if still used as shared localizers.
- [ ] Delete native-only tests; keep the C++-parity tests (now asserting diffrax == C++).
- [ ] Update docs (`tracing.py` module docstring, plan docs) to the diffrax-only reality.

## Validation Plan (API-evolution gate, Tier 3+)

- [ ] **C++-oracle parity per family** — field-line + GC-Cartesian + full-orbit + Boozer GC (`vacuum`/`no_k`/`full`) each match `simsoptpp` to documented tolerance, batched and single. This is the gate to flip each default.
- [ ] **Observable-behavior delta documented** — stop-exit terminal state, last φ/ζ hit, and `t_final` differences between native and diffrax (the ≤1-step divergence) listed before defaults flip; the chosen stop-localization path asserted against native.
- [ ] **vmap multi-lane events** — stopping criteria fire per-lane correctly under vmap/`shard_map` for every family (the diffrax root-finder-under-vmap failure must not regress; `root_finder=None` + `bracket_root_jax` localizer).
- [ ] **Stopping-criteria parity** — geometric, transit, and iteration criteria match native where meaningful; toroidal-flux criteria match native on Boozer/flux-coordinate traces and stay inactive on Cartesian field-line/particle routes where native JAX keeps them inactive.
- [ ] **Compatibility tests** — `compute_fieldlines`, `trace_particles`, `trace_particles_boozer` produce equal-quality output before/after the default flip; existing `tests/field/test_tracing_*` stay green.
- [ ] **Differentiation** — `jax.grad`/adjoint through each diffrax particle driver is finite + FD-matching (the reason for the migration); compare adjoint memory vs native `lax.scan` autodiff.
- [ ] **Toolchain** — `uvx ruff@0.15.15 check`; full suite under the diffrax env `/Users/suhjungdae/code/columbia/simsopt-jax-shared-jax/.conda/jax/bin/python` (verified py3.11.15 / jax0.10.0 / diffrax0.7.2), `JAX_PLATFORMS=cpu`, with the `repo_bootstrap` shadow-guard.
- [ ] **Rollback** — until Phase 4, every flip is reversible by setting `integrator="dopri5_native"`; record the revert recipe.

## Risks and Mitigations

- **Risk:** deleting the C++-validated native engine leaves no in-repo reference for diffrax bugs.
  **Mitigation:** end-state (B) — keep native as the deprecated fallback + parity oracle; defer deletion (A) to a separate post-soak decision.
- **Risk:** integrator-state criteria are not naturally expressible in diffrax's stateless event, and iteration count is not a continuous ODE state.
  **Mitigation:** Phase-0 criteria-specific prototypes; this is the single biggest technical unknown — resolve before committing to the default flip.
- **Risk:** diffrax event root-finders break under vmap for particle loss boundaries as they did on the field-line backend.
  **Mitigation:** assume `root_finder=None` + `bracket_root_jax`/grid post-processing until a particle prototype proves a safer exact root-finder path; cover with vmap multi-lane validation.
- **Risk:** particle-family parity drift (subtle RHS/units/normalization differences re-surface under a new driver).
  **Mitigation:** port one family at a time, each gated on its C++-oracle parity test before the next.
- **Risk:** `diffrax` becomes a hard dependency for base imports instead of only JAX tracing.
  **Mitigation:** keep the dependency scoped to `JAX`/`JAX_GPU` extras unless Phase 3 explicitly decides base install should include it; keep optional-import tests until that decision changes.

## Completion Criteria

- [ ] diffrax drivers exist + pass C++-oracle parity for all four RHS families, batched/single, under vmap.
- [ ] Field-line diffrax supports the native field-line stopping-criteria contract: geometric, transit, iteration, and Cartesian-inactive flux no-ops.
- [ ] All JAX tracing spec defaults are `dopri5_diffrax`; `diffrax` dependency scope is explicitly decided and tested; suite + benchmarks/examples green.
- [ ] (End-state A only) native machinery deleted; docs updated; no `dopri5_native` references remain.
- [ ] API-evolution artifacts produced (behavior delta, caller inventory, migration path, compatibility tests, rollback).

## Open Questions

1. **Full removal (A) or diffrax-default-native-fallback (B)?** Recommendation: ship (B); make (A) a later, separate call after soak. *User decision required.*
2. **Integrator-state criteria mechanism** — transit/flux augmented state vs post-hoc step-grid pass, plus separate iteration-count semantics. Prototype in Phase 0.
3. **Is the differentiable-particle payoff actually needed?** If JAX particle tracing has no differentiable-objective consumer yet, the migration's main benefit (adjoints) is speculative — in which case "keep native, add diffrax only where a consumer exists" may beat a full migration. Confirm the driving use case before Phase 2.
4. **Speed** — native vs diffrax has not been head-to-head benchmarked (same Dopri5 tableau ⇒ likely comparable). Measure before claiming a perf motive.
