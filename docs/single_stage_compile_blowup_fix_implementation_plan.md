# Single-Stage XLA Compile Blowup — Fix Implementation Plan

## Purpose

Define and execute the *correct* fix for the single-stage `ondevice` lane's
host-memory compile blowup (422 GiB MaxRSS, `LLVM ERROR: Unable to allocate
section memory!` from XLA's `contiguous_section_memory_manager`, job
`54341531`, SHA `521fa05f1`; see
`docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:930-955`).

This plan records the validated diagnosis, makes the host-driven `scipy-jax`
lane the sanctioned production path, and closes the three gaps between our run
path and the JAX/TORAX compile-discipline practices that were verified against
official docs and the local TORAX clone.

## Goals

- Retire the `ondevice` monolith as a production lane and make host-driven
  `scipy-jax` the default/only production single-stage path.
- Apply `--xla_cpu_opt_preset=FAST_COMPILE` in our config-before-init path, the
  way TORAX applies it at package import (`torax/__init__.py:47`).
- Add a TORAX-style compile-once invariant (`get_number_of_compiles(...) == 1`
  after warmup) onto the `scipy-jax` evaluation bundle, not just in benchmark
  probes.
- Diagnose and close the residual `scipy-jax` GPU compile-*time* cost
  (~73 min RunPod vs ~11 min Perlmutter): once-slow vs recompile-per-eval.

## Non-Goals

- Re-architecting the inner Boozer solver. Its loops are already bounded
  `lax.while_loop`/`fori_loop` (single `WhileOp`s) — not the cause.
- Putting the outer L-BFGS-B loop on-device. Host SciPy is the correct driver;
  on-device outer iteration offers no benefit for bound-constrained L-BFGS-B.
- Building new compile-shape audit tooling. The repo already has StableHLO/jaxpr
  shape counters and a cache-miss recorder (see Current Context).
- Touching upstream `src/simsopt/` (non-JAX; compile pressure is port-owned).

## Current Context

Confirmed facts (file:line evidence):

- **Crash is host-side code emission, not data.** `contiguous_section_memory_manager`
  is XLA:CPU's LLVM MCJIT section allocator; the emitted machine-code object for
  the fused graph is what exceeds host RAM, before the optimizer takes a step.
  `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:930-944`.
- **Root cause = monolithic `jit(run)` breadth** (fused outer + inner-solve +
  adjoint + polish in one compilation unit). The per-step-cost hypotheses
  (1500-DOF scaling, dense-Newton, form-K m200) were refuted; see memory
  `project_ondevice_compile_blowup_root_cause`.
- **Inner loops are already bounded `lax.while_loop`, not Python-unrolled:**
  `src/simsopt_jax/geo/optimizers/private/_bfgs.py:1,238`,
  `src/simsopt_jax/geo/optimizers/optimizer.py:3457,3496`,
  `src/simsopt_jax/geo/optimizers/private/_line_search.py:336,598`.
  Inner Boozer gradient is a separate **adjoint** stage (implicit diff), not
  backprop-through-loop. ⇒ "use `lax.scan` to stop unrolling" does **not** apply.
- **Host-driven lanes already exist and fit:** `scipy-jax` /
  `scipy-jax-fullgraph` keep the outer loop on the host and reuse a jitted
  value/grad bundle, dropping host RSS from 422 GiB → ~6 GB.
  `src/simsopt_jax/geo/optimizers/single_stage_routing.py:22,70-91`;
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4824-4846`;
  `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:953-954`.
- **The example already defaults to `scipy-jax` and gates `ondevice` behind
  explicit selection.** `--optimizer-backend` help states it "Defaults to
  'scipy' on the CPU/reference backend and 'scipy-jax' on the JAX backend" and
  that "the legacy 'ondevice' monolith must be selected explicitly"
  (`single_stage_banana_example.py:4841-4844`). A `--boozer-optimizer-backend`
  flag exists (`:4858`) and a memory hint already recommends scipy-jax
  (`:5222`). So the *example-layer* default work is effectively done; the open
  work is at the production launcher/contract layer.
- **`--xla_cpu_opt_preset=FAST_COMPILE` is NOT set in our run path** (grep of
  `src/` is empty). TORAX sets it at import (`torax/__init__.py:47`,
  `torax/run_simulation_main.py:42`). Our central config-before-init apply point
  is `apply_jax_runtime_config()` at `src/simsopt_jax/backend/runtime.py:2325`,
  with an existing XLA-flag merge helper (lines 533–576) and `jax.config.update`
  block (2336–2343).
- **Compile-count instrumentation lives only in `benchmarks/`**, never as a guard
  on the lane: `benchmarks/single_stage_outer_loop_probe.py:318`,
  `benchmarks/grouped_adjoint_memory_probe.py:399,421-431`. TORAX ships
  `get_number_of_compiles` (`torax/_src/jax_utils.py:147`) and asserts
  `== 1` across ~20 test files.
- **Closure bakes geometry as constants:** the jitted Boozer value/grad closure
  captures `coil_arrays`/`coil_set_spec`/surface runtime args then `jax.jit`s,
  `src/simsopt_jax_adapters/geo/boozer_surface.py:4761-4770`. Relevant to the
  static-vs-dynamic discipline (TORAX `torax/_src/config/runtime_params.py:17`).
- **Existing audit tooling to REUSE (do not rebuild):**
  `benchmarks/traceable_compile_shape.py`,
  `benchmarks/traceable_target_lane_compile_shape.py`,
  `benchmarks/surface_rz_geometry_hlo_probe.py:124`,
  and the example's cache-miss recorder `maybe_record_jax_compile_diagnostics`
  (`single_stage_banana_example.py:9536`), driven by the
  `--record-jax-compile-diagnostics` flag (defined `:4967`, consumed `:15060`).

## Rationale

The official JAX guidance and the TORAX architecture agree on one principle:
**compile a bounded, fixed-shape unit and drive the loop from outside it.** TORAX
never `jit`s a multi-thousand-step run; it `jit`s the step, drives the loop with
`while_loop_bounded` (scan + static `max_steps`, `torax/_src/jax_utils.py:237`),
and asserts compile-once. Our outer optimizer is L-BFGS-B, which SciPy already
runs well on the host, so the TORAX-equivalent of "drive the loop outside the
jit" *is* `scipy-jax`. The ondevice monolith is therefore the wrong compilation
boundary — not a tuning problem — and host-driving is the correct, doc-backed
fix, already implemented. Remaining work is to make it the sanctioned lane and to
add the two TORAX hygiene practices we are missing (FAST_COMPILE preset;
compile-once guard), then resolve the GPU compile-time residual.

## Assumptions

- The ~73 min `scipy-jax` GPU compile is dominated by either (a) one slow cold
  compile of the per-eval bundle, or (b) recompilation per outer evaluation from
  a cache-busting non-static arg / shifting shape. **Assumption to verify in
  Phase 3**, not yet measured.
- Setting `--xla_cpu_opt_preset=FAST_COMPILE` reduces CPU compile time/memory
  without changing optimization results (TORAX relies on it in production).
  Assumption: it does not perturb numerical parity on our lanes — gated by the
  parity validation in the Validation Plan.
- `FAST_COMPILE` is a CPU-backend preset; GPU host-compile relief comes primarily
  from the host-driven boundary, not this flag. Treat the flag as CPU-lane scoped
  unless measurement shows otherwise.

## Implementation Plan

1. **Sanction host-driven `scipy-jax`; keep `ondevice` opt-in.**
   *Current state (verified): the example already defaults `--optimizer-backend`
   to `scipy-jax` on the JAX backend and requires `ondevice` to be selected
   explicitly (`single_stage_banana_example.py:4841-4844`), exposes
   `--boozer-optimizer-backend` (`:4858`), and emits a scipy-jax memory hint
   (`:5222`). Example-layer work is largely done; remaining work is at the
   production launcher/contract layer.*
   - [ ] Confirm the production launchers/contract inherit the example's
     `scipy-jax` default (not `ondevice`) for single-stage; document the default
     in `docs/scipy_jax_11_51_matrix_implementation_plan.md`.
   - [ ] Verify an explicit `ondevice` selection surfaces a host-RAM warning —
     inspect the context of the `:5222` hint; add a warning at backend selection
     if it currently only fires from an OOM/fallback path. Cross-ref the 422 GiB
     evidence.
   - [ ] (Verify-only) The `--optimizer-backend` help already documents the
     host-driven-vs-monolith boundary (`single_stage_banana_example.py:4836-4844`);
     add an equivalent note to the production contract doc only if it is missing
     there.

2. **Apply `--xla_cpu_opt_preset=FAST_COMPILE` in config-before-init.**
   - [ ] In `apply_jax_runtime_config()` (`src/simsopt_jax/backend/runtime.py:2325`),
     compose `--xla_cpu_opt_preset=FAST_COMPILE` into `XLA_FLAGS` using the
     existing flag-merge pattern (lines 533–576) — **merge, do not overwrite**
     any user-provided `XLA_FLAGS`; idempotent (skip if already present).
   - [ ] Scope to CPU backend (or unconditionally if measurement shows it is
     inert/safe on GPU). Follow the existing determinism-flag precedent.
   - [ ] Ensure it is applied before JAX is imported/initialized (this module is
     the forced-lazy config-before-init boundary).

3. **Diagnose and close the `scipy-jax` GPU compile-time residual.**
   - [ ] Run the existing compile-count probe on the surviving host-driven
     `scipy-jax` GPU lane with `--record-jax-compile-diagnostics`
     (flag `single_stage_banana_example.py:4967`; recorder `:9536`) at maxiter
     3 vs 6; compare `cache_miss_count` / recompile sites in
     `jax_compile_diagnostics.json` via
     `benchmarks/single_stage_outer_loop_probe.py`.
   - [ ] Classify: compile-once-slow (image/box issue) vs recompile-per-eval
     (cache-token bug). If recompile-per-eval, find the non-static arg / shifting
     shape and pin it (static-vs-dynamic discipline, cf. TORAX
     `runtime_params.py:17`; closure at `boozer_surface.py:4761-4770`).
   - [ ] Write the missing decision-tree doc
     `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md` referenced by
     `docs/scipy_jax_11_51_matrix_implementation_plan.md:16,195`.

4. **Add a TORAX-style compile-once guard on the `scipy-jax` eval bundle.**
   - [ ] After warmup, assert the per-evaluation jitted bundle compiles exactly
     once across N outer steps (mirror `get_number_of_compiles(...) == 1`,
     `torax/_src/jax_utils.py:147`). Place as a lane-level invariant/test, not
     only a benchmark probe.
   - [ ] Add a regression test asserting no recompile when only `x` changes
     (stable shape), modeled on TORAX's `jit_updates_value_without_recompile`
     subtests.

5. **(Optional, after Phase 3) Inner-cap bisect for ondevice feasibility record.**
   - [ ] Run ondevice with `maxiter=1500` and the inner Boozer BFGS cap unset
     (`PROD_BOOZER_BFGS_MAXITER=` empty) on a full CPU node to confirm whether the
     `boozer_bfgs_maxiter=1500` cap (added 2026-06-12) is the dominant breadth
     multiplier vs the May runs that completed.
     `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:948-952`.
   - [ ] Record the result as a feasibility-boundary note; do not re-enable
     ondevice as production regardless of outcome.

## Validation Plan

- [ ] **FAST_COMPILE parity:** run the CPU single-stage parity lane with and
  without the preset; final objective / field-error / iota / volume relative
  diffs unchanged at the established tolerances (cf. init parity rows in
  `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:894-899`).
- [ ] **FAST_COMPILE effect:** record CPU compile wall + peak host RSS before/after
  on an mpol2 `scipy-jax` run; expect lower or equal, never higher.
- [ ] **Compile-once guard:** new test passes — bundle `get_number_of_compiles`
  (or `_cache_size()`) `== 1` after warmup across ≥3 outer evaluations.
- [ ] **GPU residual classified:** `jax_compile_diagnostics.json` shows either a
  single cold compile or a named recompile cause; decision-tree doc committed.
- [ ] **Default-lane check:** production single-stage entrypoint selects
  `scipy-jax` without an explicit flag; `ondevice` requires opt-in.
- [ ] **No-regression:** existing `tests/integration/test_single_stage_init_parity_compile_diagnostics.py`
  and `tests/jax/core/test_bounded_scan.py` still pass.

## Risks and Mitigations

- Risk: `--xla_cpu_opt_preset=FAST_COMPILE` perturbs numerics or is silently
  ignored on the installed XLA version.
  Mitigation: gate behind the FAST_COMPILE parity check; assert the flag is
  accepted (no XLA warning) on the target jaxlib pin before landing.
- Risk: overwriting a user/launcher-provided `XLA_FLAGS` (e.g. GPU determinism
  flags) when injecting the preset.
  Mitigation: reuse the existing merge/idempotent pattern at
  `runtime.py:533-576`; add a unit test for the merge.
- Risk: the GPU residual is recompile-per-eval rooted in a baked dynamic arg that
  is expensive to make static (geometry closure).
  Mitigation: Phase 3 isolates the exact arg via cache-miss sites before any
  refactor; static-vs-dynamic split scoped to that arg only.
- Risk: someone re-promotes `ondevice` for production after the cap bisect.
  Mitigation: Phase 1 makes it opt-in with an explicit host-RAM warning and a
  cross-ref to the 422 GiB feasibility-boundary evidence.

## Completion Criteria

- [ ] `scipy-jax` is the default production single-stage lane; `ondevice` is
  opt-in with a documented host-RAM warning.
- [ ] `--xla_cpu_opt_preset=FAST_COMPILE` applied via `apply_jax_runtime_config()`
  with merge-not-overwrite semantics and a unit test.
- [ ] Lane-level compile-once guard + no-recompile-on-`x`-change regression test
  passing.
- [ ] GPU compile-time residual classified (once-slow vs recompile) with
  `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md` committed.
- [ ] Parity validation green; no regression in existing compile-diagnostic tests.

## Open Questions

- Is the ~73 min `scipy-jax` GPU compile one cold compile or recompile-per-eval?
  (Owner: Phase 3 probe — decides whether any code change is needed at all.)
- Should `FAST_COMPILE` be CPU-only or also applied on GPU host-compile? (Decide
  from Phase 2 measurement; default CPU-only.)
- Does the `boozer_bfgs_maxiter=1500` inner cap dominate ondevice breadth? (Phase
  5 bisect; record-only, does not change the production decision.)
