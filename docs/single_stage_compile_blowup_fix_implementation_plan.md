# Single-Stage XLA Compile Blowup — Fix Implementation Plan

## Purpose

Define and execute the *correct* fix for the single-stage `ondevice` lane's
host-memory compile blowup (422 GiB MaxRSS, `LLVM ERROR: Unable to allocate
section memory!` from XLA's `contiguous_section_memory_manager`, job
`54341531`, SHA `521fa05f1`; see
`docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:930-955`).

This plan records the validated diagnosis, makes the host-driven `scipy-jax`
family (`scipy-jax` reduced and `scipy-jax-fullgraph` fullgraph) the sanctioned
production path, and closes the remaining gaps between our run path and the
JAX/TORAX compile-discipline practices verified against official docs and the
local TORAX clone.

## Goals

- Retire the `ondevice` monolith as a production lane and make the host-driven
  `scipy-jax` family the production single-stage path.
- Apply `--xla_cpu_opt_preset=FAST_COMPILE` in our config-before-init path, the
  way TORAX applies it at package import (`torax/__init__.py:47`).
- Add a TORAX-style compile-once invariant (`get_number_of_compiles(...) == 1`
  after warmup) onto the host-driven evaluation bundle, not just benchmark
  probes or tiny optimizer smoke tests.
- Diagnose and close the residual `scipy-jax` GPU compile-*time* cost:
  once-slow vs recompile-per-eval. The RunPod A100 production row records wall
  dominated by the quota-throttled host reference and XLA:GPU compile, but does
  not yet isolate compile minutes.

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

- **Crash is host-side XLA/LLVM code emission, not a device data allocation.**
  The target child failed inside XLA:CPU's LLVM section allocator
  (`contiguous_section_memory_manager`) while compiling the production
  `ondevice` outer-optimizer graph. The evidence proves a section-memory
  allocation failure during code emission at ~422 GiB MaxRSS; it does not
  directly measure emitted object-code bytes in isolation.
  `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:930-944`.
- **Root cause = monolithic `jit(run)` breadth** (fused outer + inner-solve +
  adjoint + polish in one compilation unit). The production evidence records the
  failing budget stack (`maxiter=1500`, `boozer_bfgs_maxiter=1500`,
  `boozer_newton_maxiter=50`, polish `run`) and the current matrix generator
  excludes the monolithic lane on that basis.
  `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:938-944`;
  `benchmarks/perlmutter/build_single_stage_matrix.py:4-16`.
- **Inner loops are already bounded `lax.while_loop`, not Python-unrolled:**
  `src/simsopt_jax/geo/optimizers/private/_bfgs.py:1,238`,
  `src/simsopt_jax/geo/optimizers/optimizer.py:3457,3496`,
  `src/simsopt_jax/geo/optimizers/private/_line_search.py:336,598`.
  Inner Boozer gradient is a separate **adjoint** stage (implicit diff), not
  backprop-through-loop. ⇒ "use `lax.scan` to stop unrolling" does **not** apply.
- **Host-driven lanes already exist and fit:** `scipy-jax` /
  `scipy-jax-fullgraph` keep the outer loop on the host and reuse a jitted
  value/grad bundle, avoiding the 422 GiB monolithic compile boundary. The
  production `scipy-jax` CPU/GPU rows ran with ~5.1 GiB and ~6.4 GiB host
  MaxRSS, respectively.
  `src/simsopt_jax/geo/optimizers/single_stage_routing.py:22,70-91`;
  `src/simsopt_jax/geo/optimizers/optimizer.py:74-78,5121-5151`;
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4824-4846`;
  `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:968-970,1016-1018`.
- **The example and production launchers already default to `scipy-jax` and gate
  `ondevice` behind explicit selection.** `--optimizer-backend` help states it
  "Defaults to 'scipy' on the CPU/reference backend and 'scipy-jax' on the JAX
  backend" and that "the legacy 'ondevice' monolith must be selected explicitly"
  (`single_stage_banana_example.py:4841-4844`). The CPU/GPU production launchers
  now default `PROD_OPTIMIZER_BACKEND` to `scipy-jax`
  (`benchmarks/perlmutter/single_stage_production_cpu.slurm:26`,
  `benchmarks/perlmutter/single_stage_production_gpu.slurm:28`). A
  `--boozer-optimizer-backend` flag exists (`:4858`) and a CPU-ondevice memory
  hint already recommends `scipy-jax` (`:5213-5222`).
- **`--xla_cpu_opt_preset=FAST_COMPILE` is NOT set in our run path** (grep of
  runtime/launchers/source is empty). TORAX sets it at import
  (`torax/__init__.py:47`). Our central config-before-JAX-import apply point is
  `apply_jax_runtime_config()` at `src/simsopt_jax/backend/runtime.py:2325`,
  with an existing XLA-flag parse/merge helper (lines 532–576) and
  `jax.config.update` block (2334–2349).
- **Compile diagnostics exist, but no production `scipy-jax` compile-once guard
  exists.** Benchmark/example diagnostics record compile/cache-miss events:
  `benchmarks/single_stage_outer_loop_probe.py:307-320`,
  `benchmarks/grouped_adjoint_memory_probe.py:396-431`, and
  `single_stage_banana_example.py:9535-9557,17045-17093`. Tiny optimizer
  compile-count smoke tests also exist
  (`tests/subprocess/jax_runtime_cases.py:186-240,302-340`;
  `tests/test_jax_import_smoke.py:809-854`), but none asserts compile-once for
  the production single-stage `scipy-jax` eval bundle. TORAX ships
  `get_number_of_compiles` (`torax/_src/jax_utils.py:147`) and uses it in tests.
- **Closure captures geometry/runtime args before `jax.jit`:** the jitted Boozer
  value/grad closure closes over `coil_arrays`/`coil_set_spec`/surface runtime
  args, then `jax.jit`s,
  `src/simsopt_jax_adapters/geo/boozer_surface.py:4761-4770`. Relevant to the
  static-vs-dynamic discipline (TORAX `torax/_src/config/runtime_params.py:17`).
- **Official JAX docs support the cache/recompile premise, not a bigger-RAM
  workaround:** `jax.jit` caches compiled code for compatible calls; static
  argument changes trigger recompiles; the persistent compilation cache stores
  compiled programs for reuse after a successful compile. That supports the
  compile-once/stable-shape guard, but does not rescue a cold monolithic compile
  that cannot finish. Sources:
  `https://github.com/jax-ml/jax/blob/main/docs/jit-compilation.md`,
  `https://github.com/jax-ml/jax/blob/main/docs/persistent_compilation_cache.md`.
- **Existing audit tooling to REUSE (do not rebuild):**
  `benchmarks/traceable_compile_shape.py`,
  `benchmarks/traceable_target_lane_compile_shape.py`,
  `benchmarks/surface_rz_geometry_hlo_probe.py:124`,
  and the example's cache-miss recorder `maybe_record_jax_compile_diagnostics`
  (`single_stage_banana_example.py:9536`), driven by the
  `--record-jax-compile-diagnostics` flag (defined `:4967`, consumed `:15060`).

## Rationale

The official JAX docs and TORAX architecture support one operational rule for
this case: compile a bounded, stable-shape unit, then prove subsequent calls
reuse the compiled executable. TORAX implements that discipline by `jit`ting a
bounded step/loop helper (`while_loop_bounded`: scan + static `max_steps`,
`torax/_src/jax_utils.py:237`) and checking compile counts
(`get_number_of_compiles`, `:147`). Our outer optimizer is L-BFGS-B, which SciPy
already runs well on the host, so the corresponding single-stage boundary is the
host-driven `scipy-jax` family: SciPy owns the outer loop and JAX owns the
fixed-shape value/grad bundle. The `ondevice` monolith is therefore the wrong
production compilation boundary. Remaining work is to add the missing TORAX
hygiene practices (FAST_COMPILE preset where safe; production eval compile-once
guard), then resolve the GPU compile-time residual.

## Assumptions

- The `scipy-jax` GPU residual is dominated by either (a) one slow cold compile
  of the per-eval bundle, or (b) recompilation per outer evaluation from a
  cache-busting non-static arg / shifting shape. **Assumption to verify in
  Phase 3**, not yet measured.
- Setting `--xla_cpu_opt_preset=FAST_COMPILE` may reduce CPU compile time/memory
  without changing optimization results (TORAX relies on it in production).
  Assumption: it is accepted by our jaxlib pin. The bit-exact `*_parity` lanes
  are excluded outright (the preset reduces XLA passes and could shift CPU
  reduction order), so the residual risk is confined to production/non-parity
  CPU lanes whose acceptance tolerances are looser.
- `FAST_COMPILE` is a CPU-backend preset; GPU host-compile relief comes primarily
  from the host-driven boundary, not this flag. Implemented as non-parity-CPU
  scoped.

## Implementation Plan

1. **Sanction the host-driven `scipy-jax` family; keep `ondevice` opt-in.**
   *Current state (verified): the example already defaults `--optimizer-backend`
   to `scipy-jax` on the JAX backend and requires `ondevice` to be selected
   explicitly (`single_stage_banana_example.py:4841-4844`), exposes
   `--boozer-optimizer-backend` (`:4858`), emits a scipy-jax memory hint
   (`:5213-5222`), and the production CPU/GPU launchers default
   `PROD_OPTIMIZER_BACKEND` to `scipy-jax`
   (`benchmarks/perlmutter/single_stage_production_cpu.slurm:26`,
   `benchmarks/perlmutter/single_stage_production_gpu.slurm:28`).*
   - [x] Confirm the production launchers inherit the example's `scipy-jax`
     default (not `ondevice`) for single-stage. Update
     `docs/scipy_jax_11_51_matrix_implementation_plan.md` if it still states the
     old launcher default.
   - [ ] Verify an explicit `ondevice` selection surfaces a host-RAM warning —
     inspect the context of the `:5222` hint; add a warning at backend selection
     if it currently only fires from an OOM/fallback path. Cross-ref the 422 GiB
     evidence.
   - [ ] (Verify-only) The `--optimizer-backend` help already documents the
     host-driven-vs-monolith boundary (`single_stage_banana_example.py:4836-4844`);
     add an equivalent note to the production contract doc only if it is missing
     there.

2. **Apply `--xla_cpu_opt_preset=FAST_COMPILE` in config-before-init.**
   *Implemented (local logic validated; CPU parity/effect pending cluster).*
   - [x] Pure merge helper `_xla_flags_with_cpu_compile_preset(xla_flags)`
     (`src/simsopt_jax/backend/runtime.py`): **merge, do not overwrite** — keeps
     existing tokens verbatim, respects a caller-supplied `--xla_cpu_opt_preset`,
     idempotent. Unit-tested in `tests/test_backend_xla_compile_preset.py`.
   - [x] Applier `_apply_cpu_compile_preset_env(config, policy)` scoped to
     **non-parity CPU lanes** — no-op on `jax_platform == "cuda"` *and* on
     `policy.parity_mode` (the preset reduces XLA passes and can shift CPU
     reduction order, so it is withheld from the bit-exact `*_parity` lanes).
     Mirrors the `_apply_jax_gpu_memory_env` precedent.
   - [x] Wired into `apply_jax_runtime_config()` in the pre-`import jax` region
     (after `_apply_jax_gpu_memory_env`), so `XLA_FLAGS` is set before XLA inits.

3. **Diagnose and close the `scipy-jax` GPU compile-time residual.**
   - [ ] Run the existing compile-count probe on the surviving host-driven
     `scipy-jax` GPU lane with `--record-jax-compile-diagnostics`
     (flag `single_stage_banana_example.py:4967`; recorder `:9536`) at maxiter
     3 vs 6; compare `compile_event_count`, `cache_miss_count`, and recompile
     sites in `jax_compile_diagnostics.json` / `results.json` via
     `benchmarks/single_stage_outer_loop_probe.py`.
   - [ ] Classify: compile-once-slow (image/box issue) vs recompile-per-eval
     (cache-token bug). If recompile-per-eval, find the non-static arg / shifting
     shape and pin it (static-vs-dynamic discipline, cf. TORAX
     `runtime_params.py:17`; closure at `boozer_surface.py:4761-4770`).
   - [ ] Write the missing decision-tree doc
     `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md` referenced by
     `docs/scipy_jax_11_51_matrix_implementation_plan.md:16,195`.

4. **Add a TORAX-style compile-once guard on the host-driven eval bundle.**
   *Authored; runs on the cluster/CI. The local pytest harness cannot collect it
   in this checkout: `tests/conftest.py:17` → `bootstrap_local_simsopt` →
   `src/simsopt/_core/util.py:19` raises `ImportError: cannot import name 'Curve'
   from 'simsoptpp' (unknown location)` — `simsoptpp` resolves to a build lacking
   `Curve` here (an environment/build artifact, not a test defect). Verified
   reproducing with and without `PYTHONPATH=src`.*
   - [x] `test_penalty_value_and_grad_bundle_reuses_compiled_executable`
     (`tests/geo/test_boozersurface_jax.py`) builds the production `scipy-jax`
     value/grad bundle via `_make_penalty_value_and_grad_cpu_ordered_with`,
     calls it at `x0` then a same-shape `x1`, and asserts (a) `_cache_size()`
     does not grow on the `x`-change (no recompile-per-step) and (b)
     `_cache_size() == 1` after warmup (compiles exactly once). Mirrors
     `get_number_of_compiles(...) == 1` (`torax/_src/jax_utils.py:147`) and the
     existing `_cache_size()` pattern (`test_boozersurface_jax.py:8884`).
   - [ ] Run on the cluster/CI to confirm green; if `_cache_size()` exceeds 1 at
     warmup, capture the sub-graph variants before tightening the assertion.

5. **(Optional, after Phase 3) Inner-cap bisect for ondevice feasibility record.**
   - [ ] Run ondevice with `maxiter=1500` and the inner Boozer BFGS cap unset
     (`PROD_BOOZER_BFGS_MAXITER=` empty) on a full CPU node to confirm whether the
     `boozer_bfgs_maxiter=1500` cap (added 2026-06-12) is the dominant breadth
     multiplier vs the May runs that completed.
     `docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:948-952`.
   - [ ] Record the result as a feasibility-boundary note; do not re-enable
     ondevice as production regardless of outcome.

## Validation Plan

- [ ] **Parity lane unaffected:** confirm the `*_parity` lanes never receive the
  preset (code-gated on `policy.parity_mode`; unit-tested by
  `test_apply_is_noop_on_cpu_parity`). The bit-exact init parity rows
  (`docs/jax_clean_reconciliation_diagnostics_2026-06-11.md:894-899`) must stay
  `0.0`.
- [ ] **Non-parity CPU effect:** on a production/`jax_cpu_fast` `scipy-jax` run,
  confirm final results stay within acceptance tolerances and that compile wall
  + peak host RSS drop (or are unchanged) with the preset active.
- [ ] **FAST_COMPILE effect:** record CPU compile wall + peak host RSS before/after
  on an mpol2 `scipy-jax` run; accept only if lower/equal or if any tradeoff is
  explicitly justified by parity and wall/RSS measurements.
- [ ] **Compile-once guard:** new test passes — bundle `get_number_of_compiles`
  (or `_cache_size()`) `== 1` after warmup across ≥3 outer evaluations.
- [ ] **GPU residual classified:** `jax_compile_diagnostics.json` shows either a
  single cold compile or a named recompile cause; decision-tree doc committed.
- [x] **Default-lane check:** production single-stage launchers select
  `scipy-jax` without an explicit flag; `ondevice` requires opt-in.
- [ ] **No-regression:** existing `tests/integration/test_single_stage_init_parity_compile_diagnostics.py`
  and `tests/jax/core/test_bounded_scan.py` still pass.

## Risks and Mitigations

- Risk: `--xla_cpu_opt_preset=FAST_COMPILE` perturbs numerics or is silently
  ignored on the installed XLA version.
  Mitigation: the bit-exact `*_parity` lanes are excluded at the source
  (`policy.parity_mode` gate); for non-parity CPU lanes, confirm acceptance
  tolerances hold and the flag is accepted (no XLA warning) on the jaxlib pin.
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

- [x] The production single-stage launchers default to `scipy-jax`; `ondevice`
  remains opt-in. A host-RAM warning exists in the example parser for explicit
  CPU-ondevice selection; cross-link it to the 422 GiB evidence if it remains
  user-facing.
- [x] `--xla_cpu_opt_preset=FAST_COMPILE` applied via `apply_jax_runtime_config()`
  with merge-not-overwrite semantics and a unit test. *(Code + unit test landed,
  local logic validated; CPU parity/effect checks pending cluster.)*
- [x] Lane-level compile-once guard + no-recompile-on-`x`-change regression test
  *authored* (`test_penalty_value_and_grad_bundle_reuses_compiled_executable`);
  pending a cluster/CI run to confirm green.
- [ ] GPU compile-time residual classified (once-slow vs recompile) with
  `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md` committed.
- [ ] Parity validation green; no regression in existing compile-diagnostic tests.

## Open Questions

- Is the `scipy-jax` GPU compile-time residual one cold compile or
  recompile-per-eval?
  (Owner: Phase 3 probe — decides whether any code change is needed at all.)
- Should `FAST_COMPILE` be CPU-only or also applied on GPU host-compile? (Decide
  from Phase 2 measurement; default CPU-only.)
- Does the `boozer_bfgs_maxiter=1500` inner cap dominate ondevice breadth? (Phase
  5 bisect; record-only, does not change the production decision.)
