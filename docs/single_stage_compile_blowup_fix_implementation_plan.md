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
- **`--xla_cpu_opt_preset=FAST_COMPILE` is now set for non-parity CPU JAX
  lanes.** TORAX sets it at import (`torax/__init__.py:47`); this repo now
  applies it in the config-before-JAX-import path via
  `_apply_cpu_compile_preset_env()` / `_xla_flags_with_cpu_compile_preset()` in
  `src/simsopt_jax/backend/runtime.py`. CUDA lanes and parity lanes are
  intentionally excluded.
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
   *Resolved on CPU (2026-06-15): classification = **once-slow**;
   recompile-per-eval **refuted**.*
   - [x] Ran the compile-count probe via `benchmarks/single_stage_init_parity.py`
     (self-compiles the seed) on CPU at maxiter 3 vs 6 with
     `--record-jax-compile-diagnostics`. Both runs: `cache_miss_count=127`,
     `compile_event_count=178`, byte-identical `cache_miss_sites` — despite
     executing **different** outer-step counts (3 vs 5). Compile count does not
     grow with the iteration budget.
   - [x] Classified: **compile-once-slow**, not recompile-per-eval. No
     cache-token/shape bug; the static-vs-dynamic closure
     (`boozer_surface.py:4761-4770`) is not triggering recompiles. The
     ~73-min RunPod figure is the single cold GPU compile being slow on that
     image (XLA:GPU autotune / cubin / container toolkit), not a lane defect.
   - [x] Wrote the decision-tree doc
     `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md`.
   - [x] Wired the persistent compilation cache into the **Perlmutter** GPU
     launcher (`benchmarks/perlmutter/single_stage_production_gpu.slurm`): sets
     `JAX_COMPILATION_CACHE_DIR` to `${RUN_ROOT}/jax_compilation_cache` (outside
     `${JOB_ROOT}`, so it persists across jobs), relocating off the `$HOME`
     default; reuses runtime.py's SAFE narrow autotune mode (no broader,
     nvlink-triggering XLA cache modes). Lets repeat jobs reuse the cold compile.
     (The fair-compare launcher is deliberately left cache-cold — it *measures*
     compile time, which a warm cache would contaminate.)
   - [x] Fixed the **RunPod/CUDA** launcher `prepare_cuda_gpu_lowres_tests.py`
     `_cuda_env`: it set `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="all"` (the broad
     mode that forces nvlink through the container toolkit — the cu1290 block);
     aligned to the SAFE narrow `xla_gpu_per_fusion_autotune_cache_dir`
     (commit `0ef2f8f76`).
   - [x] Wrote the RunPod cold→warm validation runbook
     `docs/runpod_gpu_compile_cache_validation_protocol.md` (network volume on
     `/workspace`, tmux, cold→warm wall comparison, nvlink-clean check).
   - [x] Curated + verified a 6-seed parity/walltime/memory benchmark set from
     `autoresearch/runs` (2× mpol8 + 4× mpol10, hw-valid, complete trios) →
     `docs/runpod_parity_benchmark_seeds.md`. NB: no mpol2 seeds exist there
     (lowest is mpol8); the mpol2 *mechanism* check still uses the clean-repo
     fixture via `init-parity --mpol 2`.
   - [ ] **(GPU-only, pending execution)** Provision A100 80GB, stage the 6 seeds
     on a `/workspace` network volume, and sweep
     `init-parity --platform cuda --warm-start-run-dir <seed>
     --record-jax-compile-diagnostics` → per-seed CPU-vs-GPU **precision parity**
     + **walltime** + **MaxRSS/GPU-mem**, plus the cold→warm cache-hit and
     nvlink-clean confirmation. This box can drive RunPod (`runpodctl` + API key)
     but the run is billable and not yet executed.

4. **Add a TORAX-style compile-once guard on the host-driven eval bundle.**
   *Locally verified (CPU): `1 passed in 4.46s` via real pytest in the
   `simsopt-jax/.conda/jax-0.10.0` env with `PYTHONPATH=src` (that env has a
   working `simsoptpp.Curve`). The system miniforge interpreter cannot collect
   it — its `simsoptpp` resolves without `Curve` ("unknown location") through the
   `tests/conftest.py` bootstrap chain; that is an interpreter/build artifact,
   not a test defect.*
   - [x] `test_penalty_value_and_grad_bundle_reuses_compiled_executable`
     (`tests/geo/test_boozersurface_jax.py`) builds the production `scipy-jax`
     value/grad bundle via `_make_penalty_value_and_grad_cpu_ordered_with`,
     calls it at `x0` then a same-shape `x1`, and asserts (a) `_cache_size()`
     does not grow on the `x`-change (no recompile-per-step) and (b)
     `_cache_size() == 1` after warmup (compiles exactly once). **PASSES on CPU**,
     so the host-driven bundle is confirmed compile-once for the boozer-penalty
     case. Mirrors `get_number_of_compiles(...) == 1`
     (`torax/_src/jax_utils.py:147`) and the existing `_cache_size()` pattern
     (`test_boozersurface_jax.py:8884`).

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
- [~] **Non-parity CPU effect (measured at mpol2, 2026-06-15):** A/B probe
  (`XLA_FLAGS=--xla_cpu_opt_preset=DEFAULT` vs `=FAST_COMPILE`, n=2 each) shows
  **user CPU time −18% to −19%** (353→289 s) with byte-identical compile counts
  (178/127) — the preset cuts compile *work*. But **wall (+1%) and peak RSS show
  no win at mpol2** (toy compile is too small; RSS noisy). Production-scale wall/
  RSS benefit is expected but **NOT demonstrated locally** (needs high-res seed /
  bigger box). Honest status: a real-but-modest CPU-work reduction, headline
  wall/RSS unproven.
- [ ] **FAST_COMPILE production-scale effect:** the mpol2 measurement above shows
  no wall/RSS win at toy size; the production-scale wall/RSS effect is deferred to
  the RunPod/big-box sweep (mpol8/mpol10 seeds in
  `docs/runpod_parity_benchmark_seeds.md`). Accept only if lower/equal or the
  tradeoff is justified by the parity + wall/RSS measurements there.
- [x] **Compile-once guard:** new test passes on CPU — bundle `_cache_size() == 1`
  after warmup and no growth on an `x`-change (`1 passed in 4.46s`,
  `jax-0.10.0` env). Confirms the host-driven bundle is compile-once.
- [ ] **RunPod GPU sweep (set up, pending execution):** runbook
  (`docs/runpod_gpu_compile_cache_validation_protocol.md`) + 6-seed benchmark set
  (`docs/runpod_parity_benchmark_seeds.md`) ready; the RunPod/CUDA launcher is
  nvlink-safe. Execute on A100 80GB for per-seed CPU-vs-GPU precision parity,
  walltime, MaxRSS/GPU-mem, and the cold→warm cache-hit. Billable; not yet run.
- [x] **GPU residual classified:** CPU probe shows compile count constant across
  outer budgets (127/178 at maxiter 3 vs 5 steps) → **once-slow**, recompile
  refuted; decision-tree doc `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md`
  committed.
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
  (`test_penalty_value_and_grad_bundle_reuses_compiled_executable`) — **passes on
  CPU** (`jax-0.10.0` env).
- [x] GPU compile-time residual classified: **once-slow** (recompile refuted via
  the CPU compile-count probe), decision-tree doc
  `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md` committed.
- [x] GPU compile caches made persistent + nvlink-safe (Perlmutter launcher +
  RunPod/CUDA launcher); cold→warm runbook and 6-seed benchmark set committed.
- [ ] RunPod/A100 sweep executed: per-seed CPU-vs-GPU parity green, walltime +
  MaxRSS/GPU-mem recorded, cold→warm cache-hit and nvlink-clean confirmed.
- [ ] Parity validation green; no regression in existing compile-diagnostic tests.

## Open Questions

- ~~Is the `scipy-jax` GPU compile-time residual one cold compile or
  recompile-per-eval?~~ **ANSWERED (2026-06-15): one cold compile (once-slow).**
  CPU compile-count probe is constant across outer budgets → no recompile bug →
  no lane code change needed. See
  `docs/jax_scipy_jax_gpu_compile_diagnostic_next.md`.
- ~~Should `FAST_COMPILE` be CPU-only or also applied on GPU host-compile?~~
  **RESOLVED: non-parity CPU only.** GPU relief comes from the persistent compile
  cache (the narrow nvlink-safe autotune mode), not FAST_COMPILE; CUDA and parity
  lanes are gated out. The remaining open GPU question is the *magnitude* of the
  cold→warm cache-hit, which the RunPod/A100 sweep will measure.
- Does the `boozer_bfgs_maxiter=1500` inner cap dominate ondevice breadth? (Phase
  5 bisect; record-only, does not change the production decision.)
