# TORAX-Informed JAX Porting Pattern Plan (2026-05-27)

## Review Envelope

- Target repo: `/Users/suhjungdae/code/columbia/simsopt-jax-shared-jax`
- Target repo HEAD originally reviewed: `431a517fb`
- Target repo HEAD refreshed: `b267b0d95` (2026-05-31 current-checkout refresh)
- Reference repo: `/Users/suhjungdae/code/opensource/torax`
- Reference repo HEAD reviewed: `60190df1` (clean `main`)
- Worktree note: this 2026-05-31 refresh ran from a clean tracked checkout (`git status --short` empty) on `shared-jax-clean`.
- **Refresh (2026-05-31):** the line refs in this plan were first re-verified against HEAD `21c3d517d`, then corrected against `2bcaeff28`; this pass refreshes the live status to HEAD `b267b0d95` after the MPS custom-kernel commit sequence and updates the refs that moved. Official JAX contracts for persistent cache, `lax.scan`, `lax.while_loop`, and `lax.cond` were rechecked through Context7 during this correction pass. The original review HEAD `431a517fb` is now historical, but the patterns and plan structure are unchanged.

## Purpose

Capture the five useful TORAX-derived JAX porting patterns and translate them into a concrete, repo-local implementation plan for `simsopt-jax`. The goal is not to copy TORAX abstractions directly. The goal is to use TORAX as an external stress test for patterns this repo already cares about: static/dynamic pytree contracts, cache and XLA discipline, bounded compiled control flow, branch structure, and numerical shape/stability contracts.

## Rationale

`simsopt-jax` already has several of the right foundations:

- Immutable JAX specs and explicit data/meta partitioning in `src/simsopt/jax_core/specs.py`.
- Cache and transfer policy plumbing in `benchmarks/validation_ladder_common.py` and `src/simsopt/backend/runtime.py`.
- Explicit host boundary helpers in `src/simsopt/_core/jax_host_boundary.py`.
- Fixed-size scan patterns in tracing, root solving, PM workflow, and wireframe workflow.
- Numerical stability primitives such as compensated reductions in `src/simsopt/jax_core/reductions.py`.

The remaining opportunity is to harden these surfaces with narrow contracts, proof tests, and a small amount of deduplication. We should avoid broad rewrites, silent fallbacks, or generalized type-system rollouts that do not directly remove a current failure mode.

One TORAX caution should be kept explicit: do not clone its `StaticDataclass` approach verbatim. The prior scan found a stale-value style hazard around TORAX static-field checking. If a local helper is added, it should preserve `simsopt-jax`'s existing explicit `register_dataclass` partitions and be proved with retracing tests.

## Official JAX Contracts Checked

- Persistent compilation cache is enabled by setting `jax_compilation_cache_dir` or `JAX_COMPILATION_CACHE_DIR` before the first compilation. Official docs also show small-kernel tests must lower the thresholds with `jax_persistent_cache_min_compile_time_secs=0` and `jax_persistent_cache_min_entry_size_bytes=-1`; the current programmatic runtime path imports JAX, then applies those config values before any kernel compilation in `src/simsopt/backend/runtime.py:2371` and `src/simsopt/backend/runtime.py:2384-2386`.
- Persistent-cache proof tests must stay callback-free. JAX official docs state that host callbacks make persistent caching completely avoided because the HLO includes a callback pointer that changes between runs.
- `lax.scan` is the right primitive for fixed-iteration compiled loops because it lowers to a single `WhileOp`, requires fixed carry structure, shape, and dtype, and is designed for static iteration counts.
- `lax.while_loop` is the right primitive for true dynamic termination only after the differentiation contract is checked. JAX official docs state bare `while_loop` is not reverse-mode differentiable because XLA needs static memory bounds; this repo can still use it inside an explicit custom VJP or implicit-differentiation wrapper such as `src/simsopt/jax_core/_root.py:103`.
- `lax.cond` means only one branch executes at runtime, but both branches trace in all cases. Under `vmap`, `cond` is converted to `select`, so branch-discipline tests must check trace cost and vectorized semantics separately.

References:

- JAX persistent compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- JAX `lax.scan`: https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
- JAX `lax.while_loop`: https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html
- JAX `lax.cond`: https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html

## Goals

- [ ] Make static/dynamic JAX object contracts easier to audit and harder to drift.
- [ ] Prove persistent compilation cache behavior across processes, separately from in-process JIT cache behavior.
- [ ] Reduce repeated done-gated bounded scan code where the semantics are identical.
- [ ] Classify branch decisions into static host decisions, traced runtime control flow, and host-boundary work.
- [ ] Add focused numerical shape/stability contracts at construction and test boundaries.

## Non-Goals

- [ ] Do not roll out a repo-wide `jaxtyping` or `chex` migration as part of this plan.
- [ ] Do not JAXify mutable public `Optimizable` objects directly.
- [ ] Do not add silent clamps, catch-all defensive code, or fallback paths to hide invalid states.
- [ ] Do not mutate `XLA_FLAGS` or cache configuration at library import time.
- [ ] Do not combine tracing, optimizer, PM, wireframe, and numerical changes into one broad rewrite.

## Current Evidence Map

| Surface | Current evidence | Plan relevance |
| --- | --- | --- |
| JAX spec contracts | `src/simsopt/jax_core/specs.py:1` | Explicit immutable specs and data/meta field partitions are the SSOT to harden first. |
| Validation cache policy | `benchmarks/validation_ladder_common.py:159` (`apply_compilation_cache_policy`), `benchmarks/validation_ladder_common.py:390` (`current_compilation_cache_metadata`) | Cache settings and provenance are already explicit enough to test. |
| Backend runtime cache/transfer policy | `src/simsopt/backend/runtime.py:1372`, `src/simsopt/backend/runtime.py:2381`, `src/simsopt/backend/runtime.py:2384-2386` | Runtime policy should remain opt-in and environment-driven before JAX import. |
| Existing persistent-cache write smoke | `tests/subprocess/import_smoke_cases.py:711`, `tests/test_jax_import_smoke.py:614` | Current coverage proves a small kernel writes a cache entry; the remaining gap is cross-process reuse. |
| Host boundary helpers | `src/simsopt/_core/jax_host_boundary.py:14` | Host materialization should stay explicit and direction-specific. |
| Bounded tracing scans | `src/simsopt/jax_core/tracing.py:367` (`_scan_adaptive_steps`; was `:359`) | Existing helper shape can inform a shared bounded scan utility. |
| Root solver fixed scan | `src/simsopt/jax_core/_root.py:28` | Fixed iteration counts and implicit VJP conventions should remain explicit. |
| PM done-gated scan | `src/simsopt/jax_core/pm_workflow.py:746`, `:807`, `:871`, `:969`, `:1076` | Candidate pilot for deduplicating done-gated scan structure. |
| Wireframe done-gated scan | `src/simsopt/jax_core/wireframe_workflow.py:755` | Candidate pilot paired with PM workflow. |
| Numerical reductions | `src/simsopt/jax_core/reductions.py:75` | Stability work should reuse existing primitives before adding new ones. |
| Static grouped Biot-Savart path | `src/simsopt/jax_core/biotsavart.py:704` (`group_coil_data`), `:734` (`grouped_biot_savart_B`) | Branch and specialization tests should protect hot static choices. |

## Implementation Plan

### Phase 0: Preflight And Evidence Refresh

- [ ] Run `git status --short` and record unrelated dirty files before implementation.
- [ ] Confirm target repo HEAD and TORAX reference HEAD.
- [ ] Refresh inventories with `rg` before editing:
  - [ ] `rg "register_dataclass|data_fields|meta_fields|static_arg|static_argnames" src tests`
  - [ ] `rg "persistent_cache|compilation_cache|XLA_FLAGS|JAX_COMPILATION_CACHE" src tests benchmarks`
  - [ ] `rg "lax.scan|lax.while_loop|lax.cond|fori_loop" src/simsopt/jax_core src/simsopt/geo src/simsopt/solve tests`
  - [ ] `rg "sqrt|where|nan|clip|maximum|minimum|compensated" src/simsopt/jax_core src/simsopt/geo tests`
- [ ] Decide the first implementation slice before touching code: contract tests, persistent-cache tests, or bounded-scan helper pilot.

### Phase 1: Static/Dynamic Contract Hardening

- [ ] Inventory every `jax.tree_util.register_dataclass` use under `src/simsopt`, with `src/simsopt/jax_core/specs.py` as the first SSOT slice.
- [ ] Inventory static argument use in traced optimizer, geometry, Biot-Savart, PM, and wireframe paths.
- [ ] Design the contract twice:
  - [ ] Option A: keep direct `jax.tree_util.register_dataclass` calls and add tests/docs only.
  - [ ] Option B: add a tiny `register_jax_spec` helper local to `jax_core/specs.py`.
- [ ] Choose Option B only if it removes repeated partition declarations without hiding the data/meta fields.
- [ ] Preserve existing pytree structure and field partitioning exactly.
- [ ] Add or extend tests proving:
  - [ ] Spec instances flatten as expected.
  - [ ] Dynamic data-field changes do not force static recompilation.
  - [ ] Static meta-field changes do force a distinct compiled specialization.
  - [ ] Target-lane closures do not capture device arrays accidentally.
  - [ ] Tree signatures remain stable across expected construction paths.

Recommended validation:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/jax_core/test_tree_signature.py
```

### Phase 2: Persistent Compilation Cache Proof

- [ ] Keep persistent-cache tests separate from in-process `_cache_size` tests.
- [ ] Extend the existing callback-free write smoke in `tests/subprocess/import_smoke_cases.py` into a two-process reuse proof, or add an adjacent case if keeping write and reuse coverage separate is clearer.
- [ ] Invoke the reuse proof from `tests/test_jax_import_smoke.py` or an adjacent subprocess test wrapper.
- [ ] Cover both supported cache-configuration paths: launcher env vars before JAX import, and the programmatic `simsopt_config.set_backend(..., compilation_cache_dir=...)` path that applies `jax.config.update` before first compilation.
- [ ] Set `jax_persistent_cache_min_compile_time_secs=0` and `jax_persistent_cache_min_entry_size_bytes=-1` for small-kernel proof tests, matching the runtime policy and JAX official docs.
- [ ] Use one temporary cache directory shared by the two subprocesses and assert that the second process reuses the first process's executable rather than merely repopulating an empty cache.
- [ ] Do not use host callbacks in persistent-cache proof tests; callbacks can disable or invalidate persistent-cache assumptions.
- [ ] Preserve provenance assertions in benchmark helper tests so cache mode is reported in validation output.

Recommended validation:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_backend.py -k 'compilation_cache or cuda_determinism or gpu_memory'
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_benchmark_helpers.py -k 'compilation_cache or build_provenance_includes_compilation_cache_metadata or compile_behavior'
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py -k 'persistent_cache or reuses_compiled_solver or transfer_guard'
```

CUDA follow-up, only on a CUDA host with the repo CUDA/JAX environment active:

```bash
PYTHONPATH=src SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_PLATFORMS=cuda,cpu XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true" .conda/jax/bin/python -m pytest -q tests/test_backend.py -k 'cuda_determinism or gpu_memory'
```

### Phase 3: Bounded Scan And Control-Flow Deduplication

- [ ] Define the control-flow categories before writing helpers:
  - [ ] Use `lax.scan` for fixed-capacity loops with fixed carry structure, shape, and dtype.
  - [ ] Use `lax.while_loop` only for true dynamic state machines after confirming reverse-mode differentiation is not part of the contract or an explicit custom VJP / implicit-differentiation rule owns that contract.
  - [ ] Keep host loops for I/O, callbacks, plotting, logging, and object mutation.
- [ ] Design a small helper such as `bounded_scan_until_done` under `src/simsopt/jax_core/`.
- [ ] Keep the helper narrow: explicit carry, explicit `done`, explicit status payload, explicit max steps.
- [ ] Pilot it on one repeated done-gated pair, preferably PM or wireframe workflow.
- [ ] Do not rewrite tracing and root solving in the same pass.
- [ ] Add tests for:
  - [ ] `max_steps == 0`
  - [ ] Early completion
  - [ ] Never-completed status propagation
  - [ ] Static-capacity rejection where applicable
  - [ ] Strict transfer-guard smoke coverage

Recommended validation:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_pm_workflow_jax.py tests/solve/test_wireframe_workflow_jax.py
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_tracing_jax_item14.py tests/jax_core/test_tracing_jax_conservation.py
```

### Phase 4: Branch Discipline And JAXPR Checks

- [ ] Audit expensive `lax.cond` and static-argument sites where both branches trace.
- [ ] For each site, classify the intended behavior:
  - [ ] Static host peel for compile-time choices.
  - [ ] Traced runtime branch for array-dependent choices, with explicit awareness that both branches trace even though only one branch executes.
  - [ ] Explicit host boundary for object or logging work.
- [ ] Add JAXPR or lowered-IR tests for hot paths where branch drift would be expensive or wrong.
- [ ] Add vectorized-branch tests where `vmap(lax.cond)` could change execution semantics by lowering to `select`.
- [ ] Verify compiled hot paths do not hide dense fallbacks, host callbacks, or unexpected materialization.
- [ ] Keep CPU proof and CUDA transfer proof distinct in documentation and tests.

Recommended first targets:

- [ ] `src/simsopt/jax_core/biotsavart.py`
- [ ] `src/simsopt/geo/surfaceobjectives_jax.py`
- [ ] `src/simsopt/geo/optimizer_jax.py`
- [ ] `src/simsopt/jax_core/pm_workflow.py`
- [ ] `src/simsopt/jax_core/wireframe_workflow.py`
- [ ] `src/simsopt/solve/permanent_magnet_optimization_jax.py`
- [ ] Optimizer backend static toggles and solver status paths.

### Phase 5: Numerical Shape And Stability Audit

- [ ] Scope this phase to high-risk construction and test boundaries, not every internal temporary.
- [ ] Audit known high-risk operations:
  - [ ] VMEC geometry divisions and square roots in `src/simsopt/jax_core/vmec_geometry.py`.
  - [ ] Surface curvature discriminant square roots in `src/simsopt/geo/surfaceobjectives_jax.py`.
  - [ ] Solver status, convergence, and residual conventions.
  - [ ] Compensated reductions and summation order in parity-sensitive paths.
- [ ] For each guard or transformation, document the physics or numerical contract.
- [ ] Add parity tests before changing semantics.
- [ ] Reject silent clamps unless the mathematical contract already defines the clamp.
- [ ] Prefer explicit invalid-input tests over defensive runtime fallbacks.

Recommended validation:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_objectives_jax.py tests/geo/test_boozer_residual_jax.py tests/mhd/test_vmec_compute_geometry_jax.py tests/mhd/test_vmec_diagnostics_jax.py tests/jax_core tests/solve -k 'surface or vmec or residual or stability or compensated'
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_reductions.py -k 'compensated or pairwise'
```

### Phase 6: Documentation, Provenance, And Closeout

- [ ] Update this checklist as each implementation slice lands.
- [ ] Cross-link completed work to:
  - [ ] `docs/remaining_jax_port_surfaces_impl_plan_2026-05-19.md`
  - [ ] `docs/bloat_reduction_plan_2026-05-20.md`
  - [ ] `docs/using_jax_backend.md` after refreshing its stale backend-mode table and optimizer-default guidance.
- [ ] Record exact validation commands and results before marking a phase complete.
- [ ] Keep dirty-worktree status visible in the closeout note.
- [ ] If a phase changes public behavior, add a focused migration or user-facing note.

## Acceptance Gates

- [ ] Only files in the chosen implementation slice are modified.
- [ ] `git diff --check` passes.
- [ ] Relevant focused tests pass on CPU.
- [ ] CUDA claims are made only after CUDA-host validation.
- [ ] No `.env` or `.env.*` file is created or staged.
- [ ] No dynamic imports are introduced.
- [ ] No untyped escape hatches are introduced.
- [ ] No persistent-cache test relies on host callbacks.
- [ ] No CPU-only proof is cited as GPU transfer or CUDA determinism proof.
- [ ] Any helper introduced has a smaller public surface than the duplicated code it replaces.

## Suggested Implementation Order

1. Persistent cache proof and static/dynamic contract tests.
2. Bounded scan helper pilot on PM or wireframe workflow.
3. Branch/JAXPR classification tests for hot paths.
4. Numerical shape/stability audit at construction and test boundaries.
5. Documentation cross-links and closeout.

## Risk Register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Data/meta partition drift causes stale compilation or over-recompilation. | High | Add tests that separate dynamic data-field updates from static meta-field changes. |
| Persistent-cache tests pass only because of in-process JIT cache reuse. | High | Use separate subprocesses and a fresh persistent cache directory. |
| Host callbacks invalidate persistent-cache assumptions. | High | Keep persistent-cache smoke cases callback-free. |
| Scan helper becomes a broad abstraction with unclear semantics. | Medium | Pilot on one repeated done-gated pattern and require explicit carry/done/status. |
| Silent clamps change physics behavior while hiding invalid input. | High | Require parity tests and documented numerical rationale before any guard. |
| Dirty-worktree collision mixes unrelated doc/code changes. | Medium | Recheck `git status --short` before and after each slice. |

## First Slice Recommendation

Start with Phase 1 and Phase 2 together only at the test-contract level:

- [ ] Add tests around existing spec registration behavior without changing runtime code.
- [ ] Extend the existing callback-free persistent-cache write smoke into a two-process reuse proof.
- [ ] Do not add a scan helper until the cache and static/dynamic contracts are proven.

This gives the repo higher confidence in the two easiest places to regress JAX ports: compilation specialization boundaries and false cache-proof signals.
