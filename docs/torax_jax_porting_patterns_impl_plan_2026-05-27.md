# TORAX-Informed JAX Porting Pattern Plan (2026-05-27)

## Review Envelope

- Target repo: `/Users/suhjungdae/code/columbia/simsopt-jax-shared-jax`
- Target repo HEAD originally reviewed: `431a517fb`
- Source-doc review basis: `b267b0d95` (2026-05-31 clean current-checkout refresh)
- Source-code checkpoint before docs-only gate commit: `8b94c2bbd` on `shared-jax-clean`, with a broad dirty implementation tree tracked in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`
- Docs-only drift gate introduced in commit `398b3e50d` and checkpoint basis clarified in commit `446eab365` on `shared-jax-clean`
- Reference repo: `/Users/suhjungdae/code/opensource/torax`
- Reference repo HEAD reviewed: `60190df1` (clean `main`)
- Worktree note: the 2026-05-31 source-doc refresh ran from a clean tracked checkout (`git status --short` empty) on `shared-jax-clean`; the 2026-06-01 source-code checkpoint is no longer a clean tracked checkout.
- **Refresh (2026-05-31):** the line refs in this plan were first re-verified against HEAD `21c3d517d`, then corrected against `2bcaeff28`; that pass refreshed the source-doc status to HEAD `b267b0d95` after the MPS custom-kernel commit sequence and updated the refs that moved. Official JAX contracts for persistent cache, `lax.scan`, `lax.while_loop`, and `lax.cond` were rechecked through Context7 during that correction pass. The original review HEAD `431a517fb` is now historical, but the patterns and plan structure are unchanged.
- **Execution checkpoint (2026-06-01):** implementation evidence references source-code checkpoint `8b94c2bbd`; the docs-only drift gate was introduced in `398b3e50d`, then its checkpoint basis was clarified in `446eab365`. The dirty tree must be split by the bloat-plan drift gate before any implementation commit. This TORAX plan records contract-hardening proof, not a standalone LOC-reduction closeout.

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

- Persistent compilation cache is enabled by setting `jax_compilation_cache_dir` or `JAX_COMPILATION_CACHE_DIR` before the first compilation. Official docs also show small-kernel tests must lower the thresholds with `jax_persistent_cache_min_compile_time_secs=0` and `jax_persistent_cache_min_entry_size_bytes=-1`; the current programmatic runtime path imports JAX, then applies those config values before any kernel compilation in `src/simsopt/backend/runtime.py:2418-2421`.
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
| Backend runtime cache/transfer policy | `src/simsopt/backend/runtime.py:1382` (`_resolve_kwarg`), `:2408-2417` (runtime JAX platform / transfer config), `:2418-2421` (persistent-cache config) | Runtime policy should remain opt-in and explicit before the first kernel compilation; env-only paths must still be selected before JAX import. |
| Existing persistent-cache proof | `tests/subprocess/import_smoke_cases.py:711` (write smoke), `:754` / `:768` (shared-cache subprocess cases), `tests/test_jax_import_smoke.py:615` (write smoke wrapper), `:690` (cross-process reuse wrapper) | Current coverage proves a small kernel writes a cache entry and that a second process reuses the first process's cache entry. |
| Host boundary helpers | `src/simsopt/_core/jax_host_boundary.py:14` (`host_array`), `:31` (`host_float64`) | Host materialization should stay explicit and direction-specific. |
| Bounded tracing scans | `src/simsopt/jax_core/tracing.py:375` (`_scan_adaptive_steps`; was `:359`) | Existing helper shape can inform a shared bounded scan utility. |
| Root solver fixed scan | `src/simsopt/jax_core/_root.py:28` | Fixed iteration counts and implicit VJP conventions should remain explicit. |
| PM done-gated scan | `src/simsopt/jax_core/pm_workflow.py:746`, `:807`, `:871`, `:969`, `:1076` | Candidate pilot for deduplicating done-gated scan structure. |
| Wireframe done-gated scan | `src/simsopt/jax_core/wireframe_workflow.py:755` | Candidate pilot paired with PM workflow. |
| Numerical reductions | `src/simsopt/jax_core/reductions.py:75` | Stability work should reuse existing primitives before adding new ones. |
| Static grouped Biot-Savart path | `src/simsopt/jax_core/biotsavart.py:704` (`group_coil_data`), `:734` (`grouped_biot_savart_B`) | Branch and specialization tests should protect hot static choices. |

## Implementation Plan

### Phase 0: Preflight And Evidence Refresh

- [x] Run `git status --short` and record unrelated dirty files before implementation. **2026-06-01 first slice:** tracked tree was clean before the test-contract slice, which intentionally touched only `tests/core/test_jax_core_specs.py`, `tests/subprocess/import_smoke_cases.py`, and `tests/test_jax_import_smoke.py`. Later bloat-plan T1.1/T1.2/T1.3/T1.4/T1.5/T1.6/T1.7/T1.8 slices, the T1.9 public-API reclassification, and the T1.10 probe-script classification expanded the working diff and are logged in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.
- [x] Confirm target repo HEAD and TORAX reference HEAD. **2026-06-01 slice:** target repo HEAD was `8b94c2bbd` (`shared-jax-clean`); TORAX reference evidence remains the 2026-05-31 clean `60190df1` review basis because this slice only uses official JAX contracts, not TORAX source behavior.
- [ ] Refresh inventories with `rg` before editing:
  - [x] `rg "register_dataclass|data_fields|meta_fields|static_arg|static_argnames" src tests`
  - [x] `rg "persistent_cache|compilation_cache|XLA_FLAGS|JAX_COMPILATION_CACHE" src tests benchmarks`
  - [ ] `rg "lax.scan|lax.while_loop|lax.cond|fori_loop" src/simsopt/jax_core src/simsopt/geo src/simsopt/solve tests`
  - [ ] `rg "sqrt|where|nan|clip|maximum|minimum|compensated" src/simsopt/jax_core src/simsopt/geo tests`
- [x] Decide the first implementation slice before touching code: contract tests, persistent-cache tests, or bounded-scan helper pilot. **2026-06-01 slice:** chose contract tests + persistent-cache tests only; no scan helper was added.

### Phase 1: Static/Dynamic Contract Hardening

- [x] Inventory every `jax.tree_util.register_dataclass` use under `src/simsopt`, with `src/simsopt/jax_core/specs.py` as the first SSOT slice.
- [x] Inventory static argument use in traced optimizer, geometry, Biot-Savart, PM, and wireframe paths.
- [x] Design the contract twice:
  - [x] Option A: keep direct `jax.tree_util.register_dataclass` calls and add tests/docs only. **Chosen for the first slice** because it strengthens the contract without adding another abstraction layer.
  - [x] Option B: add a tiny `register_jax_spec` helper local to `jax_core/specs.py`. **Rejected for the first slice** because no runtime helper was needed to prove the existing data/meta split; **selected for the 2026-06-01 T2.4 follow-up** after the contract tests were in place.
- [x] Choose Option B only if it removes repeated partition declarations without hiding the data/meta fields.
- [x] Preserve existing pytree structure and field partitioning exactly.
- [ ] Add or extend tests proving:
  - [x] Spec instances flatten as expected.
  - [x] Dynamic data-field changes do not force static recompilation.
  - [x] Static meta-field changes do force a distinct compiled specialization.
  - [x] Target-lane closures do not capture device arrays accidentally.
  - [x] Tree signatures remain stable across expected construction paths.

**2026-06-01 evidence:** `tests/core/test_jax_core_specs.py::test_curve_spec_data_fields_do_not_recompile_but_meta_fields_do` now pins the `CurveXYZFourierSpec` data/meta split with observable JIT cache behavior. Existing tree-signature coverage remains in `tests/jax_core/test_tree_signature.py`. Later bloat-plan T1.1/T1.3 slices also added `tests/test_jax_import_smoke.py::test_jax_core_lazy_facade_public_contract` and `tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_gpmo_public_result_wrapper_preserves_pytree_leaf_order`, protecting lazy-package exports and solve-level GPMO result pytrees without changing spec registration semantics. The target-lane closure-capture item is now covered by `tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_build_target_lane_outer_objectives_full_state_keeps_closure_constants_on_host`, which builds the CPU-order full-graph DOF map through `build_single_stage_full_graph_jax_cpu_order_dof_map`, assembles full-state target-lane scalar and value/grad wrappers through `build_target_lane_outer_objectives`, asserts the captured index constants remain host NumPy arrays with no `jax.Array` closure leaves, and executes the lifted value/grad under `jax.transfer_guard("disallow")`. Existing `test_target_lane_hardware_success_filter_keeps_closure_constants_on_host` continues to cover the target-lane hardware-success filter closure.

**2026-06-01 T2.4 follow-up evidence:** `src/simsopt/jax_core/specs.py::_register_jax_spec` now wraps the frozen dataclass plus JAX registration ceremony. All 29 spec classes use local tuple-valued `data_fields` / `meta_fields` decorator arguments, and the only direct `jax.tree_util.register_dataclass(...)` call left in `specs.py` is inside the helper. `tests/core/test_jax_core_specs.py::test_register_jax_spec_helper_preserves_data_meta_partition` proves the helper keeps dynamic data leaves and static metadata on the same JIT-cache-key behavior as the direct registrations.

Recommended validation:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/jax_core/test_tree_signature.py
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_jax_core_specs_are_pytrees
```

### Phase 2: Persistent Compilation Cache Proof

- [x] Keep persistent-cache tests separate from in-process `_cache_size` tests.
- [x] Extend the existing callback-free write smoke in `tests/subprocess/import_smoke_cases.py` into a two-process reuse proof, or add an adjacent case if keeping write and reuse coverage separate is clearer.
- [x] Invoke the reuse proof from `tests/test_jax_import_smoke.py` or an adjacent subprocess test wrapper.
- [x] Cover both supported cache-configuration paths: launcher env vars before JAX import, and the programmatic `simsopt_config.set_backend(..., compilation_cache_dir=...)` path that applies `jax.config.update` before first compilation.
- [x] Set `jax_persistent_cache_min_compile_time_secs=0` and `jax_persistent_cache_min_entry_size_bytes=-1` for small-kernel proof tests, matching the runtime policy and JAX official docs.
- [x] Use one temporary cache directory shared by the two subprocesses and assert that the second process reuses the first process's executable rather than merely repopulating an empty cache.
- [x] Do not use host callbacks in persistent-cache proof tests; callbacks can disable or invalidate persistent-cache assumptions.
- [x] Preserve provenance assertions in benchmark helper tests so cache mode is reported in validation output.

**2026-06-01 evidence:** `tests/test_jax_import_smoke.py::test_backend_persistent_cache_reuses_small_kernel_across_processes` now runs two subprocesses against one temporary cache for both programmatic and env-selected backend configuration. It asserts a first-process persistent-cache miss for `jit_small_kernel`, no second-process miss/write for the same compiled kernel, and stable cache file fingerprints.

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
  - [x] Use `lax.scan` for fixed-capacity loops with fixed carry structure, shape, and dtype.
  - [ ] Use `lax.while_loop` only for true dynamic state machines after confirming reverse-mode differentiation is not part of the contract or an explicit custom VJP / implicit-differentiation rule owns that contract.
  - [ ] Keep host loops for I/O, callbacks, plotting, logging, and object mutation.
- [x] Design a small helper such as `bounded_scan_until_done` under `src/simsopt/jax_core/`.
- [x] Keep the helper narrow: explicit carry, explicit `done`, explicit status payload, explicit max steps.
- [x] Pilot it on one repeated done-gated pair, preferably PM or wireframe workflow.
- [x] Do not rewrite tracing and root solving in the same pass.
- [ ] Add tests for:
  - [x] `max_steps == 0`
  - [x] Early completion
  - [x] Never-completed status propagation
  - [x] Static-capacity rejection where applicable
  - [x] Strict transfer-guard smoke coverage

**2026-06-01 evidence:** `src/simsopt/jax_core/_bounded_scan.py::bounded_scan_until_done` now owns the fixed-capacity scan shape: it takes an explicit carry, `max_steps`, scalar `is_done` predicate, and active-step function, then skips active work with `lax.cond` once done is true. The pilot is deliberately narrow: `pm_gpmo_live_loop_jax` and `_gsco_live_loop_unchecked` use the helper, while tracing, root solving, PM ArbVec/multi/backtracking loops, and wireframe multistep/final-adjustment loops remain untouched. `tests/jax_core/test_bounded_scan.py` covers zero steps, early completion, never-completed status propagation, and strict transfer-guard JIT execution; existing PM and wireframe workflow tests cover eager static-capacity rejection, exact restart continuation, host-loop parity, JAXPR scan presence, and transfer-guard smoke for the two piloted production loops.

Recommended validation:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_pm_workflow_jax.py tests/solve/test_wireframe_workflow_jax.py
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_tracing_jax_item14.py tests/jax_core/test_tracing_jax_conservation.py
```

### Phase 4: Branch Discipline And JAXPR Checks

- [ ] Audit expensive `lax.cond` and static-argument sites where both branches trace.
- [ ] For each site, classify the intended behavior:
  - [x] Static host peel for compile-time choices.
  - [x] Traced runtime branch for array-dependent choices, with explicit awareness that both branches trace even though only one branch executes.
  - [ ] Explicit host boundary for object or logging work.
- [x] Add JAXPR or lowered-IR tests for hot paths where branch drift would be expensive or wrong.
- [x] Add vectorized-branch tests where `vmap(lax.cond)` could change execution semantics by lowering to `select`.
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

**2026-06-01 pilot evidence:** the first branch-discipline slice classifies
two existing hot-path contracts without changing runtime behavior. The
static-host-peel contract is the cached strict scalar value/grad wrapper in
`src/simsopt/geo/surfaceobjectives_jax.py`, already pinned by
`tests/geo/test_surface_objectives_jax.py::test_cached_strict_scalar_value_and_grad_builds_stable_jit`
checking `jax.jit(..., static_argnums=(2, 3))`. The traced-runtime branch
contract is `src/simsopt/jax_core/pm_optimization.py::mwpgp_step`:
`test_step_body_uses_dynamic_branch_conditionals` keeps the scalar JAXPR at two
`cond` primitives, while the new
`test_vmap_step_body_lowers_dynamic_branches_to_selects` proves `vmap` lowers
that branch family to `select_n` and no scalar `cond` remains. This is CPU/X64
JAXPR evidence only; host-boundary classification, dense-fallback checks, and
CUDA transfer proof remain open.

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

**2026-06-01 cross-link:** T2.3 `surface_fourier.py` facade wrapper factories were executed under the bloat plan, not as a new TORAX runtime phase. The slice preserves public non-RZ surface symbols, keeps composed geometry explicit, banks 165 source LOC in `src/simsopt/jax_core/surface_fourier.py`, and records exact CPU/X64 validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. CUDA/MPS, persistent-cache, branch/JAXPR, and numerical-stability TORAX gates remain open.

**2026-06-01 cross-link:** The T2.3 tensor surface kernel wrapper fold was also executed under the bloat plan. It factors only the repeated `SurfaceXYZTensorFourier` simple `_from_dofs` wrapper bodies in `src/simsopt/jax_core/surface_fourier_kernels.py`, banks 190 additional source LOC, preserves public wrapper signatures/metadata/introspection docs, and records CPU/X64 clamping, stellsym scatter, paired-linear, XYZ adjacency, and coefficient-Jacobian validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. The `SurfaceXYZFourier` lower-level wrappers and coefficient-Jacobian families remain separate follow-ups.

**2026-06-01 cross-link:** The T2.3 `SurfaceXYZFourier` unpack fold was executed under the bloat plan. It factors only repeated flat-DOF to coefficient-matrix unpacking by calling the existing `_scatter_surface_xyzfourier_dofs(...)` helper in `src/simsopt/jax_core/surface_fourier_kernels.py`, banks 51 additional source LOC, preserves public wrapper definitions/source introspection, and records CPU/X64 `SurfaceXYZFourier` geometry, tangent, coefficient-Jacobian, paired-linear, and non-RZ fundamental-form validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. Analytic derivative formulas and coefficient-Jacobian family structure remain separate follow-ups.

**2026-06-01 cross-link:** The T2.3 coefficient-derivative wrapper-family fold was executed under the bloat plan. It centralizes only the repeated tensor/`SurfaceXYZFourier` autodiff wrapper ceremony in `src/simsopt/jax_core/surface_fourier_kernels.py`, banks 109 additional source LOC, preserves the public tensor derivative signature versus the `SurfaceXYZFourier` `coeff_template` signature, and records CPU/X64 coefficient/scalar derivative validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. `SurfaceXYZFourier` analytic derivative formulas remain separate follow-ups.

**2026-06-01 cross-link:** The T2.3 `SurfaceXYZFourier` order-hat helper slice was executed under the bloat plan. It centralizes only the repeated full-grid derivative-order basis and component-hat evaluation in `src/simsopt/jax_core/surface_fourier_kernels.py`, banks 35 additional source LOC, preserves public analytic wrapper introspection and paired-linear paths, and records CPU/X64 geometry, derivative, and full-file surface Fourier validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. Product-rule formulas remain explicit and any further formula fold is a separate bloat-plan decision, not a TORAX runtime phase.

**2026-06-01 cross-link:** The T2.3 `SurfaceXYZFourier.gammadash1` product-rule micro-slice was executed under the bloat plan. It rewrites only the full-grid `gammadash1` cosine/sine derivative spelling into explicit radial/toroidal components before calling `_surface_xyzfourier_rotate(...)` in `src/simsopt/jax_core/surface_fourier_kernels.py`, banks 5 additional source LOC, preserves public wrapper signatures, paired-linear paths, coefficient-Jacobian factories, and all other product-rule formulas, and records CPU/X64 geometry/tangent validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. Remaining product-rule folds stay separate bloat-plan decisions, not TORAX runtime phases.

**2026-06-01 cross-link:** The T2.2 Boozer radial direct-wrapper follow-up was executed under the bloat plan. It centralizes only private direct evaluator wrapper ceremony in `src/simsopt/jax_core/boozer_radial_field.py`, banks 35 source LOC, preserves the explicit formula and subset-column ownership, and records CPU/X64 routing, type-check, and metadata validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. The remaining T2.2 subset-builder profile-family question is a bloat-plan follow-up, not a TORAX runtime phase.

**2026-06-01 cross-link:** The T2.2 Boozer radial scalar-helper follow-up was executed under the bloat plan. It centralizes only repeated scalar direct-evaluator ceremony in `src/simsopt/jax_core/boozer_radial_field.py`, banks 12 additional source LOC, preserves scalar spline reads, explicit Fourier subset builders, formula ownership, RHS column reuse, and public wrapper routing, and records CPU/X64 routing, type-check, and benchmark validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. The full T2.2 formula/subset-family target remains a bloat-plan decision, not a TORAX runtime phase.

**2026-06-01 cross-link:** The T2.1 LS-Newton reporting helper follow-up was executed under the bloat plan. It centralizes only repeated Hessian/Newton-polish reporting-field extraction in `src/simsopt/geo/boozersurface_jax.py`, banks 37 source LOC, preserves solve-path branching, backend strings, VJP construction, factorization metadata, and local solve-quality overrides, and records CPU/X64 result-schema and run-code validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. This is not CUDA/MPS or TORAX Stage 2 parity proof.

**2026-06-01 cross-link:** T2.9 quantity-aware tolerance extraction was executed under the bloat plan as a numerical-contract SSOT slice. `benchmarks/validation_ladder_contract.py` now owns `QUANTITY_TOLERANCE_BUCKETS` and `quantity_parity_tolerance(...)`; the non-banana harness keeps `_tolerance_for(quantity)` as a compatibility wrapper. A 204-row pre/post snapshot proved unchanged `(bucket, rtol, atol)` results for every migrated quantity across `cpu_reference`, `parity`, `fast`, and `float32_smoke`; focused float32 diagnostic tests and source-only `mypy` for `validation_ladder_contract.py` passed. This closes the tolerance-policy contract migration but is not LOC-banked.

**2026-06-01 cross-link:** The T3.2 Biot-Savart points-helper follow-up was executed under the bloat plan, not as a TORAX runtime phase. It centralizes duplicate mutable point-state helper bodies shared by `SpecBackedBiotSavartJAX` and `BiotSavartJAX` in `src/simsopt/field/biotsavart_jax_backend.py`, banks 2 source LOC after preserving public method metadata, preserves the live no-host-round-trip JAX-array point path, and leaves `coil_cotangents_to_dofs_gradient` open because the spec-backed and live fallback contracts have diverged. CPU/X64 point/cylindrical/spec validation is recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`; this is not CUDA/MPS or Stage 2 parity proof.

**2026-06-01 cross-link:** The T2.8 `LayerDriftTracker` core helper was executed under the bloat plan as replay-diagnostic cleanup, not as a TORAX runtime phase. It centralizes the repeated layer-decomposition max/first-divergence state transitions in `benchmarks/single_stage_init_parity.py`, banks 13 benchmark LOC, preserves explicit same-candidate replay result keys and pre-Newton census gate semantics, and records CPU/X64 replay-helper validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. The larger T2.8 tracker-family cleanup remains open.

**2026-06-01 cross-link:** The T2.8 SciPy callback first-split helper follow-up was also executed under the bloat plan. It centralizes only the repeated "first split wins" assignment for Boozer SciPy callback trace divergence reporting in `benchmarks/single_stage_init_parity.py`, banks 4 additional benchmark LOC, preserves explicit callback field comparison order and `first_boozer_scipy_callback_split` schema keys, and records CPU/X64 callback-trace validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. Full single-stage parity, CUDA/MPS, and larger tracker-family cleanup remain separate gates.

**2026-06-01 cross-link:** The T2.8 target-native predicate-cache micro-slice was also executed under the bloat plan. It caches only the per-pair target-native rejection predicates inside `benchmarks/single_stage_init_parity.py`, banks 2 additional benchmark LOC, preserves replay payload keys, rejection diagnostics, gradient/hardware comparison semantics, and same-candidate gate classification, and records CPU/X64 target-native replay validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. Full single-stage parity, CUDA/MPS, and larger tracker-family cleanup remain separate gates.

**2026-06-01 cross-link:** The T2.8 comparison-scope `Counter` micro-slice was also executed under the bloat plan. It replaces only the repeated candidate/gradient comparison-scope count increments inside `benchmarks/single_stage_init_parity.py`, banks 3 additional benchmark LOC, preserves explicit replay payload keys by converting the counters back to plain dicts at the result boundary, and records CPU/X64 scope-count replay validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. Full single-stage parity, CUDA/MPS, and larger tracker-family cleanup remain separate gates.

**2026-06-01 cross-link:** The T2.8 per-pair metadata-binding micro-slice was also executed under the bloat plan. It binds repeated same-candidate event metadata inside `benchmarks/single_stage_init_parity.py`, banks 4 additional benchmark LOC, preserves explicit replay payload keys and first-failure/callback/rejection metadata fields, and records CPU/X64 replay-metadata validation in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`. Full single-stage parity, CUDA/MPS, and larger tracker-family cleanup remain separate gates.

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

- [x] Add tests around existing spec registration behavior without changing runtime code.
- [x] Extend the existing callback-free persistent-cache write smoke into a two-process reuse proof.
- [x] Do not add a scan helper until the cache and static/dynamic contracts are proven.

This gives the repo higher confidence in the two easiest places to regress JAX ports: compilation specialization boundaries and false cache-proof signals.

Validation recorded for the 2026-06-01 first slice:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/test_jax_import_smoke.py -k 'jax_core_specs or persistent_cache'
# 19 passed, 119 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/jax_core/test_tree_signature.py
# 32 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_backend.py -k 'compilation_cache or cuda_determinism or gpu_memory'
# 23 passed, 105 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_benchmark_helpers.py -k 'compilation_cache or build_provenance_includes_compilation_cache_metadata or compile_behavior'
# 13 passed, 344 deselected
.conda/jax/bin/python -m ruff check tests/core/test_jax_core_specs.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py
# passed
.conda/jax/bin/python -m ruff format --check tests/core/test_jax_core_specs.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py
# passed
git diff --check -- tests/core/test_jax_core_specs.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py
# passed
.conda/jax/bin/python -m mypy tests/core/test_jax_core_specs.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py
# blocked: No module named mypy
```
