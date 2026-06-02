# Coherent Bloat And TORAX JAX Execution Plan

## Review Envelope

- Target repo: `/Users/suhjungdae/code/columbia/simsopt-jax-shared-jax`
- Source-doc review basis: `b267b0d95` on `shared-jax-clean`
- Source-code checkpoint before docs-only gate commit: `8b94c2bbd` on `shared-jax-clean`, with a broad dirty implementation tree across `src/`, `tests/`, and `docs/`
- Docs-only drift gate introduced in commit `398b3e50d` and checkpoint basis clarified in commit `446eab365` on `shared-jax-clean`
- Latest completed T2.8 checkpoint before this scalar-close source/docs sync: `114c53685` (`docs: record T2.8 decomposition dispatch slice`), after the broad dirty tree was split into scoped source/docs commits; this sync adds the scalar-close inline slice in `benchmarks/single_stage_init_parity.py`.
- Reference TORAX repo reviewed: `/Users/suhjungdae/code/opensource/torax` at `60190df1` on clean `main`
- Historical local status at source-doc review: the two source docs were modified and this overlay was untracked; no source-code edits were part of that review. This is no longer the current working-tree state.
- Artifact note: this checkout does not contain a repo-local `.artifacts/` tree. Historical code-smell artifacts referenced by the bloat plan were found in sibling checkout `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/`.

## 2026-06-01 Drift Checkpoint

The drift-checkpoint dirty tree was validated as a contract-hardening / complexity-reduction checkpoint, not as a strict LOC-reduction checkpoint. Do not treat that whole historical tree as a generic "bloat reduction" change.

Drift-checkpoint ledger captured before the v8 doc correction (`git diff --numstat -- src tests docs` plus untracked-file `wc -l`):

- `src/` tracked: `1933 insertions / 1941 deletions`, net `-8`
- Untracked source helpers: `+53`
- Effective source net: `+45`
- `tests/` tracked: `1174 insertions / 54 deletions`, net `+1120`
- Untracked tests: `+224`
- `docs/` tracked: `938 insertions / 161 deletions`, net `+777`
- Total tracked plus untracked over `src/`, `tests/`, and `docs/`: `+2166`

Execution gate for the next pass:

- **Banked-shrink:** source LOC is net-negative in an isolated scoped slice and behavior/API compatibility is validated.
- **Foundation-only:** source LOC is flat or positive, but the slice names the exact deletion it unlocks and the next deletion task is tracked.
- **Not LOC-banked:** complexity or contract quality improved, but no current source shrink can be claimed.
- **Defer/revert-candidate:** source LOC is flat or positive and no immediate deletion payoff is identified.

The broad dirty tree has since been split through scoped banked-shrink/foundation commits. For future implementation commits, keep using this classification before staging: salvage the banked-shrink slices first; keep foundation-only slices only with their follow-up deletion target; do not count tests/docs growth as bloat reduction.

## Purpose

Coordinate execution of the refreshed bloat-reduction and TORAX-informed JAX-porting plans without treating them as independent backlogs. This file is an execution overlay for:

- `docs/bloat_reduction_plan_2026-05-20.md`
- `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md`

The source plans remain the SSOT for detailed item text, line refs, and acceptance gates. This overlay defines the order, dependency boundaries, and validation checkpoints needed to tackle the two plans coherently.

## Goals

- Execute shared JAX contract work once, then reuse it across bloat-reduction and TORAX-pattern tasks.
- Bank low-risk bloat reductions without blocking higher-value correctness gates.
- Keep persistent-cache, transfer-boundary, and MPS smoke-lane evidence separate from CPU-only proof.
- Prevent new TORAX-inspired abstractions from adding more scaffolding than they remove.
- Produce small, reviewable slices with explicit validation and rollback boundaries.

## Non-Goals

- Do not replace either source plan.
- Do not merge unrelated bloat, cache, optimizer, PM, wireframe, and MPS work into one broad refactor.
- Do not relax public APIs, parity tolerances, backend-mode contracts, or host-transfer policy to make refactors easier.
- Do not copy TORAX abstractions verbatim; only adopt patterns that fit current `simsopt-jax` contracts.
- Do not treat LOC reduction as the only success gate. Complexity reduction and preserved behavior are required.

## Current Context

- Source-doc refresh basis: `shared-jax-clean` at `b267b0d95`.
- Docs-gate history: the docs-only drift gate was introduced in `398b3e50d`, checkpoint basis was clarified in `446eab365`, the source-code checkpoint before that docs-only gate was `8b94c2bbd`, and the latest completed T2.8 checkpoint before this scalar-close sync is `114c53685`. Treat these commit hashes as historical anchors, not live-HEAD markers.
- `docs/bloat_reduction_plan_2026-05-20.md` is a tiered reduction plan: T1 mechanical wins, T2 factory introductions, T3 structural consolidations, and T4 contract decisions.
- `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md` is a pattern-hardening plan: static/dynamic pytree contracts, persistent-cache proof, bounded control flow, branch discipline, and numerical stability.
- Shared dependency surfaces include `jax_core` specs, backend runtime/cache policy, validation ladder helpers, host-boundary helpers, fixed-iteration scan code, PM/wireframe workflows, and GPU/MPS-sensitive runtime paths.

## Rationale

The two plans overlap in the places most likely to create regressions: JAX object contracts, cache/transfer configuration, compiled control flow, and parity validation. Running the bloat plan first without the TORAX contract work risks deleting or folding code before the invariants are well tested. Running the TORAX plan first without bloat discipline risks adding helper abstractions that increase long-term maintenance cost.

The right sequencing is contract-first, then mechanical deletion, then shared factories, then structural folds. Each slice should have a clear owner document, a narrow changed-file set, and validation strong enough for the touched surface.

## Assumptions

- The source docs were refreshed at `b267b0d95`; execution then advanced to source-code checkpoint `8b94c2bbd` with a broad dirty tree, the docs-only drift gate was introduced in `398b3e50d`, and its checkpoint basis was clarified in `446eab365`. Re-run evidence refresh and the drift ledger before selecting or committing any additional implementation slice.
- Code work must load and apply `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md` before implementation.
- CPU validation is not enough for GPU-sensitive or MPS-sensitive claims.
- `jax_mps_smoke` remains a smoke lane, not a production parity lane.
- TORAX is a reference for useful JAX patterns, not an upstream dependency or architectural template.

## Dependency Map

| Coherent slice | Bloat-plan source | TORAX-plan source | Why they belong together |
| --- | --- | --- | --- |
| Evidence refresh | Section 4.3, Section 9 | Phase 0 | Both plans require fresh path/caller inventories before edits. |
| JAX object contracts | T1.1, T2.4, selected T2/T3 helpers | Phase 1 | Lazy exports, spec registration, and static/dynamic pytree proof all affect import and JIT behavior. |
| Cache and transfer policy | Section 4.1, Section 9.5 | Phase 2 | Persistent-cache proof and strict-transfer proof share runtime/environment boundaries. |
| Mechanical bloat bank | T1.2 through T1.11 | Phase 0 evidence only | Low-risk deletions should proceed after caller inventory, without waiting for broader TORAX work. |
| Done-gated scan dedup | T3.7, related PM/wireframe items | Phase 3 | Bounded-loop helper work should be designed once and piloted on one workflow pair. |
| Branch discipline | T4.1, T4.2, T4.3 | Phase 4 | Static host decisions, traced branches, and optimizer-lane decisions need one classification model. |
| Numerical shape/stability | T2.8, T2.9, T3.5 | Phase 5 | Tolerance helpers, lane artifacts, and replay diagnostics must preserve independent oracles. |
| Closeout docs | Appendix/status sections | Phase 6 | Source docs should be updated only after code and validation evidence exists. |

## Implementation Plan

1. Preflight and evidence lock
   - [ ] Record `git status --short`, `git rev-parse --short HEAD`, and active branch.
   - [ ] Confirm TORAX reference checkout and HEAD if TORAX-derived claims will be touched.
   - [ ] Re-run source-doc path checks for all `path:line` refs used by the chosen slice.
   - [ ] Run caller inventories before deleting or folding any symbol.
   - [ ] Decide one execution slice and explicitly name its owner source doc.

2. Contract-first foundation
   - [ ] Start with TORAX Phase 1 only where it directly supports bloat-plan work.
   - [ ] Prove current pytree data/meta behavior before changing spec helpers.
   - [ ] Keep static metadata explicit and immutable; do not hide `data_fields` / `meta_fields` in a broad abstraction.
   - [ ] If adding a helper, require it to reduce repeated partition declarations and keep fields auditable.

3. Cache, transfer, and runtime proof
   - [ ] Complete persistent-cache write/reuse proof before claiming cache-policy hardening.
   - [ ] Keep process-local JIT cache proof separate from persistent-cache proof.
   - [ ] For transfer-sensitive changes, run strict-transfer proof on the relevant backend lane.
   - [ ] Keep MPS smoke evidence separate from CPU/CUDA parity evidence.

4. Low-risk bloat reduction
   - [ ] Execute Tier 1 bloat items after the preflight caller inventory.
   - [ ] Prefer import/export list consolidation, dead private helper deletion, and already-validated no-op cleanup before factories.
   - [ ] Preserve public compatibility kwargs and probe scripts unless caller migration is proven.
   - [ ] Commit or review each item as a bisectable slice.

5. Shared factory and loop consolidation
   - [ ] Design factory work twice before changing Tier 2 or Tier 3 surfaces.
   - [ ] Pilot done-gated scan deduplication on one PM/wireframe pair before broader rollout.
   - [ ] Keep independent oracle assertions named and separate even when surrounding setup is deduplicated.
   - [ ] Stop any abstraction that adds a second source of truth for tolerances, schemas, or backend modes.

6. Branch and optimizer decisions
   - [ ] Classify each branch as static host decision, traced runtime control flow, or explicit host-boundary work.
   - [ ] Keep `scipy-jax` / `scipy-jax-fullgraph` as a documented lane decision unless new evidence changes the source plan.
   - [ ] Treat QFM BFGS/SLSQP decisions as behavior-contract decisions, not mechanical dedupe.
   - [ ] Require tests that prove branch semantics, not just reduced branch count.

7. Numerical and parity closeout
   - [ ] Run the validation gate named by the bloat-plan tier and the TORAX-plan phase.
   - [ ] Replay parity-sensitive gates when touching Stage 2, single-stage, tolerance, or lane-artifact code.
   - [ ] Update source docs only with evidence-backed status changes.
   - [ ] Leave unresolved work unchecked and explain blockers directly.

## Validation Plan

- [x] `git diff --check -- docs/bloat_reduction_plan_2026-05-20.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md`
- [x] For every implementation slice: `git grep` or `rg` all changed/deleted symbols across `src`, `tests`, `benchmarks`, `examples`, `docs`, `.github`, scripts, and artifacts that are part of the repo contract.
- [x] For import/export or lazy-loading changes: run the relevant import smoke plus `from simsopt.<package> import *` smoke.
- [x] For field/geometry/backend-sensitive changes: run the bloat-plan Tier 1 or Tier 2 test gate exactly as scoped in the source plan.
- [x] For persistent-cache work: run a two-process cache reuse proof with a shared temporary cache directory.
- [ ] For transfer-sensitive work: run strict-transfer proof on the backend lane that the changed code affects.
- [ ] For parity-sensitive work: run the Stage 2 and single-stage gates named in the bloat plan before closing the item.

## Execution Slice Log

### 2026-06-01 — TORAX Phase 1/2 contract-first test slice

- **Owner source doc:** `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md`.
- **Selected slice:** static/dynamic JAX spec contract tests plus persistent-cache two-process reuse proof; no runtime helper, scan helper, lazy-export refactor, or bloat LOC item was attempted.
- **Changed files:** `tests/core/test_jax_core_specs.py`, `tests/subprocess/import_smoke_cases.py`, `tests/test_jax_import_smoke.py`, plus evidence updates in this plan set.
- **Caller/inventory evidence:** refreshed `register_dataclass|data_fields|meta_fields|static_arg|static_argnames` and `persistent_cache|compilation_cache|XLA_FLAGS|JAX_COMPILATION_CACHE` inventories before the slice. At this checkpoint, lazy-export explorer confirmed bloat T1.1/T1.2 remained open because several `jax_core` modules still lacked literal `__all__` and `backend.runtime` still lacked a public `__all__`.
- **Validation evidence:** CPU-only contract proof, not GPU or MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/test_jax_import_smoke.py -k 'jax_core_specs or persistent_cache'
# 19 passed, 119 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/jax_core/test_tree_signature.py
# 32 passed
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

- **Review evidence:** requirements-e2e Crucible Phase 1 ran six read-only lenses over the three-file test diff. All lenses returned PASS; no findings survived to scoring. A separate lazy-export explorer confirmed T1.1/T1.2 are still open and identified the prerequisite literal `__all__` work.
- **Remaining work at this checkpoint:** T1 bloat items were still unchecked; transfer-sensitive, branch/JAXPR, bounded-scan, numerical-stability, and full bloat tiers remained open.

### 2026-06-01 — T1.2 backend facade SSOT collapse

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.2.
- **Selected slice:** backend public facade collapse only. No `jax_core` lazy-export conversion, runtime policy behavior change, cache/transfer change, or GPU/MPS path change was attempted.
- **Changed files:** `src/simsopt/backend/runtime.py`, `src/simsopt/backend/__init__.py`, `tests/test_backend.py`, plus evidence updates in this plan set.
- **Caller/inventory evidence:** current-tree `rg` confirmed public backend imports across `src`, `tests`, and integration tests; the load-bearing public-helper smoke remains `tests/subprocess/import_smoke_cases.py::case_public_jax_helpers_are_exposed_on_package_roots`.
- **Validation evidence:** CPU-only import/API-surface proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src .conda/jax/bin/python -m pytest -q tests/test_backend.py::test_backend_public_facade_uses_runtime_all_as_ssot tests/test_backend.py::test_backend_module_guard_restores_original_backend_modules
# 2 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_backend.py -k 'eager_jax_import or compilation_cache or cuda_determinism or gpu_memory or backend_public_facade or backend_module_guard'
# 28 passed, 101 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py -k 'package_root or backend_selection or public_jax_helpers'
# 8 passed, 116 deselected
PYTHONPATH=src .conda/jax/bin/python -c "from simsopt.backend import *; import simsopt.backend as backend; import simsopt.backend.runtime as runtime; assert tuple(backend.__all__) == tuple(runtime.__all__); assert len(backend.__all__) == 54; assert callable(get_backend_config); print(len(backend.__all__))"
# 54
.conda/jax/bin/python -m ruff check src/simsopt/backend/runtime.py src/simsopt/backend/__init__.py tests/test_backend.py
# passed
.conda/jax/bin/python -m ruff format --check src/simsopt/backend/runtime.py src/simsopt/backend/__init__.py tests/test_backend.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/backend/runtime.py src/simsopt/backend/__init__.py tests/test_backend.py
# blocked: No module named mypy
git diff --check -- src/simsopt/backend/runtime.py src/simsopt/backend/__init__.py tests/test_backend.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

- **Review evidence:** requirements-e2e Crucible Phase 1 ran six read-only lenses. Two findings were confirmed and fixed: the initial facade test used a re-export identity oracle rejected by `tests/REVIEWER_ORACLE_LINT.md`, and the LOC stat drifted after the test edit. Delta-first Crucible review then returned PASS twice, confirming the literal 54-name export oracle and corrected `59 insertions(+), 115 deletions(-)` production / `129 insertions(+), 115 deletions(-)` source-test stats. Mistake-book update: 0 new entries, 0 updated; no `shared/mistake-book.md` exists and the confirmed test-quality pattern was already recorded in `tests/REVIEWER_ORACLE_LINT.md`.
- **Remaining work at this checkpoint:** T1.1 remained open here; see the next log entry for its later completion. Transfer-sensitive, branch/JAXPR, bounded-scan, numerical-stability, and larger bloat tiers remained open.

### 2026-06-01 — T1.1 `jax_core` lazy facade collapse

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.1.
- **Selected slice:** `jax_core` package lazy facade only. No spec registration helper, backend runtime behavior change, transfer/cache policy change, CUDA/MPS path change, scan helper, or numerical-kernel refactor was attempted.
- **Changed files:** `src/simsopt/_lazy_exports.py`, `src/simsopt/jax_core/__init__.py`, `src/simsopt/jax_core/{curve_geometry,field,interpolated_boozer_field,objectives_flux,specs,surface_fourier,surface_henneberg,surface_rzfourier}.py`, `src/simsopt/geo/framedcurve.py`, `tests/subprocess/import_smoke_cases.py`, `tests/test_jax_import_smoke.py`, `tests/test_lazy_exports.py`, plus evidence updates in this plan set.
- **Caller/inventory evidence:** current-tree export inventory parsed the old `_EXPORT_MODULES` from `HEAD:src/simsopt/jax_core/__init__.py` and checked every lazy source module for literal `__all__`. The live implementation probe corrected the earlier six-module prerequisite to seven modules by finding `field.py` also lacked literal `__all__`.
- **Compatibility evidence:** old and new package export counts are both 314, the old/new export order is equal, every old public export resolves on the new lazy facade, and `from simsopt.jax_core import *` exposes the same ordered name sequence.
- **Validation evidence:** CPU-only import/API-surface proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_lazy_exports.py tests/test_jax_import_smoke.py::test_jax_core_lazy_facade_public_contract
# 3 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py -k 'jax_core'
# 4 passed, 121 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/jax_core/test_tree_signature.py
# 32 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_lazy_exports.py tests/test_jax_import_smoke.py -k 'jax_core or package_root or public_jax_helpers'
# 10 passed, 117 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python - <<'PY'
import simsopt.field as field
import simsopt.geo as geo
import simsopt.jax_core as jax_core
import simsopt.mhd as mhd
import simsopt.solve as solve
import simsopt.util as util
print(len(field.__all__), len(geo.__all__), len(jax_core.__all__), len(mhd.__all__), len(solve.__all__), len(util.__all__))
PY
# 91 156 314 39 47 29
.conda/jax/bin/python -m ruff check src/simsopt/_lazy_exports.py src/simsopt/geo/framedcurve.py src/simsopt/jax_core/__init__.py src/simsopt/jax_core/curve_geometry.py src/simsopt/jax_core/field.py src/simsopt/jax_core/interpolated_boozer_field.py src/simsopt/jax_core/objectives_flux.py src/simsopt/jax_core/specs.py src/simsopt/jax_core/surface_fourier.py src/simsopt/jax_core/surface_henneberg.py src/simsopt/jax_core/surface_rzfourier.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py tests/test_lazy_exports.py
# passed
.conda/jax/bin/python -m ruff format --check src/simsopt/_lazy_exports.py src/simsopt/geo/framedcurve.py src/simsopt/jax_core/__init__.py src/simsopt/jax_core/curve_geometry.py src/simsopt/jax_core/field.py src/simsopt/jax_core/interpolated_boozer_field.py src/simsopt/jax_core/objectives_flux.py src/simsopt/jax_core/specs.py src/simsopt/jax_core/surface_fourier.py src/simsopt/jax_core/surface_henneberg.py src/simsopt/jax_core/surface_rzfourier.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py tests/test_lazy_exports.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/_lazy_exports.py src/simsopt/geo/framedcurve.py src/simsopt/jax_core/__init__.py src/simsopt/jax_core/curve_geometry.py src/simsopt/jax_core/field.py src/simsopt/jax_core/interpolated_boozer_field.py src/simsopt/jax_core/objectives_flux.py src/simsopt/jax_core/specs.py src/simsopt/jax_core/surface_fourier.py src/simsopt/jax_core/surface_henneberg.py src/simsopt/jax_core/surface_rzfourier.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py tests/test_lazy_exports.py
# blocked: No module named mypy
git diff --check -- src/simsopt/_lazy_exports.py src/simsopt/geo/framedcurve.py src/simsopt/jax_core/__init__.py src/simsopt/jax_core/curve_geometry.py src/simsopt/jax_core/field.py src/simsopt/jax_core/interpolated_boozer_field.py src/simsopt/jax_core/objectives_flux.py src/simsopt/jax_core/specs.py src/simsopt/jax_core/surface_fourier.py src/simsopt/jax_core/surface_henneberg.py src/simsopt/jax_core/surface_rzfourier.py tests/subprocess/import_smoke_cases.py tests/test_jax_import_smoke.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
git diff --no-index --check /dev/null tests/test_lazy_exports.py >/dev/null; rc=$?; test "$rc" -eq 1
# passed
```

- **Review evidence:** initial requirements-e2e Crucible Phase 1 found three T1.1 issues: the export test did not independently pin the full export set, the alias assertion used a prohibited re-export identity oracle, and the first lazy-facade draft regressed historical `__all__` order. Delta review then found same-module duplicate exports could still be masked by `package_export_order`; the helper now raises on those duplicates, `tests/test_lazy_exports.py` pins the case, and the existing duplicate `FramedCurve` module export was removed. Final delta-first review returned PASS with no concrete findings.
- **Remaining work at this checkpoint:** T1.3+ mechanical bloat items remained open here; see the next log entry for its later completion. Target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remained open.

### 2026-06-01 — T1.3 GPMO public result wrapper collapse

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.3.
- **Selected slice:** solve-level GPMO result wrappers only. No PM relax-and-split result change, core GPMO algorithm change, backend/cache/transfer policy change, CUDA/MPS path change, scan helper, or numerical-kernel refactor was attempted.
- **Changed files:** `src/simsopt/solve/permanent_magnet_optimization_jax.py`, `tests/solve/test_permanent_magnet_optimization_jax_item28.py`, plus evidence updates in this plan set.
- **Caller/inventory evidence:** current-tree grep found the five solve-level GPMO public result mirrors and confirmed `PMRelaxAndSplitResult` is a separate relax-and-split result, not a `jax_core.pm_optimization` mirror. Existing wrapper callers access result fields (`m`, `m_history`, `x`, `residual_history`, `selected_*`) through returned objects rather than importing the old result classes directly.
- **Compatibility evidence:** the old public names remain real classes, not aliases. Legacy keyword and positional constructors still build the matching core result, legacy pickle-style state without `core_result` is restored, frozen assignment behavior is preserved for delegated and new attributes, `GPMOPublicResult` is exported, and pytree flatten order is pinned as `m`, `m_history`, then the nested core result leaves.
- **Validation evidence:** CPU-only PM/import/API-surface proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_permanent_magnet_optimization_jax_item28.py -k 'gpmo_public_result'
# 2 passed, 45 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_permanent_magnet_optimization_jax_item28.py
# 47 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_runtime_dtype_policy.py -k 'pm_grid_and_solve_wrappers_follow_runtime_policy_dtype'
# 2 passed, 36 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py -k 'public_jax_helpers_are_exposed_on_package_roots'
# 1 passed, 124 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu .conda/jax/bin/python - <<'PY'
import dataclasses
import simsopt.solve as solve
import simsopt.solve.permanent_magnet_optimization_jax as pmo
print('exports', 'GPMOPublicResult' in pmo.__all__, 'GPMOPublicResult' in solve.__all__)
print('fields', tuple(field.name for field in dataclasses.fields(pmo.GPMOBaselineResult)))
print('classes', pmo.GPMOBaselineResult is pmo.GPMOPublicResult, pmo.GPMOBaselineResult.__name__)
PY
# exports True True
# fields ('m', 'm_history', 'core_result')
# classes False GPMOBaselineResult
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/solve/permanent_magnet_optimization_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/solve/permanent_magnet_optimization_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/solve/permanent_magnet_optimization_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/solve/permanent_magnet_optimization_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py
# blocked: No module named mypy
git diff --check -- src/simsopt/solve/permanent_magnet_optimization_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py
# passed
```

- **Review evidence:** the first read-only delta review found a real compatibility break: aliasing all old names to `GPMOPublicResult` dropped old public constructor shape and old pickle-state compatibility. The second review found that `__getattr__` delegation made delegated fields assignable, weakening the old frozen dataclass behavior. Both issues were fixed; final scoped review returned PASS with no concrete behavioral regressions.
- **Remaining work at this checkpoint before the next T1.4 slice:** T1.4+ mechanical bloat items, target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T1.4 host float64 boundary centralization

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.4.
- **Selected slice:** `_as_numpy_float64` host-boundary centralization only. No curve geometry algorithm change, curve-objective semantics change, surface-distance reduction change, backend/cache policy change, CUDA/MPS path change, scan helper, or numerical-kernel refactor was attempted.
- **Changed files:** `src/simsopt/_core/jax_host_boundary.py`, `src/simsopt/geo/curve.py`, `src/simsopt/geo/curveobjectives.py`, `src/simsopt/geo/curvecwsfourier.py`, `src/simsopt/geo/surfaceobjectives.py`, `tests/test_host_boundary.py`, plus evidence updates in this plan set.
- **Caller/inventory evidence:** current-tree grep found four local `_as_numpy_float64` copies in `geo/curve.py`, `geo/curveobjectives.py`, `geo/curvecwsfourier.py`, and `geo/surfaceobjectives.py`. After migration, only `geo/curve.py` keeps a one-line wrapper so it can pass `_HAS_JAX=False` without requiring a JAX import. `curveobjectives_jax.py` continues to import the curve-objective compatibility name from `curveobjectives.py`; no extra duplicate helper remains there.
- **Compatibility evidence:** `host_float64(value, has_jax=False)` materializes NumPy `float64` without calling `_require_jax()`. Default JAX-enabled calls route through `host_array(..., dtype=np.float64)`, preserving explicit `transfer_guard_device_to_host("allow")` materialization for strict transfer-guard callers. Production code changed by `16 insertions(+), 32 deletions(-)`; source/test changed by `36 insertions(+), 32 deletions(-)`.
- **Validation evidence:** CPU-only host-boundary, curve-objective, CurveCWSFourier, and surface-distance transfer-guard proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/_core/jax_host_boundary.py src/simsopt/geo/curve.py src/simsopt/geo/curveobjectives.py src/simsopt/geo/curvecwsfourier.py src/simsopt/geo/surfaceobjectives.py tests/test_host_boundary.py
# 6 files already formatted
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/_core/jax_host_boundary.py src/simsopt/geo/curve.py src/simsopt/geo/curveobjectives.py src/simsopt/geo/curvecwsfourier.py src/simsopt/geo/surfaceobjectives.py tests/test_host_boundary.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/_core/jax_host_boundary.py src/simsopt/geo/curve.py src/simsopt/geo/curveobjectives.py src/simsopt/geo/curvecwsfourier.py src/simsopt/geo/surfaceobjectives.py tests/test_host_boundary.py
# passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_host_boundary.py
# 7 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py -k 'surface_surface_distance_smoke or curvecwsfouriercpp or legacy_curve_objective'
# 17 passed, 108 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_curve_objectives_jax.py
# 12 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_objectives_jax.py -k 'surface_to_surface_distance_chunking_matches_dense_value_and_grad or surface_to_surface_chunked_gradient_respects_strict_transfer_guard'
# 2 passed, 332 deselected
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/_core/jax_host_boundary.py src/simsopt/geo/curve.py src/simsopt/geo/curveobjectives.py src/simsopt/geo/curvecwsfourier.py src/simsopt/geo/surfaceobjectives.py tests/test_host_boundary.py
# blocked: No module named mypy
git diff --check -- src/simsopt/_core/jax_host_boundary.py src/simsopt/geo/curve.py src/simsopt/geo/curveobjectives.py src/simsopt/geo/curvecwsfourier.py src/simsopt/geo/surfaceobjectives.py tests/test_host_boundary.py
# passed
```

- **Review evidence:** scoped adversarial T1.4 review found no actionable code findings after checking helper centralization, no-JAX `curve.py` behavior, strict transfer-boundary behavior, import aliases, and test quality. It did find this execution log was stale relative to the source bloat plan; this T1.4 entry and the open-question refresh are the fix.
- **Remaining work at this checkpoint before the next T1.5 slice:** T1.5+ mechanical bloat items, target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T1.5 state-token factory centralization

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.5.
- **Selected slice:** state-token counter centralization only. No Biot-Savart field evaluation behavior, Boozer solve behavior, cache-key semantics, dependency invalidation contract, backend/cache policy, CUDA/MPS path, scan helper, or numerical-kernel refactor was attempted.
- **Changed files:** `src/simsopt/_core/state_tokens.py`, `src/simsopt/field/biotsavart_jax_backend.py`, `src/simsopt/geo/boozersurface_jax.py`, `tests/core/test_state_tokens.py`, plus evidence updates in this plan set.
- **Design evidence:** two options were considered. A single global token stream would remove slightly more setup but would change current per-domain token sequences. The chosen helper, `make_state_token_factory()`, returns independent monotonic token generators, preserving the old separate Biot-Savart and Boozer streams while centralizing the duplicated `itertools.count()` wrapper mechanics.
- **Caller/inventory evidence:** current-tree grep found `_new_coil_dof_state_token()` used at Biot-Savart initialization and coil-DOF mutation sites, and `_new_traceable_solve_state_token()` used at Boozer initialization, solver-generation advancement, and `recompute_bell()`. Existing public/private token attributes remain `_coil_dof_state_token`, `_traceable_solve_state_token`, `_dof_layout_version`, and `_points_version`.
- **Compatibility evidence:** `_new_coil_dof_state_token` and `_new_traceable_solve_state_token` remain module-local callables. Each factory returns integer tokens beginning at zero and advancing monotonically, so token inequality and cache-drift tests keep the same semantics. Owner modules changed by `4 insertions(+), 12 deletions(-)` (`-8` net), but the production slice is net `+8` after the 16-line helper; source/test is net `+18` after the 10-line regression test.
- **Validation evidence:** CPU-only state-token/import proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/_core/state_tokens.py src/simsopt/field/biotsavart_jax_backend.py src/simsopt/geo/boozersurface_jax.py tests/core/test_state_tokens.py
# 4 files already formatted
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/_core/state_tokens.py src/simsopt/field/biotsavart_jax_backend.py src/simsopt/geo/boozersurface_jax.py tests/core/test_state_tokens.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/_core/state_tokens.py src/simsopt/field/biotsavart_jax_backend.py src/simsopt/geo/boozersurface_jax.py tests/core/test_state_tokens.py
# passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_state_tokens.py
# 1 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_biotsavart_jax.py -k 'coil_dof_state_token or dof_layout_version'
# 3 passed, 49 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_instantiation_assigns_traceable_solve_state_token tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_instantiation_registers_surface_and_label_invalidation_sources tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_recompute_bell
# 3 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py -k 'package_root or public_jax_helpers'
# 6 passed, 119 deselected
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/_core/state_tokens.py src/simsopt/field/biotsavart_jax_backend.py src/simsopt/geo/boozersurface_jax.py tests/core/test_state_tokens.py
# blocked: No module named mypy
git diff --check -- src/simsopt/field/biotsavart_jax_backend.py src/simsopt/geo/boozersurface_jax.py
# passed
git diff --no-index --check /dev/null src/simsopt/_core/state_tokens.py; rc=$?; test "$rc" -eq 1
# passed
git diff --no-index --check /dev/null tests/core/test_state_tokens.py; rc=$?; test "$rc" -eq 1
# passed
```

- **Review evidence:** scoped adversarial T1.5 review returned PASS with no actionable issues. The reviewer inspected the untracked helper/test directly, compared migrated call sites against HEAD, checked the `_core.state_tokens` import boundary and docs evidence, and verified independent Biot-Savart and Boozer token streams are preserved. The only observed delta is private callable metadata (`__name__ == "new_state_token"`), with no in-repo compatibility consumer.
- **Remaining work:** T1.6+ mechanical bloat items, target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T1.6 Biot-Savart current-derivative wrapper dedupe

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.6.
- **Selected slice:** Biot-Savart current-derivative wrapper body dedupe only. No Biot-Savart kernel math, coil grouping, sharding policy, backend/cache policy, CUDA/MPS path, scan helper, or numerical-kernel refactor was attempted.
- **Changed files:** `src/simsopt/field/biotsavart_jax_backend.py`, `tests/field/test_biotsavart_jax.py`, plus evidence updates in this plan set.
- **Design evidence:** two options were considered. A method-to-kernel mapping could remove the six wrapper methods but would hide the CPU-compatible public names and defaults behind indirect lookup. The chosen private mixin helper, `_per_coil_unit_current_derivative()`, centralizes only the repeated `self._points_jax`/`self.coil_set_spec()` binding while keeping all six public methods and `compute_derivatives` defaults explicit.
- **Caller/inventory evidence:** current-tree grep found exactly six JAX current-derivative methods carrying `compute_derivatives`, all on `_BiotSavartFieldEvaluationMixin`, shared by `BiotSavartJAX` and `SpecBackedBiotSavartJAX`. CPU `src/simsopt/field/biotsavart.py` still exposes the same public keyword/default pattern.
- **Compatibility evidence:** `BiotSavartJAX` and `SpecBackedBiotSavartJAX` still expose `dB_by_dcoilcurrents(compute_derivatives=0)`, `d2B_by_dXdcoilcurrents(compute_derivatives=1)`, `d3B_by_dXdXdcoilcurrents(compute_derivatives=2)`, `dA_by_dcoilcurrents(compute_derivatives=0)`, `d2A_by_dXdcoilcurrents(compute_derivatives=1)`, and `d3A_by_dXdXdcoilcurrents(compute_derivatives=2)`. The regression checks the defaults with `inspect.signature()` and calls every method with `compute_derivatives=0`, `1`, and `2` on both adapters.
- **LOC evidence:** the T1.6 source body adds the 8-line helper and replaces six 5-line call blocks with six 1-line calls (`14 insertions(+), 30 deletions(-)`, `-16` net). The focused regression adds 62 test lines. Whole-file `git diff --numstat` against HEAD also includes prior uncommitted T1.5 edits in `biotsavart_jax_backend.py`.
- **Validation evidence:** CPU-only signature/current-derivative proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/field/biotsavart_jax_backend.py tests/field/test_biotsavart_jax.py
# 2 files already formatted
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/field/biotsavart_jax_backend.py tests/field/test_biotsavart_jax.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/field/biotsavart_jax_backend.py tests/field/test_biotsavart_jax.py
# passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_biotsavart_jax.py::TestBiotSavartJAXCoilStateToken::test_current_derivative_methods_preserve_compute_derivatives_keyword_contract
# 1 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_biotsavart_jax.py -k 'current_derivative_methods_preserve_compute_derivatives_keyword_contract or dB_by_dcoilcurrents_parity_ncsx or per_coil_unit_field_contract_under_coil_group_sharding or per_coil_unit_field_vectorizes_within_quadrature_group'
# 4 passed, 49 deselected
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/field/biotsavart_jax_backend.py tests/field/test_biotsavart_jax.py
# blocked: No module named mypy
git diff --check -- src/simsopt/field/biotsavart_jax_backend.py tests/field/test_biotsavart_jax.py
# passed
```

- **Review evidence:** scoped six-lens adversarial T1.6 review returned PASS with no actionable issues. Reviewers checked AGENTS/SOFTWARE_DESIGN compliance, diff-only bug risk, public CPU/JAX/API compatibility, test quality, docs/evidence consistency, and design/runtime behavior. The review independently confirmed all six `compute_derivatives` public signatures/defaults on both adapters, compared the CPU API and official SIMSOPT docs, judged the test non-tautological because it executes real `BiotSavartJAX` and `SpecBackedBiotSavartJAX` methods, and found the private helper justified because it binds shared mixin state without changing JAX tracing/cache/threading behavior. One reviewer additionally ran all six CPU-vs-JAX current-derivative parity tests (`6 passed`). No findings survived.
- **Remaining work:** T1.7+ mechanical bloat items, target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T1.7 private optimizer dead host-helper deletion

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.7.
- **Selected slice:** dead host helper deletion in `src/simsopt/geo/optimizer_jax_private/_common.py` only. No JAX line-search math, host optimizer behavior, callback policy, backend/cache policy, CUDA/MPS path, scan helper, or numerical-kernel refactor was attempted.
- **Changed files:** `src/simsopt/geo/optimizer_jax_private/_common.py`, plus evidence updates in this plan set.
- **Caller/inventory evidence:** current-tree `rg` found `_host_cubicmin`, `_host_quadmin`, and `_line_search_sample_valid_host` only in docs after deletion: the bloat source plan and this execution log. `git grep` against HEAD showed the three names existed only as definitions in `_common.py` plus the source-plan mention; no code, tests, examples, or benchmarks called them. Package exports in `src/simsopt/geo/optimizer_jax_private/__init__.py` expose the live `_cubicmin`, `_quadmin`, and `_line_search` path, not the deleted host copies.
- **Compatibility evidence:** the live JAX line-search implementation still uses `_cubicmin`, `_quadmin`, and `_line_search_sample_valid`. The live host optimizer still owns its separate equivalents in `src/simsopt/geo/optimizer_host_lbfgs.py`.
- **LOC evidence:** deleted 47 lines from `_common.py`.
- **Validation evidence:** CPU-only private optimizer/import proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/geo/optimizer_jax_private/_common.py
# 1 file already formatted
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/geo/optimizer_jax_private/_common.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/geo/optimizer_jax_private/_common.py
# passed
rg -n "_host_cubicmin|_host_quadmin|_line_search_sample_valid_host" src tests docs benchmarks examples
# docs-only references remain in the bloat source plan and this execution log; no source/test/benchmark/example references
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_optimizer_jax_private_package_has_no_private_jax_src_usage
# 1 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax_private.py -k 'line_search_value_and_grad or line_search_zoom or line_search_promotes_integer_inputs_to_inexact_dtype'
# 8 passed, 96 deselected
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/geo/optimizer_jax_private/_common.py
# blocked: No module named mypy
git diff --check -- src/simsopt/geo/optimizer_jax_private/_common.py
# passed
```

- **Review evidence:** scoped six-lens adversarial T1.7 review returned PASS after one minor docs wording finding was fixed. Reviewers checked AGENTS/SOFTWARE_DESIGN compliance, diff-only reachability, import/export/API compatibility, docs/evidence consistency, test-quality adequacy, and design/runtime behavior. The review confirmed the deleted names were old definitions plus plan text only, current source/test/benchmark/example search has no references, live JAX line search still uses `_cubicmin`, `_quadmin`, and `_line_search_sample_valid`, host `optimizer_host_lbfgs.py` keeps its separate equivalents, package exports never exposed the deleted host copies, and validation covers the active private line-search/import surface. No findings remain.
- **Remaining work:** T1.8+ mechanical bloat items, target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T1.8 Biot-Savart dead alias deletion

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.8.
- **Selected slice:** dead alias deletion in `src/simsopt/field/biotsavart_jax_backend.py` only. No Biot-Savart kernel math, coil cotangent projection, profile timing, public derivative API, backend/cache policy, CUDA/MPS path, or transfer policy was changed.
- **Changed files:** `src/simsopt/field/biotsavart_jax_backend.py`, plus evidence updates in this plan set.
- **Caller/inventory evidence:** current-tree `rg` found `_ones_like_float64` only in docs after deletion: the bloat source plan, this execution log, and an older JAX-MPS float32 parity note about a different `jax_core/curve_geometry.py` rename. `git grep` against HEAD showed the name existed only as the old definition in `biotsavart_jax_backend.py` plus those docs. `_zero_profile_component_timings` remains live (`:257`, caller at `:2222`) and the `*_cotangents` aliases remain live native/public names (`:1957`, `:2008`, `:2017`, `:2026`).
- **Compatibility evidence:** the deletion removed only a private unused helper between `_take_positions_1d()` and `_scatter_free_values()`; no package export, public class method, compatibility kwarg, or active native pullback/cotangent path changed.
- **LOC evidence:** deleted the 6-line `_ones_like_float64` helper. The full file diff also includes earlier T1.6 edits in the same file, so `git diff --numstat` is not a pure T1.8 counter.
- **Validation evidence:** CPU-only dead-code/import proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/field/biotsavart_jax_backend.py
# 1 file already formatted
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/field/biotsavart_jax_backend.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/field/biotsavart_jax_backend.py
# passed
rg -n "_ones_like_float64" src tests docs benchmarks examples
# docs-only references remain in the bloat source plan, this execution log, and an older jax_mps_float32_parity_remediation_plan note; no source/test/benchmark/example references
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_import_biotsavart_jax tests/test_jax_import_smoke.py::test_biotsavart_jax_backend_does_not_import_coil_unwrap_helper tests/test_jax_import_smoke.py::test_field_package_import_is_lazy_with_simsoptpp
# 3 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_biotsavart_jax.py::TestBiotSavartJAXCoilStateToken::test_current_derivative_methods_preserve_compute_derivatives_keyword_contract
# 1 passed
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/field/biotsavart_jax_backend.py
# blocked: No module named mypy
git diff --check -- src/simsopt/field/biotsavart_jax_backend.py
# passed
```

- **Review evidence:** scoped six-lens adversarial T1.8 review returned PASS after one docs wording finding was fixed. Reviewers checked AGENTS/SOFTWARE_DESIGN compliance, deletion reachability, public/API/import/export compatibility, docs/evidence consistency, validation adequacy, and design/runtime behavior. The review confirmed `_ones_like_float64` has no live source/test/benchmark/example references, `HEAD` had only the old private definition plus docs, `_zero_profile_component_timings` remains live, cotangent aliases remain live, public Biot-Savart exports and `compute_derivatives` compatibility are unchanged, and the validation set is adequate for a private unused-helper deletion. No findings remain.
- **Remaining work:** T1.9+ mechanical bloat items, target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T1.9 Biot-Savart runtime-spec adapter reclassification

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.9.
- **Selected slice:** public API audit and source-plan correction only. No class deletion, in-repo constructor migration, Biot-Savart kernel math, coil cotangent projection, backend/cache policy, CUDA/MPS path, or transfer policy was changed.
- **Changed files:** evidence updates in this plan set only.
- **Caller/inventory evidence:** current-tree `rg` found `SingleStageRuntimeSpecBiotSavartJAX` in the backend `__all__` (`src/simsopt/field/biotsavart_jax_backend.py:92`), class definition (`:788`), package-export tests, production single-stage example imports/calls (`single_stage_banana_example.py:146,12810`), integration fallback imports/calls (`tests/integration/test_single_stage_physics_parity.py:455,472`), historical docs, and this source plan. `git grep` against HEAD shows the same class is already public and used, not dead.
- **Compatibility evidence:** runtime probe confirmed `from simsopt.field import SingleStageRuntimeSpecBiotSavartJAX` resolves to the backend class, the object is a class, it is a subclass of `SpecBackedBiotSavartJAX`, and its signature remains `(runtime_spec: SingleStageRuntimeSpec) -> None`. Existing tests also assert package-export identity and exercise Optimizable parent/child behavior, coil-DOF updates, full-artifact curve updates, and strict-transfer cotangent projection through the class.
- **Decision:** deletion is rejected as a Tier 1 bloat item. Although current in-repo constructor calls can be written as `SpecBackedBiotSavartJAX(make_biot_savart_spec(coil_dof_extraction=runtime_spec.seed.coil_dof_extraction, coil_dofs=runtime_spec.seed.coil_dofs))`, removing the exported subclass would be a public API change and needs Tier 3 API-evolution work before any removal.
- **LOC evidence:** 0 lines saved. The earlier ~11 LOC deletion estimate is not banked.
- **Validation evidence:** CPU-only public API and current behavior proof, not CUDA or MPS proof.

```bash
PYTHONPYCACHEPREFIX=/tmp/simsopt-jax-t19-pycache PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_runtime_spec_biotsavart_full_artifact_curves_follow_updated_dofs tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_runtime_spec_biotsavart_projects_cotangents_to_owner_dofs
# 2 passed
PYTHONPYCACHEPREFIX=/tmp/simsopt-jax-t19-pycache PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python - <<'PY'
import inspect
from simsopt.field import SingleStageRuntimeSpecBiotSavartJAX as package_cls
from simsopt.field.biotsavart_jax_backend import (
    SingleStageRuntimeSpecBiotSavartJAX,
    SpecBackedBiotSavartJAX,
)
assert package_cls is SingleStageRuntimeSpecBiotSavartJAX
assert inspect.isclass(SingleStageRuntimeSpecBiotSavartJAX)
assert issubclass(SingleStageRuntimeSpecBiotSavartJAX, SpecBackedBiotSavartJAX)
print(inspect.signature(SingleStageRuntimeSpecBiotSavartJAX))
PY
# (runtime_spec: simsopt.jax_core.specs.SingleStageRuntimeSpec) -> None
rg -n "SingleStageRuntimeSpecBiotSavartJAX" src tests benchmarks examples docs
# public/exported class plus live example/test/docs references; not dead
git diff --check -- docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

- **Review evidence:** two scoped read-only T1.9 audits returned the same result. One found deleting the exported subclass would break public API compatibility because it is in module `__all__`, exported via `simsopt.field`, is a real subclass, and has package-identity tests. The other confirmed current in-repo constructor uses could be replaced behaviorally, but that doing so would still not justify deleting the public class. No code deletion was made.
- **Remaining work:** T1.10+ mechanical bloat items, target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T1.10 benchmark probe-script classification

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, section 5.10.
- **Selected slice:** caller inventory and classification for four benchmark probe scripts only. No script deletion, helper migration, benchmark behavior change, backend/cache policy change, CUDA/MPS path, or transfer policy was changed.
- **Changed files:** evidence updates in this plan set only.
- **Caller/inventory evidence:** current-tree `rg` across `src`, `tests`, `benchmarks`, `examples`, `docs`, `.github`, and `scripts` found all four scripts still referenced by live tests or user-facing benchmark/docs surfaces. `run_code_parity_probe.py` is still the solver-parity entrypoint named by `benchmarks/cpu_run_code_benchmark.py:22`, `benchmarks/gpu_run_code_benchmark.py:21`, and `benchmarks/run_code_benchmark_common.py:331`, and imported/tested by `tests/test_benchmark_helpers.py:66` / `:9662`. `production_boozer_parity_probe.py` remains in the solve-JAX caller inventory and full-repo banana plan, and is imported/tested by `tests/test_benchmark_helpers.py:65` / `:9646`. `single_stage_surface_reprojection_probe.py` is executed by `tests/test_jax_import_smoke.py:1153`. `surface_rz_geometry_hlo_probe.py` is executed through its local script path in `tests/geo/test_surface_rzfourier_jax.py:1141` / `:1148`.
- **Decision:** retain all four scripts. T1.10 is closed as a classification-only slice; future deletion requires a separate migration/deprecation change that first removes or replaces the active tests/docs surfaces.
- **LOC evidence:** 0 lines saved. The four files total 1,254 lines in the current checkout, but none are dead Tier 1 deletion candidates.
- **Validation evidence:** CPU-only caller/classification proof, not CUDA or MPS proof.

```bash
rg -n "run_code_parity_probe|production_boozer_parity_probe|single_stage_surface_reprojection_probe|surface_rz_geometry_hlo_probe" src tests benchmarks examples docs .github scripts
# live test, benchmark, and docs references for all four scripts
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_benchmark_helpers.py::test_production_boozer_probe_defaults_jax_lane_to_ondevice tests/test_benchmark_helpers.py::test_run_code_parity_probe_defaults_jax_lane_to_ondevice
# 2 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_single_stage_surface_reprojection_probe_emits_structured_cpu_result
# 1 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_rzfourier_jax.py::test_surface_rz_geometry_hlo_probe_entrypoint_uses_local_package
# 1 passed, 1 skipped
git diff --check -- docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

- **Review evidence:** two scoped read-only subagent audits completed. Deletion-safety review returned FAIL for Tier 1 deletion of all four scripts because each still has live tests or user-facing benchmark/docs references. Validation review returned PASS for classification-only closure: the focused validators plus the full `rg` inventory are sufficient to prove the probes remain active surfaces and should be retained. Both reviews agreed this is not a safe-deletion proof and not CUDA/MPS parity evidence.
- **Remaining work:** target-lane closure-capture tests, transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — TORAX Phase 1 target-lane closure-capture regression

- **Owner source doc:** `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md`, Phase 1 test matrix.
- **Selected slice:** test-only closure-capture proof for full-state target-lane objective wrappers. No runtime behavior, optimizer policy, backend/cache policy, CUDA/MPS path, or target-lane objective math was changed.
- **Changed files:** `tests/geo/test_single_stage_example.py` plus evidence updates in this plan set.
- **Contract evidence:** `test_build_target_lane_outer_objectives_full_state_keeps_closure_constants_on_host` builds the CPU-order full-graph DOF map via `build_single_stage_full_graph_jax_cpu_order_dof_map`, routes through `build_target_lane_outer_objectives`, asserts the full-state index constants remain host NumPy arrays rather than `jax.Array`, inspects the scalar and jitted value/grad wrapper closure cells for device-array leaves, and executes the lifted value/grad under `jax.transfer_guard("disallow")`. The adjacent hardware-success filter closure regression now shares the same `_contains_jax_array` helper.
- **Validation evidence:** CPU/X64 test-only proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_build_target_lane_outer_objectives_full_state_keeps_closure_constants_on_host tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_target_lane_hardware_success_filter_keeps_closure_constants_on_host tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_full_state_target_lane_value_and_grad_lifts_compact_gradient
# 3 passed
PYTHONPATH=src .conda/jax/bin/python -m ruff check tests/geo/test_single_stage_example.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check tests/geo/test_single_stage_example.py
# 1 file already formatted
PYTHONPATH=src .conda/jax/bin/python -m py_compile tests/geo/test_single_stage_example.py
# passed
git diff --check -- tests/geo/test_single_stage_example.py
# passed
```

- **Review evidence:** scoped read-only review returned PASS. The reviewer confirmed the test is production-path aligned because it uses the real full-graph DOF map builder and the real `build_target_lane_outer_objectives` entry point while patching only the expensive runtime objective builders. The reviewer also confirmed the closure assertion is paired with strict-transfer execution plus observable value/gradient checks, and found no AGENTS/SOFTWARE_DESIGN compliance issues. Residual limit: this is not a full `test_single_stage_example.py` run or an end-to-end single-stage physics solve.
- **Remaining work:** transfer-sensitive proof, branch/JAXPR classification, bounded-scan helper work, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — TORAX Phase 3 bounded-scan helper pilot

- **Owner source doc:** `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md`, Phase 3.
- **Selected slice:** private fixed-capacity scan helper plus one PM loop and one wireframe loop. No tracing/root solver rewrite, PM ArbVec/multi/backtracking rewrite, wireframe multistep/final-adjustment rewrite, optimizer policy change, backend/cache policy change, CUDA/MPS path, or numerical math change was attempted.
- **Changed files:** `src/simsopt/jax_core/_bounded_scan.py`, `src/simsopt/jax_core/pm_workflow.py`, `src/simsopt/jax_core/wireframe_workflow.py`, `tests/jax_core/test_bounded_scan.py`, plus evidence updates in this plan set.
- **Design evidence:** two options were considered. A broader helper that also validated capacity and managed history/status arrays would have hidden loop-specific invariants from PM and wireframe callers. The chosen helper, `bounded_scan_until_done`, owns only the repeated fixed-length `lax.scan` plus scalar done-gate behavior; callers still own capacity checks, status fields, history writes, and active-step semantics.
- **Pilot evidence:** `pm_gpmo_live_loop_jax` and `_gsco_live_loop_unchecked` now use the shared helper. The unpiloted PM ArbVec/multi/backtracking loops and wireframe multistep/final-adjustment scans remain explicit until a follow-up proves their status/history contracts fit the same helper.
- **Validation evidence:** CPU/X64 workflow proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_bounded_scan.py
# 4 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_pm_workflow_jax.py::test_pm_gpmo_live_loop_matches_step_by_step_host_loop tests/solve/test_pm_workflow_jax.py::test_pm_gpmo_live_loop_restart_continuation_is_exact tests/solve/test_pm_workflow_jax.py::test_pm_gpmo_live_loop_rejects_capacity_overrun_before_scan tests/solve/test_pm_workflow_jax.py::test_pm_gpmo_live_loop_jits_under_transfer_guard
# 4 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_wireframe_workflow_jax.py::test_gsco_live_loop_matches_cpp_host_loop_for_five_steps tests/solve/test_wireframe_workflow_jax.py::test_gsco_live_loop_restart_continuation_is_exact tests/solve/test_wireframe_workflow_jax.py::test_gsco_live_loop_rejects_eager_capacity_overrun tests/solve/test_wireframe_workflow_jax.py::test_gsco_live_loop_jits_under_transfer_guard
# 4 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_pm_workflow_jax.py tests/jax_core/test_bounded_scan.py
# 42 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/solve/test_wireframe_workflow_jax.py
# 12 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_bounded_scan.py tests/solve/test_pm_workflow_jax.py tests/solve/test_wireframe_workflow_jax.py
# 54 passed
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/jax_core/_bounded_scan.py src/simsopt/jax_core/pm_workflow.py src/simsopt/jax_core/wireframe_workflow.py tests/jax_core/test_bounded_scan.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/_bounded_scan.py src/simsopt/jax_core/pm_workflow.py src/simsopt/jax_core/wireframe_workflow.py tests/jax_core/test_bounded_scan.py
# 4 files already formatted
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/jax_core/_bounded_scan.py src/simsopt/jax_core/pm_workflow.py src/simsopt/jax_core/wireframe_workflow.py tests/jax_core/test_bounded_scan.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/_bounded_scan.py src/simsopt/jax_core/pm_workflow.py src/simsopt/jax_core/wireframe_workflow.py tests/jax_core/test_bounded_scan.py
# blocked: No module named mypy
git diff --check -- src/simsopt/jax_core/pm_workflow.py src/simsopt/jax_core/wireframe_workflow.py
# passed
git diff --no-index --check /dev/null src/simsopt/jax_core/_bounded_scan.py
# passed with expected no-index diff exit
git diff --no-index --check /dev/null tests/jax_core/test_bounded_scan.py
# passed with expected no-index diff exit
```

- **Remaining work:** transfer-sensitive proof, branch/JAXPR classification, numerical-stability work, follow-up classification for unpiloted bounded scans, and larger bloat tiers remain open.

### 2026-06-01 — TORAX Phase 4 branch/JAXPR pilot

- **Owner source doc:** `docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md`, Phase 4.
- **Selected slice:** test-only scalar-versus-vectorized branch classification for the MwPGP single-step hot path. No PM optimizer math, solve policy, PM workflow loop, wireframe workflow loop, backend/cache policy, CUDA/MPS path, or host-boundary behavior was changed.
- **Changed files:** `tests/jax_core/test_pm_optimization_jax_item25.py`, plus evidence updates in this plan set.
- **Classification evidence:** `src/simsopt/jax_core/pm_optimization.py::_step_body` is intentionally a traced runtime branch for array-dependent per-geometry decisions. Existing `test_step_body_uses_dynamic_branch_conditionals` keeps the scalar path at two `cond` primitives. New `test_vmap_step_body_lowers_dynamic_branches_to_selects` batches the same entry point and proves the vectorized JAXPR has zero scalar `cond` primitives and at least one `select_n`, matching the known `vmap(lax.cond)` lowering hazard documented in Phase 4.
- **Related static-host-peel evidence:** `tests/geo/test_surface_objectives_jax.py::test_cached_strict_scalar_value_and_grad_builds_stable_jit` already pins the cached strict scalar value/grad wrapper to `static_argnums=(2, 3)` for boolean compile-time choices. This pilot did not rerun that test.
- **Validation evidence:** CPU/X64 JAXPR proof, not CUDA or MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_pm_optimization_jax_item25.py::TestMwPGPSingleStep::test_step_body_uses_dynamic_branch_conditionals tests/jax_core/test_pm_optimization_jax_item25.py::TestMwPGPSingleStep::test_vmap_step_body_lowers_dynamic_branches_to_selects
# 2 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_pm_optimization_jax_item25.py::TestMwPGPSingleStep
# 4 passed
PYTHONPATH=src .conda/jax/bin/python -m ruff check tests/jax_core/test_pm_optimization_jax_item25.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check tests/jax_core/test_pm_optimization_jax_item25.py
# 1 file already formatted
PYTHONPATH=src .conda/jax/bin/python -m py_compile tests/jax_core/test_pm_optimization_jax_item25.py
# passed
git diff --check -- tests/jax_core/test_pm_optimization_jax_item25.py docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md
# passed
```

- **Remaining work:** transfer-sensitive proof, explicit host-boundary classification, dense-fallback/materialization checks, branch/JAXPR follow-up for `biotsavart.py`, `surfaceobjectives_jax.py`, `optimizer_jax.py`, PM/wireframe workflow sites beyond this pilot, numerical-stability work, and larger bloat tiers remain open.

### 2026-06-01 — T2.1 Boozer result-dict factory pilot and follow-up

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.1.
- **Selected slice:** schema-core result packing plus named public LS-Newton/exact-Newton result envelopes. Solver math, dense factorization, VJP callback construction, exact/LS reporting fields, backend selection, and result-record required/forbidden key constants were not changed.
- **Changed files:** `src/simsopt/geo/boozersurface_jax.py`, `tests/geo/test_boozersurface_jax.py`, plus this plan set.
- **Design-it-twice gate:** rejected a generic record-mode builder because it would hide exact/LS failure payload fields and centralize too much solve-specific knowledge. Landed narrow helpers for the repeated traceable core, public core, and public linearized core fields, then keyword-only LS-Newton/exact-Newton envelope factories. The information-hiding test is `test_boozer_result_core_helpers_match_schema_sources`, which ties helper key sets to the existing schema constants, fixed type/linearization invariants, and forbidden-key contracts.
- **Scope status:** partial and not LOC-banked. This is validated schema/factory hardening for T2.1, but the `~130` LOC reduction remains open because the keyword-only envelope factories prioritize auditable field mapping and leave solve-quality/reporting blocks at the owning solve paths.
- **Review verdict:** scoped adversarial review PASS for schema-key preservation, no defensive fallbacks, no dynamic imports, no `any` casts, and no solver behavior edits. The remaining concern is deliberately tracked as scope status, not a code finding: this work should not be counted as the full T2.1 LOC closeout.
- **Validation evidence:** CPU/X64 result-packaging proof, not CUDA/MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::test_public_solver_result_record_registry_is_mode_aware tests/geo/test_boozersurface_jax.py::test_boozer_result_core_helpers_match_schema_sources
# 2 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::test_public_solver_result_record_registry_is_mode_aware tests/geo/test_boozersurface_jax.py::test_boozer_result_core_helpers_match_schema_sources tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_result_dict_keys tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_exact_uses_operator_only_newton tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_ls_skip_policy_does_not_call_newton tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_ls_skips_lu_for_nonfinite_newton_result tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_functional_aliases_run_code_traceable_schema tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_exact_skips_lu_for_nonfinite_newton_result tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_newton_api_routes_without_legacy_vectorize_kwarg tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_sdofs_syncs_surface_on_ls_newton_failure tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_skip_policy_preserves_failed_ls_state_without_newton tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_invalid_newton_iterate_aborts_adjoint_state tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_unsuccessful_finite_newton_exit_aborts_adjoint_state
# 13 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py -k "result_record or result_dict_keys or run_code_traceable"
# 20 passed, 2 skipped, 457 deselected
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py
# All checks passed!
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py
# 2 files already formatted
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py
# blocked: No module named mypy
git diff --check -- src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md
# passed
```

### 2026-06-01 — T2.1 LS-Newton reporting LOC-banking follow-up

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.1.
- **Selected slice:** repeated Hessian/Newton-polish reporting-field extraction used by traceable LS success and public LS Newton failure/success result dictionaries. Solver math, dense factorization, backend strings, VJP callback construction, solve-quality override fields, success/failure branching, and result-record schema constants were not changed.
- **Changed files:** `src/simsopt/geo/boozersurface_jax.py`, `tests/geo/test_boozersurface_jax.py`, plus this plan set.
- **Design-it-twice gate:** Option A, pushing the reporting keys into `_boozer_ls_newton_result_core(...)`, was rejected because failure-path and success-path solve-quality overrides would become less visible at the owning solve sites. Option B, selected here, adds `_ls_newton_reporting_fields(...)` beside `_exact_newton_reporting_fields(...)` and uses it only where the repeated `result.get(...)` payload is identical.
- **Scope status:** T2.1 now has a small LOC-banked reporting follow-up, but the full historical `~130 LOC` target remains open. `boozersurface_jax.py` is source-negative by 37 LOC for this slice (`31 insertions / 68 deletions`; 7,255 -> 7,218 LOC). The tests tie each helper key to a distinct sentinel from `_BOOZER_HESSIAN_REPORTING_RESULT_KEYS` and prove a public LS-Newton result preserves those distinct reporting values.
- **Validation evidence:** CPU/X64 result-packaging proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::test_public_solver_result_record_registry_is_mode_aware tests/geo/test_boozersurface_jax.py::test_boozer_result_core_helpers_match_schema_sources
# 2 passed
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py::test_public_solver_result_record_registry_is_mode_aware tests/geo/test_boozersurface_jax.py::test_boozer_result_core_helpers_match_schema_sources tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_result_dict_keys tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_exact_uses_operator_only_newton tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_ls_skip_policy_does_not_call_newton tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_ls_skips_lu_for_nonfinite_newton_result tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_functional_aliases_run_code_traceable_schema tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_exact_skips_lu_for_nonfinite_newton_result tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_newton_api_routes_without_legacy_vectorize_kwarg tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_sdofs_syncs_surface_on_ls_newton_failure tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_skip_policy_preserves_failed_ls_state_without_newton tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_invalid_newton_iterate_aborts_adjoint_state tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_unsuccessful_finite_newton_exit_aborts_adjoint_state
# 13 passed
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py -k "result_record or result_dict_keys or run_code_traceable"
# 20 passed, 2 skipped, 457 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py
# All checks passed
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py
# 2 files already formatted
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/geo/boozersurface_jax.py
# pre-existing blocker: callable annotations, quadpoint signatures, local guarded redefinition, grouped-field call arity
git diff --check -- src/simsopt/geo/boozersurface_jax.py tests/geo/test_boozersurface_jax.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

### 2026-06-01 — T2.2 Boozer radial evaluator formula dedup

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.2.
- **Selected slice:** direct `boozer_radial_field.py` evaluator formula dedup plus radial Boozer RHS column reuse. No public wrapper API, CPU Boozer implementation, Fourier math, tracing integrator, backend/cache policy, CUDA/MPS path, or transfer policy was changed.
- **Changed files:** `src/simsopt/jax_core/boozer_radial_field.py`, `src/simsopt/jax_core/tracing.py`, `tests/field/test_trace_boozer_analytic_jax.py`, `tests/field/test_boozermagneticfield_jax_item33.py`, plus this plan set.
- **Design-it-twice gate:** the simple full-column wrapper was implemented first and rejected by benchmark evidence because it made standalone direct evaluators and the RHS path evaluate too many radial profiles. The landed design keeps formula ownership in `_eval_*_from_columns`, uses typed subset columns for direct evaluators, and adds `_eval_radial_rhs_columns` plus `_RADIAL_RHS_COLUMN_EVALUATORS` so radial Boozer guiding-centre RHS evaluates one column bundle per point.
- **Scope status:** formula deduped, not LOC-banked. The old `~400 LOC` T2.2 estimate is not subtracted because preserving the benchmark gate required subset-builder scaffolding (`boozer_radial_field.py` is `262 insertions / 264 deletions`; `tracing.py` adds `114 insertions / 30 deletions`). Future T2.2 LOC banking needs a separate profile-family parametrization and fresh benchmark proof.
- **Validation evidence:** CPU/X64 radial routing, column-reuse, wrapper parity, benchmark, and tracing proof, not an independent formula-oracle or CUDA/MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_trace_boozer_analytic_jax.py
# 27 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_boozermagneticfield_jax_item33.py -k "radial_columns_cached_once_per_points_cycle or direct_radial_evaluators_reuse_column_evaluators"
# 2 skipped, 19 deselected
.conda/jax/bin/python -m ruff check src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py tests/field/test_boozermagneticfield_jax_item33.py tests/field/test_trace_boozer_analytic_jax.py
# All checks passed
.conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py tests/field/test_boozermagneticfield_jax_item33.py tests/field/test_trace_boozer_analytic_jax.py
# 4 files already formatted
.conda/jax/bin/python -m py_compile src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py tests/field/test_boozermagneticfield_jax_item33.py tests/field/test_trace_boozer_analytic_jax.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py tests/field/test_trace_boozer_analytic_jax.py
# blocked: No module named mypy
git diff --check -- src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py tests/field/test_boozermagneticfield_jax_item33.py tests/field/test_trace_boozer_analytic_jax.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md
# passed
```

- **Benchmark evidence:** saved pre-change baseline was `direct_modB 0.000578229`, `direct_dmodBds 0.001090641`, `direct_G 0.000272629`, `rhs_vacuum 0.003945453`. Final five-trial medians on the same synthetic non-JIT shape were `direct_modB 0.000612696`, `direct_dmodBds 0.001073811`, `direct_G 0.000250935`, `rhs_vacuum 0.003285109`.
- **Review evidence:** direct scoped review found no added dynamic imports, untyped casts, defensive exception handling, or secret-file references in the source/test diff. The remaining issue is accounting, not correctness: this should not be counted as the full T2.2 bloat-LOC closeout.

### 2026-06-01 — T2.2 Boozer radial direct-wrapper LOC-banking follow-up

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.2.
- **Selected slice:** private direct evaluator wrapper ceremony in `src/simsopt/jax_core/boozer_radial_field.py`. No Fourier formula, subset-column factory, tracing dispatch, public wrapper API, CPU Boozer implementation, CUDA/MPS path, transfer policy, or benchmark policy was changed.
- **Changed files:** `src/simsopt/jax_core/boozer_radial_field.py`, plus this plan set.
- **Design-it-twice gate:** Option A, dynamically parametrizing all subset-column builders, is the larger remaining T2.2 opportunity but risks hiding profile-specific stellsym and derivative-factor requirements behind field-name tables. Option B, selected at this checkpoint, folded only the repeated direct private wrappers into typed `_direct_radial_evaluator(...)` and `_direct_modB_value_evaluator(...)` factories, preserving the explicit subset builders and formulas.
- **Scope status:** direct-wrapper follow-up LOC-banked, full T2.2 still open. `boozer_radial_field.py` is source-negative by 35 LOC for this slice (`99 insertions / 134 deletions`; 1,191 -> 1,156 LOC). The old `~400 LOC` estimate remains unbanked because the benchmark-sensitive subset builders are still explicit.
- **Validation evidence:** CPU/X64 wrapper routing, private evaluator metadata, source lint/format, py_compile, source-only mypy, dependency consistency, and diff hygiene proof; not CUDA/MPS proof and not a fresh RHS benchmark.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py src/simsopt/field/boozermagneticfield_jax.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/boozer_radial_field.py
# 1 file already formatted
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py src/simsopt/field/boozermagneticfield_jax.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/boozer_radial_field.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_trace_boozer_analytic_jax.py
# 27 passed
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_boozermagneticfield_jax_item33.py
# 21 skipped
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- src/simsopt/jax_core/boozer_radial_field.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

- **Review evidence:** six-lens adversarial review returned PASS after fixing docs-only stale line-anchor findings. API/behavior, test-quality, and design/performance lenses found no source findings: private evaluator metadata and import identity are preserved, JIT/Jacobian probes and pickle-by-module/name probes passed, and the helper abstraction is scoped to repeated private direct-wrapper ceremony. Docs/history/accounting lenses initially found stale anchors for the new helper locations; delta reviewers confirmed the corrected anchors match the live source.

### 2026-06-01 — T2.2 Boozer radial scalar-helper LOC-banking follow-up

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.2.
- **Selected slice:** repeated scalar direct-evaluator construction in `src/simsopt/jax_core/boozer_radial_field.py`. No Fourier formula, Fourier subset builder, tracing dispatch, public wrapper API, CPU Boozer implementation, CUDA/MPS path, transfer policy, or benchmark policy was changed.
- **Changed files:** `src/simsopt/jax_core/boozer_radial_field.py`, `tests/field/test_trace_boozer_analytic_jax.py`, plus this plan set.
- **Design-it-twice gate:** Option A, table-driving the Fourier subset builders, was rejected after adversarial review because it hid field ownership behind string-key dispatch for only a 2-LOC bank. Option B, selected here, keeps all Fourier subset builders explicit and factors only the seven scalar direct evaluators through `_direct_scalar_evaluator(...)` with typed profile selectors.
- **Scope status:** scalar-helper follow-up LOC-banked, full T2.2 still open. `boozer_radial_field.py` is source-negative by 12 LOC for this slice (`26 insertions / 38 deletions`). The old `~400 LOC` estimate remains unbanked because the larger formula/subset-family redesign has not been proven readable and benchmark-safe.
- **Validation evidence:** CPU/X64 routing/cache tests, full radial tracing tests, scalar metadata/pickle proof, source lint/format, py_compile, source-only mypy, and fresh non-JIT benchmark proof; not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/boozer_radial_field.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/boozer_radial_field.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# 3 files already formatted
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/boozer_radial_field.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile src/simsopt/jax_core/boozer_radial_field.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py -k "radial_columns_cached_once_per_points_cycle or direct_radial_evaluators_reuse_column_evaluators"
# 2 passed, 2 skipped, 45 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/field/test_trace_boozer_analytic_jax.py
# 28 passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/field/test_boozermagneticfield_jax_item33.py
# 21 skipped
```

- **Benchmark evidence:** the same `stellsym=True` synthetic non-JIT gate was rerun against the documented final medians from the direct-wrapper follow-up. Recorded five-trial medians were `direct_modB 0.000637098` (limit `0.000673966`), `direct_dmodBds 0.001141614` (limit `0.001181192`), `direct_G 0.000267225` (limit `0.000276029`), and `rhs_vacuum 0.003449284` (limit `0.003613620`). Delta review reproduced the gate in the current checkout with 25-trial medians `direct_modB 0.000594125`, `direct_dmodBds 0.001056167`, `direct_G 0.000243000`, and `rhs_vacuum 0.003481042`; all passed the no >10% regression gate.
- **Review evidence:** initial adversarial review rejected the attempted string-key `_profile_subset(...)` helper as hidden dynamic field dispatch and flagged weak same-valued optional-profile test coverage. The landed delta removes `_profile_subset(...)`, restores explicit Fourier subset builders, adds `_direct_scalar_evaluator(...)`, and gives the non-stellsym synthetic state distinct optional-mode profiles before re-running validation.

### 2026-06-01 — T2.2 Boozer radial pass-through inline follow-up

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.2.
- **Selected slice:** the one-use `_eval_with_radial_columns(...)` helper in `src/simsopt/jax_core/boozer_radial_field.py`. No Fourier formula, scalar spline path, subset-column factory, tracing dispatch, public wrapper API, CPU Boozer implementation, CUDA/MPS path, transfer policy, or benchmark policy was changed.
- **Changed files:** `src/simsopt/jax_core/boozer_radial_field.py`, plus this plan set.
- **Design-it-twice gate:** Option A, keep the helper as an interface seam for possible future column-factory changes, was rejected because `_direct_radial_evaluator(...)` is already the narrower abstraction boundary that preserves generated evaluator names and metadata. Option B, selected here, deletes the pass-through and keeps the column-factory call next to the supplied column evaluator in `_direct_radial_evaluator(...)`.
- **Scope status:** pass-through inline follow-up LOC-banked, full T2.2 still open. `boozer_radial_field.py` is source-negative by 10 LOC for this slice (`2 insertions / 12 deletions`; 1,144 -> 1,134 LOC). The old `~400 LOC` estimate remains unbanked because the larger formula/subset-family redesign has not been proven readable and benchmark-safe.
- **Validation evidence:** CPU/X64 focused radial routing/cache tests, radial direct-evaluator metadata probe, source lint/format, py_compile, source-only mypy, dependency check, source/test call-site grep, and diff whitespace checks passed; not CUDA/MPS proof.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# All checks passed!
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/boozer_radial_field.py
# 1 file already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/jax_core/boozer_radial_field.py
# passed
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/boozer_radial_field.py
# Success: no issues found in 1 source file
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/field/test_trace_boozer_analytic_jax.py -k "radial_boozer_rhs_evaluates_one_column_bundle_per_point or direct_radial_evaluators_reuse_column_evaluators or rhs_key_contract"
# 3 passed, 25 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python - <<'PY'
from simsopt.jax_core import boozer_radial_field as brf

names = [
    "_eval_modB", "_eval_dmodBdtheta", "_eval_dmodBdzeta", "_eval_dmodBds",
    "_eval_R", "_eval_dRdtheta", "_eval_dRdzeta", "_eval_dRds",
    "_eval_Z", "_eval_dZdtheta", "_eval_dZdzeta", "_eval_dZds",
    "_eval_nu", "_eval_dnudtheta", "_eval_dnudzeta", "_eval_dnuds",
    "_eval_K", "_eval_dKdtheta", "_eval_dKdzeta",
]
for name in names:
    fn = getattr(brf, name)
    assert fn.__name__ == name
    assert fn.__qualname__ == name
    assert fn.__module__ == brf.__name__
print(f"{len(names)} radial evaluator metadata entries preserved")
PY
# 19 radial evaluator metadata entries preserved
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found.
git grep -n "_eval_with_radial_columns" -- src tests benchmarks examples
# no source/test/benchmark/example call sites
git diff --check -- src/simsopt/jax_core/boozer_radial_field.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

### 2026-06-01 — T2.2 Boozer radial typed modB direct-factory follow-up

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.2.
- **Selected slice:** the modB value/theta/zeta direct-evaluator factory in `src/simsopt/jax_core/boozer_radial_field.py`. No Fourier formula, modB subset-column factory, scalar spline path, tracing dispatch, public wrapper API, CPU Boozer implementation, CUDA/MPS path, transfer policy, or benchmark policy was changed.
- **Changed files:** `src/simsopt/jax_core/boozer_radial_field.py`, plus this plan set.
- **Design-it-twice gate:** Option A, routing modB value/theta/zeta through the full `_eval_radial_columns(...)` bundle, was rejected because earlier benchmark evidence showed full-bundle direct wrappers regress the direct/RHS hot path. Option B, selected here, makes `_direct_radial_evaluator(...)` generic over its column type and keeps `_eval_modB_value_radial_columns(...)` as the subset factory, deleting only `_direct_modB_value_evaluator(...)` and its one-use `_ModBValueEvaluator` alias.
- **Scope status:** typed modB direct-factory follow-up LOC-banked, full T2.2 still open. `boozer_radial_field.py` is source-negative by 14 LOC for this slice (`24 insertions / 38 deletions`; 1,134 -> 1,120 LOC). Together with the earlier T2.2 follow-ups, T2.2 has 71 source LOC banked; the old `~400 LOC` estimate remains unbanked because the larger formula/subset-family redesign has not been proven readable and benchmark-safe.
- **Validation evidence:** CPU/X64 focused radial routing/cache tests, full radial tracing tests, direct-evaluator metadata proof, source lint/format, py_compile, source-only mypy, dependency check, source/test call-site grep, and diff whitespace checks passed; not CUDA/MPS proof and not fresh benchmark proof.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/jax_core/boozer_radial_field.py src/simsopt/jax_core/tracing.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# All checks passed!
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/boozer_radial_field.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# 3 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/jax_core/boozer_radial_field.py tests/field/test_trace_boozer_analytic_jax.py tests/field/test_boozermagneticfield_jax_item33.py
# passed
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/boozer_radial_field.py
# Success: no issues found in 1 source file
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/field/test_trace_boozer_analytic_jax.py -k "radial_boozer_rhs_evaluates_one_column_bundle_per_point or direct_radial_evaluators_reuse_column_evaluators or dispatch_exposes_complete_key_set"
# 4 passed, 24 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/field/test_boozermagneticfield_jax_item33.py -k "direct_radial_evaluators_reuse_column_evaluators or radial_columns_cached_once_per_points_cycle"
# 2 skipped, 19 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/field/test_trace_boozer_analytic_jax.py
# 28 passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python - <<'PY'
from simsopt.jax_core import boozer_radial_field as radial

names = (
    "_eval_modB", "_eval_dmodBdtheta", "_eval_dmodBdzeta", "_eval_dmodBds",
    "_eval_R", "_eval_dRdtheta", "_eval_dRdzeta", "_eval_dRds",
    "_eval_Z", "_eval_dZdtheta", "_eval_dZdzeta", "_eval_dZds",
    "_eval_nu", "_eval_dnudtheta", "_eval_dnudzeta", "_eval_dnuds",
    "_eval_K", "_eval_dKdtheta", "_eval_dKdzeta",
    "_eval_psip", "_eval_G", "_eval_I", "_eval_iota",
    "_eval_dGds", "_eval_dIds", "_eval_diotads",
)
for name in names:
    evaluator = getattr(radial, name)
    assert evaluator.__name__ == name
    assert evaluator.__qualname__ == name
    assert evaluator.__module__ == radial.__name__
print(f"metadata preserved for {len(names)} direct evaluators")
PY
# metadata preserved for 26 direct evaluators
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found.
rg -n "_ModBValueEvaluator|_direct_modB_value_evaluator" src tests benchmarks
# no source/test call sites
git diff --check -- src/simsopt/jax_core/boozer_radial_field.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

### 2026-06-01 — T2.3 surface Fourier facade factory slice

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.3.
- **Selected slice:** `surface_fourier.py` facade wrapper factory only. No lower-level Fourier formulas, `surface_fourier_kernels.py` coefficient-Jacobian code, CPU geometry code, backend/cache policy, CUDA/MPS path, transfer policy, or public symbol deletion was changed.
- **Changed files:** `src/simsopt/jax_core/surface_fourier.py`, plus this plan set.
- **Design-it-twice gate:** a broad table covering every public geometry function was rejected because composed quantities such as `normal`, fundamental forms, curvatures, area, and volume encode readable geometry composition. The landed design factors only the kernel-backed spec wrappers and paired-linear dof wrappers, preserving explicit composition where the function body carries mathematical meaning.
- **Scope status:** facade LOC-banked, full T2.3 still open. `surface_fourier.py` is source-negative by 165 LOC (`281 insertions / 446 deletions`), but the lower-level `surface_fourier_kernels.py` `_from_dofs` wrapper fold remains a separate follow-up before the old full-item `~550` estimate can be banked.
- **Validation evidence:** CPU/X64 non-RZ surface wrapper parity and JIT proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py -k 'higher_paired_lin_wrappers_match_cpp or spec_geometry_and_normals_match_cpp or high_order_spec_geometry_is_finite_and_matches_cpp or spec_second_coordinate_derivatives_match_cpp or non_rz_spec_wrappers_match_cpu or spec_wrappers_are_jittable'
# 14 passed, 136 deselected
.conda/jax/bin/python -m ruff check src/simsopt/jax_core/surface_fourier.py
# All checks passed
.conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/surface_fourier.py
# 1 file already formatted
.conda/jax/bin/python -m py_compile src/simsopt/jax_core/surface_fourier.py
# passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found.
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/surface_fourier.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy tests/geo/test_surface_fourier_jax.py
# blocked by pre-existing benchmarks/validation_ladder_contract.py type errors (18 errors) imported by the test helper path.
git diff --check -- src/simsopt/jax_core/surface_fourier.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

- **Review evidence:** scoped adversarial review returned PASS after fixing two review-found issues: transient tensor paired-linear dof wrappers were initially bound before tensor spec wrappers existed, and the T2.3 source LOC ledger still cited the stale pre-slice 909 LOC count. Delta reviewers then confirmed the refreshed `/tmp/t2_3_surface_fourier_facade.diff` byte-matches the live scoped diff, public imports/package facade/`__all__` order and paired-linear wrapper metadata are preserved, math/kernel routing is unchanged, guardrails are clean, source-only `mypy` passes, the test-file `mypy` blocker is pre-existing in `benchmarks/validation_ladder_contract.py`, and the corrected 978-to-813 LOC accounting banks only 165 source LOC.

### 2026-06-01 — T2.3 tensor surface kernel wrapper fold

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.3.
- **Selected slice:** the ten simple `SurfaceXYZTensorFourier` `surface_*_from_dofs` wrappers in `surface_fourier_kernels.py`: `gamma`, paired `gamma_lin`, first/second coordinate derivatives, and `normal`. No `SurfaceXYZFourier` scatter/template wrapper, coefficient-Jacobian wrapper, composed area/volume/unit-normal helper, CPU geometry code, backend/cache policy, CUDA/MPS path, transfer policy, or public symbol deletion was changed.
- **Changed files:** `src/simsopt/jax_core/surface_fourier_kernels.py`, plus this plan set.
- **Design-it-twice gate:** a broad factory across tensor wrappers, `SurfaceXYZFourier` wrappers, and coefficient-Jacobian wrappers was rejected because those families have different scatter/template and Jacobian signatures. The first narrow factory also was rejected after review because public introspection regressed (`__code__.co_name` and `inspect.getsource(...)`). The landed design keeps explicit public wrapper definitions and factors only their shared evaluator body through `_eval_surface_tensor_from_dofs(...)`, including `clamped_dims`.
- **Scope status:** tensor-kernel LOC-banked, full T2.3 still open. `surface_fourier_kernels.py` is source-negative by 190 LOC (`58 insertions / 248 deletions`), bringing completed T2.3 banked source reduction to 355 LOC across the prior facade slice and this tensor-kernel slice. The old full-item `~550` estimate still is not banked because `SurfaceXYZFourier` lower-level wrappers and coefficient-Jacobian families remain explicit follow-ups.
- **Validation evidence:** CPU/X64 tensor wrapper, clamping, stellsym scatter, XYZ scatter/template adjacency, paired-linear, and coefficient-Jacobian behavior proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python - <<'PY'
import inspect
from simsopt.jax_core import surface_fourier_kernels as k
expected = '(dofs, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym, scatter_indices=None, *, clamped_dims=(False, False, False))'
for name in [
    'surface_gamma_from_dofs',
    'surface_gamma_lin_from_dofs',
    'surface_gammadash1_from_dofs',
    'surface_gammadash1_lin_from_dofs',
    'surface_gammadash2_from_dofs',
    'surface_gammadash2_lin_from_dofs',
    'surface_gammadash1dash1_from_dofs',
    'surface_gammadash1dash2_from_dofs',
    'surface_gammadash2dash2_from_dofs',
    'surface_normal_from_dofs',
]:
    fn = getattr(k, name)
    assert str(inspect.signature(fn)) == expected
    assert fn.__name__ == name
    assert fn.__qualname__ == name
    assert fn.__module__ == 'simsopt.jax_core.surface_fourier_kernels'
    assert fn.__code__.co_name == name
    assert inspect.getsource(fn).lstrip().startswith(f'def {name}(')
    doc = inspect.getdoc(fn) or ''
    assert 'scatter_indices' in doc and 'clamped_dims' in doc and 'Returns' in doc
print('signature-and-introspection-preservation: PASS')
PY
# signature-and-introspection-preservation: PASS
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py tests/geo/test_surface_xyz_tensor_clamped_jax.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/surface_fourier_kernels.py
# 1 file already formatted
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py tests/geo/test_surface_xyz_tensor_clamped_jax.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/surface_fourier_kernels.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_xyz_tensor_clamped_jax.py
# 38 passed
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py -k 'coefficient_derivatives_match_cpp or second_coordinate_derivatives_match_cpp or geometry_and_tangents_match_cpp or second_coordinate_derivative_dcoeff_match_cpp or tangent_derivative_columns_match_cpp or gamma_and_tangent_lin_match_cpp or higher_paired_lin_wrappers_match_cpp or non_rz_fundamental_form_derivatives_match_cpp'
# 60 passed, 90 deselected
```

- **Known non-slice tooling note:** `ruff format --check` over the two unmodified parity test files still reports pre-existing formatting drift (`Would reformat`); source-only format for the changed file passes.
- **Review evidence:** initial API/behavior review found a real public-introspection regression in the first narrow factory draft: generated wrappers preserved signatures and names but exposed `_from_dofs` through `__code__.co_name` / `inspect.getsource(...)` and had weaker public docs. The final implementation fixed that by keeping explicit public wrappers while factoring only the shared evaluator body. Delta API/behavior, docs/accounting, and design/tooling reviewers then returned strict PASS: public signatures, `__name__`, `__qualname__`, `__module__`, `__code__.co_name`, source introspection, and docs are preserved; LOC accounting is `58 insertions / 248 deletions` (`-190`) for this source slice and `355` banked across completed T2.3 slices; the narrow `fmt: off` blocks are accepted as table-like wrapper/doc assignments; and no runtime, tooling, or scope findings remain.

### 2026-06-01 — T2.3 `SurfaceXYZFourier` unpack fold

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.3.
- **Selected slice:** only the repeated `SurfaceXYZFourier` flat-DOF to `(xc, xs, yc, ys, zc, zs)` coefficient unpack inside the nine analytic lower-level wrappers. The wrappers now call the existing `_scatter_surface_xyzfourier_dofs(...)` helper directly. No analytic derivative formula, paired derivative helper, coefficient-Jacobian factory, composed area/volume/unit-normal helper, CPU geometry code, backend/cache policy, CUDA/MPS path, transfer policy, or public symbol deletion was changed.
- **Changed files:** `src/simsopt/jax_core/surface_fourier_kernels.py`, plus this plan set.
- **Design-it-twice gate:** a formula-table factory was rejected because the explicit `SurfaceXYZFourier` derivatives carry distinct rotation/product-rule terms. A broader coefficient-Jacobian merge was rejected because Jacobian/Hessian wrappers already have separate tensor and XYZ signatures. A same-layer helper around `_scatter_surface_xyzfourier_dofs(...)` also was rejected after design review as a shallow pass-through. The landed design factors only by calling the existing coefficient-unpack SSOT from each analytic wrapper and leaves public wrappers and formulas readable.
- **Scope status:** unpack LOC-banked, full T2.3 still open. `surface_fourier_kernels.py` is source-negative by 51 LOC for this slice (`15 insertions / 66 deletions`), bringing completed T2.3 banked source reduction to 406 LOC across the facade, tensor-kernel, and XYZ unpack slices. The old full-item `~550` estimate still is not banked because `SurfaceXYZFourier` analytic formulas and coefficient-Jacobian families remain explicit follow-ups.
- **Validation evidence:** CPU/X64 `SurfaceXYZFourier` geometry, tangent, coefficient-Jacobian, paired-linear, and non-RZ fundamental-form proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python - <<'PY'
import inspect
from simsopt.jax_core import surface_fourier_kernels as k
expected = '(dofs, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym, scatter_indices=None, coeff_template=None)'
for name in [
    'surface_xyzfourier_gamma_from_dofs',
    'surface_xyzfourier_gamma_lin_from_dofs',
    'surface_xyzfourier_gammadash1_from_dofs',
    'surface_xyzfourier_gammadash1_lin_from_dofs',
    'surface_xyzfourier_gammadash2_from_dofs',
    'surface_xyzfourier_gammadash2_lin_from_dofs',
    'surface_xyzfourier_gammadash1dash1_from_dofs',
    'surface_xyzfourier_gammadash1dash2_from_dofs',
    'surface_xyzfourier_gammadash2dash2_from_dofs',
]:
    fn = getattr(k, name)
    assert str(inspect.signature(fn)) == expected
    assert fn.__code__.co_name == name
    assert inspect.getsource(fn).lstrip().startswith(f'def {name}(')
print('xyzfourier-wrapper-introspection: PASS')
PY
# xyzfourier-wrapper-introspection: PASS
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/surface_fourier_kernels.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/surface_fourier_kernels.py
# 1 file already formatted
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile src/simsopt/jax_core/surface_fourier_kernels.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/surface_fourier_kernels.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py -k 'geometry_and_tangents_match_cpp or second_coordinate_derivative_dcoeff_match_cpp or tangent_derivative_columns_match_cpp or gamma_and_tangent_lin_match_cpp or higher_paired_lin_wrappers_match_cpp or non_rz_fundamental_form_derivatives_match_cpp'
# 34 passed, 116 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py::TestSurfaceXYZFourierJaxCppParity
# 26 passed
```

- **Review evidence:** initial design/tooling review found and fixed one real issue: the first draft introduced a shallow `_surface_xyzfourier_coeffs_from_dofs(...)` pass-through around the existing `_scatter_surface_xyzfourier_dofs(...)` helper. The final implementation deletes that helper and calls the existing unpack SSOT directly from the nine explicit public wrappers. Delta API/behavior, docs/accounting, and design/tooling reviewers then returned strict PASS: public wrapper signatures/source introspection are preserved, stellsym/scatter/template routing remains on the existing helper, paired derivative and coefficient-Jacobian families are untouched, LOC accounting is `15 insertions / 66 deletions` (`-51`) for this source slice and `406` banked across completed T2.3 slices, and no runtime/tooling/scope findings remain.

### 2026-06-01 — T2.3 coefficient-derivative wrapper-family fold

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.3.
- **Selected slice:** only the repeated coefficient-derivative wrapper ceremony in `surface_fourier_kernels.py`: tensor and `SurfaceXYZFourier` `jax.jacfwd`, explicit heavy Hessian, scalar `jax.grad`, and scalar `jax.hessian` wrapper builders. No analytic derivative formula, public export name, scalar tolerance, composed geometry formula, CPU geometry code, backend/cache policy, CUDA/MPS path, transfer policy, or public symbol deletion was changed.
- **Changed files:** `src/simsopt/jax_core/surface_fourier_kernels.py`, plus this plan set.
- **Design-it-twice gate:** a broad `*args` / `**kwargs` wrapper was rejected because it would erase the public distinction between tensor derivative wrappers and `SurfaceXYZFourier` wrappers that accept `coeff_template`. Fully separate tensor/XYZ helpers were also rejected because they preserved the same repeated derivative-transform ceremony. The landed design uses one `_surface_dof_transform(...)` helper with two explicit inner signatures and small transform functions for `jax.jacfwd`, explicit Hessian, `jax.grad`, and `jax.hessian`.
- **Scope status:** coefficient-derivative wrapper LOC-banked, full T2.3 still open. `surface_fourier_kernels.py` is source-negative by 109 LOC for this slice (`52 insertions / 161 deletions`), bringing completed T2.3 banked source reduction to 515 LOC across the facade, tensor-kernel, XYZ unpack, and coefficient-derivative wrapper slices. The old full-item `~550` estimate still is not closed because the `SurfaceXYZFourier` analytic formulas remain explicit follow-ups.
- **Validation evidence:** CPU/X64 tensor and `SurfaceXYZFourier` coefficient/scalar derivative proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/surface_fourier_kernels.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/surface_fourier_kernels.py
# 1 file already formatted
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile src/simsopt/jax_core/surface_fourier_kernels.py
# passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m mypy src/simsopt/jax_core/surface_fourier_kernels.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python - <<'PY'
import inspect
from simsopt.jax_core import surface_fourier_kernels as sf
tensor_names = [
    'dgamma_by_dcoeff',
    'dgammadash1_by_dcoeff',
    'dgammadash2_by_dcoeff',
    'dgammadash1dash1_by_dcoeff',
    'dgammadash1dash2_by_dcoeff',
    'dgammadash2dash2_by_dcoeff',
    'dnormal_by_dcoeff',
    'd2normal_by_dcoeffdcoeff',
    'dunitnormal_by_dcoeff',
    'darea_by_dcoeff',
    'd2area_by_dcoeffdcoeff',
    'dvolume_by_dcoeff',
    'd2volume_by_dcoeffdcoeff',
]
xyz_names = [
    'surface_xyzfourier_dgamma_by_dcoeff',
    'surface_xyzfourier_dgammadash1_by_dcoeff',
    'surface_xyzfourier_dgammadash2_by_dcoeff',
    'surface_xyzfourier_dgammadash1dash1_by_dcoeff',
    'surface_xyzfourier_dgammadash1dash2_by_dcoeff',
    'surface_xyzfourier_dgammadash2dash2_by_dcoeff',
    'surface_xyzfourier_dnormal_by_dcoeff',
    'surface_xyzfourier_d2normal_by_dcoeffdcoeff',
    'surface_xyzfourier_dunitnormal_by_dcoeff',
    'surface_xyzfourier_darea_by_dcoeff',
    'surface_xyzfourier_d2area_by_dcoeffdcoeff',
    'surface_xyzfourier_dvolume_by_dcoeff',
    'surface_xyzfourier_d2volume_by_dcoeffdcoeff',
]
expected_tensor = '(dofs, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym, scatter_indices=None)'
expected_xyz = '(dofs, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym, scatter_indices=None, coeff_template=None)'
for name in tensor_names:
    fn = getattr(sf, name)
    assert str(inspect.signature(fn)) == expected_tensor
    assert fn.__code__.co_name == 'wrapper'
for name in xyz_names:
    fn = getattr(sf, name)
    assert str(inspect.signature(fn)) == expected_xyz
    assert fn.__code__.co_name == 'wrapper'
print('surface-derivative-wrapper-signatures: PASS')
PY
# surface-derivative-wrapper-signatures: PASS
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py -k 'coefficient_derivatives_match_cpp or second_coordinate_derivative_dcoeff_match_cpp or tangent_derivative_columns_match_cpp or normal_derivative_columns_match_cpp or dnormal_by_dcoeff_vjp_matches_cpp or d2normal_by_dcoeffdcoeff_matches_cpp or non_rz_fundamental_form_derivatives_match_cpp or scalar_derivatives_match_cpp or scalar_hessians_match_cpp or scalar_derivative_vjp_matches_cpp'
# 48 passed, 102 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py
# 150 passed
```

- **Review evidence:** six-lens adversarial review returned strict PASS after one docs-scope correction. The test-quality lens found that the first evidence wording overstated the focused selector as scalar-derivative coverage; the final docs now attribute scalar area/volume wrapper coverage to the full `tests/geo/test_surface_fourier_jax.py` run. API/behavior, design/tooling, docs/accounting, repo-guardrail, and history/comment lenses found no remaining issues: public tensor and `SurfaceXYZFourier` derivative signatures are preserved, `coeff_template` arity is intact, no `*args` / `**kwargs` escape hatch or shallow pass-through helper was introduced, LOC accounting is `52 insertions / 161 deletions` (`-109`) for this source slice and `515` banked across completed T2.3 slices, and validation evidence remains CPU/X64 only.

### 2026-06-01 — T2.3 `SurfaceXYZFourier` order-hat helper slice

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.3.
- **Selected slice:** only the repeated derivative-order to separable-basis to component-hat evaluation in the six full-grid analytic `SurfaceXYZFourier` wrappers: `gamma`, `gammadash1`, `gammadash2`, `gammadash1dash1`, `gammadash1dash2`, and `gammadash2dash2`. No paired-linear wrapper, coefficient-Jacobian wrapper, composed area/volume/unit-normal helper, CPU geometry code, backend/cache policy, CUDA/MPS path, transfer policy, public symbol, rotation term, or product-rule formula was changed.
- **Changed files:** `src/simsopt/jax_core/surface_fourier_kernels.py`, plus this plan set.
- **Design-it-twice gate:** a deeper formula-table factory was rejected because it would obscure the derivative product-rule and rotation algebra that reviewers need to audit locally. A two-helper variant (`one order` plus `many orders`) was also rejected as too shallow for this slice. The landed design uses one `_surface_xyzfourier_component_hat_derivatives(...)` helper that hides only the repeated representation rule `(phi_order, theta_order) -> separable basis -> (xhat, yhat, zhat)`, while all product-rule formulas remain in the public wrapper bodies.
- **Scope status:** order-hat helper LOC-banked, full T2.3 still open for any future product-rule formula fold. `surface_fourier_kernels.py` is source-negative by 35 LOC for this slice (`73 insertions / 108 deletions`), bringing completed T2.3 banked source reduction to 550 LOC across the facade, tensor-kernel, XYZ unpack, coefficient-derivative wrapper, and order-hat slices. The old full-item `~550` estimate is not closed as a scope claim because the remaining product-rule formulas are intentionally still explicit.
- **Validation evidence:** CPU/X64 `SurfaceXYZFourier` analytic geometry/derivative proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/surface_fourier_kernels.py
# 1 file already formatted
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/surface_fourier_kernels.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python - <<'PY'
import inspect
from simsopt.jax_core import surface_fourier_kernels as k
names = [
    'surface_xyzfourier_gamma_from_dofs',
    'surface_xyzfourier_gammadash1_from_dofs',
    'surface_xyzfourier_gammadash2_from_dofs',
    'surface_xyzfourier_gammadash1dash1_from_dofs',
    'surface_xyzfourier_gammadash1dash2_from_dofs',
    'surface_xyzfourier_gammadash2dash2_from_dofs',
]
for name in names:
    fn = getattr(k, name)
    sig = inspect.signature(fn)
    source = inspect.getsource(fn)
    assert fn.__name__ == name, name
    assert fn.__qualname__ == name, name
    assert fn.__module__ == 'simsopt.jax_core.surface_fourier_kernels', name
    assert fn.__code__.co_name == name, name
    assert source.startswith(f'def {name}('), name
    assert 'coeff_template=None' in str(sig), (name, sig)
print('surface xyz analytic wrapper introspection ok')
PY
# surface xyz analytic wrapper introspection ok
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py -k 'geometry_and_tangents_match_cpp or second_coordinate_derivative_dcoeff_match_cpp or tangent_derivative_columns_match_cpp or gamma_and_tangent_lin_match_cpp or higher_paired_lin_wrappers_match_cpp or non_rz_fundamental_form_derivatives_match_cpp'
# 34 passed, 116 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py::TestSurfaceXYZFourierJaxCppParity
# 26 passed
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_surface_fourier_jax.py
# 150 passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- src/simsopt/jax_core/surface_fourier_kernels.py
# passed
```

- **Review evidence:** six-lens adversarial review returned strict PASS after one docs-planning correction. The docs/accounting and history lenses found that the open-question list still stopped at the coefficient-derivative wrapper slice; the final docs now include the `SurfaceXYZFourier` order-hat helper slice and keep the remaining work framed as a product-rule formula fold only if it stays readable. API/behavior, AGENTS/guardrail, mistake-pattern, and comment/test-quality lenses found no remaining issues: derivative-order tuple destructuring is correct, paired-linear paths remain separate, public wrapper introspection is preserved, LOC accounting is `73 insertions / 108 deletions` (`-35`) for this source slice and `550` banked across completed T2.3 slices, and validation evidence remains CPU/X64 only.

### 2026-06-01 — T2.3 `SurfaceXYZFourier.gammadash1` product-rule micro-slice

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.3.
- **Selected slice:** only the full-grid `SurfaceXYZFourier.gammadash1` product-rule spelling in `surface_fourier_kernels.py`. The expanded cosine/sine derivative components are now represented as explicit radial/toroidal components and rotated through `_surface_xyzfourier_rotate(...)`, matching the local representation already used by the paired-linear `gammadash1` wrapper. No paired-linear wrapper, coefficient-Jacobian wrapper, composed area/volume/unit-normal helper, CPU geometry code, backend/cache policy, CUDA/MPS path, transfer policy, public symbol, derivative-order helper, or other product-rule formula was changed.
- **Changed files:** `src/simsopt/jax_core/surface_fourier_kernels.py`, plus this plan set.
- **Design-it-twice gate:** a broad product-rule table/factory remains rejected because it would obscure the derivative algebra. The selected micro-slice changes only one formula to a local radial/toroidal spelling that is shorter and matches the neighboring paired-linear formula. Leaving the other explicit formulas untouched is intentional until each can prove a clearer source-negative representation.
- **Scope status:** `gammadash1` product-rule micro-slice LOC-banked; full T2.3 remains open for any future readable product-rule folds. `surface_fourier_kernels.py` is source-negative by 5 LOC for this slice (`3 insertions / 8 deletions`), bringing completed T2.3 banked source reduction to 555 LOC across the facade, tensor-kernel, XYZ unpack, coefficient-derivative wrapper, order-hat, and `gammadash1` micro-slices. The old full-item `~550` estimate still is not closed as a scope claim because the remaining product-rule formulas are intentionally explicit.
- **Validation evidence:** CPU/X64 `SurfaceXYZFourier` analytic geometry/tangent proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/surface_fourier_kernels.py
# 1 file already formatted
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/jax_core/surface_fourier_kernels.py
# Success: no issues found in 1 source file
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/geo/test_surface_fourier_jax.py -k 'geometry_and_tangents_match_cpp or gamma_and_tangent_lin_match_cpp or second_coordinate_derivative_dcoeff_match_cpp or non_rz_fundamental_form_derivatives_match_cpp'
# 26 passed, 124 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/geo/test_surface_fourier_jax.py::TestSurfaceXYZFourierJaxCppParity
# 26 passed
```

### 2026-06-01 — T2.9 quantity-aware tolerance contract helper

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.9.
- **Selected slice:** parity quantity-to-tolerance policy only. No parity tolerance values, backend-mode routing, fixture construction, comparison verdict semantics, CUDA/MPS runtime behavior, or `parity_ladder_tolerances(lane)` API was changed.
- **Changed files:** `benchmarks/validation_ladder_contract.py`, `benchmarks/non_banana_example_cpp_jax_cpu_parity.py`, `tests/test_benchmark_helpers.py`, plus this plan set.
- **Design-it-twice gate:** duplicating the old `_tolerance_for(...)` policy in tests was rejected because it would keep a second source of truth. The landed design puts `QUANTITY_TOLERANCE_BUCKETS` and `quantity_parity_tolerance(...)` beside `PARITY_LADDER_TOLERANCES`, leaves the harness `_tolerance_for(quantity)` as a compatibility wrapper, and uses a 204-row pre/post snapshot as the oracle.
- **Scope status:** T2.9 complete, not LOC-banked. `non_banana_example_cpp_jax_cpu_parity.py` shrank by 117 LOC, but `validation_ladder_contract.py` grew by 185 LOC and focused tests added 49 LOC. The slice reduces tolerance-policy duplication and closes the previous `validation_ladder_contract.py` mypy blocker; it should not be counted as a bloat LOC reduction.
- **Validation evidence:** CPU/X64 helper and harness-policy proof, not CUDA/MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python - <<'PY'
import json
from pathlib import Path
import benchmarks.non_banana_example_cpp_jax_cpu_parity as harness
rows = []
for tier in ("cpu_reference", "parity", "fast", "float32_smoke"):
    harness.get_tolerance_tier = lambda tier=tier: tier
    for quantity in sorted(harness._TOLERANCE_BUCKETS):
        bucket, rtol, atol = harness._tolerance_for(quantity)
        rows.append({"runtime_tier": tier, "quantity": quantity, "bucket": bucket, "rtol": rtol, "atol": atol})
Path("/tmp/t2_9_tolerance_snapshot_before.json").write_text(json.dumps(rows, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(len(rows), rows[0], rows[-1])
PY
# 204 rows
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python - <<'PY'
import json
from pathlib import Path
from benchmarks.validation_ladder_contract import QUANTITY_TOLERANCE_BUCKETS, quantity_parity_tolerance
rows = []
for tier in ("cpu_reference", "parity", "fast", "float32_smoke"):
    for quantity in sorted(QUANTITY_TOLERANCE_BUCKETS):
        bucket, rtol, atol = quantity_parity_tolerance(quantity, runtime_tier=tier)
        rows.append({"runtime_tier": tier, "quantity": quantity, "bucket": bucket, "rtol": rtol, "atol": atol})
Path("/tmp/t2_9_tolerance_snapshot_after.json").write_text(json.dumps(rows, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(len(rows), rows[0], rows[-1])
PY
# 204 rows
diff -u /tmp/t2_9_tolerance_snapshot_before.json /tmp/t2_9_tolerance_snapshot_after.json
# passed, no diff
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_benchmark_helpers.py -k 'quantity_parity_tolerance or parity_ladder_tolerances'
# 6 passed, 354 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py -k 'float32_smoke_tolerance_tier_routes_by_quantity or unknown_runtime_tolerance_tier_fails_closed or float32_smoke_keeps_gradient_as_diagnostic_failure'
# 3 passed, 59 deselected
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check benchmarks/validation_ladder_contract.py benchmarks/non_banana_example_cpp_jax_cpu_parity.py tests/test_benchmark_helpers.py tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check benchmarks/validation_ladder_contract.py benchmarks/non_banana_example_cpp_jax_cpu_parity.py tests/test_benchmark_helpers.py tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py
# 4 files already formatted
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m py_compile benchmarks/validation_ladder_contract.py benchmarks/non_banana_example_cpp_jax_cpu_parity.py tests/test_benchmark_helpers.py tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy benchmarks/validation_ladder_contract.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy tests/geo/test_surface_fourier_jax.py
# Success: no issues found in 1 source file
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy benchmarks/non_banana_example_cpp_jax_cpu_parity.py
# existing broader benchmark mypy blockers remain: 36 errors in 5 files
git diff --check -- benchmarks/validation_ladder_contract.py benchmarks/non_banana_example_cpp_jax_cpu_parity.py tests/test_benchmark_helpers.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

### 2026-06-01 — T3.2 Biot-Savart points-helper LOC-banking follow-up

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T3.2.
- **Selected slice:** point-management duplication only. `coil_cotangents_to_dofs_gradient`, coil/introspection layout, field-evaluation kernels, state-token generation, uniform `CurveXYZFourier` fast path, and Stage 2 objective behavior were not changed.
- **Changed files:** `src/simsopt/field/biotsavart_jax_backend.py`, plus this plan set.
- **Design-it-twice gate:** Option A, hoisting both points and cotangent projection into one larger mixin, was rejected because the cotangent bodies now encode different contracts: the spec-backed class uses the jitted extraction-spec helper, while the live class keeps fallback-compatible projection. Option B, selected here, hoists only duplicate point-state helper bodies into private module-level helpers, keeps public methods class-local so introspection/type metadata is preserved, and leaves `BiotSavartJAX.clear_points()` live-class-only.
- **Scope status:** points-only LOC-banked, full T3.2 still open. `biotsavart_jax_backend.py` moved from 2,301 to 2,299 LOC for this slice (`29 insertions / 31 deletions`, `-2`). Do not bank the old `~250` estimate until cotangent reconciliation has fallback coverage and proves source-negative.
- **Caller/inventory evidence:** `rg` found point API callers in flux objectives, wireframe optimization, interpolated/dipole/poloidal/Reiman/Dommaschk/wireframe field wrappers, import smokes, Biot-Savart field tests, single-stage integration tests, and Stage 2 integration tests. The only current `clear_points()` caller remains `wireframe_optimization_jax.py`, guarded by `isinstance(field, BiotSavartJAX)`.
- **Compatibility evidence:** both adapters still expose `set_points`, `set_points_cart`, `set_points_cyl`, `set_points_from_spec`, `get_points_cart_ref`, `get_points_cart`, `get_points_cyl`, and `field_eval_spec`; the live adapter alone exposes `clear_points`. Public methods remain class-local, so `BiotSavartJAX.set_points.__qualname__`, `SpecBackedBiotSavartJAX.set_points.__qualname__`, signatures, and `SpecBackedBiotSavartJAX` `FieldEvalSpec` annotations are preserved. `BiotSavartJAX.set_points(...)` also preserves its no-host-round-trip path for JAX arrays.
- **Validation evidence:** CPU/X64 point-contract proof, not CUDA/MPS or full Stage 2 parity proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::test_cylindrical_public_accessors_parity_ncsx tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::test_cylindrical_public_accessors_use_cached_phi_basis_ncsx tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::test_cartesian_public_accessors_normalize_cylindrical_phi_ncsx tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::test_spec_backed_cylindrical_public_accessors_parity_ncsx
# 4 passed
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/integration/test_single_stage_jax_cpu_reference.py::TestAdjointSolveConsistency::test_field_eval_spec_round_trip_uses_immutable_points tests/integration/test_single_stage_jax_cpu_reference.py::TestAdjointSolveConsistency::test_set_points_promotes_float32_inputs_to_float64
# 2 passed
PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=1 .conda/jax/bin/python - <<'PY'
import inspect
import jax.numpy as jnp
import numpy as np
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX, SpecBackedBiotSavartJAX
from simsopt.field.coil import Coil, Current
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.jax_core.specs import FieldEvalSpec, make_biot_savart_spec
curve = CurveXYZFourier(16, 1)
coil = Coil(curve, Current(1.0))
field = BiotSavartJAX([coil])
points = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float64)
assert field.set_points(points) is field
field_eval_spec = field.field_eval_spec()
assert isinstance(field_eval_spec, FieldEvalSpec)
spec = make_biot_savart_spec(coil_dof_extraction=field.coil_dof_extraction_spec(), coil_dofs=np.asarray(field.x, dtype=np.float64))
spec_field = SpecBackedBiotSavartJAX(spec)
assert spec_field.set_points_from_spec(field_eval_spec) is spec_field
np.testing.assert_allclose(np.asarray(spec_field.get_points_cart()), np.asarray(points))
assert BiotSavartJAX.set_points.__qualname__ == "BiotSavartJAX.set_points"
assert SpecBackedBiotSavartJAX.set_points.__qualname__ == "SpecBackedBiotSavartJAX.set_points"
assert inspect.signature(BiotSavartJAX.set_points) == inspect.Signature(
    parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD), inspect.Parameter("points", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
)
assert "field_eval_spec" in SpecBackedBiotSavartJAX.set_points_from_spec.__annotations__
assert "return" in SpecBackedBiotSavartJAX.field_eval_spec.__annotations__
assert hasattr(field, "clear_points")
assert not hasattr(spec_field, "clear_points")
print(BiotSavartJAX.set_points.__qualname__, SpecBackedBiotSavartJAX.set_points.__qualname__)
PY
# BiotSavartJAX.set_points SpecBackedBiotSavartJAX.set_points
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check src/simsopt/field/biotsavart_jax_backend.py tests/field/test_biotsavart_jax.py tests/integration/test_single_stage_jax_cpu_reference.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check src/simsopt/field/biotsavart_jax_backend.py
# 1 file already formatted
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/field/biotsavart_jax_backend.py
# passed
PYTHONNOUSERSITE=1 MYPYPATH=src .conda/jax/bin/python -m mypy src/simsopt/field/biotsavart_jax_backend.py
# pre-existing blocker: SpecBackedBiotSavartJAX.x property and save override errors
git diff --check -- src/simsopt/field/biotsavart_jax_backend.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

- **Review evidence:** initial Crucible Phase 1 found one real public-metadata regression: moving public point methods directly onto `_BiotSavartPointsMixin` changed public `__qualname__` values and dropped `SpecBackedBiotSavartJAX` `FieldEvalSpec` annotations. Delta review then found that the first fix had dropped public `BiotSavartJAX` docstrings. The final implementation keeps the public methods class-local, restores their docstrings/annotations/signatures, and factors only shared helper bodies into private module-level functions. This reduces the banked LOC from the draft 32-line fold to 2 source LOC, but preserves the API metadata.

### 2026-06-01 — T2.4 spec dataclass auto-registration helper

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.4, with TORAX Phase 1 static/dynamic contract overlap.
- **Selected slice:** local `specs.py` helper for frozen dataclass plus JAX registration. No public `jax_core` export, backend/cache policy, surface/field kernel, CUDA/MPS path, or transfer policy was changed.
- **Changed files:** `src/simsopt/jax_core/specs.py`, `tests/core/test_jax_core_specs.py`, plus this plan set.
- **Design-it-twice gate:** central partition-table registration was rejected because it would split one class's data/meta decision across a table and a class definition. The landed local decorator keeps each explicit partition beside the class while hiding only the repeated frozen-registration ceremony.
- **Scope status:** T2.4 complete and small LOC-banked. `src/simsopt/jax_core/specs.py` now has one `_register_jax_spec(...)` helper, 29 decorator uses, and one direct `jax.tree_util.register_dataclass(...)` call inside the helper. The file count moved from the pre-slice 1,622 LOC cited in the source plan to 1,570 LOC, so bank about 52 net LOC rather than the old gross `~140` estimate.
- **Validation evidence:** CPU/X64 spec contract proof, not CUDA/MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py::test_curve_spec_data_fields_do_not_recompile_but_meta_fields_do tests/core/test_jax_core_specs.py::test_register_jax_spec_helper_preserves_data_meta_partition tests/test_jax_import_smoke.py::test_jax_core_specs_are_pytrees
# 3 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/core/test_jax_core_specs.py tests/jax_core/test_tree_signature.py
# 33 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_jax_core_specs_are_pytrees
# 1 passed
.conda/jax/bin/python -m py_compile src/simsopt/jax_core/specs.py tests/core/test_jax_core_specs.py
# passed
.conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/specs.py tests/core/test_jax_core_specs.py
# 2 files already formatted
.conda/jax/bin/python -m ruff check src/simsopt/jax_core/specs.py tests/core/test_jax_core_specs.py
# All checks passed
.conda/jax/bin/python -m mypy --version
# blocked: No module named mypy
git diff --check -- src/simsopt/jax_core/specs.py tests/core/test_jax_core_specs.py
# passed
```

- **Review evidence:** direct scoped review found no added dynamic imports, untyped casts, defensive exception handling, or secret-file references in the source/test diff. The helper preserves the existing data/meta partitions by construction and by the cache-key regression.

### 2026-06-01 — T2.5 leading-axis sharding helper factory

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.5.
- **Selected slice:** leading-axis batch sharding helper fold only. No coil-group collective policy, backend sharding tuning parser, field kernel, surface integral math, tracing integrator, CUDA/MPS path, or transfer policy was changed.
- **Changed files:** `src/simsopt/jax_core/sharding.py`, `tests/jax_core/test_sharding_helpers.py`, plus this bloat plan set.
- **Design-it-twice gate:** one-public-class/alias collapse was rejected because the three config names are exported and directly constructed in tests. The landed design keeps `TrajectoryBatchShardingConfig`, `SeedBatchShardingConfig`, and `SurfaceQuadratureShardingConfig` as concrete dataclass subclasses while sharing the leading-axis predicate/config, maybe-shard, and summary implementations underneath.
- **Scope status:** T2.5 complete and small LOC-banked. The helper fold reduces `src/simsopt/jax_core/sharding.py` from 725 to 719 LOC (`102 insertions / 108 deletions`). The old `~170 LOC` estimate is not banked because preserving public config class identity and explicit public wrappers offsets the internal dedupe.
- **Validation evidence:** CPU/X64 forced-device sharding proof, not CUDA/MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_sharding_helpers.py tests/jax_core/test_tracing_jax_item14.py::test_trajectory_batch_sharding_summary_surfaces_axis_contract
# 4 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_surface_seed_sharding.py
# 6 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/jax_core/test_points_coils_sharding.py
# 2 passed
.conda/jax/bin/python -m py_compile src/simsopt/jax_core/sharding.py tests/jax_core/test_sharding_helpers.py
# passed
.conda/jax/bin/python -m ruff format --check src/simsopt/jax_core/sharding.py tests/jax_core/test_sharding_helpers.py
# 2 files already formatted
.conda/jax/bin/python -m ruff check src/simsopt/jax_core/sharding.py tests/jax_core/test_sharding_helpers.py
# All checks passed
.conda/jax/bin/python -m mypy --version
# blocked: No module named mypy
git diff --check -- src/simsopt/jax_core/sharding.py tests/jax_core/test_sharding_helpers.py
# passed
```

- **Review evidence:** direct scoped review found no added dynamic imports, untyped casts, defensive exception handling, or secret-file references in the source/test diff. Residual risk is backend coverage: these checks force CPU multi-device sharding and do not prove CUDA/MPS placement behavior.

### 2026-06-01 — T2.7 SciPy adapter closure factory

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.7.
- **Selected slice:** SciPy host value/gradient objective closure dedupe in `optimizer_jax_reference.py`. No public optimizer method registry, SciPy option truth table, target-lane routing, backend runtime policy, or on-device/private optimizer code was changed.
- **Changed files:** `src/simsopt/geo/optimizer_jax_reference.py`, plus this bloat plan set.
- **Design-it-twice gate:** replacing the three adapter entrypoints with one externally parameterized function was rejected because it would hide the important guard distinction between private reference adapters and the explicit target SciPy-control lane. The landed design centralizes the host-array value/grad objective and result attachment while keeping `_scipy_minimize(...)`, `_scipy_minimize_value_and_grad(...)`, and `target_scipy_minimize_value_and_grad(...)` as explicit wrappers.
- **Scope status:** T2.7 complete and small LOC-banked. `src/simsopt/geo/optimizer_jax_reference.py` moved from 569 to 539 LOC (`63 insertions / 93 deletions`). The old `~220 LOC` estimate is not banked because preserving readable guard wrappers keeps some ceremony by design.
- **Validation evidence:** CPU/X64 optimizer-reference adapter proof, not CUDA/MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax.py -k 'reference_scipy_adapter or target_scipy_jax or scipy_minimize_does_not_cache_unmarked_objective or target_scipy_jax_marks_host_optimizer_transfer_boundaries'
# 10 passed, 469 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_boozersurface_jax_private.py::test_private_scipy_adapters_reject_all_jax_backend_modes
# 8 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/geo/test_optimizer_jax_reference.py
# 4 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_optimizer_jax_reference_methods_reject_all_jax_backend_modes
# 1 passed
.conda/jax/bin/python -m py_compile src/simsopt/geo/optimizer_jax_reference.py
# passed
.conda/jax/bin/python -m ruff format --check src/simsopt/geo/optimizer_jax_reference.py
# 1 file already formatted
.conda/jax/bin/python -m ruff check src/simsopt/geo/optimizer_jax_reference.py
# All checks passed
.conda/jax/bin/python -m mypy --version
# blocked: No module named mypy
git diff --check -- src/simsopt/geo/optimizer_jax_reference.py
# passed
```

- **Review evidence:** direct scoped review found no added dynamic imports, untyped casts, defensive exception handling, or secret-file references in the source diff. The remaining risk is validation scope: this proves the CPU/reference SciPy-control contract, not GPU/MPS behavior.

### 2026-06-01 — T2.6 backend runtime resolver fold

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.6.
- **Selected slice:** backend runtime kwarg/env/default precedence refactor only. No public backend mode, legacy env synchronization, JAX runtime application, CUDA determinism policy, sharding policy, chunk policy, cache policy values, or GPU memory runtime env application was changed.
- **Changed files:** `src/simsopt/backend/runtime.py`, `tests/test_backend.py`, plus this bloat plan set.
- **Design-it-twice gate:** a fully heterogenous table plus dict/`**kwargs` builder was rejected because it would obscure `BackendConfig` field types and parser/default source labels. The landed design uses one `_resolve_kwarg(...)` helper for the precedence rule, then keeps explicit typed field resolution at `_config_from_mode(...)`.
- **Scope status:** T2.6 complete and small LOC-banked. The eight old private `_resolve_*` runtime kwarg helpers are gone. The old `~100 LOC` estimate is not banked; relative to the already-applied T1.2 runtime export context, this is approximately a low-twenties source reduction and mainly reduces precedence-rule change amplification.
- **Regression evidence:** `tests/test_backend.py::test_runtime_kwargs_override_env_before_mode_defaults` sets every runtime-kwarg env override family and verifies explicit kwargs win for debug-nans, disable-jit, transfer guard, compilation cache, GPU preallocate, GPU memory fraction, XLA GPU allocator, and TF GPU allocator; the TF allocator proof uses an invalid env value plus a valid explicit kwarg so precedence is distinguished. `tests/test_backend.py::test_simsopt_debug_env_applies_runtime_debug_overlay` now also pins the old debug-overlay short-circuit for invalid `SIMSOPT_JAX_DISABLE_JIT` and `SIMSOPT_JAX_TRANSFER_GUARD` env values.
- **Validation evidence:** CPU/X64 backend runtime policy proof, not CUDA/MPS execution proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_backend.py -k 'backend_mode_policy_helpers or fast_mode_policy_helpers or mps_smoke_mode_policy_helpers or cpu_float32_smoke_mode_policy_helpers or parity_modes_default_transfer_guard_to_log or debug_overlay or compilation_cache or gpu_memory_env_overrides_mode_defaults or runtime_kwargs_override_env_before_mode_defaults or apply_jax_runtime_config_applies_gpu_memory_policy or apply_jax_runtime_config_accepts_preimported_jax_with_matching_gpu_memory_env or apply_jax_runtime_config_rejects_preimported_jax_with_missing_gpu_memory_env'
# 17 passed, 113 deselected
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_backend.py
# 130 passed, 2 warnings after fixing the review-found debug-overlay regression
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/test_jax_import_smoke.py::test_programmatic_backend_selection_configures_jax_runtime tests/test_jax_import_smoke.py::test_parity_mode_defaults_transfer_guard_and_keeps_x64_enabled tests/test_jax_import_smoke.py::test_backend_runtime_module_has_no_private_jax_src_usage
# 3 passed
PYTHONPATH=src .conda/jax/bin/python -m py_compile src/simsopt/backend/runtime.py tests/test_backend.py
# passed
PYTHONPATH=src .conda/jax/bin/python -m ruff format --check src/simsopt/backend/runtime.py tests/test_backend.py
# 2 files already formatted
PYTHONPATH=src .conda/jax/bin/python -m ruff check src/simsopt/backend/runtime.py tests/test_backend.py
# All checks passed
PYTHONPATH=src .conda/jax/bin/python -m mypy --version
# blocked: No module named mypy
git diff --check -- src/simsopt/backend/runtime.py tests/test_backend.py
# passed
git diff --unified=0 -- src/simsopt/backend/runtime.py tests/test_backend.py | rg -n "^\+.*(import\(|\bAny\b|cast\(|try:|except )"
# no matches
```

- **Review evidence:** scoped review found and fixed one behavior regression: the first `_resolve_kwarg(...)` draft eagerly validated `disable_jit` and `transfer_guard` before the `SIMSOPT_DEBUG=1` overlay could force those fields, whereas HEAD lazily skipped those resolvers under the overlay. The final source restores that short-circuit, adds the regression coverage above, and still has no added dynamic imports, untyped casts, defensive exception handling, or secret-file references in the source/test diff. The important accounting constraint is that this should be counted as a small T2.6 complexity/LOC win, not the original `~100 LOC` bank.

### 2026-06-01 — T2.8 `LayerDriftTracker` core helper

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** same-candidate replay layer-decomposition tracker state only. No candidate matching, solver-contract diagnostics, objective-component comparison, scipy callback comparison, hardware/failure comparison, parity-census schema, or gate semantics were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** a generic key-prefix result builder for every replay tracker was rejected because it would make the output schema depend on string construction. The landed design introduces `LayerDriftTracker` only for layer-decomposition families and keeps the final replay payload keys explicit in `compare_same_candidate_objective_replay`.
- **Scope status:** core helper LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 13 LOC for this slice (`79 insertions / 92 deletions`). Do not bank the old `~200 LOC` estimate until the remaining tracker families are folded without obscuring the replay schema.
- **Regression evidence:** same-candidate replay tests still cover iota layer reporting, parity bug census ordering, pre-Newton census gate failure messages, and strict gate classification.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay.

```bash
.conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed
.conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate_replay_reports_iota_decomposition_layer or same_candidate_replay_reports_parity_bug_census or pre_newton_census'
# 9 passed, 351 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate'
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: pre-existing benchmark/example typing debt; the new LayerDriftTracker produced no reported mypy error
git diff --check -- benchmarks/single_stage_init_parity.py
# passed
```

### 2026-06-01 — T2.8 SciPy callback first-split helper

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** same-candidate replay Boozer SciPy callback trace first-split recording only. No callback field comparison tolerance, max-diff accumulation, length/presence mismatch handling, outer replay schema keys, solver metadata capture, candidate matching, or gate semantics were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** table-driving all callback fields was rejected because the ordering and tolerance policy for `evaluation_index`, `decision_vector`, `fun`, and `gradient` are intentionally explicit. The landed design extracts only the repeated "first split wins" assignment into `_record_first_scipy_callback_split(...)` and leaves the field checks in their original order.
- **Scope status:** SciPy callback first-split helper LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 4 LOC for this slice (`34 insertions / 38 deletions`). Together with the earlier `LayerDriftTracker` slice, T2.8 has 17 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** `tests/test_benchmark_helpers.py` still pins `first_boozer_scipy_callback_split` field/callback/evaluation-index payloads, max callback absolute diff, and length-mismatch reason. The broader same-candidate selector still exercises the replay helper contract.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate_replay_compares_boozer_scipy_callback_trace or same_candidate_replay_reports_boozer_scipy_callback_trace_length'
# 2 passed, 358 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate'
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: pre-existing benchmark/example typing debt; the new SciPy callback helper produced no distinct new gate-clean path
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
```

### 2026-06-01 — T2.8 target-native predicate-cache slice

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** per-pair target-native rejection predicate reuse inside `compare_same_candidate_objective_replay(...)`. No replay payload keys, rejection diagnostics, gradient/hardware comparison tolerances, parity-census schema, same-candidate gate classification, or solver metadata behavior were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** a generic replay-event state object was rejected because it would hide output-schema ownership and mix unrelated candidate, failure, callback, and hardware facts. The landed design caches only `cpu_rejected_by_contract` / `jax_rejected_by_contract` for the current pair and reuses those booleans in the existing explicit branches.
- **Scope status:** target-native predicate cache LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 2 LOC for this slice (`9 insertions / 11 deletions`). Together with the earlier `LayerDriftTracker` and SciPy callback slices, T2.8 has 19 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** target-native scope, target-native prefix, and target-native rejection diagnostics remain covered by focused tests; the broader same-candidate selector still exercises the replay helper contract.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate_replay_accepts_target_native_scope or same_candidate_replay_accepts_target_native_prefix or same_candidate_replay_classifies_target_native_rejections'
# 3 passed, 357 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate'
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: pre-existing benchmark/example typing debt; the new target-native predicate cache produced no distinct new gate-clean path
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
```

### 2026-06-01 — T2.8 comparison-scope `Counter` slice

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** candidate/gradient comparison-scope count bookkeeping inside `compare_same_candidate_objective_replay(...)`. No replay payload keys, target-native prefix/full-vector scope labels, target-native rejection diagnostics, gradient/objective comparison tolerances, parity-census schema, same-candidate gate classification, or solver metadata behavior were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** a broad replay-count accumulator object was rejected because it would hide the public result-key mapping. The landed design uses local `Counter[str]` instances only for the two repeated increment sites and preserves the explicit `dict(...)` conversion at the return boundary.
- **Scope status:** comparison-scope counter LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 3 LOC for this slice (`6 insertions / 9 deletions`). Together with the earlier `LayerDriftTracker`, SciPy callback, and target-native predicate-cache slices, T2.8 has 22 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** target-native prefix/full-vector scope-count payloads remain covered by focused tests; the broader same-candidate selector still exercises the replay helper contract.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'single_stage_init_same_candidate_replay_accepts_target_native_prefix or single_stage_init_same_candidate_replay_keeps_full_vector_shape_gate or single_stage_init_same_candidate_replay_rejects_target_prefix_mismatch or single_stage_init_same_candidate_replay_classifies_target_native_rejections'
# 4 passed, 356 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate'
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 132 pre-existing benchmark/example typing errors; the new comparison-scope Counter slice produced no distinct new gate-clean path
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
```

### 2026-06-01 — T2.8 per-pair loop metadata-binding slice

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** local per-pair bindings for repeated `event_index`, `accepted_iteration_target`, and `line_search_evaluation` reads inside `compare_same_candidate_objective_replay(...)`. No replay payload keys, metadata field names, target-native rejection diagnostics, SciPy callback split schema, layer-decomposition summaries, parity-census schema, same-candidate gate classification, or solver metadata behavior were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** a generic replay-event metadata object was rejected because it would hide the result dict keys and introduce another schema surface. The landed design uses plain local bindings inside the existing loop and leaves every outgoing result dict explicit.
- **Scope status:** per-pair metadata binding LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 4 LOC for this slice (`25 insertions / 29 deletions`). Together with the earlier `LayerDriftTracker`, SciPy callback, target-native predicate-cache, and comparison-scope `Counter` slices, T2.8 has 26 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** first-failure event metadata, target-native rejection diagnostics, SciPy callback split metadata, and parity-census path metadata remain covered by focused tests; the broader same-candidate selector still exercises the replay helper contract.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay.

```bash
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate_replay_reports_first_mismatch or same_candidate_replay_classifies_target_native_rejections or same_candidate_replay_compares_boozer_scipy_callback_trace or same_candidate_replay_reports_parity_bug_census'
# 4 passed, 356 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate'
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 132 pre-existing benchmark/example typing errors; the new per-pair metadata-binding slice produced no distinct new gate-clean path
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
```

### 2026-06-01 — T2.8 target-native component-summary reuse

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** reuse `_compare_same_candidate_objective_components(...)` for the target-native replay "no objective components" summary and bind the helper's repeated slice-owner predicate once. No replay payload keys, target-native rejection diagnostics, objective/gradient tolerances, component-owner reporting, parity-census schema, same-candidate gate classification, or solver metadata behavior were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** a broader objective-summary object was rejected because it would add another schema surface for a two-line source reduction. The landed design keeps `_compare_same_candidate_objective_components(...)` as the single owner for objective-component summary shape, including the target-native empty summary, while the public replay result dict remains explicit.
- **Scope status:** component-summary reuse LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 2 LOC for this slice (`19 insertions / 21 deletions`). Together with the earlier `LayerDriftTracker`, SciPy callback, target-native predicate-cache, comparison-scope `Counter`, and per-pair metadata-binding slices, T2.8 has 28 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** target-native replay still skips objective-component comparison without adding a presence mismatch, normal component-owner reporting still records pair/index metadata, the broader same-candidate selector still exercises replay helper contracts, and the pre-Newton census gate remains pinned.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay, CUDA, or MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'target_native or component_owner'
# 5 passed, 355 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'pre_newton_census or same_candidate_classifies_pre_newton_census_only or same_candidate_gate_requires_census_recording'
# 8 passed, 352 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate'
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 128 pre-existing benchmark/example typing errors; the new component-summary reuse slice produced no distinct new gate-clean path
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py
# passed
```

### 2026-06-01 — T2.8 target-native flag inline

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** delete the single-use `_same_candidate_target_native_replay_event(...)` helper and read `target_native_replay` directly at the same-candidate replay loop site. No replay payload keys, target-native rejection diagnostics, candidate/gradient comparison scopes, objective-component summaries, parity-census schema, same-candidate gate classification, or solver metadata behavior were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** keeping the one-line helper was rejected because it had one caller and hid no reusable policy after the target-native predicates were already cached locally. A broader target-native event object was rejected because it would add another schema surface. The landed design keeps the flag read local to the loop that owns the rest of the per-pair replay predicates.
- **Scope status:** target-native flag inline LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 6 LOC for this slice (`1 insertion / 7 deletions`). Together with the earlier `LayerDriftTracker`, SciPy callback, target-native predicate-cache, comparison-scope `Counter`, per-pair metadata-binding, and component-summary reuse slices, T2.8 has 34 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** target-native scope/prefix/rejection classification still passes, the broader same-candidate selector still exercises replay helper contracts, and `_pre_newton_census_gate_failures` remains untouched apart from line-number drift from the helper deletion.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay, CUDA, or MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate_replay_accepts_target_native_scope or same_candidate_replay_accepts_target_native_prefix or same_candidate_replay_classifies_target_native_rejections'
# 3 passed, 357 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k 'same_candidate'
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 128 pre-existing benchmark/example typing errors; the new target-native flag inline slice produced no distinct new gate-clean path
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py
# passed
```

### 2026-06-01 — T2.8 SciPy callback `fun` threshold slice

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** callback `fun` first-split threshold bookkeeping inside `_compare_same_candidate_scipy_callback_trace(...)`. No callback trace schema keys, field comparison order, scalar tolerance constants, callback length/presence handling, candidate matching, parity-census schema, same-candidate gate classification, or solver metadata behavior was changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** Option A, duplicating the scalar tolerance formula directly in the callback branch, was rejected because it created a second tolerance-policy spelling and shifted mypy's existing tolerance-constant debt onto a new line. Option B, selected here, records the failure-list length before `_compare_same_candidate_scalar(...)` and uses that comparator's own mismatch emission as the first-split predicate for finite `fun` values. This keeps `_compare_same_candidate_scalar(...)` as the tolerance SSOT and preserves the old missing-value behavior by retaining the `cpu_fun is not None and jax_fun is not None` guard.
- **Scope status:** callback `fun` threshold slice LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 4 LOC for this slice (`2 insertions / 6 deletions`). Together with the earlier T2.8 micro-slices, T2.8 has 38 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** a pre/post one-off probe for a callback `fun` mismatch preserved the first-split payload with `field="fun"`, `callback_index=1`, and `max_abs_diff=0.20000000000000018`; the focused callback selector and broad `same_candidate` selector still pass.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay, CUDA, or MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed!
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "single_stage_init_same_candidate_replay and scipy_callback"
# 2 passed, 358 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "same_candidate"
# 22 passed, 338 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python - <<'PY'
from benchmarks import single_stage_init_parity as mod

scalar = lambda value: {"value": float(value), "finite": True}
vector = lambda values: {"values": [float(v) for v in values], "all_finite": True}
entry = {
    "evaluation_index": 1,
    "decision_vector": vector([1.0, 2.0]),
    "fun": scalar(3.0),
    "gradient": vector([0.5, -0.25]),
}
jax_entry = dict(entry, fun=scalar(3.2))
failures = []
summary = mod._compare_same_candidate_scipy_callback_trace(
    failures,
    field="boozer_solver_metadata.pre_newton_scipy_callback_trace",
    cpu_trace=[entry],
    jax_trace=[jax_entry],
)
print(summary["first_split"])
print(summary["max_abs_diff"])
print(failures[0])
PY
# {'field': 'fun', 'callback_index': 1, 'cpu_evaluation_index': 1, 'jax_evaluation_index': 1, 'max_abs_diff': 0.20000000000000018}
# 0.20000000000000018
# boozer_solver_metadata.pre_newton_scipy_callback_trace[1].fun mismatch: cpu=3.0000000000000000e+00, jax=3.2000000000000002e+00, abs_diff=2.000e-01.
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 128 pre-existing benchmark/example typing errors; the callback fun threshold slice produced no distinct new gate-clean path
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py
# passed
```

### 2026-06-01 — T2.8 SciPy callback split payload inline

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** callback split payload construction under `_record_first_scipy_callback_split(...)`. No callback trace schema keys, field comparison order, first-split precedence, length/presence mismatch handling, scalar/vector tolerance policy, candidate matching, parity-census schema, same-candidate gate classification, or solver metadata behavior was changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** Option A, keeping both `_same_candidate_scipy_callback_split(...)` and `_record_first_scipy_callback_split(...)`, was rejected because the lower helper became a one-use pass-through with no independent policy after the first-split helper landed. Option B, selected here, deletes the pass-through and keeps the explicit callback split payload literal inside `_record_first_scipy_callback_split(...)`, so the schema remains visible at the single state-transition site.
- **Scope status:** callback split payload-inline slice LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 12 LOC for this slice (`7 insertions / 19 deletions`; 4,789 -> 4,777 LOC). Together with the earlier T2.8 micro-slices, T2.8 has 50 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** the focused callback selector still covers gradient split payloads and callback trace length mismatch. The broad `same_candidate` selector still covers callback, target-native, component-summary, decomposition, parity-census, and gate-classification replay behavior.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay, CUDA, or MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed!
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "single_stage_init_same_candidate_replay and scipy_callback"
# 2 passed, 358 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "same_candidate"
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 128 pre-existing benchmark/example typing errors; no error was introduced on the payload-inline diff lines
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

### 2026-06-01 — T2.8 layer diagnostic wrapper inline

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** layer-decomposition diagnostic field-diff dispatch inside `_compare_same_candidate_layer_decomposition(...)`. No layer field tables, scalar/vector path diff helpers, layer divergence threshold, `parity_bug_census` schema, target-native skip behavior, same-candidate gate classification, or final replay result keys were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** Option A, keeping `_diagnostic_scalar_abs_diff(...)` and `_diagnostic_vector_abs_diff(...)`, was rejected because each wrapper had one call site and only added a same-level layer around the shared `None`/`None` zero-drift policy. Option B, selected here, keeps that policy explicit in the layer loop and dispatches directly to `_path_scalar_abs_diff(...)` / `_path_vector_abs_diff(...)`, preserving the scalar/vector kind decision at the layer-field comparison site.
- **Scope status:** layer diagnostic wrapper-inline slice LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 16 LOC for this slice (`5 insertions / 21 deletions`; 4,777 -> 4,761 LOC). Together with the earlier T2.8 micro-slices, T2.8 has 66 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** the focused decomposition/parity-census selector covers both `iota_penalty_decomposition` first-layer reporting and the `parity_bug_census` divergent-layer path. The broad `same_candidate` selector still covers callback, target-native, component-summary, decomposition, parity-census, and gate-classification replay behavior.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay, CUDA, or MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed!
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "decomposition or parity_bug_census"
# 2 passed, 358 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "same_candidate"
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 128 pre-existing benchmark/example typing errors; no error was introduced on the layer-diagnostic inline diff lines
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

### 2026-06-01 — T2.8 decomposition dispatch wrapper inline

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** layer-decomposition domain dispatch for `boozer_solve_decomposition` and `iota_penalty_decomposition`. No layer field tables, field names, scalar/vector path diff helpers, layer divergence threshold, `parity_bug_census` schema, target-native skip behavior, same-candidate gate classification, or final replay result keys were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** Option A, keeping `_compare_same_candidate_iota_decomposition(...)` and `_compare_same_candidate_boozer_solve_decomposition(...)`, was rejected because both helpers had one call site and only forwarded domain constants into `_compare_same_candidate_layer_decomposition(...)`. Option B, selected here, deletes the pass-through wrappers and keeps `field_name` plus `layer_fields` explicit at the two generic call sites.
- **Scope status:** decomposition dispatch wrapper-inline slice LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 34 LOC for this slice (`6 insertions / 40 deletions`; 4,761 -> 4,727 LOC). Together with the earlier T2.8 micro-slices, T2.8 has 100 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** the focused decomposition/parity-census selector covers both `iota_penalty_decomposition` first-layer reporting and the `parity_bug_census` divergent-layer path. The broad `same_candidate` selector still covers callback, target-native, component-summary, decomposition, parity-census, and gate-classification replay behavior.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay, CUDA, or MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed!
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "decomposition or parity_bug_census"
# 2 passed, 358 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "same_candidate"
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 128 pre-existing benchmark/example typing errors; no error was introduced on the decomposition-dispatch inline diff lines
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

### 2026-06-01 — T2.8 scalar-close inline

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.8.
- **Selected slice:** scalar tolerance predicate inside `_compare_same_candidate_scalar(...)`. No scalar tolerance constants, mismatch text, callback first-split behavior, parity-census schema, replay payload keys, target-native behavior, vector tolerance policy, or final replay result keys were changed.
- **Changed files:** `benchmarks/single_stage_init_parity.py`, plus this bloat plan set.
- **Design-it-twice gate:** Option A, keeping `_scalar_close(...)`, was rejected because it had one call site and only hid the same tolerance formula from `_compare_same_candidate_scalar(...)`. Option B, selected here, deletes the wrapper and checks `not diff <= (atol + rtol * abs(float(cpu_value)))` next to the already computed `diff`, preserving NaN mismatch behavior while keeping scalar mismatch emission in one function.
- **Scope status:** scalar-close inline slice LOC-banked small; full T2.8 remains open. `benchmarks/single_stage_init_parity.py` is source-negative by 4 LOC for this slice (`1 insertion / 5 deletions`; 4,727 -> 4,723 LOC). Together with the earlier T2.8 micro-slices, T2.8 has 104 benchmark LOC banked so far; the old `~200 LOC` estimate remains unbanked.
- **Regression evidence:** the broad `same_candidate` selector still covers callback, target-native, component-summary, decomposition, parity-census, gate-classification, and scalar objective replay behavior. The inline predicate is algebraically the old helper body using the existing `diff`; `not diff <= ...` intentionally preserves the old false-on-NaN close check as a mismatch.
- **Validation evidence:** CPU/X64 replay-helper proof, not full single-stage parity replay, CUDA, or MPS proof.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# All checks passed!
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m ruff format --check benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# 2 files already formatted
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m py_compile benchmarks/single_stage_init_parity.py tests/test_benchmark_helpers.py
# passed
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "same_candidate"
# 22 passed, 338 deselected
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark_helpers.py -k "scalar or same_candidate"
# 22 passed, 338 deselected
PYTHONNOUSERSITE=1 PYTHONPATH=src .conda/jax/bin/python -m mypy benchmarks/single_stage_init_parity.py
# blocked: 128 pre-existing benchmark/example typing errors; no error was introduced on the scalar-close inline diff line
PYTHONNOUSERSITE=1 .conda/jax/bin/python -m pip check
# No broken requirements found
git diff --check -- benchmarks/single_stage_init_parity.py docs/bloat_reduction_plan_2026-05-20.md docs/bloat_torax_coherent_execution_plan_2026-05-31.md docs/torax_jax_porting_patterns_impl_plan_2026-05-27.md
# passed
```

## Risks and Mitigations

- Risk: A TORAX-inspired helper creates another abstraction layer without deleting real complexity.
  Mitigation: Require a before/after caller map and reject helpers that do not remove repeated declarations or enforce a tested invariant.

- Risk: Low-risk bloat work accidentally removes public compatibility or oracle coverage.
  Mitigation: Run full caller inventory and preserve public kwargs, probe scripts, and independent assertions unless migration evidence is explicit.

- Risk: CPU-only validation is mistaken for GPU, CUDA, or MPS proof.
  Mitigation: Label each validation result by backend lane and do not close GPU-sensitive work without the relevant strict-transfer or smoke evidence.

- Risk: Source docs drift again during concurrent commits.
  Mitigation: Treat all line refs as snapshots and re-grep before every implementation slice.

- Risk: Multiple plans become conflicting sources of truth.
  Mitigation: Keep this file as an overlay only; update the source plan that owns the detailed item when status changes.

## Completion Criteria

- [x] One execution slice is selected with a named source-doc owner and validation gate.
- [x] All changed symbols in that slice have caller inventories.
- [x] The implemented dirty-tree slices preserve public APIs, parity tolerances, backend modes, and transfer/cache contracts in their recorded validation scope.
- [x] The source doc owning the completed work is updated with evidence, not just checkbox changes.
- [x] Validation output is recorded with backend lane and exact command.
- [x] Remaining work is still traceable to the source docs and not duplicated into an unsorted backlog.
- [x] The broad dirty tree was split into scoped banked-shrink/foundation commits before continuing; future implementation commits must keep using the same classification.

## Open Questions

- Which slice should be executed next after the completed TORAX Phase 1/2 contract-first proof, T1.1/T1.2/T1.3/T1.4/T1.5/T1.6/T1.7/T1.8 bloat collapses, T1.9 public-API reclassification, T1.10 probe-script classification, TORAX Phase 1 target-lane closure-capture regression, TORAX Phase 3 bounded-scan helper pilot, TORAX Phase 4 branch/JAXPR pilot, T2.1 Boozer schema/envelope factory pilot, T2.1 LS-Newton reporting LOC-banking follow-up, T2.2 Boozer radial formula dedup, T2.2 direct-wrapper LOC-banking follow-up, T2.2 scalar-helper LOC-banking follow-up, T2.2 radial pass-through inline follow-up, T2.3 surface Fourier facade slice, T2.3 tensor kernel wrapper fold, T2.3 `SurfaceXYZFourier` unpack fold, T2.3 coefficient-derivative wrapper-family fold, T2.3 `SurfaceXYZFourier` order-hat helper slice, T2.3 `SurfaceXYZFourier.gammadash1` product-rule micro-slice, T2.4 spec dataclass registration helper, T2.5 leading-axis sharding helper, T2.6 backend runtime resolver fold, T2.7 SciPy adapter closure factory, T2.8 `LayerDriftTracker` core helper, T2.8 SciPy callback first-split helper, T2.8 target-native predicate cache, T2.8 comparison-scope Counter slice, T2.8 per-pair metadata-binding slice, T2.8 target-native component-summary reuse slice, T2.8 target-native flag inline slice, T2.8 SciPy callback `fun` threshold slice, T2.8 callback split payload-inline slice, T2.8 layer diagnostic wrapper-inline slice, T2.8 decomposition dispatch wrapper-inline slice, T2.8 scalar-close inline slice, T2.9 quantity-tolerance contract helper, and T3.2 Biot-Savart points-helper follow-up: finish only a larger T2.8 tracker-family cleanup if it stays schema-explicit, attempt only a larger T2.2 formula/subset-family redesign if it stays readable and benchmark-safe, continue T2.3 product-rule formula folds only when each stays readable, pursue T3.2 cotangent reconciliation only with fallback coverage, branch/JAXPR follow-up for non-piloted hot paths, transfer-sensitive proof, or select another untouched item?
- Should completed slices be committed one checkbox at a time, or grouped by validation gate when multiple tiny doc-only updates are adjacent?
- What backend lane is available for strict-transfer proof in the current machine context when a GPU-sensitive item is selected?
