# Coherent Bloat And TORAX JAX Execution Plan

## Review Envelope

- Target repo: `/Users/suhjungdae/code/columbia/simsopt-jax-shared-jax`
- Source-doc review basis: `b267b0d95` on `shared-jax-clean`
- Source-code checkpoint before docs-only gate commit: `8b94c2bbd` on `shared-jax-clean`, with a broad dirty implementation tree across `src/`, `tests/`, and `docs/`
- Docs-only drift gate introduced in commit `398b3e50d` and checkpoint basis clarified in commit `446eab365` on `shared-jax-clean`
- Reference TORAX repo reviewed: `/Users/suhjungdae/code/opensource/torax` at `60190df1` on clean `main`
- Historical local status at source-doc review: the two source docs were modified and this overlay was untracked; no source-code edits were part of that review. This is no longer the current working-tree state.
- Artifact note: this checkout does not contain a repo-local `.artifacts/` tree. Historical code-smell artifacts referenced by the bloat plan were found in sibling checkout `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/`.

## 2026-06-01 Drift Checkpoint

The current dirty tree is validated as a contract-hardening / complexity-reduction checkpoint, not as a strict LOC-reduction checkpoint. Do not commit the whole tree under a generic "bloat reduction" label.

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

Before any commit, split the dirty tree by this classification. Salvage the banked-shrink slices first; keep foundation-only slices only with their follow-up deletion target; do not count tests/docs growth as bloat reduction.

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
- Docs-gate history: the docs-only drift gate was introduced in `398b3e50d`, checkpoint basis was clarified in `446eab365`, the source-code checkpoint before that docs-only gate was `8b94c2bbd`, and source-doc edit status from the original review envelope is historical only. Treat these commit hashes as historical anchors, not live-HEAD markers.
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

### 2026-06-01 — T2.2 Boozer radial evaluator formula dedup

- **Owner source doc:** `docs/bloat_reduction_plan_2026-05-20.md`, T2.2.
- **Selected slice:** direct `boozer_radial_field.py` evaluator formula dedup plus radial Boozer RHS column reuse. No public wrapper API, CPU Boozer implementation, Fourier math, tracing integrator, backend/cache policy, CUDA/MPS path, or transfer policy was changed.
- **Changed files:** `src/simsopt/jax_core/boozer_radial_field.py`, `src/simsopt/jax_core/tracing.py`, `tests/field/test_trace_boozer_analytic_jax.py`, `tests/field/test_boozermagneticfield_jax_item33.py`, plus this plan set.
- **Design-it-twice gate:** the simple full-column wrapper was implemented first and rejected by benchmark evidence because it made standalone direct evaluators and the RHS path evaluate too many radial profiles. The landed design keeps formula ownership in `_eval_*_from_columns`, uses typed subset columns for direct evaluators, and adds `_eval_radial_rhs_columns` plus `_RADIAL_RHS_COLUMN_EVALUATORS` so radial Boozer guiding-centre RHS evaluates one column bundle per point.
- **Scope status:** formula deduped, not LOC-banked. The old `~400 LOC` T2.2 estimate is not subtracted because preserving the benchmark gate required subset-builder scaffolding (`boozer_radial_field.py` is `262 insertions / 264 deletions`; `tracing.py` adds `114 insertions / 30 deletions`). Future T2.2 LOC banking needs a separate profile-family parametrization and fresh benchmark proof.
- **Validation evidence:** CPU/X64 radial formula and tracing proof, not CUDA/MPS proof.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_trace_boozer_analytic_jax.py
# 27 passed
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .conda/jax/bin/python -m pytest -q tests/field/test_boozermagneticfield_jax_item33.py -k "radial_columns_cached_once_per_points_cycle or direct_radial_evaluators_match_column_evaluators"
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
- [ ] Before the next commit, split the dirty tree into banked-shrink, foundation-only, not-LOC-banked, and defer/revert-candidate slices.

## Open Questions

- Which slice should be executed next after the completed TORAX Phase 1/2 contract-first proof, T1.1/T1.2/T1.3/T1.4/T1.5/T1.6/T1.7/T1.8 bloat collapses, T1.9 public-API reclassification, T1.10 probe-script classification, TORAX Phase 1 target-lane closure-capture regression, TORAX Phase 3 bounded-scan helper pilot, TORAX Phase 4 branch/JAXPR pilot, T2.1 Boozer schema/envelope factory pilot, T2.2 Boozer radial formula dedup, T2.4 spec dataclass registration helper, T2.5 leading-axis sharding helper, T2.6 backend runtime resolver fold, and T2.7 SciPy adapter closure factory: finish a LOC-banked T2.1 reporting fold, do a T2.2 LOC-banking follow-up, branch/JAXPR follow-up for non-piloted hot paths, transfer-sensitive proof, or select a new untouched T2 factory such as T2.3 surface Fourier wrappers?
- Should completed slices be committed one checkbox at a time, or grouped by validation gate when multiple tiny doc-only updates are adjacent?
- What backend lane is available for strict-transfer proof in the current machine context when a GPU-sensitive item is selected?
