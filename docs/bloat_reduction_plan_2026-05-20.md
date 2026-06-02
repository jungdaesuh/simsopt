# simsopt-jax Bloat Reduction Plan

**Status:** Draft v8 — docs-only drift gate introduced in commit `398b3e50d` and checkpoint basis clarified in commit `446eab365` (2026-06-01) on branch `shared-jax-clean`, recording the source-code checkpoint before that docs gate as `8b94c2bbd`. Supersedes Draft v7 (clean source-doc refresh basis `b267b0d95`, 2026-05-31), Draft v6 (basis `2bcaeff28`, dirty correction pass), Draft v5 (basis `21c3d517d`, 2026-05-30), and Draft v4 (basis `5bcd9061c`, 2026-05-29). v8 does not recalculate every historical line ref; it records that the dirty source/test tree at `8b94c2bbd` drifted from strict LOC reduction into contract-hardening / foundation work. Treat old aggregate estimates as stale until salvage commits are split and re-measured.
**Author:** orchestrator synthesis of 8-lane parallel audit (2026-05-20); v4 reconciliation from a 6-agent re-verification (2026-05-29); v5 reconciliation from a 4-agent re-verification (2026-05-30); v6 doc-review correction from a 3-agent drift audit (2026-05-30); v7 current-checkout doc-review refresh (2026-05-31); v8 drift checkpoint after dirty-tree LOC validation (2026-06-01).
**Branch:** `shared-jax-clean`; docs-only drift gate was introduced in `398b3e50d`, checkpoint basis was clarified in `446eab365`, and the source-code checkpoint before that docs-only gate was `8b94c2bbd`. These commit hashes are historical anchors, not live-HEAD markers.
**Audit basis:** 8 parallel subagent reports plus current-tree Crucible review covering Boozer/objectives, optimizer JAX, jax_core kernels, field/backend, PM/QFM/wireframe, benchmarks/parity, tests, cross-cutting duplication, official JAX/SciPy/SIMSOPT API documentation checks, and a post-v2 codebase-delta validation.

---

## Table of Contents

0. [2026-06-01 Execution Drift Gate](#0-2026-06-01-execution-drift-gate)
1. [Goals & Purpose](#1-goals--purpose)
2. [Success Criteria](#2-success-criteria)
3. [Guiding Principles](#3-guiding-principles)
4. [Risk Model & Guardrails](#4-risk-model--guardrails)
5. [Tier 1 — Mechanical Wins](#5-tier-1--mechanical-wins)
6. [Tier 2 — Factory Introductions](#6-tier-2--factory-introductions)
7. [Tier 3 — Structural Consolidations](#7-tier-3--structural-consolidations)
8. [Tier 4 — Decision Points](#8-tier-4--decision-points)
9. [Cross-Cutting Validation Gates](#9-cross-cutting-validation-gates)
10. [Sequencing Rationale](#10-sequencing-rationale)
11. [Rollback Strategy](#11-rollback-strategy)
12. [Open Questions](#12-open-questions)
13. [Estimated Totals](#13-estimated-totals)
14. [Appendix A — How to Use This Document](#14-appendix-a--how-to-use-this-document)
15. [Appendix B — Audit Source Trace](#15-appendix-b--audit-source-trace)

---

## 0. 2026-06-01 Execution Drift Gate

The drift-checkpoint dirty tree must not be treated as a single "bloat reduction" change. It contained real source shrinkers, but the checkout as a whole was larger once untracked helpers, tests, and docs were counted.

Drift-checkpoint ledger captured before the v8 doc correction (`git diff --numstat -- src tests docs` plus untracked-file `wc -l`):

- `src/` tracked: `1933 insertions / 1941 deletions`, net `-8`
- Untracked source helpers: `+53`
- Effective source net: `+45`
- `tests/` tracked: `1174 insertions / 54 deletions`, net `+1120`
- Untracked tests: `+224`
- `docs/` tracked: `938 insertions / 161 deletions`, net `+777`
- Total tracked plus untracked over `src/`, `tests/`, and `docs/`: `+2166`

Classify each dirty slice before commit:

- **Banked-shrink:** source LOC is net-negative in the isolated slice and behavior/API compatibility is validated.
- **Foundation-only:** source LOC is flat or positive, but the slice names the exact follow-up deletion it unlocks.
- **Not LOC-banked:** complexity or contract quality improved, but no current source shrink can be claimed.
- **Defer/revert-candidate:** source LOC is flat or positive and no immediate deletion payoff is identified.

The next execution pass should salvage banked-shrink commits first, keep foundation-only commits only when their deletion target is explicit, and reclassify T2.1/T2.2-style pilots as not LOC-banked until they pay down source.

## 1. Goals & Purpose

### Why this work matters

The JAX port grew via per-milestone accretion (M1→M6), each milestone adding correct scaffolding without retiring the templates from prior layers. The result: roughly 9,000–11,000 candidate LOC of duplication, scaffolding, and decision-gated variants buried inside ~115k LOC of JAX-related code (src + benchmarks + tests). Maintenance is harder than it needs to be; every cross-cutting change (a new backend mode, a new result-dict field, a new solver variant) has to be applied in 5–7 places.

### Primary goal

**Reduce JAX-port LOC while preserving 100% of features, public APIs, and CLAUDE.md contracts.**

### Secondary goals

1. Convert hand-maintained templates into factories driven by data (mode tables, schema frozensets, history-spec dicts) so future changes touch one place.
2. Move diagnostic/profile machinery out of hot production files into sibling diagnostic modules.
3. Preserve subprocess skip visibility by verifying the current `_skip_case(...)` sentinel coverage rather than re-fixing already converted sites.
4. Surface and resolve five Tier-4 contract clarifications (`lbfgs-trace`, `scipy-jax` / `scipy-jax-fullgraph`, `qfm_solver` BFGS reuse, QFM SLSQP compatibility alias, surviving tautological tests) that have been ambiguous since 2026-05-13.

### Explicit non-goals

- Algorithmic changes to any solver, kernel, or parity oracle.
- New features.
- Deleting `_cpu_ordered` byte-identity twins (contractually required).
- Touching `optimizer_jax_private/_lbfgsb_scipy.py` SciPy 1.17.1-compatible port (vendored algorithm; load-bearing for `optimizer_backend="ondevice"`).
- Public API renames (private helpers may rename freely).
- Removing public compatibility kwargs, including `compute_derivatives` on Biot-Savart current-derivative methods.
- Deleting probe scripts that are still imported, executed, documented, or used as parity oracles.
- Tolerance changes (`PARITY_LADDER_TOLERANCES` is frozen).
- Speed optimization (LOC reduction is the goal; speed must not regress, but no performance work).

---

## 2. Success Criteria

A tier is "complete" when **all** of:

1. [ ] The change measurably reduces a named complexity symptom (change-amplification, cognitive load, or duplicated edit-sites), per `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md` — complexity is explicitly *not* line count (SD:107) and refactors must "measurably reduce complexity, not just rearrange it" (SD:81). Net LOC reduction toward the tier estimate is a secondary indicator, not the gate.
2. [ ] `tests/test_jax_import_smoke.py` + `tests/field/test_biotsavart_jax.py` + `tests/geo/test_*_jax.py` pass.
3. [ ] `tests/integration/test_jax_native_path.py` + `tests/integration/test_stage2_jax.py` + `tests/integration/test_single_stage_jax_cpu_reference.py` pass.
4. [ ] `tests/test_benchmark_helpers.py` + `tests/test_run_code_benchmark_common.py` pass.
5. [ ] `_pre_newton_census_gate_failures` produces byte-identical output on a pinned input (single_stage_init_parity gate).
6. [ ] The applicable `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md` tier gate is complete before implementation: design-it-twice + information-hiding test for Tier 2+ structural work, and API-evolution gate for public/widely-used API changes.
7. [ ] GPU/strict-transfer proof exists before tier completion for any change touching CUDA/parity-sensitive surfaces (`sharding.py`, `biotsavart_jax_backend.py`, Stage 2/single-stage parity infrastructure, optimizer routing, or transfer/cache runtime config).
8. [ ] `ruff check` + `ruff format` clean on touched files.
9. [ ] No new mypy errors on touched files (pre-existing upstream errors expected).
10. [ ] Git diff reviewed; no accidental deletion of independent-oracle test assertions; no accidental relaxation of tolerance values.

### Aggregate success

Historical target: total net deletion >= 8,000 LOC after all tiers; zero feature regressions; one ratified contract decision per Tier-4 item. Under the v8 drift gate, this target was not satisfied by the drift-checkpoint dirty tree. It can only be claimed after banked-shrink slices are isolated, validated, and re-measured.

### Closure-mode success

The next pass is a closure pass, not another open-ended micro-slice pass. Use the live unchecked section count as the execution queue; the old aggregate "33 items" count is historical until re-estimated.

1. [ ] Classify each remaining unchecked section as `do-now`, `decision-only`, or `defer/no-bank` before implementation.
2. [ ] For already-partial T2 items (`T2.1`, `T2.2`, `T2.3`, `T2.8`), spend one bounded scan looking for a source-negative follow-up. If none is obvious and contract-safe, close the remaining scope as `defer/no-bank` with rationale instead of continuing tiny helper slices indefinitely.
3. [ ] For T4 items, write a docs-backed contract decision first. Only implement code if that decision identifies a safe deletion or quarantine target.
4. [ ] For private-helper, source-negative refactors that do not touch public APIs, backend modes, GPU/transfer policy, or tolerance contracts, use the light review gate: local validation plus one adversarial reviewer. Reserve four-reviewer passes for public API, math-heavy, cross-module, GPU/transfer-sensitive, or parity-oracle changes.
5. [ ] A closure commit should either bank meaningful source LOC, close/defer a remaining task, or record a contract decision. Do not create more micro-commits just to bank single-digit LOC unless the slice also removes a repeated edit-site.

---

## 3. Guiding Principles

Drawn from `/Users/suhjungdae/code/columbia/AGENTS.md` + `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md` + this codebase's `CLAUDE.md`:

1. **SSOT.** Every result-dict schema, every tolerance table, every mode dispatch ladder should live in exactly one place.
2. **DRY at the factory level, not the assert level.** Templates fold into factories; oracle assertions stay independent and named.
3. **Immutable inputs preferred.** Use `dataclasses.replace`, `jax.tree.map`, frozen dataclasses; avoid mutation-as-update.
4. **Functional core, imperative shell.** Pure JAX kernels stay pure; impure plumbing (env vars, host conversions, profiling) lives at the boundary.
5. **One-shot wins before factories before structural folds.** Mechanical deletions don't change behavior; factories change one shape; structural folds change many shapes — sequence by risk.
6. **Trust contracts, distrust validators of values we ourselves wrote.** Validators at module boundaries are needed; validators of constants in our own tables are not.
7. **Diagnostic ≠ production.** Hot files should not carry profiling helpers; one test caller does not justify 260 LOC in a 2208-LOC file.
8. **No new abstractions for hypothetical futures.** If the second instance is hypothetical, skip the factory.
9. **Per-touch type/format checks.** Run `ruff` on touched files only; ignore pre-existing upstream noise.
10. **Atomic commits per item.** Each checkbox should normally map to one bisectable commit. v8 exception: the current dirty-tree checkmarks must be salvage-split into isolated commits before they count as committed progress.
11. **Compatibility beats deletion.** Public signature compatibility, documented CLIs, and parity-oracle scripts stay unless a caller inventory plus migration plan proves the surface is truly retired.

**On rule conflict**, the SOFTWARE_DESIGN.md tie-breaker hierarchy governs: correctness/safety > minimize total reader cognitive load > match local convention > Boy-Scout (file/function scope only; cross-module needs explicit user scope expansion) (SD:24-33).

---

## 4. Risk Model & Guardrails

### 4.1 Load-bearing contracts — DO NOT TOUCH

Every PR review must re-confirm these survive bit-identical post-refactor.

- [ ] `_cpu_ordered` byte-identity oracles (`biotsavart_cpu_ordered.py`, `surface_fourier_jax_cpu_ordered.py`, `boozer_residual_jax` cpu_ordered branch).
- [ ] Forward/adjoint PLU factor reuse and reporting: `boozersurface_jax.py` `_traceable_plu_or_dummy` (`:3090`) / `_traceable_lu_piv_or_dummy` (`:3120`) and `_build_runtime_linear_solve_callbacks` (`:3938`); the LS-lane PLU adjoint lives in the **now-tracked** `surfaceobjectives_traceable_jax.py` `_traceable_solve_plu_linearization` (`:420`, dense matrix materialization `:454`). (`surfaceobjectives_jax.py` is now 3,110 LOC and the logic relocated to `surfaceobjectives_traceable_jax.py` (3,543 LOC), which is now a committed file.)
- [ ] `_pre_newton_census_gate_failures` at `single_stage_init_parity.py:3276` (def; used `:3353`/`:3363`; same-candidate replay gate at `:3324`; same-candidate run wiring `:4451`/`:4488`; result payload wiring `:4691`). Release blocker. Earlier v3/v4/v6/SUMMARY/T2.8 drift-note anchor sets were superseded by this current-checkout refresh.
- [ ] `PARITY_LADDER_TOLERANCES` and all sibling tolerance tables in `benchmarks/validation_ladder_contract.py`.
- [ ] 7 backend modes (`native_cpu`, `jax_cpu_fast`, `jax_cpu_parity`, `jax_cpu_float32_smoke`, `jax_gpu_fast`, `jax_gpu_parity`, `jax_mps_smoke`) + hard rejection of removed `jax_metal_smoke` / `metal` selectors.
- [ ] `XLA_FLAGS` validation BEFORE `import jax`; `XLA_PYTHON_CLIENT_*` env writes before JAX init.
- [ ] `_coil_dof_state_token` semantics (advances on aggregate writes AND SIMSOPT ancestor invalidation).
- [ ] `SquaredFluxJAX` JIT closure capture at construction + 3 drift detectors.
- [ ] `get_adjoint_runtime_state()` runtime SSOT for exact-lane adjoint.
- [ ] `_normalize_solver_options` exact strip: function at `boozersurface_jax.py:3845` (still exact); the strip itself (`if boozer_type == "exact": normalized_options.pop("optimizer_backend", None)`) is at `:3919-3920`. (v3 cited `:3122 / 3185-3186 / 3419-3420`, all stale.)
- [ ] `int()` / `bool()` / `float()` host casts at SciPy/NumPy boundary + `linear_solve_status.iterations` device-placed `int32`.
- [ ] Mixed-quadrature grouped dispatch with static `group_count`.
- [ ] Stellsym DOF scatter convention (cos-cos + sin-sin for x; cos-sin + sin-cos for y, z).
- [ ] Multi-machine FP tolerance floors (`sdofs_inf ≤ 1e-11`, `rtol=1e-12` reserved for same-state direct-kernel single-machine).
- [ ] Independent-oracle test assertions (C++ symbol parity, FD/Taylor convergence, closed-form analytic).

### 4.2 Validation gates per tier

| Tier | Required passing test suites | Mode flips required | GPU/strict-transfer proof |
|------|------------------------------|---------------------|---------------------------|
| T1 | `jax_import_smoke` + all `test_*_jax*.py` unit suites + ruff/format on touched files | none for pure host-only items | required if touched files are GPU-sensitive |
| T2 | T1 + `test_stage2_jax` + `test_jax_native_path` + `test_run_code_benchmark_common` | `native_cpu` + `jax_cpu_parity` for parity-facing helpers | required for backend/runtime/sharding/field/optimizer changes |
| T3 | T2 + `test_single_stage_jax_cpu_reference` (parity-strict) + `single_stage_init_parity.py` gate replay on pinned fixture | `jax_cpu_parity` lane manual run | required for tier exit |
| T4 | tier-specific; see Section 8 | tier-specific | tier-specific |

### 4.3 Per-item pre-flight check (apply to every change)

1. ☐ `git grep` for all callers of the symbol being changed/deleted.
2. ☐ Confirm no live caller or user contract remains in `src/`, `examples/`, `benchmarks/`, `tests/`, `docs/`, `.github/`, slurm scripts, and repo-local artifacts if present. In the `simsopt-jax-shared-jax` checkout reviewed at `b267b0d95`, `.artifacts/` is absent, so artifact-backed claims must first verify the external artifact path.
3. ☐ Classify the change under `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md`.
4. ☐ For Tier 2+ or new-module work, write the design-it-twice comparison and information-hiding test before implementation.
5. ☐ For public/widely-used API surfaces, complete the API-evolution gate (SD:224-235): observable-behavior delta (Hyrum's Law), caller inventory, migration path, compatibility tests, **deprecation timeline** (when removing a surface), and rollback plan.
6. ☐ Classify any new config parameter under SD's three categories (infrastructure → externalize / externally-owned → typed+documented+tested / internally-owned → **refuse**), and run the SD post-flight new-direct-dependency checklist for any new import.
7. ☐ Identify which CLAUDE.md contract (if any) the change touches.
8. ☐ Run the validation gate for the appropriate tier.
9. ☐ Diff review: confirm no incidental deletions, no tolerance changes, no public API renames, and no new pass-through methods or implementation-inheritance state leaks (SD red-flag table).

### 4.4 Official-doc constraints checked during review

- **SIMSOPT Biot-Savart compatibility:** [official SIMSOPT field docs](https://simsopt.readthedocs.io/v0.9.1/simsopt.field.html) expose `dB_by_dcoilcurrents(compute_derivatives=0)`, `d2B_by_dXdcoilcurrents(compute_derivatives=1)`, `d3B_by_dXdXdcoilcurrents(compute_derivatives=2)`, and vector-potential siblings. JAX wrappers must keep those compatibility kwargs.
- **JAX dataclass pytrees:** [official JAX `register_dataclass` docs](https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_dataclass.html) require exact `data_fields` / `meta_fields` semantics; metadata participates in JIT cache keys and must stay static, hashable, and immutable.
- **JAX transfer guard:** [official JAX transfer-guard docs](https://docs.jax.dev/en/latest/transfer_guard.html) distinguish host/device transfer directions, note CPU device fetches are always allowed, and expose thread-local guard contexts. CUDA strict-transfer proof is therefore required for GPU purity claims; CPU-only tests cannot prove device-to-host purity.
- **JAX persistent compilation cache:** [official JAX persistent-cache docs](https://docs.jax.dev/en/latest/persistent_compilation_cache.html) require shared filesystems or remote storage for multi-process cache reuse; runtime/cache refactors must not imply local rank-0 cache paths are enough for multi-node proof.
- **SciPy BFGS/L-BFGS-B:** official SciPy docs expose [BFGS Armijo (`c1`) and curvature (`c2`) conditions](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html) and [L-BFGS-B termination semantics](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html). QFM Armijo-only solver reuse is a mathematical behavior decision, not a mechanical dedupe.

### 4.5 Current-tree status and concurrency checkpoint

**v7 correction basis (2026-05-31): HEAD `b267b0d95`, tracked tree clean before this doc edit.** The v6 basis `2bcaeff28` is now historical: `HEAD` advanced through committed MPS custom-kernel routing, MPS float32/reference-lane, status-masking, while-profile, and custom-Metal planning work. At the start of this v7 pass, `git status --short` was empty. There are no pre-existing tracked edits to preserve in this checkout.

**The v4 "uncommitted overlap" table is RETIRED.** It claimed tiny deltas (e.g. `jax_core/__init__.py` `+0/-13`) that were already wrong when written — those files had large working-tree edits (`jax_core/__init__.py` was actually `+3/-316` vs the then-HEAD) that v4 had silently folded into its LOC numbers while still calling the basis "committed." All of that is now genuinely committed, so the table has no meaning. Do not reintroduce it.

**`surfaceobjectives_traceable_jax.py` (3,543 LOC) is now a TRACKED, committed file** (v4 called it "UNTRACKED, 3,426 LOC"). It is the real split of `surfaceobjectives_jax.py` (now 3,110 LOC in the current checkout) and owns the diagnostics/profile-suite, the `_make_traceable_*` / `_ensure_traceable_runtime_*` families (T3.3 / T3.8 targets), and the LS-lane PLU adjoint `_traceable_solve_plu_linearization` (`:420`) referenced in §4.1.

**Concurrency caveat (still live).** The repo is under an active commit cadence by concurrent agents. Since v6, `git diff --name-only 2bcaeff28..HEAD -- src/simsopt tests benchmarks examples docs/...` shows committed drift in `benchmarks/single_stage_init_parity.py`, `benchmarks/validation_ladder_common.py`, the single-stage example, MPS kernel-contract code/tests, `surfaceobjectives_jax.py`, `surfaceobjectives_traceable_jax.py`, `tests/test_benchmark_helpers.py`, and these docs. **Re-grep before executing any item**, and do not assume an item is un-started just because the box reads `[ ]` — several were silently completed or re-scoped in the committed batch (see §4.7/§4.8/§4.9).

### 4.5.1 - Closure-mode acceleration update (2026-06-02)

Live unchecked-section scan after source checkpoint `4fcd33b05` originally showed 16 open sections: partial Tier 2 (`T2.1`, `T2.2`, `T2.3`, `T2.8`), all eight Tier 3 sections (`T3.1`-`T3.8`, with `T3.2` / `T3.3` / `T3.8` partially overtaken by committed work), and four Tier 4 decisions (`T4.1`, `T4.3`, `T4.4`, `T4.5`). After 2026-06-02 closure triage, bounded T2/T3 residual scans, and two additional Tier 3 helper slices, all 16 sections are closed for this bloat pass. The last active source queue was resolved as follows: `T3.1` and `T3.6` banked source-negative helper follow-ups, `T3.2` / `T3.3` / `T3.5` / `T3.7` are closed as `defer/no-bank` or `needs-design/no-bank`, and `T3.4` / `T3.8` were already closed no-bank.

Execution buckets:

- **Do now:** completed for this bloat pass. `T3.1` banked the remaining backtracking `record_every` recorder fold, and `T3.6` banked the component-scalar / surface-geometry comparison helper fold. No source section remains active without a separate design gate.
- **Closed decisions:** `T4.1`, `T4.3`, `T4.4`, and `T4.5` are docs-backed no-deletion/no-bank decisions as of 2026-06-02. Do not reopen them as LOC targets unless new workflow evidence exposes a safe migration, quarantine, or independent-oracle rewrite with its own acceptance gate.
- **Defer/no-bank:** already-partial T2/T3 residuals where another slice would preserve contracts but not bank source LOC. The current no-bank set is partial Tier 2 residuals, `T3.2`, `T3.3`, `T3.5`, `T3.7`, `T3.4`, `T3.8`, and Tier 4 decisions. Record the rationale and stop counting the residual as active bloat work.

Review gate override for closure mode:

- Use one adversarial reviewer for private-helper/source-negative refactors that stay within one module family and do not touch public APIs, backend modes, parity tolerances, GPU/transfer policy, or independent-oracle tests.
- Use the full multi-review gate for public API, math-heavy, cross-module, GPU/transfer-sensitive, parity-oracle, or benchmark/runtime contract work.
- Always keep `git diff --check`, touched-file `ruff`, touched-file `mypy` where applicable, and the narrowest behavior test that covers the edited route.

Parallelization plan:

- **Sequence barrier 0:** finish and commit the docs-only closure baseline before source implementation, so every worker uses the same `4fcd33b05` plus closure-mode anchor.
- **Parallel read-only wave:** complete for the closure pass. Partial T2 bankability scouts and bounded T3 scouts returned buckets, write sets, expected source bank, validation gates, and conflict sets.
- **Parallel implementation allowed only with disjoint write sets:** an optimizer-only `T3.1` slice and a benchmark-driver-only `T3.6` slice can proceed concurrently only if each worker owns separate files and docs staging stays serialized. Do not let parallel workers edit the same plan docs at the same time.
- **Sequential implementation required for overlap clusters:** `T3.3` with any future reopened `T3.8`; `T3.5` with `T3.6`; any future reopened `T2.8` work with `T3.5` / `T3.6`; any future reopened `T2.1` work with linear-solve-callback work; and any GPU/transfer, parity-oracle, backend-mode, or tolerance-sensitive item.
- **Sequence barrier 1:** after each source commit, refresh `git status`, touched-file inventories, and the relevant owner-doc lines before starting another source commit in the same file family.
- **Sequence barrier 2:** run grouped closure validation and adversarial review before tagging any Tier 3/Tier closure complete. Strict-transfer validation remains required only for future GPU/transfer-sensitive changes; no CUDA proof is claimed by this pass.

### 4.6 — v4 reconciliation summary (2026-05-29)

Findings from the 6-agent re-verification against HEAD `5bcd9061c`:

- **Scaffolding is the stale part, not the substance.** Almost none of the 33 refactors has been executed, so the consolidation work is still wanted. What drifted is the basis commit, the §4.5 tables, and nearly every line ref/count.
- **Items already done / overtaken (now marked):**
  - **T2.2** — v4 said ALREADY DONE, but v6 corrects this to **PARTIAL / NOT BANKED**: `boozer_radial_field.py` still carries parallel `_eval_*_from_columns` implementations (`:452-787`) and separate `_eval_*` implementations (`:807-1190`). Keep the item open until the thin-wrapper fold lands; do not remove its ~400 LOC from the T2 target.
  - **T3.3 / T3.8** — PARTIALLY DONE / RELOCATED into `surfaceobjectives_traceable_jax.py` (diagnostics + `_make_traceable_*` family), not the `_diagnostics` sibling the items name. Both residuals have since been closed for this pass: T3.3 as `defer/no-bank` for bloat and T3.8 as `defer/no-bank`.
  - **T4.2** — DECISION MADE by Plan A (HANDOFF.md, 2026-05-29), with v6 wording corrected against live defaults: `scipy-jax` is the default JAX optimizer lane on both CPU and CUDA; `scipy-jax-fullgraph` remains an explicit stress/parity lane. This collapses to docs, not removal.
  - **T4.4** — CLOSED KEEP: `SLSQP`/`slsqp` has 0 occurrences in `qfm_solver.py`, but the public compatibility alias is alive in the qfm *surface* wrappers (`qfmsurface_jax.py:281`, `qfmsurface.py:147`) and remains supported. See §8.4.
  - **T1.11** — VERIFIED satisfied (`_skip_case` used 19×).
- **Corrected counts/figures:** T1.1 `jax_core/__init__.py` is 363 LOC not 676 (saving ~300 not ~620); T1.3 has five true GPMO result mirrors plus a separate `PMRelaxAndSplitResult`, not six true GPMO mirrors; T1.6 covers 6 methods not 12; T1.8 over-counted dead aliases (`_zero_profile_component_timings` is live at `:2222`).
- **SOFTWARE_DESIGN.md alignment:** the plan is substantively compliant (cited gate names match SD verbatim; SD mtime 2026-05-19 predates the plan, so refs are current). v4 added: the complexity-not-LOC success gate (§2.1), the tie-breaker precedence note (§3), and the config-param/dependency/deprecation-timeline pre-flight steps (§4.3). One item to watch during execution: **T3.2 mixin extraction** vs SD's "implementation inheritance leaks state — prefer composition" red flag (SD:283).

### 4.7 — v5 reconciliation summary (2026-05-30)

Findings from the historical v5 4-agent re-verification against clean HEAD `21c3d517d`, plus v6/v7 current-checkout corrections:

- **The bloat refactors themselves are still almost entirely un-started.** The committed batch (`f287bde96` and successors) was *audit remediation* — correctness / strict-CUDA / test-oracle fixes — not *bloat reduction*; the consolidation work in §5–§8 remains wanted. v4's structural conclusions stand; v5 only refreshes the basis, the line refs, and a few item statuses.
- **~17–19 of the 2026-05-20 code-smell audit's High-cluster findings are now fixed in-tree** (verified by re-grep, not just re-pointed; a handful are only *partially* fixed — see the per-finding `RESOLVED`/`PARTIAL` annotations in the refreshed reports). Fully resolved: A1-F1 (`_safe_radius_squared`→`_radius_squared`, clamp removed), A2-F3 (`_set_global_coil_dofs` token-after-mutation), A2-F12 (gradient delegation), A4-F1/A10-M2 (`recompute_bell` token+cache), A4-F2 (precomputed surface signature), A5-F2 (all `jax.debug.callback` now `ordered=False`), A6-H1 (`f_bwd` NaN routing — was a CLAUDE.md contract FAIL, now PASS), A6-H2 (`sdofs` kwarg), A6-M5 (column-batched adjoint), A7-H1 (conftest no longer registers the deleted `backend.py`), A7-M1 (`metal`→`jax_mps_smoke`), A7-M2 (CLAUDE.md backend row), A8-H4 (vectorized `modB`), A9-H3 (vectorized coil unroll), A9-M1 (runner cache re-keyed to `_cached_traceable_runner`), A10-H1 (QfmResidualJAX hot-path read + `append_parent` removed), A8-H3 (`jax.local_devices()` at both `surfaceobjectives_jax.py:2151` and `src/simsopt/solve/mpi_jax.py:114`), A11-H2/H3/H4/M3 (test-oracle fixes). V6 correction: A2-F2/A10-H2 is **resolved** by `SpecBackedBiotSavartJAX.update_free_dof_size_indices` advancing `_dof_layout_version`; the missing spec-backed `set_recompute_flag` override is moot because the spec-backed adapter owns its DOFs directly and has no upstream `depends_on` cascade. A8-H1 remains partial: `wireframe_workflow.py:390-411` still has `jnp.asarray(max_steps/history_capacity, dtype=...)` paths, and `:410-411` plus `:1167` remain outside the tracer-only branch. These touch parity-sensitive surfaces but are orthogonal to the bloat plan. Artifact status correction: `.artifacts/` and `verification/` are absent from the `simsopt-jax-shared-jax` checkout at `b267b0d95`; the refreshed audit artifacts were found in sibling checkout `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/`, with final retraction status at `verification/CORRECTIONS_round5.md` under that artifact tree.
- **Item already done / overtaken (addition to v4's list):**
  - **T3.2** — the field-evaluation duplication is **ALREADY DONE** via `_BiotSavartFieldEvaluationMixin` and the 2026-06-01 points-only follow-up now banks 2 LOC through private point-state helpers while preserving public method metadata. The remaining T3.2 work is only the diverged `coil_cotangents_to_dofs_gradient`; re-scope down from the old ~250 LOC estimate.
- **Corrections to v4 item text:**
  - **T2.9** — v4/v3 over-counted the migration sites. Before the 2026-06-01 T2.9 slice, `_tolerance_for` existed in **only one** file (`non_banana_example_cpp_jax_cpu_parity.py`, then around `:323`). After that slice, the benchmark keeps a compatibility wrapper and the quantity policy lives in `validation_ladder_contract.py::quantity_parity_tolerance(...)`; see §6.9. `single_stage_parity_matrix.py` has **no** `_tolerance_for` (the cited `:292-302` is wrong), and `stage2_e2e_comparison.py` uses a different mechanism (`optimizer_drift_tolerances:71/74`). The ~100 LOC estimate was overstated and was not banked.
  - **T4.4** — **NOT obsolete and now closed as KEEP.** `minimize_qfm_exact_constraints_SLSQP` has 0 occurrences in `qfm_solver.py` (true; the exact path is augmented-Lagrangian), but the public alias **still exists** in the surface wrappers `qfmsurface_jax.py:281` and `qfmsurface.py:147`, with live example and test callers. No quarantine is planned in this bloat pass.
  - **T2.8** — "16 trackers" undercounted the pre-helper inventory. The 2026-06-01 core helper now owns the two layer-decomposition drift families; remaining tracker cleanup should re-grep from `LayerDriftTracker` at `:2969` and `compare_same_candidate_objective_replay` at `:3395`, not v3's `:2300-2700`.
  - **T1.1** — the `jax_core/__init__.py` 676→363 LOC drop is **NOT** the planned lazy-export-map conversion; the file still carries the full explicit dual-list. T1.1 remains genuinely `[ ]` with ~300 LOC available. (Do not infer "done" from the LOC delta.)
- **Two stale LOC counts fixed:** `tracing.py` is 4,287 (v4 said 4,299); `surfaceobjectives_traceable_jax.py` was 3,428 in v5/v6 and is 3,543 in the v7 checkout.

### 4.8 — v6 doc-review correction summary (2026-05-30, historical)

- **Current HEAD/status at v6:** live HEAD was `2bcaeff28`, not `21c3d517d`; the tracked worktree had pre-existing source/test edits outside this document. Treat v5's clean-tree language as historical only.
- **Artifact status correction:** the sibling artifact-tree file `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/verification/CORRECTIONS_round5.md` retracts A6-H3/item 12 and A1-F2/item 17 as active bugs. A6-H3's tuple-arity future-proofing assert is landed at `surfaceobjectives_traceable_jax.py:399-402`; A1-F2 matches SciPy/FITPACK repeated-knot behavior. This checkout does not contain a repo-local `verification/` directory.
- **DOF invalidation correction:** A2-F2/A10-H2 is fixed by the spec-backed `_dof_layout_version` bump; do not require or cite a spec-backed `set_recompute_flag` override. The live mutable `BiotSavartJAX` override remains at `biotsavart_jax_backend.py:1370-1377`.
- **Execution-accounting correction:** T2.2 is not already banked. The column bundle exists, but the duplicated `_eval_*_from_columns` and `_eval_*` families remain in `boozer_radial_field.py`; totals below now count it as remaining candidate work.

### 4.9 — v7 current-checkout refresh summary (2026-05-31)

- **Current HEAD/status:** live HEAD is `b267b0d95`, not `2bcaeff28`; `git status --short` was empty before this edit.
- **MPS custom-kernel drift classification:** commits from `20e0d3aa8` through `b267b0d95` landed MPS custom-kernel routing, status-masking, float32-reference, while-profile, and custom-Metal planning work. These are adjacent JAX/MPS changes, not completed bloat-reduction checklist items.
- **Moved refs corrected:** `single_stage_init_parity.py` release-gate refs, `surfaceobjectives_traceable_jax.py` diagnostics/lazy-cache refs, `single_stage_banana_example.py` optimizer-default refs, and `validation_ladder_common.py` cache-policy refs were refreshed against `b267b0d95`.
- **Still-open bloat status at the 2026-05-31 refresh:** T1.1 remained un-started, T2.2 remained partial/open, T3.2 remained partial, and T4.2 remained resolved/doc-only. This line is superseded by the 2026-06-02 closure-mode classification in §4.5.1 and the per-item statuses below.

### 4.10 — Contract-first prerequisite slice (2026-06-01)

- **Current HEAD/status:** execution started from `8b94c2bbd` on `shared-jax-clean`. The first implemented slice was deliberately not a bloat LOC deletion: it hardened the TORAX-derived prerequisite contracts before import/export consolidation.
- **Completed prerequisite:** added a static/dynamic JAX spec specialization test in `tests/core/test_jax_core_specs.py` and two-process persistent-cache reuse proof in `tests/subprocess/import_smoke_cases.py` / `tests/test_jax_import_smoke.py`.
- **Bloat status at this checkpoint:** T1.1 and T1.2 remained open. A read-only lazy-export inventory confirmed T1.1 first needs literal `__all__` definitions in six `jax_core` submodules, and T1.2 first needs `src/simsopt/backend/runtime.py` to own the public backend `__all__` before the facade can collapse safely.
- **Validation evidence:** see `docs/bloat_torax_coherent_execution_plan_2026-05-31.md` execution log for exact commands. This was CPU-only contract proof; it is not CUDA or MPS evidence.

### 4.11 — T1.2 backend facade collapse (2026-06-01)

- **Current HEAD/status:** execution continued from `8b94c2bbd` on `shared-jax-clean`, after the contract-first prerequisite slice.
- **Completed bloat item:** `src/simsopt/backend/runtime.py` now owns the literal 54-name public backend `__all__`, and `src/simsopt/backend/__init__.py` is a four-line facade driven by `runtime.__all__`.
- **Regression proof added:** `tests/test_backend.py::test_backend_public_facade_uses_runtime_all_as_ssot` pins the exact 54-name public export tuple for both `runtime.__all__` and the package facade, and verifies `from simsopt.backend import *` exposes exactly that namespace.
- **Bloat status at this checkpoint:** T1.2 is complete. T1.1 remained open and still needed literal `__all__` definitions before `jax_core/__init__.py` could safely switch to `_lazy_exports`; see §4.12 for the later T1.1 completion.
- **Validation evidence:** see `docs/bloat_torax_coherent_execution_plan_2026-05-31.md` execution log for exact commands. This was CPU import/API-surface proof; it is not CUDA or MPS evidence.

### 4.12 — T1.1 `jax_core` lazy facade collapse (2026-06-01)

- **Current HEAD/status:** execution continued from `8b94c2bbd` on `shared-jax-clean`, after the contract-first prerequisite slice and T1.2 backend facade collapse.
- **Completed bloat item:** `src/simsopt/jax_core/__init__.py` now uses `_lazy_exports.build_lazy_export_map(...)` and lazy `resolve_lazy_export(...)` instead of the eager fourteen-module import block, explicit `_EXPORT_MODULES`, duplicated `__all__`, `_EXPORT_MODULE_OBJECTS`, and manual resolver.
- **Live inventory correction:** the prerequisite inventory originally identified six missing literal `__all__` modules. The implementation probe found `src/simsopt/jax_core/field.py` also lacked a literal `__all__`; T1.1 therefore materialized seven module-local export lists (`curve_geometry`, `field`, `objectives_flux`, `specs`, `surface_fourier`, `surface_henneberg`, `surface_rzfourier`).
- **Compatibility detail:** the package-level public export sequence remains the historical 314-name order and every old exported name resolves. The old facade advertised `evaluate_interpolated_boozer_scalar` but that attribute did not exist on the target submodule; T1.1 adds it as an alias of `evaluate_scalar`.
- **Helper-contract cleanup:** final delta review found the shared lazy-export helper still accepted same-module duplicate export names. T1.1 now rejects any duplicate package export before package-order overrides can mask it, adds a focused regression in `tests/test_lazy_exports.py`, and removes an existing duplicate `FramedCurve` entry from `src/simsopt/geo/framedcurve.py`.
- **Regression proof added:** `tests/subprocess/import_smoke_cases.py::case_jax_core_lazy_facade_public_contract` and `tests/test_jax_import_smoke.py::test_jax_core_lazy_facade_public_contract` verify the facade stays lazy before first attribute resolution, preserves the 314-name curated package export set, excludes broader direct-submodule-only exports, resolves the interpolated-Boozer alias, and exposes the same set through `from simsopt.jax_core import *`. `tests/test_lazy_exports.py` pins same-module duplicate rejection with and without `package_export_order`.
- **LOC/accounting:** `src/simsopt/jax_core/__init__.py` is now 146 LOC (363 → 146, `-217` in the facade) while preserving historical `__all__` order. Production-file diff for the full T1.1 slice is `475 insertions(+), 358 deletions(-)` because the missing module-local literal `__all__` declarations, exact package-order table, shared-helper package override support, duplicate-export hardening, and one duplicate direct-module export cleanup were added in the same slice; full source/test diff is `750 insertions(+), 358 deletions(-)` including the new 20-line helper regression test.
- **Bloat status:** T1.1 implementation and validation are complete, but the full production slice is net-positive LOC because public export order and independent-oracle constraints are preserved. Count only the facade-local `-217` as the direct `jax_core/__init__.py` simplification; do not bank a net T1 LOC reduction for the full slice. T1.2 is also complete; remaining Tier 1 items start at T1.3 unless the next slice deliberately selects another still-open item.
- **Validation evidence:** see `docs/bloat_torax_coherent_execution_plan_2026-05-31.md` execution log for exact commands. This was CPU import/API-surface proof; it is not CUDA or MPS evidence.

### 4.13 — T2.2 radial-evaluator dedup slice (2026-06-01)

- **Current status:** the direct `boozer_radial_field.py` `_eval_*` Fourier/scalar formula duplication is folded. Fourier direct evaluators now route through the canonical `_eval_*_from_columns` family via the typed generic `_direct_radial_evaluator(...)` (`src/simsopt/jax_core/boozer_radial_field.py:1018`), including the modB value/theta/zeta subset-column factory at `:542`; scalar direct evaluators use `_direct_scalar_evaluator(...)` (`:1037`, assignments at `:1114`) to preserve the original scalar spline path. `_eval_radial_columns` remains the public-wrapper points-cycle cache path (`:405`).
- **Hot-path guard:** the first cheap full-bundle wrapper version regressed the Boozer RHS benchmark, so the landed design adds RHS-specific radial columns (`_eval_radial_rhs_columns`, `:521`) and tracing dispatch through `_RADIAL_RHS_COLUMN_EVALUATORS` (`src/simsopt/jax_core/tracing.py:2711`) instead of calling each direct evaluator separately.
- **Bloat accounting:** do **not** bank the old `~400 LOC` estimate. The formula-dedup slice was not LOC-banked because benchmark-preserving subset builders offset the direct-formula deletion (`boozer_radial_field.py` diff was `262 insertions / 264 deletions`; `tracing.py` added the RHS dispatch path). The 2026-06-01 direct-wrapper follow-up banks 35 source LOC in `boozer_radial_field.py` (`99 insertions / 134 deletions`; 1,191 -> 1,156 LOC), the scalar-helper follow-up banks 12 more source LOC (`26 insertions / 38 deletions`) by factoring only scalar direct-evaluator ceremony into `_direct_scalar_evaluator(...)`, the pass-through inline follow-up banks 10 more source LOC (`2 insertions / 12 deletions`; 1,144 -> 1,134 LOC) by deleting the one-use `_eval_with_radial_columns(...)` helper, and the typed modB direct-factory follow-up banks 14 more source LOC (`24 insertions / 38 deletions`; 1,134 -> 1,120 LOC) by routing value/theta/zeta through the generic factory. The full T2.2 estimate remains unbanked.

### 4.14 — T2.4 spec dataclass auto-registration helper (2026-06-01)

- **Current status:** completed. `src/simsopt/jax_core/specs.py` now has one private `_register_jax_spec(...)` helper at `:102` and 29 local decorator uses. The only remaining direct `jax.tree_util.register_dataclass(...)` call is inside that helper at `:109`.
- **Contract shape:** each spec class still declares its own `data_fields` and `meta_fields` beside the class definition, now as immutable tuple literals. This keeps the traced-data vs static-meta split locally auditable while removing the repeated frozen-dataclass plus manual-registration blocks.
- **Regression proof added:** `tests/core/test_jax_core_specs.py::test_register_jax_spec_helper_preserves_data_meta_partition` proves the helper preserves JIT cache behavior: data-field changes reuse the compiled specialization, while meta-field changes force a new specialization. Existing `CurveXYZFourierSpec` cache-key and tree-signature coverage still exercises a real production spec class.
- **Bloat accounting:** `specs.py` is now 1,570 LOC versus the pre-slice 1,622 LOC cited for T2.4. Bank about 52 net LOC for the file; do not bank the earlier `~140` estimate as net source reduction because preserving explicit per-class partitions requires readable multiline decorators and the helper itself.
- **Validation evidence:** focused helper/real-spec/import smoke passed (`3 passed`); full core spec plus tree-signature suite passed (`33 passed`); import smoke for `test_jax_core_specs_are_pytrees` passed (`1 passed`); scoped `ruff check`, `ruff format --check`, `py_compile`, and `git diff --check` passed. `mypy` remains blocked in `.conda/jax` with `No module named mypy`.

### 4.15 — T2.5 leading-axis sharding helper slice (2026-06-01)

- **Current status:** completed as a contract-preserving helper fold. `src/simsopt/jax_core/sharding.py` now owns `_LeadingAxisBatchShardingConfig`, `_leading_axis_sharding_config(...)`, `_maybe_shard_leading_axis_inputs(...)`, and `_leading_axis_sharding_summary(...)`; the public `TrajectoryBatchShardingConfig`, `SeedBatchShardingConfig`, and `SurfaceQuadratureShardingConfig` names remain concrete dataclass subclasses with the same four public fields.
- **Contract shape:** the three public config classes still construct with `mesh`, `axis_name`, `device_count`, and `strategy`; the three summary functions still emit the old JSON keys (`trajectory_sharded`, `seed_batch_sharded`, `surface_quadrature_sharded`, and their device-count keys).
- **Regression proof added:** `tests/jax_core/test_sharding_helpers.py::test_leading_axis_sharding_configs_preserve_public_contract` pins the public class names, dataclass fields, shared placement path, unsharded summary shape, sharded summary key names, axis, strategy, mesh shape, and device-count keys.
- **Bloat accounting:** bank only 6 net production LOC (`src/simsopt/jax_core/sharding.py` 725 -> 719; diff `102 insertions / 108 deletions`). The earlier `~170` estimate would require collapsing exported config classes into one public class/alias, which this slice deliberately avoided as a public API change.
- **Validation evidence:** focused helper/trajectory summary proof passed (`4 passed`); forced CPU surface/seed sharding subprocess cases passed (`6 passed`); forced CPU points-coils subprocess cases passed (`2 passed`); scoped `ruff check`, `ruff format --check`, `py_compile`, and `git diff --check` passed. `mypy` remains blocked in `.conda/jax` with `No module named mypy`. This is CPU forced-device sharding proof, not CUDA/GPU proof.

### 4.16 — T2.7 SciPy adapter closure factory (2026-06-01)

- **Current status:** completed. `src/simsopt/geo/optimizer_jax_reference.py` now shares SciPy host objective construction through `_make_scipy_host_value_and_grad_objective(...)` at `:192` and shared dispatch attachment through `_scipy_minimize_value_and_grad_core(...)` at `:230`.
- **Contract shape:** `_scipy_minimize(...)` and `_scipy_minimize_value_and_grad(...)` still call `_require_native_cpu_reference_backend_for_scipy_adapter(...)` under their own component names before SciPy can run; `target_scipy_minimize_value_and_grad(...)` still keeps the explicit target-lane `method='lbfgs'` check and does not add the private-reference backend guard.
- **Bloat accounting:** bank 30 net production LOC (`src/simsopt/geo/optimizer_jax_reference.py` 569 -> 539; diff `63 insertions / 93 deletions`). The earlier `~220` estimate is not banked; preserving the explicit guard wrappers and target-lane method check keeps the public adapter contract readable.
- **Validation evidence:** SciPy adapter host-contract and target-lane selectors passed (`10 passed, 469 deselected`); private adapter JAX-backend rejection passed (`8 passed`); optimizer-reference result tests passed (`4 passed`); import-smoke backend rejection passed (`1 passed`); scoped `ruff check`, `ruff format --check`, `py_compile`, and `git diff --check` passed. `mypy` remains blocked in `.conda/jax` with `No module named mypy`.

### 4.17 — T2.6 backend runtime resolver fold (2026-06-01)

- **Current status:** completed as a behavior-preserving small refactor. `src/simsopt/backend/runtime.py` now owns one `_resolve_kwarg(...)` helper for explicit-kwarg → env override → mode-default precedence, and the eight old private `_resolve_*` runtime kwarg helpers are gone.
- **Contract shape:** `_config_from_mode(...)` still resolves every `BackendConfig` field explicitly; there is no heterogenous dict/`**kwargs` builder. This keeps the public `set_backend(...)` keyword names, mode defaults, env validation sources, debug overlay, and cache/transfer/gpu allocator semantics visible at the construction site.
- **Regression proof added:** `tests/test_backend.py::test_runtime_kwargs_override_env_before_mode_defaults` sets every runtime-kwarg env override family and proves explicit kwargs win over those env values before mode defaults, including an invalid TF allocator env value overridden by a valid explicit kwarg. `tests/test_backend.py::test_simsopt_debug_env_applies_runtime_debug_overlay` now also pins the old debug-overlay short-circuit: invalid `SIMSOPT_JAX_DISABLE_JIT` and `SIMSOPT_JAX_TRANSFER_GUARD` env values are ignored when `SIMSOPT_DEBUG=1` owns those fields. Existing backend tests still cover mode defaults, transfer guard, compilation cache fallbacks, GPU memory env application, and eager-JAX import behavior.
- **Bloat accounting:** bank only a small runtime simplification, not the old `~100 LOC` estimate. Current `runtime.py` is 2,535 LOC from a 2,500-LOC HEAD baseline, but that includes the earlier T1.2 runtime `__all__` addition; relative to the already-applied T1.2 context, T2.6 is approximately a low-twenties net source reduction and mainly reduces change amplification.
- **Validation evidence:** full `tests/test_backend.py` passed (`130 passed, 2 expected CUDA-flag warnings`) after the review-found debug-overlay regression was fixed; backend import-smoke selectors passed (`3 passed`); scoped `ruff check`, `ruff format --check`, `py_compile`, guardrail diff scan, and `git diff --check` passed. `mypy` remains blocked in `.conda/jax` with `No module named mypy`.

---

## 5. Tier 1 — Mechanical Wins

**Target:** ~630–670 LOC guaranteed reduction, plus up to ~1,245 LOC only if probe-script migration proves safe. **Effort:** ~2 days. **Risk:** Low-Medium.

Goal: bank low-risk LOC reduction first; pattern-validate factory ideas in tiny scope.

### 5.1 — [x] T1.1: `jax_core/__init__.py` → lazy export map

- **Files:** `src/simsopt/jax_core/__init__.py` (363 → 146 LOC), `src/simsopt/_lazy_exports.py`, seven `jax_core` submodules that now own literal package-relevant `__all__` declarations, `src/simsopt/geo/framedcurve.py`, `tests/subprocess/import_smoke_cases.py`, `tests/test_jax_import_smoke.py`, `tests/test_lazy_exports.py`.
- **Change:** Replaced the explicit dual list (`_EXPORT_MODULES`, duplicated `__all__`, `_EXPORT_MODULE_OBJECTS`, manual `__getattr__`) with `_lazy_exports.build_lazy_export_map(...)` matching the sibling lazy-facade pattern. The shared helper now accepts a package-level export override for modules whose direct submodule `__all__` is intentionally broader than the package facade.
- **LOC saved/accounted:** the facade itself saves 217 LOC. Production-file net for the complete T1.1 slice is `475 insertions(+), 358 deletions(-)` because missing submodule-local literal `__all__` lists, historical package export-order preservation, duplicate-export hardening, and one duplicate direct-module export cleanup had to be materialized; full source/test diff is `750 insertions(+), 358 deletions(-)` including the new helper regression test.
- **Risk:** Low-to-medium public-facade risk, covered by import-smoke regression. The package public export sequence remains the historical 314-name order.
- **Contracts:** All fourteen facade source modules now have a literal `__all__` or a package override. The helper raises on cross-module and same-module duplicates; package overrides preserve the old curated facade subset for `biotsavart` and `interpolated_boozer_field` without removing broader direct-submodule exports.
- **Validation gate:** T1 complete; exact commands recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.2 — [x] T1.2: `backend/__init__.py` dual-list collapse

- **Files:** `src/simsopt/backend/__init__.py` (117 LOC -> 4 LOC), `src/simsopt/backend/runtime.py`, `tests/test_backend.py`.
- **Change:** Replaced the facade's explicit import list and duplicated `__all__` with `from .runtime import *` driven by a literal `runtime.__all__`.
- **LOC saved:** production files are `59 insertions(+), 115 deletions(-)`; the full source/test slice is `129 insertions(+), 115 deletions(-)` because the test now carries an independent literal public-export oracle.
- **Risk:** Low public-facade risk. Existing public names and object identities are pinned by test.
- **Validation gate:** T1 complete; exact commands recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.3 — [x] T1.3: Retire GPMO `Result` dataclass mirrors

- **Files:** `src/simsopt/solve/permanent_magnet_optimization_jax.py:116-314` (file now 880 LOC; v3 cited `:99-303`), `tests/solve/test_permanent_magnet_optimization_jax_item28.py`.
- **Change:** Replaced the five true solve-level GPMO result mirrors (`GPMOBaselineResult`, `GPMOMultiResult`, `GPMOBacktrackingResult`, `GPMOArbVecResult`, `GPMOArbVecBacktrackingResult`) with concrete public subclasses of one `GPMOPublicResult(m, m_history, core_result)` storage contract. `PMRelaxAndSplitResult` was left unchanged because live code showed it is a separate relax-and-split result, not a mirror of `jax_core.pm_optimization`.
- **LOC saved:** production slice is `163 insertions(+), 204 deletions(-)` (`-41` net). The source/test slice is `336 insertions(+), 206 deletions(-)` because regression tests now pin constructor, legacy-state, frozen-assignment, export, and pytree-order compatibility.
- **Risk:** Low-to-medium public result compatibility risk, covered by PM wrapper tests and adversarial review.
- **Contracts:** Pytree registration persists with flatten order `m`, `m_history`, then `core_result` leaves. The old public class names remain real classes, old keyword/positional constructor fields are accepted, old pickle-style state without `core_result` rebuilds the matching core result, and frozen assignment behavior is preserved.
- **Validation gate:** T1 + PM tests passed; exact commands recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.4 — [x] T1.4: Centralize `_as_numpy_float64`

- **Files:** `src/simsopt/_core/jax_host_boundary.py` now owns `host_float64`; `geo/curveobjectives.py` and `geo/curvecwsfourier.py` import it directly as their local compatibility name; `geo/surfaceobjectives.py` calls it directly for VJP materialization; `geo/curve.py` keeps the only remaining one-line `_as_numpy_float64` wrapper to pass `_HAS_JAX` and preserve the no-JAX short-circuit.
- **LOC saved:** production slice is `16 insertions(+), 32 deletions(-)` (`-16` net). The source/test slice is `36 insertions(+), 32 deletions(-)` because `tests/test_host_boundary.py` now pins explicit device-host transfer and no-JAX no-import behavior.
- **Risk:** Low. The curve↔jax_core import-cycle story did not change; the shared helper lives in `_core` and does not import JAX unless a JAX boundary is requested.
- **Contracts:** `host_float64(value, has_jax=False)` returns a NumPy `float64` array without importing JAX. Default JAX-enabled calls route through `host_array(..., dtype=np.float64)` so strict transfer-guard device-to-host materialization remains explicit.
- **Validation gate:** T1 complete; exact commands recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.5 — [x] T1.5: Centralize state-token counters

- **Files:** `src/simsopt/_core/state_tokens.py`, `src/simsopt/field/biotsavart_jax_backend.py`, `src/simsopt/geo/boozersurface_jax.py`, `tests/core/test_state_tokens.py`.
- **Change:** Added `make_state_token_factory()` as the shared source for independent monotonic integer token generators. `_new_coil_dof_state_token` and `_new_traceable_solve_state_token` remain private module-local callables, preserving the existing separate token streams while removing duplicated `itertools.count()` setup.
- **LOC saved/accounted:** the two owner modules changed by `4 insertions(+), 12 deletions(-)` (`-8` net), but the production slice is net `+8` after adding the 16-line shared helper. The source/test slice is net `+18` after adding the 10-line helper regression. Count this as complexity/SSOT reduction, not a LOC bank, until another token stream migrates to the helper.
- **Risk:** Low. Token attribute names and token advancement call sites are unchanged.
- **Contracts:** Token attribute names accessible to consumers (`_coil_dof_state_token`, `_traceable_solve_state_token`, `_dof_layout_version`, `_points_version`) are unchanged. Each factory returns an independent monotonic sequence starting at zero, matching the previous per-module `count()` counters.
- **Validation gate:** T1 complete; exact commands recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.6 — [x] T1.6: Preserve `compute_derivatives=N` compatibility while deduplicating Biot-Savart current-derivative wrappers

- **Files:** `biotsavart_jax_backend.py:488-590` (**6** current-derivative methods carrying `compute_derivatives`, at `:568, 572, 576, 580, 584, 588`) and `tests/field/test_biotsavart_jax.py:1789-1849` (signature/keyword regression).
- **Change:** Kept the public `compute_derivatives` kwarg and defaults on every JAX method because SIMSOPT CPU APIs and official docs expose them. Added `_per_coil_unit_current_derivative()` on `_BiotSavartFieldEvaluationMixin` and routed the six current-derivative methods through it.
- **LOC saved/accounted:** T1.6 source body adds the 8-line helper and replaces six 5-line call blocks with six 1-line calls (`14 insertions(+), 30 deletions(-)`, `-16` net). The focused regression adds 62 test lines. The file still contains uncommitted T1.5 state-token edits, so whole-file `git diff --numstat` against HEAD also includes that earlier slice.
- **Risk:** Low. The compatibility signature is the contract; in-repo caller absence is not proof that external callers do not pass the kwarg.
- **Contracts:** `BiotSavartJAX` and `SpecBackedBiotSavartJAX` still expose all six current-derivative methods with defaults `0/1/2` matching the CPU API. The regression calls every method with `compute_derivatives=0`, `1`, and `2` on both adapters.
- **Validation gate:** T1 complete; exact commands recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.7 — [x] T1.7: Delete dead `_host_cubicmin` / `_host_quadmin` / `_line_search_sample_valid_host`

- **Files:** `src/simsopt/geo/optimizer_jax_private/_common.py:117` now jumps directly from `_line_search_sample_valid()` to `_emit_debug_callback()` after deleting the three dead host helper definitions. The deleted HEAD lines were `_host_cubicmin:117`, `_host_quadmin:144`, and `_line_search_sample_valid_host:158`; v4 said `:117-161`.
- **Change:** Deleted the private JAX optimizer's unused host-side `_host_cubicmin`, `_host_quadmin`, and `_line_search_sample_valid_host` copies. The live JAX path still uses `_cubicmin`, `_quadmin`, and `_line_search_sample_valid`; the live host optimizer keeps its own equivalents in `optimizer_host_lbfgs.py`.
- **LOC saved:** 47 lines.
- **Risk:** Very low.
- **Validation gate:** T1 complete; current grep found no source/test/benchmark/example references after deletion, and exact commands are recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.8 — [x] T1.8: Delete dead aliases in `biotsavart_jax_backend.py`

- **Files:** `src/simsopt/field/biotsavart_jax_backend.py:890` now jumps directly from `_take_positions_1d()` to `_scatter_free_values()` after deleting `_ones_like_float64`.
- **v4 NOTE resolved:** `_zero_profile_component_timings` (now `:257`) has a **live caller at `:2222`** — it was not deleted. The 4 `*_cotangents` aliases remain live public/native names (`:1957`, `:2008`, `:2017`, `:2026`) and were not changed.
- **LOC saved:** 6 lines (only `_ones_like_float64`; v3's ~15 over-counted).
- **Risk:** Very low.
- **Validation gate:** T1 complete; current grep found no source/test/benchmark/example references after deletion, and exact commands are recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.9 — [x] T1.9: Reclassify `SingleStageRuntimeSpecBiotSavartJAX` as public compatibility API

- **Files:** `src/simsopt/field/biotsavart_jax_backend.py:92` exports the name and `:788-800` defines it as a real subclass; `src/simsopt/field/__init__.py:35` re-exports the JAX backend through the package facade. Live code imports/calls it in `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:146,12810`, `tests/integration/test_single_stage_physics_parity.py:455,472`, and `tests/geo/test_single_stage_example.py:26,4354,4434,4624`.
- **Decision:** Do **not** delete in Tier 1. Current in-repo constructor calls can be mechanically expressed as `SpecBackedBiotSavartJAX(make_biot_savart_spec(...))`, but removing this exported class would be a public API break. Future removal requires Tier 3 API-evolution work: observable behavior delta, caller inventory, migration path, compatibility tests, deprecation plan, and rollback plan.
- **LOC saved:** 0. The earlier ~11 LOC bank is rejected.
- **Risk:** Low to keep; high to delete without deprecation because tests already assert package export identity and external callers may import the class.
- **Validation gate:** Reclassification complete; public class/import/subclass signature probe passed and exact commands are recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.10 — [x] T1.10: Classify probe scripts; retain active parity/smoke entrypoints

- **Files:** `benchmarks/run_code_parity_probe.py` (174), `benchmarks/production_boozer_parity_probe.py` (281), `benchmarks/single_stage_surface_reprojection_probe.py` (460), `benchmarks/surface_rz_geometry_hlo_probe.py` (339; v3 said 330).
- **Current-tree status:** Completed as classification-only on 2026-06-01. All four scripts are active; no Tier 1 deletion is safe without a separate migration/deprecation slice.
- **Classification:**
  - `benchmarks/run_code_parity_probe.py`: active parity oracle. `benchmarks/cpu_run_code_benchmark.py:22`, `benchmarks/gpu_run_code_benchmark.py:21`, and `benchmarks/run_code_benchmark_common.py:331` direct solver-parity users to this script; `tests/test_benchmark_helpers.py:66` imports it and `tests/test_benchmark_helpers.py:9662` tests its default JAX lane.
  - `benchmarks/production_boozer_parity_probe.py`: active parity oracle. `docs/solve_jax_api_caller_inventory_2026-05-19.md:266` and `:320` list it as a benchmark surface, `docs/full_repo_banana_e2e_cpu_gpu_test_plan_2026-05-19.md:865` gives an invocation, and `tests/test_benchmark_helpers.py:65` / `:9646` import and test its default JAX lane.
  - `benchmarks/single_stage_surface_reprojection_probe.py`: active CPU smoke entrypoint. `tests/test_jax_import_smoke.py:48` keeps the script path and `tests/test_jax_import_smoke.py:1153` executes it and asserts structured output.
  - `benchmarks/surface_rz_geometry_hlo_probe.py`: active HLO/instrumentation smoke entrypoint. `tests/geo/test_surface_rzfourier_jax.py:1141` executes the local script path at `:1148`; historical docs also cite this instrumentation style in `docs/jax_native_round3_performance_todos_2026-05-18.md:109` and `:475`.
- **Change:** No code deletion. The T1.10 work is the caller inventory and classification; any future retirement must be a separate migration slice that first replaces tests/docs and removes the active entrypoint contract.
- **LOC saved:** 0. The earlier up-to-~1,245 LOC possible deletion is not banked.
- **Risk:** Low to keep; medium/high to delete because each script still has live test or documentation surfaces.
- **Validation gate:** Completed with full caller inventory over `src`, `tests`, `benchmarks`, `examples`, `docs`, `.github`, and `scripts`, plus focused CPU validation. Exact commands are recorded in `docs/bloat_torax_coherent_execution_plan_2026-05-31.md`.

### 5.11 — [x] T1.11: Verify subprocess skip sentinels remain complete — **VERIFIED (v4)**: `_skip_case` used 19× in `tests/subprocess/jax_runtime_cases.py`; no additions needed

- **Files:** `tests/subprocess/jax_runtime_cases.py`.
- **Current-tree status:** The previously identified skip returns already call `_skip_case(...)` before returning. The old line inventory was stale; one cited line was a helper no-GPU return, not a test-case skip.
- **Change:** Keep this as a verification task only: run/update the AST audit and refresh any stale audit artifact text. Do not add duplicate sentinels.
- **LOC saved:** ~0 net (each grows by 1 line), but restores honest skip visibility.
- **Risk:** Low.
- **Validation gate:** T1 + `tests/test_pytest_skip_xfail_audit.py` AST check.

### 5.12 — Deferred, not Tier 1: traceable-runner cache dictionaries

- **Files:** `src/simsopt/geo/optimizer_jax.py:301-303, 415-478, 1824-1828, 4330-4334, 4648-4652`.
- **Current-tree status:** Three cache dictionaries exist for LM, Newton-polish, and exact-Newton traceable runners. `_cached_traceable_runner(...)` already shares weakref/token ownership logic across them.
- **Do not do:** Do not collapse the dictionaries mechanically into one dict keyed by `(kind, callable_fn, cache_key)`. The current cache contract uses identity keys for bare callables, semantic tokens for marked callables, weakref cleanup for weakrefable callables, and strong refs for nonweakrefable callables. Tests cover nonweakrefable, unhashable, equal-but-distinct, and explicit semantic-token callable objects.
- **If pursued:** Treat as a Tier 1b local abstraction change, not a T1 deletion. Write two designs, preserve the existing ownership semantics exactly, and prove the same tests still cover per-kind isolation.
- **Validation gate:** `tests/geo/test_lm_damping_parity.py` cache tests plus focused `optimizer_jax.py` caller inventory for `_TRACEABLE_*_RUNNER_CACHE`.

### Tier 1 exit gate

All required T1 source items are closed for this pass. Closure-mode revision
(2026-06-02): complexity reduction and validated source-queue closure are the
gate; net LOC is a secondary indicator per §2, so no extra T1 LOC-bank task
remains. Do not tag `bloat-reduction-T1-complete` until the post-CUDA/GPU
validated commit exists.

---

## 6. Tier 2 — Factory Introductions

**Target:** ~3,500 LOC reduction. **Effort:** ~3–4 days. **Risk:** Low-Medium.

Goal: convert repeated templates into data-driven factories. Each item proves the pattern at small scope before Tier 3 attempts larger folds.

### 6.1 — [x] T2.1: Boozer result-dict factories — **SKIP-PAYLOAD LOC-BANKED / REMAINING TARGET DEFER-NO-BANK (2026-06-02)**

- **Files:** `boozersurface_jax.py` result-dict packaging sites now include traceable skip/result paths at `:5554`, `:5686`, `:5776`, public LS skip/success paths at `:6070`, `:6313`, `:6431`, and exact/public fallback paths at `:6822`, `:6924`, `:7045`; schema/factory helpers live at `:533`, `:569`, `:593`, `:608`, `:663`, the LS-Newton reporting packer at `:3341`, and the skipped-polish packer at `:3369`. The remaining solve-quality fields stay beside the owning solve paths.
- **Change:** Introduce small result-pack helpers such as `_boozer_traceable_result_core(...)`, `_boozer_public_result_core(...)`, `_boozer_public_linearized_result_core(...)`, `_boozer_ls_newton_result_core(...)`, and `_boozer_exact_newton_result_core(...)`; keep `_BOOZER_TRACEABLE_RESULT_KEYS` (`:312`, still exact) as the traceable schema SSOT. **v5 NOTE:** `_BOOZER_RESULT_SCHEMAS` is **confirmed ABSENT** from the tree (not merely renamed) — the only traceable result-dict schema SSOT is `_BOOZER_TRACEABLE_RESULT_KEYS:312`.
- **Historical LOC target:** ~130. **Current LOC banked:** 39 total: 37 from the LS-Newton reporting helper follow-up (`src/simsopt/geo/boozersurface_jax.py` `31 insertions / 68 deletions`) plus 2 from the skipped-polish payload follow-up (`36 insertions / 38 deletions`). Do not bank the full old target: solve-quality overrides and failure-path diagnostics remain intentionally local.
- **Risk:** Medium after the post-v2 committed drift. This item touches user-visible result dict schemas and must be semantically inventoried from the latest committed tree before implementation.
- **Contracts:** Result-dict required/forbidden keys; success-vs-failure `linear_solve_backend` strings; `adjoint_linear_solve_available` flag.
- **Validation gate:** T2 + result-dict schema tests in `test_boozersurface_jax.py`.
- **2026-06-01 partial slice:** Introduced schema-core helpers `_boozer_traceable_result_core(...)`, `_boozer_public_result_core(...)`, and `_boozer_public_linearized_result_core(...)`, then added `test_boozer_result_core_helpers_match_schema_sources` so the helpers stay tied to `_BOOZER_TRACEABLE_RESULT_KEYS`, `_BOOZER_SOLVER_RESULT_CORE_KEYS`, `_BOOZER_RUNTIME_RESULT_KEYS`, and `_BOOZER_LINEARIZED_RESULT_KEYS`.
- **2026-06-01 follow-up:** Added keyword-only public LS-Newton and exact-Newton envelope factories (`_boozer_ls_newton_result_core(...)`, `_boozer_exact_newton_result_core(...)`) and extended the same helper test to cover their key sets, fixed `"ls"`/`"exact"` type invariants, linearization-kind invariants, and forbidden-key contracts.
- **2026-06-01 LOC-banking follow-up:** Added `_ls_newton_reporting_fields(...)` for the repeated Hessian/Newton-polish reporting keys and reused it at the traceable LS success path plus public LS failure/success paths (`:5774`, `:6334`, `:6456`). The helper and public-result tests now tie the packed key set and distinct value forwarding to `_BOOZER_HESSIAN_REPORTING_RESULT_KEYS`.
- **2026-06-02 LOC-banking follow-up:** Added `_skipped_newton_polish_fields(...)` for the repeated traceable/public Newton-polish skip payloads. It delegates the Hessian-reporting key shape to `_ls_newton_reporting_fields(...)`, supplies only the skip-specific dynamic values and policy flags, and is used by the traceable skip path (`:5697`) plus public skipped-polish path (`:6089`). The helper test now ties the returned key set to `_BOOZER_HESSIAN_REPORTING_RESULT_KEYS` plus `newton_polish_policy` / `newton_polish_skipped`.
- **Design-it-twice gate:** Option A, a generic record-builder keyed by record mode, was rejected because it would hide solve-specific payload fields and make exact/LS failure paths harder to audit. Option B, narrow schema-core helpers, was selected for the first slice because only duplicated core fields move while residuals, callbacks, dense factors, reporting fields, and failure metadata remain visible at each solve site. Option C, named LS-Newton/exact-Newton envelope factories, was selected for the follow-up after rejecting positional factories; the landed helpers are keyword-only so field mapping stays auditable at every call site.
- **Information-hiding test:** The schema constants remain the expected-key SSOT, while the helpers manually pack the matching values. A future public or traceable core key-set change still needs the schema constant and helper implementation updated together; the helper test fails if those drift or if a helper admits forbidden traceable/public keys. The envelope factories hide only the repeated public record envelope, and `_ls_newton_reporting_fields(...)` hides only repeated `result.get(...)` reporting extraction; solve-quality overrides, factorization backend strings, callbacks, and failure-path control flow remain local to the owning solve site.
- **2026-06-02 residual decision:** close the remaining T2.1 target as `defer/no-bank` for this bloat pass after the skipped-polish helper. A broader record-mode builder would hide backend strings, failure-stage/category fields, dense-factor choices, and solve-quality overrides for no obvious source-negative win beyond this narrow payload fold.
- **Remaining target:** closed no-bank. T2.1 banked 39 source LOC across the reporting and skipped-polish follow-ups; retire the unbanked remainder of the old `~130` estimate unless a future API/result-schema migration supplies a separate acceptance gate.
- **Validation evidence:** focused schema/result selectors passed (`2 passed`, `13 passed`, and `20 passed, 2 skipped, 457 deselected`) for earlier slices; the skipped-polish helper passed `test_boozer_result_core_helpers_match_schema_sources`, `test_run_code_skip_policy_returns_ls_state_without_newton`, and `test_run_code_traceable_ls_skip_policy_does_not_call_newton` (`3 passed`). `ruff check`, `ruff format --check`, `py_compile`, guardrail scan, and `git diff --check` passed for the touched source/test/doc files. `mypy` is now installed/runnable, but source-only `mypy src/simsopt/geo/boozersurface_jax.py` remains blocked by pre-existing errors at the runtime-state callable annotations, quadpoint signatures, local `guarded` redefinition, and grouped-field call arity.

### 6.2 — [x] T2.2: `boozer_radial_field` 16×2 evaluator collapse — **FORMULA DEDUPED + WRAPPERS LOC-BANKED / REMAINING TARGET DEFER-NO-BANK (2026-06-02)**

- **2026-06-01 status:** The duplicated direct Fourier formulas are folded. `_eval_modB` / derivative siblings route through `_eval_*_from_columns` via direct or typed subset-column wrappers (`src/simsopt/jax_core/boozer_radial_field.py:542`, `:556`, `:1018`), and scalar direct evaluators use `_direct_scalar_evaluator(...)` (`:1037`, assignments at `:1114`) to keep the original scalar spline path. Follow-up slices now remove the repeated private Fourier/value direct-evaluator ceremony, scalar direct-evaluator ceremony, one-use radial-column pass-through, and the modB-specific direct factory without changing formula ownership or RHS column reuse.
- **Files:** `src/simsopt/jax_core/boozer_radial_field.py`; `src/simsopt/jax_core/tracing.py`; active tests in `tests/field/test_trace_boozer_analytic_jax.py`.
- **Change landed:** canonical formula ownership is now the `_eval_X_from_columns` family. Direct evaluators construct only the radial columns they need, and radial Boozer guiding-centre RHS calls evaluate one RHS column bundle per point rather than re-evaluating each field scalar separately. The direct-wrapper follow-up introduced typed direct evaluator factories and the typed modB follow-up now routes all private direct evaluators through `_direct_radial_evaluator(...)`, preserving `__name__`, `__qualname__`, `__module__`, and the modB value/theta/zeta subset-column factory. The scalar-helper follow-up uses `_direct_scalar_evaluator(...)` with typed profile selectors to build the seven scalar direct evaluator objects while preserving their scalar spline reads and metadata. The pass-through inline follow-up deletes `_eval_with_radial_columns(...)`; `_direct_radial_evaluator(...)` evaluates the selected radial columns directly before calling the supplied column evaluator.
- **LOC saved:** 71 banked across the four T2.2 LOC follow-ups: 35 for the direct-wrapper follow-up (`99 insertions / 134 deletions`; `src/simsopt/jax_core/boozer_radial_field.py` 1,191 -> 1,156 LOC), 12 for the scalar-helper follow-up (`26 insertions / 38 deletions`), 10 for the pass-through inline follow-up (`2 insertions / 12 deletions`; 1,144 -> 1,134 LOC), plus 14 for the typed modB direct-factory follow-up (`24 insertions / 38 deletions`; 1,134 -> 1,120 LOC). The old `~400` estimate is still not banked because the formula-dedup slice was effectively flat once benchmark-preserving scaffolding was included.
- **Risk:** Low-to-medium. The first simple full-bundle wrapper was simpler but regressed the RHS/direct microbenchmarks; the landed subset-column design preserves the benchmark gate at the cost of extra helper scaffolding.
- **Contracts:** `BoozerRadialColumnBundle` field ordering (pytree flattening); `state.stellsym` static branch; `inverse_fourier_transform_{even,odd}` switch.
- **Validation evidence:** `tests/field/test_trace_boozer_analytic_jax.py` passed (`28 passed`), including `stellsym=True/False` direct-vs-column routing coverage with distinct optional-mode profiles; focused routing/cache tests passed (`2 passed, 2 skipped, 45 deselected`); `tests/field/test_boozermagneticfield_jax_item33.py` remains skipped in this env (`21 skipped`); the direct-wrapper metadata preservation script passed for 19 generated private evaluators and the scalar-helper metadata/pickle probe passed for the 7 generated scalar evaluators; the pass-through inline follow-up passed the focused routing selector (`3 passed, 25 deselected`), radial direct-evaluator metadata probe (`19 radial evaluator metadata entries preserved`), and left no source/test call sites for `_eval_with_radial_columns`; the typed modB direct-factory follow-up passed the focused routing selector (`4 passed, 24 deselected`), full radial tracing tests (`28 passed`), metadata proof for 26 direct evaluators, source-only `mypy`, dependency check, and left no source/test call sites for `_direct_modB_value_evaluator` or `_ModBValueEvaluator`; `ruff check`, `ruff format --check`, `py_compile`, `pip check`, and `git diff --check` passed across the T2.2 follow-ups.
- **Benchmark evidence:** saved pre-change baseline was `direct_modB 0.000578229`, `direct_dmodBds 0.001090641`, `direct_G 0.000272629`, `rhs_vacuum 0.003945453`. Direct-wrapper final medians were `direct_modB 0.000612696`, `direct_dmodBds 0.001073811`, `direct_G 0.000250935`, `rhs_vacuum 0.003285109`. The scalar-helper follow-up reran the same `stellsym=True` synthetic non-JIT gate against the documented final medians and passed: `direct_modB 0.000637098`, `direct_dmodBds 0.001141614`, `direct_G 0.000267225`, `rhs_vacuum 0.003449284`, all below the +10% limit. Delta review reproduced the gate in the current checkout with 25-trial medians `direct_modB 0.000594125`, `direct_dmodBds 0.001056167`, `direct_G 0.000243000`, `rhs_vacuum 0.003481042`, also below the same limits.
- **Review clarification:** The direct-vs-column tests prove routing to the column SSOT, not an independent formula oracle; formula correctness remains covered by wrapper parity, closed-form analytic, and benchmark evidence.
- **2026-06-02 residual decision:** close the remaining T2.2 target as `defer/no-bank` for this bloat pass. The current file already centralizes formulas in `_eval_*_from_columns` (`:663-1000`), routes direct evaluators through typed `_direct_radial_evaluator(...)` / `_direct_scalar_evaluator(...)` (`:1018`, `:1037`, assignments at `:1053-1120`), and has no remaining `_eval_with_radial_columns`, `_direct_modB_value_evaluator`, or `_ModBValueEvaluator` source target. The surviving subset-column builders are benchmark-preserving and field-specific; collapsing them further would trade explicit stellsym/profile requirements for table-driven adapter code with no obvious net source bank.
- **Remaining LOC-banking follow-up:** closed no-bank. Future T2.2 work should not reopen formula correctness unless the direct-vs-column parity tests fail, and any larger subset-family redesign needs a fresh benchmark gate before source edits.

### 6.3 — [x] T2.3: Surface fourier `_from_dofs` / `_from_spec` factory — **LOC-BANKED / REMAINING PRODUCT-RULE FORMULAS DEFER-NO-BANK (2026-06-02)**

- **Files:** `surface_fourier_kernels.py` (pre tensor-kernel slice 3,139 LOC; current 2,692 LOC; the tensor simple `_*_from_dofs` wrappers spanned ~`:2208-2516`, the `SurfaceXYZFourier` analytic wrappers span ~`:1427-2150`, and the coefficient-derivative wrapper helpers now sit near `:2522-2706`) + `surface_fourier.py` (pre facade slice 978 LOC; current 813 LOC; the facade `_*_from_spec` / paired-linear `_*_from_dofs` wrappers spanned ~`:171-864`). (v4's `kernels:2200-2799` was stale but overlapped the tensor wrapper family.)
- **2026-06-01 facade slice:** `surface_fourier.py` now uses typed local factories for the `SurfaceXYZFourier` / `SurfaceXYZTensorFourier` kernel-backed spec wrappers and paired-linear dof wrappers. Public names remain exported symbols with assigned `__name__` / `__qualname__` / `__module__`, while composed geometry (`normal`, fundamental forms, curvatures, area, volume) stays explicit.
- **2026-06-01 tensor-kernel slice:** `surface_fourier_kernels.py` now uses `_eval_surface_tensor_from_dofs(...)` for the ten simple `SurfaceXYZTensorFourier` evaluator wrappers: `gamma`, paired `gamma_lin`, first/second coordinate derivatives, and `normal`. The public wrapper functions remain explicit so `__code__.co_name`, `inspect.getsource(...)`, signatures, and docs stay public-introspection compatible. The concrete evaluator functions, `_dofs_to_xyzc_any(...)`, stellsym scatter construction, `SurfaceXYZFourier` scatter/template wrappers, composed area/volume/unit-normal helpers, and coefficient-Jacobian wrappers stay explicit.
- **2026-06-01 `SurfaceXYZFourier` unpack slice:** the nine analytic `SurfaceXYZFourier` wrappers now call the existing `_scatter_surface_xyzfourier_dofs(...)` helper directly instead of repeating the six-line flat-DOF to `(xc, xs, yc, ys, zc, zs)` unpack block. Public wrappers, source introspection, analytic formulas, paired derivative helper, composed area/volume/unit-normal helpers, and coefficient-Jacobian factories stay explicit.
- **2026-06-01 coefficient-derivative wrapper slice:** `surface_fourier_kernels.py` now uses one `_surface_dof_transform(...)` helper for the repeated `jax.jacfwd` / explicit Hessian / `jax.grad` / `jax.hessian` wrapper ceremony across tensor and `SurfaceXYZFourier` coefficient-derivative families. The helper has two explicit inner signatures, preserving the tensor signature without `coeff_template` and the `SurfaceXYZFourier` signature with `coeff_template`. No derivative formula, public export name, scalar tolerance, CPU geometry code, CUDA/GPU path, or composed geometry formula was changed.
- **2026-06-01 `SurfaceXYZFourier` order-hat slice:** the six full-grid analytic `SurfaceXYZFourier` wrappers now share `_surface_xyzfourier_component_hat_derivatives(...)` for the repeated derivative-order to separable-basis to `(xhat, yhat, zhat)` component evaluation. Public wrappers, paired-linear wrappers, coefficient-Jacobian factories, rotation terms, and product-rule formulas stay explicit.
- **2026-06-01 `SurfaceXYZFourier.gammadash1` product-rule micro-slice:** the full-grid `gammadash1` wrapper now uses the same explicit radial/toroidal product-rule representation already used by the paired-linear path; after the rotate-helper sharing slice, that rotation routes through `_rotate_hat_components(...)`. Public wrapper signature, source ownership, coefficient-Jacobian factories, paired-linear wrappers, and the other product-rule formulas are unchanged.
- **2026-06-01 tensor `surface_gammadash1` product-rule micro-slice:** the full-grid tensor `surface_gammadash1(...)` and its clamped path now share the explicit radial/toroidal product-rule representation already used by `surface_gammadash1_lin(...)`, then call `_rotate_hat_components(...)`. Tensor public wrapper signatures, clamped-dimension semantics, coefficient-Jacobian factories, and `SurfaceXYZFourier` formulas are unchanged.
- **2026-06-01 `SurfaceXYZFourier` rotate-helper slice:** the private `SurfaceXYZFourier` full-grid and paired-linear rotate helpers were deleted; those call sites now share the tensor `_rotate_hat_components(...)` / `_rotate_hat_components_lin(...)` helpers. Public wrappers, derivative formulas, radial/toroidal product-rule spelling, scatter/template handling, coefficient-Jacobian factories, tensor formulas, and composed geometry stay explicit.
- **LOC saved:** 612 banked across completed T2.3 slices: 165 in `src/simsopt/jax_core/surface_fourier.py` for the facade slice (`281 insertions / 446 deletions`), 190 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the tensor-kernel slice (`58 insertions / 248 deletions`), 51 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the `SurfaceXYZFourier` unpack slice (`15 insertions / 66 deletions`), 109 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the coefficient-derivative wrapper slice (`52 insertions / 161 deletions`), 35 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the `SurfaceXYZFourier` order-hat slice (`73 insertions / 108 deletions`), 5 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the `SurfaceXYZFourier.gammadash1` radial/toroidal product-rule micro-slice (`3 insertions / 8 deletions`), 28 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the tensor `surface_gammadash1` radial/toroidal product-rule micro-slice (`7 insertions / 35 deletions`; 2,749 -> 2,721 LOC), plus 29 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the `SurfaceXYZFourier` rotate-helper sharing slice (`9 insertions / 38 deletions`; 2,721 -> 2,692 LOC). Do **not** close the old full-item estimate merely because the banked total now exceeds 550 actual LOC: the remaining product-rule formulas are intentionally still explicit, and any further formula fold requires a separate readability and parity gate.
- **Risk:** Low-to-medium. `__all__` listings + cross-imports stay unchanged; factory-generated wrappers must keep public symbol names and JIT behavior. The tensor-kernel slice preserves public signatures, `__name__`, `__qualname__`, and `__module__` for all ten folded wrappers. The coefficient-derivative wrapper slice preserves public tensor and `SurfaceXYZFourier` derivative signatures, including the `coeff_template` distinction. The order-hat slice must preserve derivative-order tuple destructuring and full-grid versus paired-linear rotation paths. The `SurfaceXYZFourier.gammadash1` and tensor `surface_gammadash1` product-rule micro-slices must preserve the same `d/dphi` cylindrical rotation algebra while removing only expanded cosine/sine spelling. The rotate-helper sharing slice must preserve the same cylindrical rotation algebra for full-grid and paired-linear `SurfaceXYZFourier` call sites without changing wrapper signatures or derivative formulas.
- **Contracts:** Stellsym scatter indices; every public symbol name; tensor conventions (`dgamma_by_dcoeff[i,j,l,k]`).
- **Validation gate:** T2 + `tests/geo/test_surface_fourier_jax.py`.
- **2026-06-01 tensor-kernel validation evidence:** signature-and-introspection preservation script for the ten folded wrappers passed, including `__code__.co_name`, `inspect.getsource(...)`, and public doc checks; `ruff check` passed for `surface_fourier_kernels.py` plus the two parity test files; `ruff format --check src/simsopt/jax_core/surface_fourier_kernels.py` passed; `py_compile` passed for the changed source plus parity tests; `mypy src/simsopt/jax_core/surface_fourier_kernels.py` passed; `tests/geo/test_surface_xyz_tensor_clamped_jax.py` passed (`38 passed`); the focused Fourier parity selector passed (`60 passed, 90 deselected`). `ruff format --check` over the two unmodified parity test files remains a pre-existing blocker (`Would reformat`) and is not part of this LOC-banked source slice.
- **2026-06-01 `SurfaceXYZFourier` unpack validation evidence:** `ruff check`, source-only `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/surface_fourier_kernels.py`, and public wrapper introspection passed. CPU/X64 tests passed: focused selector for geometry, dcoeff, paired-linear, and non-RZ fundamental form behavior (`34 passed, 116 deselected`) plus the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`).
- **2026-06-01 coefficient-derivative wrapper validation evidence:** derivative wrapper signature script passed for all 13 tensor derivative functions and all 13 `SurfaceXYZFourier` derivative functions, including the `coeff_template` arity distinction and `__code__.co_name == "wrapper"`; `ruff check`, source-only `ruff format --check`, `py_compile`, and `mypy src/simsopt/jax_core/surface_fourier_kernels.py` passed; focused CPU/X64 coefficient/normal derivative parity selector passed (`48 passed, 102 deselected`); scalar area/volume derivative wrappers are covered by the full CPU/X64 `tests/geo/test_surface_fourier_jax.py` file (`150 passed`).
- **2026-06-01 `SurfaceXYZFourier` order-hat validation evidence:** public wrapper introspection passed for the six touched full-grid analytic functions, including `__name__`, `__qualname__`, `__module__`, `__code__.co_name`, source prefix, and `coeff_template` signature preservation; `ruff check`, source-only `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/surface_fourier_kernels.py`, `pip check`, and `git diff --check` passed. CPU/X64 tests passed: focused selector for geometry, tangents, second derivatives, paired-linear wrappers, and non-RZ fundamental-form behavior (`34 passed, 116 deselected`), the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`), and the full `tests/geo/test_surface_fourier_jax.py` file (`150 passed`).
- **2026-06-01 `SurfaceXYZFourier.gammadash1` product-rule validation evidence:** `ruff check src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py`, source-only `ruff format --check`, `py_compile`, and `mypy src/simsopt/jax_core/surface_fourier_kernels.py` passed. CPU/X64 tests passed: focused selector for geometry/tangent and adjacent derivative coverage (`26 passed, 124 deselected`) and the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`).
- **2026-06-01 tensor `surface_gammadash1` product-rule validation evidence:** `ruff check src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py tests/geo/test_surface_xyz_tensor_clamped_jax.py`, source-only `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/surface_fourier_kernels.py`, and `pip check` passed. CPU/X64 tests passed: focused clamped tensor selector (`16 passed, 22 deselected`), focused surface Fourier geometry/tangent selector (`14 passed, 136 deselected`), and the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`).
- **2026-06-01 `SurfaceXYZFourier` rotate-helper validation evidence:** `ruff check src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py tests/geo/test_surface_xyz_tensor_clamped_jax.py`, source-only `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/surface_fourier_kernels.py`, `pip check`, `git diff --check`, and `rg -n '_surface_xyzfourier_rotate' src/simsopt/jax_core/surface_fourier_kernels.py tests src benchmarks` passed. CPU/X64 tests passed: focused `SurfaceXYZFourier` geometry/tangent/second-derivative selector (`26 passed, 124 deselected`), the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`), and the focused tensor clamped selector (`16 passed, 22 deselected`).
- **2026-06-02 residual decision:** close the remaining T2.3 target as `defer/no-bank` for this bloat pass. The current banked total already exceeds the old net source estimate, but the surviving product-rule formulas are intentionally explicit: for example, `surface_xyzfourier_gammadash1dash1_from_dofs` and `surface_xyzfourier_gammadash1dash2_from_dofs` keep the radial/toroidal second-derivative algebra visible near `:1611-1682` after the rotate-helper sharing slice. A broader product-rule factory would obscure derivative algebra and public-wrapper introspection for little or no safe source reduction.
- **Remaining LOC-banking follow-up:** closed no-bank. Future formula work should be a separate math/readability patch only if it is clearer than the current local formulas and has CPU/X64 parity plus introspection proof.

### 6.4 — [x] T2.4: Spec dataclass auto-registration helper — **COMPLETED / LOC-BANKED SMALL (2026-06-01)**

- **Files:** `src/simsopt/jax_core/specs.py` now owns `_register_jax_spec(...)` at `:102`; the file contains 29 spec-class decorator uses and one direct `jax.tree_util.register_dataclass(...)` call inside the helper. `tests/core/test_jax_core_specs.py` owns the helper regression at `:374`.
- **Change:** Defined private `_register_jax_spec(data_fields=(...), meta_fields=(...))` to wrap `@dataclass(frozen=True)` plus JAX dataclass registration. Each spec keeps its data/meta partition next to its class definition.
- **LOC saved:** about 52 net LOC in `specs.py` (1,622 -> 1,570). The earlier `~140` estimate was a gross repeated-block estimate, not the validated net source reduction.
- **Risk:** Low after validation. JAX `register_dataclass` treats `meta_fields` as static JIT-cache-key material; the helper preserves every explicit partition and converts tuple declarations to the list form accepted by JAX at the registration boundary.
- **Contracts:** Field names + data/meta partition (the explicit decorator arguments drive what's a traced array vs JIT static); metadata remains static cache-key material.
- **Design-it-twice gate:** Option A, centralizing all partitions in a module table, was rejected because a field split change would require a coordinated table edit plus class audit. Option B, local per-class decorator arguments, was selected because the class remains the single place a reviewer sees the partition while the repeated frozen-registration ceremony is hidden.
- **Information-hiding test:** changing a spec's data/meta decision still requires editing only that class's decorator, and tests fail if data-field changes recompile or meta-field changes stop recompiling.
- **Validation gate:** completed with `tests/core/test_jax_core_specs.py::test_curve_spec_data_fields_do_not_recompile_but_meta_fields_do`, `tests/core/test_jax_core_specs.py::test_register_jax_spec_helper_preserves_data_meta_partition`, `tests/test_jax_import_smoke.py::test_jax_core_specs_are_pytrees`, full `tests/core/test_jax_core_specs.py tests/jax_core/test_tree_signature.py`, scoped `ruff`, `py_compile`, and `git diff --check`; `mypy` blocked with `No module named mypy`.

### 6.5 — [x] T2.5: Batch-axis sharding helper factory — **COMPLETED / LOC-BANKED SMALL (2026-06-01)**

- **Files:** `src/simsopt/jax_core/sharding.py`; new focused regression `tests/jax_core/test_sharding_helpers.py`.
- **Change:** Replaced the three duplicate point-axis batch predicates/config builders, maybe-shard wrappers, and summary bodies with shared private helpers. Preserved the exported `TrajectoryBatchShardingConfig`, `SeedBatchShardingConfig`, and `SurfaceQuadratureShardingConfig` as concrete dataclass subclasses rather than collapsing them into a single public class.
- **LOC saved:** 6 net production LOC (`102 insertions / 108 deletions` in `sharding.py`; 725 -> 719). The old `~170` estimate is not banked because preserving public config class identity and readable wrapper functions offset most of the helper extraction.
- **Risk:** Low for CPU behavior after validation; CUDA/GPU-specific placement proof remains outside this CPU forced-device slice.
- **Contracts:** JSON summary key names (`trajectory_sharded`, `seed_batch_sharded`, `surface_quadrature_sharded`, `field_collective`, etc.); public class names and field order; leading-axis sharding threshold and strategy checks.
- **Design-it-twice gate:** Option A, aliasing all three public classes to one implementation class, would save more LOC but change `type(config).__name__` / concrete public class identity. Option B, private base plus concrete public subclasses, was selected because it centralizes the internal policy while preserving exported class contracts.
- **Validation gate:** completed with `tests/jax_core/test_sharding_helpers.py`, `tests/jax_core/test_tracing_jax_item14.py::test_trajectory_batch_sharding_summary_surfaces_axis_contract`, `tests/jax_core/test_surface_seed_sharding.py`, `tests/jax_core/test_points_coils_sharding.py`, scoped `ruff`, `py_compile`, and `git diff --check`; `mypy` blocked with `No module named mypy`.

### 6.6 — [x] T2.6: 8 `_resolve_*` env/kwarg helpers → 1 generic resolver — **COMPLETED / LOC-BANKED SMALL (2026-06-01)**

- **Files:** `src/simsopt/backend/runtime.py`, `tests/test_backend.py`.
- **Change:** Replaced the eight runtime-kwarg resolver ladders with `_resolve_kwarg(...)`, which owns explicit-kwarg → env override → mode-default precedence. `_config_from_mode(...)` keeps explicit typed field resolution rather than using a heterogenous dict builder; `_MODE_POLICY_DEFAULTS` remains data-only.
- **LOC saved:** small, approximately low-teens net LOC relative to the already-applied T1.2 runtime export context. The old `~100` estimate is not banked because preserving typed explicit construction and per-field parser source labels keeps some ceremony by design.
- **Risk:** Low after validation. Pure refactor; same truth table.
- **Contracts:** Mode-default precedence; env-value validation; `set_backend(...)` kwarg names.
- **Design-it-twice gate:** Option A, a fully heterogenous table plus dict/`**kwargs` builder, was rejected because it would hide `BackendConfig` field types and make parser/default mismatches easier. Option B, one generic precedence helper with explicit typed field resolution at `_config_from_mode(...)`, was selected because it removes repeated precedence ladders while keeping field-level contracts readable.
- **Information-hiding test:** changing precedence now edits `_resolve_kwarg(...)` once; changing a field's parser, env source, or mode default still edits only that field's line in `_config_from_mode(...)`.
- **Validation gate:** completed with full `tests/test_backend.py`, backend import-smoke selectors, scoped `ruff`, `py_compile`, guardrail diff scan, and `git diff --check`; `mypy` blocked with `No module named mypy`.

### 6.7 — [x] T2.7: SciPy adapter unification in `optimizer_jax_reference.py` — **COMPLETED / LOC-BANKED SMALL (2026-06-01)**

- **Files:** `src/simsopt/geo/optimizer_jax_reference.py` (569 -> 539 LOC).
- **Change:** Collapsed the three duplicated SciPy host value/gradient objective closures into `_make_scipy_host_value_and_grad_objective(...)` and `_scipy_minimize_value_and_grad_core(...)`. Kept the three public/private adapter entrypoints as explicit wrappers so their guard and method contracts remain visible.
- **LOC saved:** 30 net production LOC (`63 insertions / 93 deletions`). The old `~220` estimate is not banked because the component-specific guard wrappers and target-lane method check are preserved.
- **Risk:** Low after validation. All three adapters still route through `_scipy_dispatch_core(...)`; `_scipy_dispatch(...)` remains available and unchanged.
- **Contracts:** `_require_native_cpu_reference_backend_for_scipy_adapter` guard remains on `_scipy_minimize` and `_scipy_minimize_value_and_grad`; `target_scipy_minimize_value_and_grad` keeps no backend guard and still rejects non-`lbfgs`; `scipy_call_contract`, `scipy_initial_call`, and `scipy_callback_trace` fields stay populated.
- **Design-it-twice gate:** Option A, replacing the three entrypoints with one public parameterized function, was rejected because it would hide the target-lane guard distinction. Option B, shared private host-objective/core helpers under explicit wrappers, was selected because it removes duplicated host conversion logic without changing adapter contracts.
- **Validation gate:** completed with focused `tests/geo/test_boozersurface_jax.py` SciPy adapter selectors, `tests/geo/test_boozersurface_jax_private.py::test_private_scipy_adapters_reject_all_jax_backend_modes`, `tests/geo/test_optimizer_jax_reference.py`, `tests/test_jax_import_smoke.py::test_optimizer_jax_reference_methods_reject_all_jax_backend_modes`, scoped `ruff`, `py_compile`, and `git diff --check`; `mypy` blocked with `No module named mypy`.

### 6.8 — [x] T2.8: `LayerDriftTracker` dataclass for `single_stage_init_parity` — **HELPERS LOC-BANKED / REMAINING TARGET DEFER-NO-BANK (2026-06-02)**

- **Files:** `benchmarks/single_stage_init_parity.py` — `_compare_same_candidate_scalar` at `:2415`, `_record_first_scipy_callback_split` at `:2723`, `LayerDriftTracker` at `:2969`, `_first_parity_bug_census_divergence` at `:3006`, `_update_parity_bug_census` at `:3169`, `_pre_newton_census_gate_failures` at `:3284`, and `compare_same_candidate_objective_replay` at `:3376` (scope counters around `:3441`, per-pair metadata bindings around `:3459`, target-native flag inline around `:3458`, target-native rejection cache around `:3502`, objective-component summary reuse around `:3662`, first-divergence helper calls around `:3624`/`:3714`, parity-census result-schema conversion around `:3812`).
- **Change:** Landed `LayerDriftTracker` for the two layer-decomposition families (`boozer_solve_decomposition` and `iota_penalty_decomposition`), a narrow `_record_first_scipy_callback_split(...)` helper for the repeated "first split wins" bookkeeping in the Boozer SciPy callback trace comparison, a target-native rejection-predicate cache for the repeated per-pair contract checks, `Counter[str]` bookkeeping for the candidate/gradient comparison-scope counts, local per-pair metadata bindings for repeated event-index/iteration/line-search fields, reuse of `_compare_same_candidate_objective_components(...)` for the target-native no-component summary plus the shared slice-owner predicate, deletion of the single-use target-native replay-flag helper in favor of the direct local flag read, reuse of `_compare_same_candidate_scalar(...)`'s mismatch emission to avoid the second `_scalar_close(...)` call in the callback `fun` first-split branch, deletion of the one-use `_same_candidate_scipy_callback_split(...)` pass-through so the explicit callback split payload now lives directly in `_record_first_scipy_callback_split(...)`, deletion of the one-use `_diagnostic_scalar_abs_diff(...)` / `_diagnostic_vector_abs_diff(...)` wrappers by keeping the layer-diagnostic `None`/`None` zero-drift policy explicit in `_compare_same_candidate_layer_decomposition(...)`, deletion of the one-use `_compare_same_candidate_iota_decomposition(...)` / `_compare_same_candidate_boozer_solve_decomposition(...)` wrappers by passing the explicit `field_name` and `layer_fields` bindings directly to `_compare_same_candidate_layer_decomposition(...)` at the two call sites, deletion of the one-use `_scalar_close(...)` wrapper by spelling the scalar tolerance predicate inside `_compare_same_candidate_scalar(...)` with the already computed `diff`, and deletion of the one-use `_finalize_parity_bug_census(...)` wrapper by keeping the final `parity_bug_census` recorded/not-applicable schema literal at the replay result boundary. The helpers/cached predicates/counters/bindings own repeated local state transitions while the final replay result dict and callback/layer schemas remain explicit at the public schema boundary.
- **Latest residual slice:** `_first_parity_bug_census_divergence(...)` now owns the repeated "first divergence wins" family/payload promotion for the `boozer_solve` and `iota_penalty` tracker updates. The helper keeps the `family` key first, copies the tracker divergence payload, and preserves the nested `layer_diffs` copy; no tracker update rule, divergent-layer threshold, final replay result key, or `parity_bug_census` result schema changed.
- **Remaining target:** closed no-bank. These slices intentionally did **not** hide outgoing max-slice result keys, metadata result keys, hardware, failure, candidate, scope-count result keys, or broader callback trackers behind a broad dynamic key-prefix helper.
- **LOC saved:** 118 net benchmark LOC across completed T2.8 micro-slices: 13 for the `LayerDriftTracker` slice (`79 insertions / 92 deletions`), 4 for the SciPy callback first-split slice (`34 insertions / 38 deletions`), 2 for the target-native predicate-cache slice (`9 insertions / 11 deletions`), 3 for the comparison-scope `Counter` slice (`6 insertions / 9 deletions`), 4 for the per-pair metadata-binding slice (`25 insertions / 29 deletions`), 2 for the component-summary reuse slice (`19 insertions / 21 deletions`), 6 for the target-native flag inline slice (`1 insertion / 7 deletions`), 4 for the callback `fun` threshold slice (`2 insertions / 6 deletions`), 12 for the callback split payload-inline slice (`7 insertions / 19 deletions`), 16 for the layer diagnostic wrapper-inline slice (`5 insertions / 21 deletions`), 34 for the decomposition dispatch wrapper-inline slice (`6 insertions / 40 deletions`; 4,761 -> 4,727 LOC), 4 for the scalar-close inline slice (`1 insertion / 5 deletions`; 4,727 -> 4,723 LOC), 11 for the parity-census finalize inline slice (`24 insertions / 35 deletions`; 4,723 -> 4,712 LOC), plus 3 for the first-divergence payload helper slice (`18 insertions / 21 deletions`; 4,712 -> 4,709 LOC). Retire the unbanked remainder of the old `~200 LOC` estimate for this pass.
- **Risk:** Low. Internal trackers; external dict shape preserved.
- **Contracts:** `_pre_newton_census_gate_failures` untouched; `parity_bug_census["divergent_layers"]` schema; `parity_bug_census["first_divergence"]` key order and nested `layer_diffs` copy preserved; all replay payload keys preserved.
- **Design-it-twice gate:** Option A, a generic key-prefix `.summary_dict(...)` emitter for every replay tracker, was rejected because it would hide the output schema in string construction. Option B, a typed state helper for only layer-decomposition drift plus explicit return-key mapping, was selected because it removes repeated state transitions while keeping the public replay payload auditable. For the latest residual slice, keeping the duplicated first-divergence literals was rejected because the two families consume the same tracker output under the same first-wins guard; the selected helper is private and only promotes that tracker output into the existing public payload shape.
- **2026-06-02 residual decision:** close the remaining T2.8 target as `defer/no-bank` for this bloat pass. The replay loop still keeps public result keys explicit at the return boundary (`candidate_comparison_scope_counts`, slice maxima, decomposition maxima, solve-quality probes, `parity_bug_census`, hardware/failure summaries, solver summaries, and `first_failure_event` around `:3812-3890`). The only apparent larger reduction is a broad dynamic replay-summary builder or schema-key factory, which would make the release-blocker gate less auditable and conflict with the closure-mode instruction to stop continuing tiny helper slices indefinitely.
- **Validation gate:** focused same-candidate replay tests passed (`22 passed, 338 deselected`); the target-native flag selector passed (`3 passed, 357 deselected`); the component-summary selector passed (`5 passed, 355 deselected`); the focused replay/pre-Newton selector passed (`8 passed, 352 deselected`); the loop-metadata selector passed (`4 passed, 356 deselected`); the scope-counter selector passed (`4 passed, 356 deselected`); the SciPy callback split tests passed (`2 passed, 358 deselected`); the callback `fun` split probe preserved the `field="fun"` first-split payload and `max_boozer_scipy_callback_abs_diff=0.2`; the callback split payload-inline slice reran the same focused callback selector (`2 passed, 358 deselected`); the layer diagnostic wrapper-inline slice reran the decomposition/parity-census selector (`2 passed, 358 deselected`); the decomposition dispatch wrapper-inline slice reran the decomposition/parity-census selector (`2 passed, 358 deselected`) and the broad same-candidate selector (`22 passed, 338 deselected`); the scalar-close inline slice reran the broad same-candidate selector (`22 passed, 338 deselected`) and the scalar-or-same-candidate selector (`22 passed, 338 deselected`); the parity-census finalize inline slice reran the focused replay/census selector (`28 passed, 332 deselected`); the first-divergence payload helper slice reran the focused replay/census selector (`28 passed, 332 deselected`); the target-native predicate selector passed (`3 passed, 357 deselected`); earlier pinned pre-Newton census failure-list tests passed (`9 passed, 351 deselected` for the focused replay/pre-Newton selector); scoped `ruff`, `ruff format --check`, `py_compile`, `pip check`, and `git diff --check` passed. `mypy benchmarks/single_stage_init_parity.py` remains blocked by 128 pre-existing benchmark/example typing errors, including missing example modules and existing scalar/slice-summary typing debt, not by the new tracker, callback helper, target-native predicate cache, scope counters, loop metadata bindings, component-summary reuse, target-native flag inline, callback `fun` threshold slice, callback split payload-inline slice, layer diagnostic wrapper-inline slice, decomposition dispatch wrapper-inline slice, scalar-close inline slice, parity-census finalize inline slice, or first-divergence payload helper slice.

### 6.9 — [x] T2.9: Quantity-aware tolerance helper in `validation_ladder_contract.py` — **COMPLETED / NOT LOC-BANKED (2026-06-01)**

- **Files:** `_tolerance_for` lives in **only one** file — `non_banana_example_cpp_jax_cpu_parity.py` (now a compatibility wrapper). **v5 CORRECTION:** v4's other two sites are wrong — `single_stage_parity_matrix.py` has **no** `_tolerance_for` (the cited `:292-302` does not exist) and `stage2_e2e_comparison.py` uses a **different** mechanism (`optimizer_drift_tolerances:71/74`). So this was a single-file local helper migration, not a 3-file consolidation.
- **Change:** Landed `QUANTITY_TOLERANCE_BUCKETS`, `quantity_parity_tolerance(...)`, and `quantity_uses_gradient_tolerance(...)` in `benchmarks/validation_ladder_contract.py`; `non_banana_example_cpp_jax_cpu_parity.py::_tolerance_for(quantity)` now delegates to the contract helper. The helper preserves `event_time_tracing` state-vector tolerances, `qfm_gradient` derivative-heavy tolerances, and float32 value/objective/gradient branches. It does not collapse to a generic `kind="rtol_atol"` bucket.
- **LOC saved:** 0 banked. `non_banana_example_cpp_jax_cpu_parity.py` shrank by 117 LOC, but `validation_ladder_contract.py` grew by 185 LOC and focused tests added 49 LOC, so this is an SSOT/contract-hardening slice, not bloat reduction.
- **Risk:** Low-Medium. Pure helper only if every quantity maps to the same bucket and pair as before.
- **Contracts:** `PARITY_LADDER_TOLERANCES` values frozen; `parity_ladder_tolerances(lane)` API unchanged.
- **Validation gate:** T2 + pre/post snapshot of every migrated quantity's `(bucket, rtol, atol)`.
- **2026-06-01 evidence:** 204-row pre/post snapshot over every migrated quantity and runtime tier was byte-identical. Focused tests passed: `tests/test_benchmark_helpers.py -k 'quantity_parity_tolerance or parity_ladder_tolerances'` (`6 passed, 354 deselected`) and `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py -k 'float32_smoke_tolerance_tier_routes_by_quantity or unknown_runtime_tolerance_tier_fails_closed or float32_smoke_keeps_gradient_as_diagnostic_failure'` (`3 passed, 59 deselected`). `ruff`, `ruff format --check`, `py_compile`, `mypy benchmarks/validation_ladder_contract.py`, `mypy tests/geo/test_surface_fourier_jax.py`, and `git diff --check` passed. Broader `mypy benchmarks/non_banana_example_cpp_jax_cpu_parity.py` still reports unrelated existing errors in neighboring benchmark modules, but the previous `validation_ladder_contract.py` blocker is closed.

### Tier 2 exit gate

Tier 2 source-bloat residuals are closed for this pass after T2.1/T2.2/T2.3/T2.8 were either banked or explicitly deferred no-bank. Do not tag `bloat-reduction-T2-complete` until the T2 suite, contract checklist, and `_pre_newton_census_gate_failures` replay gate are rerun as a grouped tier-exit proof.

---

## 7. Tier 3 — Structural Consolidations

**Target:** ~4,000–6,000 LOC reduction. **Effort:** ~1–2 weeks. **Risk:** Medium.

Goal: large-scale folds across multiple files. Higher risk because they touch hot code paths. Each item gets its own pre-flight design doc commit before implementation.

### 7.1 — [x] T3.1: GPMO 5-way fold (largest single win) — **RECORDING HELPERS LOC-BANKED / FULL FOLD NEEDS-DESIGN (2026-06-02)**

- **Current status:** source-negative T3.1 helper slices are banked and the remaining full driver factory is closed as `needs-design/no-bank` for this pass. `_gpmo_recording_scan(...)` now lives at `pm_optimization.py:566`, owns the shared `record_every` cadence, slot advancement, `x_history`, `residual_history`, and per-trace extra-history recording, and passes the scan iteration into variant step callbacks. It is used by `gpmo_baseline_solve:867`, `gpmo_arbvec_solve:1112`, `gpmo_arbvec_backtracking_solve:1847`, `gpmo_multi_solve:2237`, and `gpmo_backtracking_solve:2598`.
- **Files:** `src/simsopt/jax_core/pm_optimization.py` is now 3,175 LOC after the simple-driver and backtracking-recorder helper slices, down from 3,400 at `HEAD` (`158 insertions / 383 deletions`, net `-225` source LOC). Public wrapper file `src/simsopt/solve/permanent_magnet_optimization_jax.py` was not changed.
- **Remaining target decision:** close the old full five-way fold target as `needs-design/no-bank`. The bucketed ArbVec active-prefix path, deeper solve-driver/spec factory work, and cross-variant result-shape unification are not closure-safe without a separate design packet and numerical equivalence matrix.
- **LOC saved:** 225 source LOC banked across the simple recording-scan and backtracking-recorder helper slices. Do not bank the stale full `~1,500` estimate.
- **Risk:** Low for the banked recorder slices; medium for any future full driver factory. JIT cache keys must include variant-specific step/history behavior; tie-break order and `_UNAVAILABLE_CANDIDATE_COST` sentinel must hold; `selected_groups`, `num_nonzeros_history`, `removed_pair_count_history`, and `done_history` shapes must stay pinned.
- **Contracts:** `_record_rows = np.arange(record_every - 1, K, record_every)` off-by-one plus forced final row; sampled histories byte-equivalent to full histories at the selected rows; backtracking `x_snapshot` remains `next_state[0]`; `selected_groups` shape preserved; candidate-cost and step functions untouched; live-loop history capacity validators untouched.
- **Validation gate:** banked helper slices passed scoped `ruff`, `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/pm_optimization.py`, focused record-every selector `tests/solve/test_permanent_magnet_optimization_jax_item28.py -k "record_every"` (`11 passed, 36 deselected`), baseline/multi/ArbVec classes in `tests/jax_core/test_pm_optimization_jax_item25.py` (`26 passed`), and backtracking classes `TestGPMOBacktracking` / `TestGPMOArbVecBacktracking` (`11 passed`).

### 7.2 — [x] T3.2: `SpecBackedBiotSavartJAX` ↔ `BiotSavartJAX` mixin extraction — **POINT HELPERS LOC-BANKED / COTANGENTS DEFER-NO-BANK (2026-06-02)**

- **Current status:** the field-evaluation duplication was already folded through `_BiotSavartFieldEvaluationMixin` (`field/biotsavart_jax_backend.py:505`) inherited by both `SpecBackedBiotSavartJAX` (`:642`) and `BiotSavartJAX` (`:1281`). The 2026-06-01 follow-up now also folds the shared point-state mutation bodies through `_set_biot_savart_points(...)`, `_set_biot_savart_points_cyl(...)`, and `_get_biot_savart_points_cyl(...)` (`:485-499`).
- **Files:** `src/simsopt/field/biotsavart_jax_backend.py` (2,301 -> 2,299 LOC for this points-only slice). Public methods remain class-local so `__qualname__`, signatures, and `FieldEvalSpec` annotations stay on `SpecBackedBiotSavartJAX` / `BiotSavartJAX`; shared private helpers cover only the duplicated point-state mutation and cylindrical readback bodies. `BiotSavartJAX.clear_points()` remains live-class-only (`:1758`) for wireframe temporary mutation.
- **Change:** Hoisted only the duplicate mutable point-state helper bodies into private module-level helpers. The public wrappers preserve method metadata, `BiotSavartJAX.set_points(...)` preserves the live no-host-round-trip path for JAX arrays, and `clear_points()` stays out of `SpecBackedBiotSavartJAX`.
- **2026-06-02 residual decision:** close the remaining cotangent target as `defer/no-bank` for this bloat pass. The spec-backed implementation delegates to the jitted extraction-spec helper (`:775`), while the live class owns fallback-compatible projection, external-surface cotangent addition, and `profile_B_vjp` timing behavior (`:2118`, `:2205`). A source-negative reconciliation would touch behavior-bearing fallback/profile paths without enough acceptance coverage.
- **LOC saved:** 2 banked for the metadata-preserving points-only slice (`29 insertions / 31 deletions`). Do **not** bank the old `~250` estimate.
- **Risk:** Medium. Token plumbing must stay live-class only; point-version increments must stay identical; `_coils` property, `_uses_uniform_curve_xyz_fourier_fastpath`, and `coil_dof_extraction_spec()` remain class-owned; cotangent reconciliation needs separate fallback coverage before any hoist.
- **Validation gate:** points-only gate completed with focused CPU/X64 cylindrical accessor parity for live/spec-backed adapters (`4 passed`), `FieldEvalSpec` round-trip and float64 promotion selectors (`2 passed`), a runtime probe confirming public method `__qualname__` / signatures / `FieldEvalSpec` annotations and `clear_points` absence on `SpecBackedBiotSavartJAX`, scoped `ruff`, `py_compile`, and `git diff --check`. Source-only `mypy` now runs in `.conda/jax` but remains blocked by pre-existing `SpecBackedBiotSavartJAX.x` / `save` override errors in this file.

### 7.3 — [x] T3.3: Profile machinery → sibling diagnostic modules — **CLOSED / DEFER-NO-BANK FOR BLOAT (2026-06-02)**

- **v4/v5/v7 status (PARTIALLY DONE / RELOCATED):** The `surfaceobjectives_jax.py` half already moved — `diagnose_traceable_objective_runtime` and `make_traceable_objective_profile_suite` are no longer in `surfaceobjectives_jax.py` (now 3,110 LOC in the current checkout); they live in the **now-tracked** `surfaceobjectives_traceable_jax.py` (`diagnose…:2599`, `make_…_profile_suite:3493`), but are NOT yet isolated to a dedicated `_diagnostics` sibling. The v3 refs `surfaceobjectives_jax.py:5382-5650 / 5670-5959 / 6270-6284` are **dead**.
- **Files (remaining):**
  - `biotsavart_jax_backend.py` profile helpers (~`:215-303`, `:2125-2212`) → new `biotsavart_jax_profile.py` (still pending; re-derive refs against HEAD).
  - Finish the `surfaceobjectives` split: extract the diagnostics now living in `surfaceobjectives_traceable_jax.py` into a `_diagnostics` sibling, if that isolation is still wanted.
- **Decision:** close as `defer/no-bank` for this bloat pass. Moving the remaining Biot-Savart and traceable-objective profile/diagnostic helpers into sibling modules would improve modularity but is not a source-negative closure slice after import/re-export compatibility, public profile payload preservation, and runtime-cache boundaries are preserved. Reopen only as a dedicated modularity/API-compatibility design, not as bloat banking.
- **LOC saved:** 0 banked for this residual. Retire the old `~620 moved` estimate as a bloat target.
- **Risk:** Medium. Maintain redirect imports in original modules for backward compat.
- **Contracts:** External test caller signatures (`tests/integration/test_stage2_jax.py:1779`).
- **Validation gate:** T3 + profile-related tests.

### 7.4 — [x] T3.4: `_build_runtime_linear_solve_callbacks` 4-branch refactor — **CLOSED / DEFER-NO-BANK (2026-06-01)**

- **Files:** `boozersurface_jax.py` — `_build_runtime_linear_solve_callbacks` now starts at `:4166` in a 7,216 LOC file. The existing `pack_callbacks(...)` at `:4186` already owns the reusable tuple shape, backend label, dense-factor staging, and `_with_nan_status` wrapping. The live branches are `dense-plu-shared` at `:4229`, scipy host `dense-plu` at `:4313`, hessian operator callbacks at `:4390`, and exact-jacobian operator callbacks at `:4452`.
- **Decision:** Close as `defer/no-bank`, not a source refactor. The original extraction target is stale: sharing the dense PLU branches beyond `pack_callbacks(...)` would need residency/status/solver knobs that hide branch-specific contracts. The shared-LU path is device-resident, consumes `LU_PIV`, uses `_lu_solve_dense_hessian(...)`, and applies extra backward-error status repair; the scipy-PLU path is host-resident, explicitly marks host bridge transfers, and uses `scipy.linalg.solve_triangular(...)`. The hessian and exact-jacobian operator branches are also asymmetric enough that a shared `_pack_operator_callbacks(...)` would be boolean-adapter churn.
- **LOC saved:** 0 banked. The old `~120` estimate is no longer valid in the current tree.
- **Risk:** Medium. Must preserve byte-identity of shared `(lu, piv)` factor reuse; `_with_nan_status` wrap-once semantics; `apply_forward` / `apply_transpose` wired to device-resident `H_dev` directly. Re-derive the range from the latest committed tree before implementation.
- **Contracts:** `linear_solve_backend` reporting strings (`dense-plu-shared`, `dense-plu`, `operator`); `linear_solve_factors` payload shape (tuple of `jnp` arrays).
- **Validation gate:** Read-only current-tree revalidation plus exact line/contract inventory; no source changed, so no runtime validator was required. If revisited, run focused runtime-state selectors covering scipy PLU host bridge, shared-LU factor reuse, hessian operator callbacks, exact-jacobian operator callbacks, and strict transfer status behavior before touching code.

### 7.5 — [x] T3.5: LaneArtifact-builder coverage extension — **CLOSED / NEEDS-DESIGN NO-BANK (2026-06-02)**

- **Files:** `non_banana_example_parity_fixtures.py` — `LaneArtifact:151`, `_build_cpu_lane:368`, `_build_jax_lane:450`, `_build_scalar_lane:544` (v4's `:354-535` is stale). Current inline `LaneArtifact(...)` construction extends through surface scalar, QFM, PM, wireframe, Boozer, tracing, and `_tracing_lane_artifact` paths (`:806`, `:843`, `:991`, `:1025`, `:1256`, `:1299`, `:1509`, `:1774`, `:2097`, `:2303`, `:2343`, `:2524`, `:2567`, `:2769`, `:2797`, `:2971`, `:3008`, `:3184`, `:3215`, `:3994`, `:4132`, `:4333`, `:4479`, `:4663`, `:4685`, `:4873`, `:4895`, `:4956`). `_METADATA_RAW_ARRAY_KEYS_BY_FIXTURE_KIND` lives in the dirty driver `non_banana_example_cpp_jax_cpu_parity.py:353` (the old "promote from driver" target).
- **Decision:** close as `needs-design/no-bank` for this bloat pass. The remaining inline `LaneArtifact(...)` sites span distinct fixture families and metadata semantics; promoting `_METADATA_RAW_ARRAY_KEYS_BY_FIXTURE_KIND` out of the dirty driver would be a cross-file contract migration, not a safe helper slice. It also overlaps `T3.6`, whose comparison driver is currently dirty from source-negative helper work.
- **LOC saved:** 0 banked for this residual. Retire the old `~1,650` estimate until a separate design proves byte-identical `LaneArtifact` output and source-negative migration.
- **Risk:** Medium. `LaneArtifact` output must be byte-identical; hash kwargs, fixture IDs, classifications, unsupported-component lists, raw-array keys, and timing payloads must stay pinned.
- **Contracts:** Fixture IDs, classifications, FixtureBuild contract, comparison row order, raw-array hash fields, and release-blocker replay gates. `_pre_newton_census_gate_failures` remains untouched.
- **Validation gate if reopened:** T3 + full `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`, release-blocker gate replay, and per-fixture canonical JSON identity for `LaneArtifact` fields and comparison row `(quantity, component, source_example)` ordering.

### 7.6 — [x] T3.6: Table-driven `_*_comparisons` driver — **COMPARISON HELPERS LOC-BANKED / TABLE DRIVER DEFER-NO-BANK (2026-06-02)**

- **Current status:** source-negative T3.6 helper slices are banked and the full table-driven driver is closed as `defer/no-bank` for this pass. `_compare_raw_array(...)` now lives at `non_banana_example_cpp_jax_cpu_parity.py:605`; `_surface_geometry_comparisons(...)`, `_compare_component_scalar(...)`, `_required_lane_scalar(...)`, and `_compare_native_subtotal(...)` live at `:621`, `:634`, `:650`, and `:656`. These helpers centralize same-key raw-array access, canonical surface geometry rows, required component-scalar comparisons, required-lane scalar preconditions, and native subtotal comparisons while leaving fixture dispatch, optional rows, derived normal-projection checks, JSON lane comparisons, and PM algorithm labels explicit.
- **Files:** `non_banana_example_cpp_jax_cpu_parity.py` is now 2,789 LOC after the helper slices, down from 3,138 at `HEAD` (`419 insertions / 768 deletions`, net `-349` benchmark LOC). Current comparison anchors: `_supported_comparisons:961`, `_surface_scalar_comparisons:1045`, `_strain_comparisons:1081`, `_coil_force_energy_comparisons:1118`, `_qfm_comparisons:1172`, `_pm_comparisons:1205`, `_pm_relax_and_split_comparisons:1268`, `_wireframe_comparisons:1340`, `_wireframe_gsco_comparisons:1401`, `_boozer_fixed_state_comparisons:1496`, `_tracing_comparisons:1540`, `_boozer_qa_wrappers_comparisons:1603`.
- **Remaining target decision:** close the full `ComparisonSpec` table-driver rewrite as `defer/no-bank`. A safe table driver now needs typed specs plus pre/post canonical row-identity evidence, and likely becomes source-positive or hides fixture-family contracts after preserving optional-row and derived-budget special cases.
- **LOC saved:** 349 benchmark LOC banked across the raw-array and component/surface helper slices. Do not bank the stale full table-driver estimate.
- **Risk:** Low for the banked helper slices; medium for any future table-driver rewrite because row ordering and fixture-family routing become data-driven.
- **Contracts:** `_tolerance_for` lookup unchanged; `PARITY_LADDER_TOLERANCES`; bucket assignments; fixture dispatch and quantity ordering; derived normal-projection and JSON-lane comparisons remain specialized.
- **Validation gate:** banked helper slices passed scoped `ruff`, `ruff format --check`, `py_compile`, `pip check`, `git diff --check`, added-line guardrail scan, and full CPU/X64 parity integration file `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` (`62 passed, 3 warnings`). `mypy benchmarks/non_banana_example_cpp_jax_cpu_parity.py` remains blocked by broader benchmark/import typing debt (`82 errors in 7 files`), so no mypy proof is claimed. Future table-driver work still needs parity matrix row byte-identity.

### 7.7 — [x] T3.7: Generic adaptive-step trace driver factory — **CLOSED / NEEDS-DESIGN NO-BANK (2026-06-02)**

- **Files:** `jax_core/tracing.py` (**4,287 LOC**; v4 said 4,299) — drivers `trace_fieldline:1217`, `trace_guiding_center:1879`, `trace_guiding_center_boozer:2982`, `trace_fullorbit:3758` (plus batched variants); only `_run_dopri5_4state:2407` exists today (v4's `:1883/:2994/:3770/:2419` drifted ~10-12 lines). The bounded-scan helper is `_scan_adaptive_steps:367`.
- **Decision:** close as `needs-design/no-bank` for this bloat pass. There is plausible high-yield work here, but not a current-tree closure-safe helper slice: the drivers diverge on fieldline vs particle state, Boozer-axis invalid handling, Python pre-step field-state resolution, event localization, batched transfer behavior, and public result shapes.
- **LOC saved:** 0 banked. Retire the stale `~1,100` estimate until a dedicated tracing-driver design proves source-negative parity.
- **Risk:** Medium-high. Drivers diverge: `trace_fieldline` has no `_boozer_axis_invalid`; `trace_guiding_center_boozer` resolves field state in Python pre-step.
- **Contracts:** Stopping criteria (`_stopping_criterion_should_stop` + `is_boozer_state` switch); `phi_init` continuous-branch contract; exit status codes (`_BOOZER_AXIS_STATUS`).
- **Validation gate if reopened:** T3 + `tests/field/test_tracing_jax_item16.py` + `tests/field/test_tracing_jax_item16_extended.py` + `tests/jax_core/test_tracing_jax_*.py` + tracing benchmark sanity, including public signature/result-shape checks and event/status priority probes.

### 7.8 — [x] T3.8: Lazy-cache helper for `_make_traceable_*` family — **CLOSED / DEFER-NO-BANK (2026-06-02)**

- **Files:** `surfaceobjectives_traceable_jax.py` (**now-tracked, 3,543 LOC**) already owns the central cache abstraction: `_traceable_runtime_cache_key:1479`, `_get_cached_traceable_runtime_entry:1504`, and runtime-entry slot initialization. The remaining `_ensure_traceable_runtime_*` family is not uniform enough for a safe small `_lazy(entry, key, builder)` fold: reporting metrics are simple slots, public boundaries expose six public wrappers, optimizer value/grad uses a `general_only_forward=True` bundle, seeded value/grad caches baseline gradient state, and host wrappers preserve explicit host-boundary / baseline-peel behavior.
- **Decision:** close as `defer/no-bank`, not a source refactor. The original small-helper estimate is stale because the important cache-key and cache-entry work is already present, and the surviving ensure/make paths carry distinct contracts.
- **LOC saved:** 0 banked. Retire the old `~80` estimate for this bloat pass.
- **Risk:** Medium if reopened. Cache key contract is load-bearing (CLAUDE.md "Traceable runtime bundle cache contract"). Must preserve deterministic signatures of solved baseline state / objective kwargs / coil set spec; `is`-comparison for success-filter callables via `_TraceableCallableSignature`; no `id()` in any cache key; public reporting boundaries stay lazy until invoked; host wrappers preserve explicit host-boundary/baseline behavior.
- **Validation evidence:** read-only current-tree scout and line/contract inventory only; no source changed. If reopened as a larger rewrite, keep the distributed cache tests for key hashing, cache invalidation, public boundary laziness, seeded value/grad, optimizer value/grad, and integration reuse/rebuild coverage.

### Tier 3 exit gate

All 8 Tier 3 items are closed for this bloat pass, but this is not the old "all merged" target. The closure outcome is: T3.1 and T3.6 banked source-negative helper slices; T3.2, T3.3, T3.5, and T3.7 are no-bank / needs-design decisions; T3.4 and T3.8 were closed no-bank. Do not tag `bloat-reduction-T3-complete` until grouped closure validation, adversarial review, and contract checklist re-affirmation pass against the final dirty tree.

---

## 8. Tier 4 — Decision Points

These items deliver significant LOC reduction (potentially 1,640+ LOC) but require contract clarifications the audit alone can't resolve.

### 8.1 — [x] T4.1: Is `lbfgs-trace` host engine still needed? — KEEP / NO-DELETION DECISION (2026-06-02)

- **Decision:** Keep `lbfgs-trace` and stop counting `optimizer_host_lbfgs.py` as an active deletion target in this bloat pass.
- **Current-tree evidence:** `optimizer_host_lbfgs.py` is 1,628 LOC and still owns `record_optimizer_state_trace`, `failure_callback`, rejected-step sampling, and `host_invalid_step_log_to_list`. The route is reachable through `optimizer_jax_reference.py` (`reference_minimize(method="lbfgs-trace")`), `optimizer_jax.py` method validation and diagnostics, `benchmarks/single_stage_init_parity.py`, the production single-stage example CLI, and tests that pin failure-callback / invalid-step-log behavior.
- **Why no deletion:** `docs/solve_jax_api_spec_2026-05-19.md` records that callers cannot silently migrate this surface to a plain SciPy callback because the rejected-line-search-sample data is not surfaced by SciPy. On-device `invalid_step_log` and `scipy-jax` do not prove replacement of the host reference trace contract.
- **LOC saved:** 0 banked. Retire the old `~1,628 LOC` deletion estimate until an explicit API-evolution migration replaces `record_optimizer_state_trace`, `failure_callback`, and `invalid_step_log` consumers.
- **Future-only refactor:** A non-closure dedupe with `_line_search.py` may be investigated only if it preserves the host trace payload and has caller migration evidence.
- **Validation gate:** Docs-backed current-tree inventory only; no source changed, so no runtime validator is claimed.

### 8.2 — [x] T4.2: Reconcile `scipy-jax` and `scipy-jax-fullgraph` outer backends with CLAUDE.md spec — **DECISION MADE (v4): KEEP + document**

- **v7 status:** Plan A (HANDOFF.md, 2026-05-29) answered the open question, but v4/v5 misstated the default split. Live defaults route the JAX optimizer lane to `scipy-jax` on both CPU and CUDA (`tests/test_cli_defaults.py:36-41`, `:51-56`, `:133-145`; single-stage resolver `single_stage_banana_example.py:8346-8355`; Stage 2 resolver `banana_coil_solver.py:803-809`). `scipy-jax-fullgraph` remains an explicit stress/parity lane. This is the settled keep branch; the only remaining work is the CLAUDE.md / user-doc update. Both lanes are live in `optimizer_jax.py` (mapping `:227-228`, dispatch `:703`).
- **Current-tree status:** Alive user-facing surfaces. `scipy-jax` and `scipy-jax-fullgraph` are exposed by Stage 2 and single-stage example CLIs, mapped in integration tests, and routed in benchmark helper tests.
- **Decision status:** None on removal. Keep both lanes; no LOC is banked from this item.
- **Follow-up:** Closed on 2026-06-02. `CLAUDE.md` now separates the inner
  Boozer LS backend vocabulary from the outer Stage 2 / single-stage optimizer
  lanes, documents `scipy-jax` as the implicit/default JAX outer route,
  documents `scipy-jax-fullgraph` as the full-graph stress/parity route, points
  the runtime-cache token contract at `src/simsopt/_core/state_tokens.py`, and
  records that the private on-device L-BFGS-B helpers are a SciPy
  1.17.1-compatible port. `docs/using_jax_backend.md` mirrors the default/stress
  lane split in copy-paste examples.

### 8.3 — [x] T4.3: Should `qfm_solver._bfgs_minimize` reuse `optimizer_jax_private._bfgs._minimize_bfgs_private`? — REJECT REUSE / NO-BANK (2026-06-02)

- **Decision:** Reject mechanical reuse. Keep the QFM-local BFGS implementation unless a separate QFM retune experiment proves the solver behavior can change.
- **Current-tree evidence:** `qfm_solver.py:497-634` still owns a QFM-local BFGS path with Armijo-style acceptance and QFM-specific convergence defaults. `optimizer_jax_private/_bfgs.py` exposes a private optimizer BFGS route with different line-search / curvature semantics. Existing tests compare curvature helpers, but they do not prove full solver replacement equivalence.
- **Why no reuse:** This is a behavior-contract decision, not a duplicate-code decision. Replacing the Armijo-only inner BFGS with the private optimizer path can change iteration counts, acceptance, and warm-start behavior in the QFM exact/AL wrappers.
- **LOC saved:** 0 banked. Retire the old `~138 LOC` reuse estimate until a retune experiment exists.
- **If revisited:** Run QFM acceptance covering natural-equality KKT success, feasible-nonstationary rejection, branch-stability invariants, host-SLSQP diagnostics, infeasible/warm-start perturbation cases, and the existing QFM JAX/host comparison tests before touching source.
- **Validation gate:** Docs-backed current-tree inventory only; no source changed.

### 8.4 — [x] T4.4: Rename or quarantine `minimize_qfm_exact_constraints_SLSQP` alias — KEEP COMPATIBILITY ALIAS (2026-06-02)

- **Decision:** Keep the alias. Do not quarantine or deprecate it in this bloat pass.
- **Current-tree evidence:** `SLSQP` / `slsqp` has 0 occurrences in `qfm_solver.py` (true; the solver-level exact path is augmented-Lagrangian / exact-KKT), but the public surface wrappers are live: host CPU `qfmsurface.py:147` remains the true SLSQP entrypoint, and JAX `qfmsurface_jax.py:281` is documented as a compatibility alias for the augmented-Lagrangian exact path. Examples and tests still call the alias and method form.
- **Why keep:** The alias is public compatibility surface, not dead solver internals. Renaming would be API evolution and likely LOC-neutral or source-positive.
- **LOC saved:** 0 banked.
- **Validation gate:** Docs-backed current-tree inventory only; no source changed.

### 8.5 — [x] T4.5: Surviving tautological tests — CLASSIFIED / NO-DELETION (2026-06-02)

- **Decision:** Accept the surviving cases as Tier 4 routing/health tests for now; no deletion or mock-invariant consolidation in this bloat pass.
- **Current-tree evidence:** `tests/field/test_trace_boozer_analytic_jax.py` still carries explicit routing assertions. `tests/integration/test_single_stage_jax_cpu_reference.py` labels option propagation, same-helper routing, cached-transform identity, branch-divergent health, adjoint self-consistency, and VJP health checks. `tests/objectives/test_fluxobjective_jax_parity.py` has fast-path identity assertions, and `tests/geo/test_curve_objectives_jax.py` retains signature identity checks.
- **Why no deletion:** These tests are weak as independent numerical oracles, but several are explicit routing or health checks. Deleting them would remove named route guards without adding independent oracle coverage. Rewriting them would likely be LOC-neutral or source-positive and belongs in a test-oracle improvement pass, not bloat closure.
- **LOC saved:** 0 banked.
- **If revisited:** Convert per case to an independent oracle or a stronger invariant, then update `tests/REVIEWER_ORACLE_LINT.md` classification. Do not count identity-test removal as LOC bank unless route coverage is replaced.
- **Validation gate:** Docs-backed current-tree inventory only; no source changed.

---

## 9. Cross-Cutting Validation Gates

Run these after EACH tier (not just at exit). The baseline local gate is CPU-only; GPU/parity-sensitive items must also produce the CUDA/strict-transfer proof in Section 9.5 before the tier can close. Activate `jax` env first:

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=cpu
```

### 9.1 Smoke + unit (after every commit)

The command below is the canonical §9.1 file set. It may be run either as one
pytest invocation or as split chunks with the same environment and
`-m "not private_optimizer_runtime"` marker when walltime or session durability
makes a single invocation impractical. Record split evidence as split evidence;
do not present it as an all-in-one command pass unless the all-in-one command was
actually run.

```bash
.conda/jax/bin/python -m pytest tests/test_jax_import_smoke.py \
    tests/field/test_biotsavart_jax.py \
    tests/geo/test_surface_fourier_jax.py \
    tests/geo/test_boozer_residual_jax.py \
    tests/objectives/test_integral_bdotn_jax.py \
    tests/geo/test_boozer_derivatives_jax.py \
    tests/geo/test_boozersurface_jax.py \
    tests/integration/test_jax_native_path.py \
    -m "not private_optimizer_runtime" -v
```

**2026-06-02 local CPU/X64 split rerun:** the full §9.1 file set passed when
run in split chunks under `PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True
JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu`. Results: import smoke `114 passed, 11
skipped in 807.65s`; Biot-Savart JAX `53 passed in 65.16s`;
surface/boozer-residual/integral chunk `255 passed, 105 skipped in 87.30s`;
Boozer derivatives/BoozerSurface chunk `502 passed, 4 skipped in 565.20s`;
native-path integration `14 passed in 5.24s`. The CI `jax-public-unit` job still
contains the same §9.1 file set, with `tests/test_jax_import_smoke.py` as the
separate import-smoke step and the remaining files in the public pure-JAX unit
step; that job also includes extra public checks outside §9.1.

### 9.2 Private optimizer (after Tier 1, before any optimizer touch in Tier 2/3)

```bash
.conda/jax/bin/python -m pytest tests/geo/test_boozersurface_jax_private.py \
    tests/integration/test_section6_public_lane_split.py \
    tests/integration/test_single_stage_jax_cpu_reference.py \
    -m "private_optimizer_runtime" -v
```

### 9.3 Benchmark/runtime helper regressions

```bash
.conda/jax/bin/python -m pytest tests/test_run_code_benchmark_common.py tests/test_benchmark_helpers.py -v
```

### 9.4 Integration (Stage 2 + parity)

```bash
.conda/jax/bin/python -m pytest tests/integration/ -v
```

### 9.5 CUDA + strict-transfer proof (tier-exit for GPU-sensitive changes)

Required when a tier touches `sharding.py`, `biotsavart_jax_backend.py`, Stage 2/single-stage parity infrastructure, optimizer routing, or backend transfer/cache runtime config.

Existing CI gates that satisfy this proof on a CUDA runner:

- `.github/workflows/jax_smoke.yml` job `jax-gpu-e2e`.
- `.github/workflows/jax_smoke.yml` job `jax-gpu-strict-purity`.

Minimum proof contents:

- CUDA runtime and x64 contract verified.
- `SIMSOPT_BACKEND_MODE=jax_gpu_parity`.
- `SIMSOPT_BACKEND_STRICT=1`.
- `SIMSOPT_JAX_TRANSFER_GUARD=disallow`.
- Stage 2 CUDA e2e artifact and single-stage CUDA init parity artifact when optimizer/parity code changed.
- Strict transfer-guard smoke slice plus Boozer CUDA inner-solve and single-stage outer-loop proof when field/objective/runtime code changed.

CPU-only test success is not sufficient to mark these tiers complete.

### 9.6 Release-blocker gate replay (T2.8, T3.5 mandatory)

Pinned fixture replay of `_pre_newton_census_gate_failures` must produce byte-identical failure list pre/post.

### 9.7 Ruff per touched file

```bash
.conda/jax/bin/python -m ruff check <changed-files>
.conda/jax/bin/python -m ruff format <changed-files>
```

---

## 10. Sequencing Rationale

Why this order:

1. **Tier 1 first because risk is lowest.** Pattern-validate the `_lazy_exports` adoption on `jax_core/__init__.py` (single file, well-trodden pattern) before applying anything similar at scale.
2. **Tier 2 second because factories are local.** Each factory introduction is bounded to one or two files; pattern-validates the larger structural folds in Tier 3.
3. **Tier 3 third because risk concentrates.** GPMO 5-way fold and trace driver factory touch hot code paths the parity gates depend on. Doing these last means Tier 1 + Tier 2 trust has built up confidence in the validation discipline.
4. **Tier 4 alongside Tier 1.** Open Tier-4 questions early. This is now done for the current pass: T4 closes as no-bank behavior/compatibility decisions, and `lbfgs-trace` is no longer a deletion-gated LOC target.
5. **One commit per item.** Bisectable; if a parity test regresses, `git bisect` to the offending commit in under 10 minutes.

---

## 11. Rollback Strategy

- Every item is its own commit with the format `refactor(scope): <change> — preserve <contract>`.
- Tier exit: tag (`bloat-reduction-T1-complete`, etc.).
- If a contract breaks in Tier 3: revert the offending commit only; Tier 1 + Tier 2 wins stay.
- If validation gate fails mid-Tier-3: stop, do not proceed to next item; either fix-forward or revert that single commit.
- No squash merges across tiers; preserve atomic history.

---

## 12. Open Questions

These need user answers before starting or deliberately reopening scope:

1. ☑ **Branch strategy.** Closed on 2026-06-02 for this dirty-tree closeout: stay on current branch `shared-jax-clean`; no new branch is created.
2. ☑ **Time horizon.** Closed on 2026-06-02: this is the current closeout pass; Tier 4, partial Tier 2 residuals, and Tier 3 residuals are closed for this pass, and no sprint/background follow-up is opened unless new scope is deliberately reopened.
3. ☐ **GPU proof venue.** Use the existing self-hosted GitHub CUDA runner, Perlmutter, or another CUDA host for Section 9.5 tier-exit proof?
4. ☑ **Tier 4 decisions.** Closed as of 2026-06-02: T4.1 keep `lbfgs-trace`; T4.2 keep/document optimizer lanes; T4.3 reject QFM BFGS reuse without retune proof; T4.4 keep QFM SLSQP compatibility alias; T4.5 classify surviving routing/health tests with no deletion.
5. ☑ **CLAUDE.md updates.** Closed on 2026-06-02 in this effort: `CLAUDE.md`
   now documents the `scipy-jax` default outer lane, the
   `scipy-jax-fullgraph` stress/parity lane, `src/simsopt/_core/state_tokens.py`,
   and the SciPy 1.17.1-compatible private on-device L-BFGS-B port.
6. ☑ **Memory/project note.** Closed on 2026-06-02: no T1.4 curve↔jax_core import-cycle behavior changed in this closeout, so no project note is required. Do not edit `MEMORY.md` directly unless explicitly requested.

---

## 13. Estimated Totals

| Tier | LOC reduction (est.) | Items | Effort | Risk |
|------|---------------------:|------:|--------|------|
| T1 — Mechanical Wins | ~630-670 guaranteed (v3 ~950; T1.1 cut ~620→~300 as file already shrank to 363 LOC) (+ up to ~1,245 probe-script decision-gated) | 11 (T1.11 verified) | 2 days | Low-Med |
| T2 — Factory Introductions | Closed for this pass: partial residuals now explicitly banked or deferred no-bank. Do not count unbanked old T2.1/T2.2/T2.3/T2.8 estimates without a new acceptance gate. | 0 active source sections | Tier-exit proof only | Low-Med |
| T3 — Structural Consolidations | Closure pass complete for active source queue. Banked: T3.1 records 225 source LOC, T3.2 points 2 LOC, T3.6 comparison helpers 349 benchmark LOC. No-bank/needs-design: T3.3, T3.5, T3.7, T3.4, T3.8 residuals. Historical ~4,000-5,500 estimate is retired. | 0 active source sections for this pass | Grouped validation/review only | Med |
| T4 — Decision Points | All five decisions closed. T4.1 keeps `lbfgs-trace`; T4.2 keeps/documents optimizer lanes; T4.3 rejects QFM BFGS reuse without retune proof; T4.4 keeps the QFM SLSQP compatibility alias; T4.5 classifies surviving routing/health tests. No LOC banked. | 5 closed | done for this pass | No-bank / behavior-bound |
| **Aggregate** | **Closure-mode aggregate.** Historical estimate was ~7,900-9,800 guaranteed candidate LOC remaining pending T2.2 re-estimate; Tier 4 deletion gates, partial T2 residuals, and no-bank Tier 3 residuals are retired for this pass. Do not reuse the old drift-checkpoint ledger as a current banked-reduction claim. | **Historical 33-item ledger is superseded by the 0-active-source closure state after partial T2, all T3 items, and Tier 4 closure** | **Grouped validation/review; fresh triage only for reopened scope** | **manageable only with closure mode and the v8 drift gate** |

**Closure-mode replacement for the stale aggregate:** after commit `4fcd33b05`, the initial live queue was 16 unchecked sections. Partial Tier 2, all Tier 3 sections, and Tier 4 decisions are now classified for this bloat pass. The active source queue is 0; remaining work is grouped validation, adversarial review, and any follow-up fixes, not more unscouted source implementation.

---

## 14. Appendix A — How to Use This Document

### As a checklist during execution

1. In normal tier mode, work tiers sequentially: T1 → T2 → T3. For the current closure pass after `4fcd33b05`, §4.5.1 supersedes this ordering: partial Tier 2, all Tier 3 items, and Tier 4 decisions are now classified, so the active source queue is 0 unless new scope is deliberately reopened.
2. Within a tier or closure bucket, items are mostly independent but ordered by risk and expected banked source reduction.
3. Under closure mode, each reopened `- [ ]` item maps to one of three outcomes before implementation: `do-now`, `decision-only`, or `defer/no-bank`.
4. Each commit should close a task, bank meaningful source LOC, or ratify a contract decision. Do not keep creating tiny source-negative commits when the remaining scope is better closed as `defer/no-bank`.
5. After each item: run the per-item validation gate; do not proceed if it fails.
6. At tier exit: run the tier exit gate and the contract checklist in Section 4.1; tag the commit.

### As a status report

The current state of execution is encoded in the section checkboxes plus the closure-mode bucket assigned to each remaining section. The old dirty-tree caveat is historical after the scoped split through `4fcd33b05`; future status reports should cite committed checkpoints, not broad uncommitted implementation piles.

### As a contract reference

Section 4.1 is the authoritative "do not touch" list during this refactor. If anything not on that list looks load-bearing, escalate before editing.

### If a contract breaks

1. Revert the offending commit (single commit, atomic).
2. File a local incident note describing what broke and why. If this checkout has no artifact tree, create one at `.artifacts/bloat-reduction-2026-05-20/incidents.md` or record the incident in the execution handoff that owns the rollback.
3. Update Section 4.1 if the audit missed a contract.
4. Resume the next item.

---

## 15. Appendix B — Audit Source Trace

This plan was generated from 8 parallel subagent reports (2026-05-20):

| Lane | Scope | Reducible LOC |
|------|-------|--------------:|
| 1 | Boozer surface + objectives (`boozersurface_jax`, `surfaceobjectives_jax`, `boozer_residual_jax`) | 700–900 |
| 2 | Optimizer JAX (`optimizer_jax`, `optimizer_jax_private/*`, `optimizer_jax_reference`, `optimizer_host_lbfgs`) | 800–1,000; T4.1 deletion retired as no-bank |
| 3 | jax_core kernels (`tracing`, `surface_fourier_kernels`, `biotsavart`, `specs`, `sharding`, etc.) | 2,600 |
| 4 | Field/backend layer (`backend/runtime`, `biotsavart_jax_backend`, `field/*`) | ~700 |
| 5 | PM / QFM / wireframe (`pm_optimization`, `pm_workflow`, `qfm_solver`, `boozer_radial_field`) | 3,100 |
| 6 | Benchmarks/parity infrastructure (`non_banana_*`, `single_stage_*`, `validation_ladder_*`) | 2,300–3,000 (+ ~1,245 probe scripts only if caller migration retires them) |
| 7 | JAX tests (`test_boozersurface_jax`, `test_single_stage_jax_cpu_reference`, etc.) | ~500 historical; T4.5 identity/routing cleanup is no-bank for this pass |
| 8 | Cross-cutting duplication (sibling-variant files, re-export shims, dtype helpers, state tokens) | ~850 |

Historical aggregate candidate estimate after lane overlap consolidation: **~8,300–9,800 LOC**, plus optional decision-gated deletions and partial Tier 2 residuals that are now retired no-bank for this pass. Under the v8 drift gate, this is not a current banked-reduction claim; re-measure only if new scope is deliberately reopened after the 0-active-source closure state.

Full audit transcripts available in the orchestrator session log (2026-05-20). The original bloat plan was generated from 8 bloat-reduction lanes; the separate code-smell artifact contains 11 per-lane reports + `SUMMARY.md` + 24 verification-round logs. In the `simsopt-jax-shared-jax` checkout reviewed at `b267b0d95`, `.artifacts/` is not present; the historical artifact tree was found at `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/`. Those logs are historical records with 2026-05-30 line-ref annotations originally refreshed against HEAD `21c3d517d`; use `SUMMARY.md` plus `verification/CORRECTIONS_round5.md` inside that artifact tree as the final status for retracted items, and re-grep current HEAD/dirty files before executing.

---

*End of plan v8. Crucible-reviewed; v5 basis + §4.1 / §5–§8 line refs were re-derived against clean HEAD `21c3d517d` on 2026-05-30, corrected against live HEAD `2bcaeff28` and a dirty worktree by a 3-agent doc-review pass, refreshed against clean HEAD `b267b0d95` on 2026-05-31, then updated with the 2026-06-01 `8b94c2bbd` execution-drift gate. Re-grep before executing — the repo is under active concurrent commits.*
