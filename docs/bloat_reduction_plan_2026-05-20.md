# simsopt-jax Bloat Reduction Plan

**Status:** Draft v8 — execution-drift checkpoint (2026-06-01) against current checkout `8b94c2bbd` on branch `shared-jax-clean`. Supersedes Draft v7 (clean source-doc refresh basis `b267b0d95`, 2026-05-31), Draft v6 (basis `2bcaeff28`, dirty correction pass), Draft v5 (basis `21c3d517d`, 2026-05-30), and Draft v4 (basis `5bcd9061c`, 2026-05-29). v8 does not recalculate every historical line ref; it records that the current dirty tree has drifted from strict LOC reduction into contract-hardening / foundation work. Treat old aggregate estimates as stale until salvage commits are split and re-measured.
**Author:** orchestrator synthesis of 8-lane parallel audit (2026-05-20); v4 reconciliation from a 6-agent re-verification (2026-05-29); v5 reconciliation from a 4-agent re-verification (2026-05-30); v6 doc-review correction from a 3-agent drift audit (2026-05-30); v7 current-checkout doc-review refresh (2026-05-31); v8 drift checkpoint after dirty-tree LOC validation (2026-06-01).
**Branch:** current checkout `shared-jax-clean` at `8b94c2bbd`.
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

The current dirty tree must not be committed as a single "bloat reduction" change. It contains real source shrinkers, but the checkout as a whole is larger once untracked helpers, tests, and docs are counted.

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

Historical target: total net deletion >= 8,000 LOC after all tiers; zero feature regressions; one ratified contract decision per Tier-4 item. Under the v8 drift gate, this target is not satisfied by the current dirty tree. It can only be claimed after banked-shrink slices are isolated, validated, and re-measured.

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
- [ ] `_pre_newton_census_gate_failures` at `single_stage_init_parity.py:3344` (def; used `:3421`/`:3431`; `SystemExit` wiring `:4515`/`:4822`). Release blocker. (v4 cited def `:2877`/used `:2954`-`:2964`; the SUMMARY's `:2198`, the v3 `:3275-3279`, and v6's `:2854` / `:2931` / `:2941` / `:4150-4153` refs are all stale.)
- [ ] `PARITY_LADDER_TOLERANCES` and all sibling tolerance tables in `benchmarks/validation_ladder_contract.py`.
- [ ] 7 backend modes (`native_cpu`, `jax_cpu_fast`, `jax_cpu_parity`, `jax_cpu_float32_smoke`, `jax_gpu_fast`, `jax_gpu_parity`, `jax_mps_smoke`) + hard rejection of removed `jax_metal_smoke` / `metal` selectors.
- [ ] `XLA_FLAGS` validation BEFORE `import jax`; `XLA_PYTHON_CLIENT_*` env writes before JAX init.
- [ ] `_coil_dof_state_token` semantics (advances on aggregate writes AND SIMSOPT ancestor invalidation).
- [ ] `SquaredFluxJAX` JIT closure capture at construction + 3 drift detectors.
- [ ] `get_adjoint_runtime_state()` runtime SSOT for exact-lane adjoint.
- [ ] `_normalize_solver_options` exact strip: function at `boozersurface_jax.py:3617` (still exact); the strip itself (`if boozer_type == "exact": normalized_options.pop("optimizer_backend", None)`) is at `:3691-3692`. (v3 cited `:3122 / 3185-3186 / 3419-3420`, all stale.)
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

### 4.6 — v4 reconciliation summary (2026-05-29)

Findings from the 6-agent re-verification against HEAD `5bcd9061c`:

- **Scaffolding is the stale part, not the substance.** Almost none of the 33 refactors has been executed, so the consolidation work is still wanted. What drifted is the basis commit, the §4.5 tables, and nearly every line ref/count.
- **Items already done / overtaken (now marked):**
  - **T2.2** — v4 said ALREADY DONE, but v6 corrects this to **PARTIAL / NOT BANKED**: `boozer_radial_field.py` still carries parallel `_eval_*_from_columns` implementations (`:452-787`) and separate `_eval_*` implementations (`:807-1190`). Keep the item open until the thin-wrapper fold lands; do not remove its ~400 LOC from the T2 target.
  - **T3.3 / T3.8** — PARTIALLY DONE / RELOCATED into `surfaceobjectives_traceable_jax.py` (diagnostics + `_make_traceable_*` family), not the `_diagnostics` sibling the items name.
  - **T4.2** — DECISION MADE by Plan A (HANDOFF.md, 2026-05-29), with v6 wording corrected against live defaults: `scipy-jax` is the default JAX optimizer lane on both CPU and CUDA; `scipy-jax-fullgraph` remains an explicit stress/parity lane. This collapses to docs, not removal.
  - **T4.4** — LIKELY OBSOLETE: `SLSQP`/`slsqp` has 0 occurrences in `qfm_solver.py`; confirm with a full-repo grep, then close. **[v5 correction: this was WRONG — the alias is alive in the qfm *surface* wrappers (`qfmsurface_jax.py:281`, `qfmsurface.py:147`). See §4.7 / §8.4.]**
  - **T1.11** — VERIFIED satisfied (`_skip_case` used 19×).
- **Corrected counts/figures:** T1.1 `jax_core/__init__.py` is 363 LOC not 676 (saving ~300 not ~620); T1.3 has 6 Result classes not 5; T1.6 covers 6 methods not 12; T1.8 over-counted dead aliases (`_zero_profile_component_timings` is live at `:2249`).
- **SOFTWARE_DESIGN.md alignment:** the plan is substantively compliant (cited gate names match SD verbatim; SD mtime 2026-05-19 predates the plan, so refs are current). v4 added: the complexity-not-LOC success gate (§2.1), the tie-breaker precedence note (§3), and the config-param/dependency/deprecation-timeline pre-flight steps (§4.3). One item to watch during execution: **T3.2 mixin extraction** vs SD's "implementation inheritance leaks state — prefer composition" red flag (SD:283).

### 4.7 — v5 reconciliation summary (2026-05-30)

Findings from the historical v5 4-agent re-verification against clean HEAD `21c3d517d`, plus v6/v7 current-checkout corrections:

- **The bloat refactors themselves are still almost entirely un-started.** The committed batch (`f287bde96` and successors) was *audit remediation* — correctness / strict-CUDA / test-oracle fixes — not *bloat reduction*; the consolidation work in §5–§8 remains wanted. v4's structural conclusions stand; v5 only refreshes the basis, the line refs, and a few item statuses.
- **~17–19 of the 2026-05-20 code-smell audit's High-cluster findings are now fixed in-tree** (verified by re-grep, not just re-pointed; a handful are only *partially* fixed — see the per-finding `RESOLVED`/`PARTIAL` annotations in the refreshed reports). Fully resolved: A1-F1 (`_safe_radius_squared`→`_radius_squared`, clamp removed), A2-F3 (`_set_global_coil_dofs` token-after-mutation), A2-F12 (gradient delegation), A4-F1/A10-M2 (`recompute_bell` token+cache), A4-F2 (precomputed surface signature), A5-F2 (all `jax.debug.callback` now `ordered=False`), A6-H1 (`f_bwd` NaN routing — was a CLAUDE.md contract FAIL, now PASS), A6-H2 (`sdofs` kwarg), A6-M5 (column-batched adjoint), A7-H1 (conftest no longer registers the deleted `backend.py`), A7-M1 (`metal`→`jax_mps_smoke`), A7-M2 (CLAUDE.md backend row), A8-H4 (vectorized `modB`), A9-H3 (vectorized coil unroll), A9-M1 (runner cache re-keyed to `_cached_traceable_runner`), A10-H1 (QfmResidualJAX hot-path read + `append_parent` removed), A8-H3 (`jax.local_devices()` at both `surfaceobjectives_jax.py:2151` and `src/simsopt/solve/mpi_jax.py:114`), A11-H2/H3/H4/M3 (test-oracle fixes). V6 correction: A2-F2/A10-H2 is **resolved** by `SpecBackedBiotSavartJAX.update_free_dof_size_indices` advancing `_dof_layout_version`; the missing spec-backed `set_recompute_flag` override is moot because the spec-backed adapter owns its DOFs directly and has no upstream `depends_on` cascade. A8-H1 remains partial: `wireframe_workflow.py:390-411` still has `jnp.asarray(max_steps/history_capacity, dtype=...)` paths, and `:410-411` plus `:1167` remain outside the tracer-only branch. These touch parity-sensitive surfaces but are orthogonal to the bloat plan. Artifact status correction: `.artifacts/` and `verification/` are absent from the `simsopt-jax-shared-jax` checkout at `b267b0d95`; the refreshed audit artifacts were found in sibling checkout `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/`, with final retraction status at `verification/CORRECTIONS_round5.md` under that artifact tree.
- **Item already done / overtaken (addition to v4's list):**
  - **T3.2** — the field-evaluation duplication is **ALREADY DONE**: a `_BiotSavartFieldEvaluationMixin` (`biotsavart_jax_backend.py:489-515`) is now inherited by both `SpecBackedBiotSavartJAX` and `BiotSavartJAX`. The remaining T3.2 dup is only `set_points`/`get_points*` and the diverged `coil_cotangents_to_dofs_gradient`; re-scope down from ~250 LOC.
- **Corrections to v4 item text:**
  - **T2.9** — v4/v3 over-counted the migration sites. `_tolerance_for` exists in **only one** file (`non_banana_example_cpp_jax_cpu_parity.py:323`). `single_stage_parity_matrix.py` has **no** `_tolerance_for` (the cited `:292-302` is wrong), and `stage2_e2e_comparison.py` uses a different mechanism (`optimizer_drift_tolerances:71/74`). The ~100 LOC estimate is overstated.
  - **T4.4** — **NOT obsolete.** `minimize_qfm_exact_constraints_SLSQP` has 0 occurrences in `qfm_solver.py` (true; the exact path is augmented-Lagrangian), but the public alias **still exists** in the surface wrappers `qfmsurface_jax.py:281` and `qfmsurface.py:147`, with live test callers. The clarity/quarantine decision is still open — just re-scoped to the surface wrappers.
  - **T2.8** — "16 trackers" undercounts: there are ≥27 distinct `max_*`/`first_*` accumulators (~75 tracker-related vars) in `compare_same_candidate_objective_replay` (now at `:2946`, not v3's `:2300-2700`).
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
- **Still-open bloat status:** T1.1 remains un-started, T2.2 remains partial/open, T3.2 remains partial, and T4.2 remains resolved/doc-only.

---

## 5. Tier 1 — Mechanical Wins

**Target:** ~630–670 LOC guaranteed reduction, plus up to ~1,245 LOC only if probe-script migration proves safe. **Effort:** ~2 days. **Risk:** Low-Medium.

Goal: bank low-risk LOC reduction first; pattern-validate factory ideas in tiny scope.

### 5.1 — [ ] T1.1: `jax_core/__init__.py` → lazy export map

- **Files:** `src/simsopt/jax_core/__init__.py` (**363 LOC**, observed @`b267b0d95` — down from 676, but the shrink was unrelated churn, **not** this refactor: the explicit dual-list is still fully present and there is no `build_lazy_export_map` call, so T1.1 is genuinely un-started. Do not infer "done" from the LOC delta).
- **Change:** Replace the explicit dual list (`_EXPORT_MODULES` at `:19`, `__all__` at `:336`, `_EXPORT_MODULE_OBJECTS` at `:339`, manual `__getattr__` at `:357`) with `_lazy_exports.build_lazy_export_map(...)` matching `src/simsopt/geo/__init__.py:75` and `src/simsopt/field/__init__.py:56`.
- **LOC saved:** **~300** (revised down from v3's ~620 — the file is now 363 LOC, so ~620 is impossible).
- **Risk:** Low. Five sibling packages already use this pattern.
- **Contracts:** Each of the 14 submodules in `_EXPORT_MODULE_OBJECTS` must have a literal `__all__`. The helper raises on duplicates; verify no cross-module name collisions.
- **Validation gate:** T1 + manual `python -c "from simsopt.jax_core import *"` smoke.

### 5.2 — [ ] T1.2: `backend/__init__.py` dual-list collapse

- **Files:** `src/simsopt/backend/__init__.py` (117 LOC).
- **Change:** Replace explicit dual-list with `from .runtime import *` driven by `runtime.__all__`, or use `build_lazy_export_map`.
- **LOC saved:** ~50.
- **Risk:** Trivial.
- **Validation gate:** T1.

### 5.3 — [ ] T1.3: Retire GPMO `Result` dataclass mirrors

- **Files:** `src/simsopt/solve/permanent_magnet_optimization_jax.py:116-314` (file now 921 LOC; v3 cited `:99-303`).
- **Change:** Replace **6** cloned `*Result` dataclasses (`PMRelaxAndSplitResult:117`, `GPMOBaselineResult:143`, `GPMOMultiResult:175`, `GPMOBacktrackingResult:209`, `GPMOArbVecResult:247`, `GPMOArbVecBacktrackingResult:279`; mirrors of `pm_optimization.py` plus 2 fields `m`, `m_history`) with one `GPMOPublicResult(core_result, m, m_history)` shim or `@dataclass(frozen=True)` extension. (v3 said "5 cloned" — there are 6.)
- **LOC saved:** ~180.
- **Risk:** Low.
- **Contracts:** Pytree registration must persist; field order load-bearing for any pickled state.
- **Validation gate:** T1 + PM tests if present.

### 5.4 — [ ] T1.4: Centralize `_as_numpy_float64`

- **Files:** `src/simsopt/_core/jax_host_boundary.py` (add `host_float64` — the module already has `host_array:14` / `host_float:27` / `host_int:31` / `host_bool:35`, but **no** `host_float64` and **no** `_as_numpy_float64`, so this is un-started); migrate 4 local copies: `geo/curve.py:64`, `geo/curveobjectives.py:76`, `geo/curvecwsfourier.py:23`, `geo/surfaceobjectives.py:1224` (last is a nested local def).
- **LOC saved:** ~40.
- **Risk:** Low. The curve↔jax_core import cycle is on the `_as_jax_float64` side, not numpy.
- **Contracts:** Preserve `_HAS_JAX` short-circuit in curve.py (move into helper).
- **Validation gate:** T1.

### 5.5 — [ ] T1.5: Centralize state-token counters

- **Files:** New `src/simsopt/_core/state_tokens.py` (does not yet exist). Migrate `_new_coil_dof_state_token` (`biotsavart_jax_backend.py:100`; v3 said `:79-93`) and `_new_traceable_solve_state_token` (`boozersurface_jax.py:139`; v3 said `:135-139`).
- **LOC saved:** ~20.
- **Risk:** Trivial.
- **Contracts:** Token attribute names accessible to consumers (`_coil_dof_state_token`, `_traceable_solve_state_token`, `_dof_layout_version`, `_points_version`).
- **Validation gate:** T1.

### 5.6 — [ ] T1.6: Preserve `compute_derivatives=N` compatibility while deduplicating Biot-Savart current-derivative wrappers

- **Files:** `biotsavart_jax_backend.py:564-604` (**6** current-derivative methods carrying `compute_derivatives`, at `:564, 572, 580, 588, 596, 604`; v3 cited `:712-765, 1806-1859` and "12 methods" — both stale).
- **Change:** Keep the public `compute_derivatives` kwarg on every JAX method because SIMSOPT CPU APIs and official docs expose it. Deduplicate only the shared wrapper/docstring text or route through a common helper that still accepts the kwarg.
- **LOC saved:** ~10-20.
- **Risk:** Low. The compatibility signature is the contract; in-repo caller absence is not proof that external callers do not pass the kwarg.
- **Validation gate:** T1 + signature regression that `BiotSavartJAX` and `SpecBackedBiotSavartJAX` accept `compute_derivatives=0/1/2` on all six current-derivative methods.

### 5.7 — [ ] T1.7: Delete dead `_host_cubicmin` / `_host_quadmin` / `_line_search_sample_valid_host`

- **Files:** `src/simsopt/geo/optimizer_jax_private/_common.py:117-163` (`_host_cubicmin:117`, `_host_quadmin:144`, `_line_search_sample_valid_host:158`; v4 said `:117-161`).
- **Change:** Delete — confirmed **zero callers repo-wide** (host duplicates live in `optimizer_host_lbfgs.py`).
- **LOC saved:** ~44.
- **Risk:** Very low.
- **Validation gate:** T1.

### 5.8 — [ ] T1.8: Delete dead aliases in `biotsavart_jax_backend.py`

- **Files:** `biotsavart_jax_backend.py:915` (`_ones_like_float64` — confirmed dead, zero callers).
- **v4 NOTE:** `_zero_profile_component_timings` (now `:261`) has a **live caller at `:2249`** — it is NOT dead; drop it from this item. The 4 `*_cotangents` "re-exports" (v3 `:1905, 1956, 1965, 1974`) are live names used throughout the file (e.g. `:786, 2147, 2179`) — re-audit before deleting any.
- **LOC saved:** ~5 (only `_ones_like_float64`; v3's ~15 over-counted).
- **Risk:** Very low.
- **Validation gate:** T1 + grep audit.

### 5.9 — [ ] T1.9: Delete `SingleStageRuntimeSpecBiotSavartJAX` 11-line subclass

- **Files:** `src/simsopt/field/biotsavart_jax_backend.py:808-818` (now an **11-line** subclass; v4 said `:836-847`); 3 call sites (`single_stage_banana_example.py`, `tests/.../test_single_stage_physics_parity.py`, `tests/.../test_single_stage_example.py`).
- **Change:** Replace with 1-line `SpecBackedBiotSavartJAX(make_biot_savart_spec(...))`.
- **LOC saved:** ~11.
- **Risk:** Low.
- **Validation gate:** T1.

### 5.10 — [ ] T1.10: Classify probe scripts; retire only after live caller migration

- **Files:** `benchmarks/run_code_parity_probe.py` (174), `benchmarks/production_boozer_parity_probe.py` (281), `benchmarks/single_stage_surface_reprojection_probe.py` (460), `benchmarks/surface_rz_geometry_hlo_probe.py` (339; v3 said 330).
- **Current-tree status:** These are not dead as of 2026-05-20. `tests/test_benchmark_helpers.py` imports/tests `production_boozer_parity_probe` and `run_code_parity_probe`; `tests/test_jax_import_smoke.py` executes `single_stage_surface_reprojection_probe.py`; `tests/geo/test_surface_rzfourier_jax.py` executes `surface_rz_geometry_hlo_probe.py`; benchmark/docs surfaces direct users to run-code probes.
- **Change:** Build a caller inventory and classify each script as (a) active parity oracle, (b) active smoke entrypoint, (c) migrated to a smaller test helper, or (d) actually retired. Delete only class (d), and only after replacing or updating tests/docs that reference it.
- **LOC saved:** 0 guaranteed; up to ~1,245 only after migration/retirement is approved.
- **Risk:** Medium until caller migration is complete.
- **Validation gate:** T1 + `tests/test_benchmark_helpers.py` + `tests/test_jax_import_smoke.py` + `tests/geo/test_surface_rzfourier_jax.py` + full caller inventory grep over `src/`, `examples/`, `benchmarks/`, `tests/`, `docs/`, `.github/`, slurm scripts, and repo-local artifacts if present.

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

All required T1 items merged; guaranteed net LOC reduction ≥ 600; probe-script deletion excluded unless T1.10 migration retires a script safely; full T1 suite green; contract checklist re-affirmed; tag `bloat-reduction-T1-complete`.

---

## 6. Tier 2 — Factory Introductions

**Target:** ~3,500 LOC reduction. **Effort:** ~3–4 days. **Risk:** Low-Medium.

Goal: convert repeated templates into data-driven factories. Each item proves the pattern at small scope before Tier 3 attempts larger folds.

### 6.1 — [ ] T2.1: Boozer result-dict factories

- **Files:** `boozersurface_jax.py` result-dict packaging sites are now at `:5068`, `:5093` (early returns), `:5316`, `:5449`, `:5554`, `:5879`, `:6135`, `:6275`, `:6670`, `:6769`, `:6916` (v4's `:5098`-`:6511` list was stale by hundreds of lines — the file's max packaging site is now `:6916`, not `:6511`). Newton/quality reporting duplication sits inside those dicts.
- **Change:** Introduce small result-pack helpers such as `_boozer_traceable_result_core(...)`, `_boozer_ls_result_core(...)`, and `_boozer_exact_result_core(...)`; drive from `_BOOZER_TRACEABLE_RESULT_KEYS` (`:316`, still exact) as constructor SSOT. **v5 NOTE:** `_BOOZER_RESULT_SCHEMAS` is **confirmed ABSENT** from the tree (not merely renamed) — the only result-dict SSOT is `_BOOZER_TRACEABLE_RESULT_KEYS:316`.
- **Historical LOC target:** ~130. **Current LOC banked:** 0; this slice is partial / not LOC-banked until a follow-up reduces source LOC without hiding failure-path diagnostics.
- **Risk:** Medium after the post-v2 committed drift. This item touches user-visible result dict schemas and must be semantically inventoried from the latest committed tree before implementation.
- **Contracts:** Result-dict required/forbidden keys; success-vs-failure `linear_solve_backend` strings; `adjoint_linear_solve_available` flag.
- **Validation gate:** T2 + result-dict schema tests in `test_boozersurface_jax.py`.

### 6.2 — [ ] T2.2: `boozer_radial_field` 16×2 evaluator collapse — **PARTIAL / NOT BANKED (v6 correction)**

- **v7 status:** The column bundle and `_eval_radial_columns` structure exist, but the fold is not complete. `boozer_radial_field.py` still has parallel `_eval_*_from_columns` implementations (`:452-787`) and separate `_eval_*` implementations (`:807-1190`) that call `inverse_fourier_transform_{even,odd}` / `_scalar_at` directly. Residual duplication remains; the ~400 LOC should not be counted as already banked.
- **Files:** `src/simsopt/jax_core/boozer_radial_field.py` (1,193 LOC).
- **Change:** Keep `_eval_X_from_columns` only; add cheap wrapper `_eval_X(state, points) = _eval_X_from_columns(state, _eval_radial_columns(state, points[:, 0]), points)` or parametrize via per-quantity tuple `(cnc_field, sns_field, deriv_factor, radial_kind)`.
- **LOC saved:** ~400.
- **Risk:** Low. `_eval_radial_columns` evaluates 27 mode profiles per call — single-scalar callers pay slightly more; benchmark before/after on a small fixture.
- **Contracts:** `BoozerRadialColumnBundle` field ordering (pytree flattening); `state.stellsym` static branch; `inverse_fourier_transform_{even,odd}` switch.
- **Validation gate:** T2 + benchmark sanity (no regression > 10% on `boozer_radial_field` benchmarks).

### 6.3 — [ ] T2.3: Surface fourier `_from_dofs` / `_from_spec` factory

- **Files:** `surface_fourier_kernels.py` (3,139 LOC; the `_*_from_dofs` kernel wrappers begin ~`:1415`) + `surface_fourier.py` (909 LOC; the `_*_from_spec` wrappers span ~`:102-905`). (v4's `kernels:2200-2799` was stale.)
- **Change:** Build `_make_from_spec_xyz_fourier(kernel)` + `_make_from_dofs_xyz_fourier(spec_to_args, kernel)` factories mirroring existing `_dcoeff_jacobian` at `:2820` (v3 said `:2812-2864`). Each public name becomes one line: `surface_xyz_fourier_gamma_from_spec = _make_from_spec_xyz_fourier(_kernel_gamma)`.
- **LOC saved:** ~550.
- **Risk:** Medium. `__all__` listings + cross-imports must stay byte-identical; docstrings need `__doc__` assignment.
- **Contracts:** Stellsym scatter indices; every public symbol name; tensor conventions (`dgamma_by_dcoeff[i,j,l,k]`).
- **Validation gate:** T2 + `tests/geo/test_surface_fourier_jax.py`.

### 6.4 — [ ] T2.4: Spec dataclass auto-registration helper

- **Files:** `src/simsopt/jax_core/specs.py:28-688` (confirmed **29** spec classes, each a `@dataclass(frozen=True)` + manual `register_dataclass(...)`; file 1,622 LOC; v3 cited `:26-740`).
- **Change:** Define `@register_jax_spec(data_fields=[...], meta_fields=[...])` decorator wrapping `@dataclass(frozen=True)` + `register_dataclass(...)`.
- **LOC saved:** ~140.
- **Risk:** Low. JAX `register_dataclass` treats `meta_fields` as static JIT-cache-key material, so the helper must preserve each explicit data/meta partition exactly and keep metadata hashable/immutable.
- **Contracts:** Field names + data/meta partition (the explicit lists drive what's a traced array vs JIT static).
- **Validation gate:** T2 + `tests/test_jax_import_smoke.py::test_jax_core_specs_are_pytrees` + representative JIT cache-key/static-field regression.

### 6.5 — [ ] T2.5: Batch-axis sharding helper factory

- **Files:** `sharding.py` (725 LOC) — the 3 batch dataclasses targeted here are at `:94` / `:104` / `:114`; their builder / maybe_shard / summary triplets at `:301,:309,:317` · `:466,:490,:514` · `:538,:558,:578`. (NOTE: there is a 4th frozen dataclass, `CoilGroupCollectiveConfig:47`, outside this triplet — leave it out of scope.)
- **Change:** Replace `TrajectoryBatchShardingConfig` / `SeedBatchShardingConfig` / `SurfaceQuadratureShardingConfig` plus their 3 builder / maybe_shard / summary triplets with one parameterized triplet taking `(predicate, axis_name, config_kind)`.
- **LOC saved:** ~170.
- **Risk:** Low. CLAUDE.md confirms sharding policy fields are reporting metadata, not load-bearing for kernel execution.
- **Contracts:** JSON summary key names (`trajectory_sharded`, `field_collective`, etc.) — callers may grep them.
- **Validation gate:** T2.

### 6.6 — [ ] T2.6: 8 `_resolve_*` env/kwarg helpers → 1 generic + table-driven config builder

- **Files:** `backend/runtime.py` (2,500 LOC) — 22 separate `_resolve_*` functions; `_config_from_mode` at `:1466`; `_MODE_POLICY_DEFAULTS` at `:208`. (v3 cited `:1313-1441` / `:1444-1488`, drifted.)
- **Change:** Define `_resolve_kwarg(kwarg, env_name, default, *, parser, source)`; replace 8 ladders with table-driven loop. Have `_MODE_POLICY_DEFAULTS` carry parser refs.
- **LOC saved:** ~100.
- **Risk:** Low. Pure refactor; same truth table.
- **Contracts:** Mode-default precedence; env-value validation; `set_backend(...)` kwarg names.
- **Validation gate:** T2 + `tests/test_backend.py`.

### 6.7 — [ ] T2.7: SciPy adapter unification in `optimizer_jax_reference.py`

- **Files:** `optimizer_jax_reference.py` (569 LOC) — `_scipy_minimize:192`, `target_scipy_minimize_value_and_grad:240`, `_scipy_minimize_value_and_grad:296`, all routing through `_scipy_dispatch_core:138` (closures span `:138-340`; v3 cited `:182-340`).
- **Change:** Collapse `_scipy_minimize`, `_scipy_minimize_value_and_grad`, `target_scipy_minimize_value_and_grad` into one closure-factory parameterized by `(strict_backend_guard, accept_value_and_grad_callable)`.
- **LOC saved:** ~220.
- **Risk:** Low. All 3 route to `_scipy_dispatch_core` already.
- **Contracts:** `_require_native_cpu_reference_backend_for_scipy_adapter` guard on the 2 lanes that have it; `target_scipy_minimize_value_and_grad` keeps no guard; `scipy_call_contract`, `scipy_initial_call`, `scipy_callback_trace` fields stay populated.
- **Validation gate:** T2.

### 6.8 — [ ] T2.8: `LayerDriftTracker` dataclass for `single_stage_init_parity`

- **Files:** `single_stage_init_parity.py` — `compare_same_candidate_objective_replay` now at `:2946` (v4's `:2300-2700` is stale; the tracker block runs roughly `:2968-3260`).
- **Change:** Replace the parallel `max_*` / `first_*` trackers (**≥27 distinct accumulators, ~75 tracker-related vars** — v4's "16" undercounts) with `LayerDriftTracker` instances per family; `.update(summary, *, pair_index, line_search_evaluation)`; `.summary_dict()` returns the same keys.
- **LOC saved:** ~200.
- **Risk:** Low. Internal trackers; external dict shape preserved.
- **Contracts:** `_pre_newton_census_gate_failures` untouched; `parity_bug_census["divergent_layers"]` schema; all `*_summary_dict` payload keys.
- **Validation gate:** T2 + replay `_pre_newton_census_gate_failures` on a pinned fixture and confirm byte-identical failure list.

### 6.9 — [ ] T2.9: Quantity-aware tolerance helper in `validation_ladder_contract.py`

- **Files:** `_tolerance_for` lives in **only one** file — `non_banana_example_cpp_jax_cpu_parity.py:323`. **v5 CORRECTION:** v4's other two sites are wrong — `single_stage_parity_matrix.py` has **no** `_tolerance_for` (the cited `:292-302` does not exist) and `stage2_e2e_comparison.py` uses a **different** mechanism (`optimizer_drift_tolerances:71/74`). So this is a single-file local helper, not a 3-file consolidation; the ~100 LOC estimate below is overstated (realistically ~30–50).
- **Change:** Add a quantity-aware helper that preserves the existing `_tolerance_for(quantity)` semantics, including `event_time_tracing` state-vector tolerances, `qfm_gradient` derivative-heavy tolerances, and float32 objective/gradient branches. Do not collapse to a generic `kind="rtol_atol"` bucket that can apply the wrong tolerance floor.
- **LOC saved:** ~100.
- **Risk:** Low-Medium. Pure helper only if every quantity maps to the same bucket and pair as before.
- **Contracts:** `PARITY_LADDER_TOLERANCES` values frozen; `parity_ladder_tolerances(lane)` API unchanged.
- **Validation gate:** T2 + pre/post snapshot of every migrated quantity's `(bucket, rtol, atol)`.

### Tier 2 exit gate

All 9 items merged; net LOC reduction target recomputed from live item estimates; T2 suite green; contract checklist re-affirmed; `_pre_newton_census_gate_failures` replay byte-identical; tag `bloat-reduction-T2-complete`.

---

## 7. Tier 3 — Structural Consolidations

**Target:** ~4,000–6,000 LOC reduction. **Effort:** ~1–2 weeks. **Risk:** Medium.

Goal: large-scale folds across multiple files. Higher risk because they touch hot code paths. Each item gets its own pre-flight design doc commit before implementation.

### 7.1 — [ ] T3.1: GPMO 5-way fold (largest single win)

- **Files:** `pm_optimization.py` (3,400 LOC) — 6 solve drivers `gpmo_baseline_solve:801`, `gpmo_arbvec_solve:1094`, `gpmo_arbvec_solve_bucketed:1239`, `gpmo_arbvec_backtracking_solve:1877`, `gpmo_multi_solve:2339`, `gpmo_backtracking_solve:2751` (each with its own `*_step`) + `solve/permanent_magnet_optimization_jax.py` (921 LOC, 6 `*Result` at `:143-279`). (v3 cited `:753-2906` / `pm_workflow.py:746-1142` / `:371-637`, all drifted.)
- **Change:** One `_gpmo_solve(step_fn, initial_state, spec, history_spec, K, record_every)` driver + one `_gpmo_recording_scan_body(step_fn, history_spec)`. Each variant supplies `step_fn`, `history_spec`, `*Spec` dataclass.
- **LOC saved:** ~1,500.
- **Risk:** Medium. JIT cache keys must include `step_fn` / `history_spec`; tie-break order and `_UNAVAILABLE_CANDIDATE_COST` sentinel must hold; `selected_groups`, `num_nonzeros_history`, `removed_pair_count_history`, `done_history` shapes preserved.
- **Contracts:** GPMO C++ tie-break order (`+` before `-`); `_record_rows = np.arange(record_every - 1, K, record_every)` off-by-one; live-loop history capacity validators.
- **Validation gate:** T3 + PM acceptance fixtures + GPMO numerical equivalence on baseline + arbvec + arbvec_bucketed + arbvec_backtracking + multi + backtracking variants.

### 7.2 — [ ] T3.2: `SpecBackedBiotSavartJAX` ↔ `BiotSavartJAX` mixin extraction

- **v5 status (PARTIALLY DONE):** the field-evaluation duplication is **already folded** — a `_BiotSavartFieldEvaluationMixin` (`field/biotsavart_jax_backend.py:489-515`, B/A/dA_by_dX/d2A/dB_by_dX) is now inherited by **both** `SpecBackedBiotSavartJAX` (`:642`) and `BiotSavartJAX` (`:1299`). Re-scope T3.2 down from ~250 LOC.
- **Files (remaining dup):** `src/simsopt/field/biotsavart_jax_backend.py` (2,328 LOC) — `set_points`/`get_points*` (`:751-784` vs `:1747-1790`) still unshared; `coil_cotangents_to_dofs_gradient` has **diverged** (Spec class `:786-802` now delegates to a jitted helper; the live class starts at `:2147` and still uses the coil-dofs idiom) — reconcile before hoisting.
- **Change:** Hoist the remaining point-management methods into a `_BiotSavartPointsMixin`; reconcile the two `coil_cotangents_to_dofs_gradient` bodies. Live class keeps `_introspect_coils`, token plumbing; Spec class keeps the spec-driver constructor.
- **LOC saved:** ~250.
- **Risk:** Medium. Token plumbing must stay live-class only; `_coils` property; `_uses_uniform_curve_xyz_fourier_fastpath`; `coil_dof_extraction_spec()` callable on both.
- **Validation gate:** T3 + full Stage 2 parity matrix.

### 7.3 — [ ] T3.3: Profile machinery → sibling diagnostic modules

- **v4/v5/v7 status (PARTIALLY DONE / RELOCATED):** The `surfaceobjectives_jax.py` half already moved — `diagnose_traceable_objective_runtime` and `make_traceable_objective_profile_suite` are no longer in `surfaceobjectives_jax.py` (now 3,110 LOC in the current checkout); they live in the **now-tracked** `surfaceobjectives_traceable_jax.py` (`diagnose…:2599`, `make_…_profile_suite:3493`), but are NOT yet isolated to a dedicated `_diagnostics` sibling. The v3 refs `surfaceobjectives_jax.py:5382-5650 / 5670-5959 / 6270-6284` are **dead**.
- **Files (remaining):**
  - `biotsavart_jax_backend.py` profile helpers (~`:215-303`, `:2125-2212`) → new `biotsavart_jax_profile.py` (still pending; re-derive refs against HEAD).
  - Finish the `surfaceobjectives` split: extract the diagnostics now living in `surfaceobjectives_traceable_jax.py` into a `_diagnostics` sibling, if that isolation is still wanted.
- **LOC saved:** ~620 moved (net same, but hot files shrink dramatically; logical separation improves audit-ability).
- **Risk:** Medium. Maintain redirect imports in original modules for backward compat.
- **Contracts:** External test caller signatures (`tests/integration/test_stage2_jax.py:1779`).
- **Validation gate:** T3 + profile-related tests.

### 7.4 — [ ] T3.4: `_build_runtime_linear_solve_callbacks` 4-branch refactor

- **Files:** `boozersurface_jax.py` — `_build_runtime_linear_solve_callbacks` now at `:3938` (file is 7,090 LOC; v3 cited `:3730-4063`). Re-derive the inner branch ranges from `:3938`.
- **Change:** Extract `_pack_dense_plu_callbacks(matrix, lu_piv, status_fn, backend_label, residency)` shared between the `shared_lu_piv` (`:4001`, reports `dense-plu-shared` at `:4078`) and scipy-PLU (`:4086`, reports `dense-plu` at `:4153`) branches. Extract `_pack_operator_callbacks(operator_dict, system_solver, system_solver_with_status, stab)` shared between the hessian-operator (`:4163`) and exact_jacobian (`:4225`) branches. (v4's `:3793/:3876/:3954/:4016` are all stale.)
- **LOC saved:** ~120.
- **Risk:** Medium. Must preserve byte-identity of shared `(lu, piv)` factor reuse; `_with_nan_status` wrap-once semantics; `apply_forward` / `apply_transpose` wired to device-resident `H_dev` directly. Re-derive the range from the latest committed tree before implementation.
- **Contracts:** `linear_solve_backend` reporting strings (`dense-plu-shared`, `dense-plu`, `operator`); `linear_solve_factors` payload shape (tuple of `jnp` arrays).
- **Validation gate:** T3 + `test_boozersurface_jax_private.py` private optimizer lane + parity-mode CPU test.

### 7.5 — [ ] T3.5: LaneArtifact-builder coverage extension

- **Files:** `non_banana_example_parity_fixtures.py` (7,730 LOC) — `_build_cpu_lane:368`, `_build_jax_lane:450`, `_build_scalar_lane:544` (v4's `:354-535` is stale); remaining inline `LaneArtifact(...)` sites at `:806/:843/:991/:1025`. `_METADATA_RAW_ARRAY_KEYS_BY_FIXTURE_KIND` lives in the driver `non_banana_example_cpp_jax_cpu_parity.py:474` (the "promote from driver" target).
- **Change:** Extend `_build_cpu_lane` / `_build_jax_lane` / `_build_scalar_lane` to accept arbitrary `raw_arrays` + `fixture_kind`; compute the 7 hash fields from raw arrays via `_METADATA_RAW_ARRAY_KEYS_BY_FIXTURE_KIND` (promote from driver file).
- **LOC saved:** ~1,650.
- **Risk:** Low. `LaneArtifact` output byte-identical; all hash kwargs already derived from raw arrays.
- **Contracts:** Fixture IDs, classifications, FixtureBuild contract, `_pre_newton_census_gate_failures` (untouched).
- **Validation gate:** T3 + `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py` + per-fixture `compare-array` row byte-identity.

### 7.6 — [ ] T3.6: Table-driven `_*_comparisons` driver

- **Files:** `non_banana_example_cpp_jax_cpu_parity.py` (3,255 LOC) — the per-fixture comparison block now spans ~`:1008-1900` (v4's `:951-1843` is stale): `_supported_comparisons:1008`, `_surface_scalar_comparisons:1130`, `_strain_comparisons:1193`, `_coil_force_energy_comparisons:1241`, `_qfm_comparisons:1301`, `_pm_comparisons:1369`, `_pm_relax_and_split_comparisons:1483`, `_wireframe_comparisons:1624`, `_wireframe_gsco_comparisons:1725`, `_boozer_fixed_state_comparisons:1883`.
- **Change:** Define `COMPARISON_PLAN_BY_FIXTURE_KIND: Mapping[str, Sequence[ComparisonSpec]]`; single `_apply_comparison_plan(plan, cpu, jax_lane)` driver.
- **LOC saved:** ~700.
- **Risk:** Low. Dict-equivalent output.
- **Contracts:** `_tolerance_for` lookup unchanged; `PARITY_LADDER_TOLERANCES`; bucket assignments.
- **Validation gate:** T3 + parity matrix row byte-identity.

### 7.7 — [ ] T3.7: Generic adaptive-step trace driver factory

- **Files:** `jax_core/tracing.py` (**4,287 LOC**; v4 said 4,299) — drivers `trace_fieldline:1217`, `trace_guiding_center:1879`, `trace_guiding_center_boozer:2982`, `trace_fullorbit:3758` (plus batched variants); only `_run_dopri5_4state:2407` exists today (v4's `:1883/:2994/:3770/:2419` drifted ~10-12 lines). The bounded-scan helper is `_scan_adaptive_steps:367`.
- **Change:** Extend `_run_dopri5_4state` into general `_run_dopri5(rhs, y0, *, state_dim, traj_width, phi_hits_width, phi_recording, axis_invalid_guard, pre_step_resolver, ...)`. Inline drivers as thin shims.
- **LOC saved:** ~1,100.
- **Risk:** Medium-high. Drivers diverge: `trace_fieldline` has no `_boozer_axis_invalid`; `trace_guiding_center_boozer` resolves field state in Python pre-step.
- **Contracts:** Stopping criteria (`_stopping_criterion_should_stop` + `is_boozer_state` switch); `phi_init` continuous-branch contract; exit status codes (`_BOOZER_AXIS_STATUS`).
- **Validation gate:** T3 + `tests/field/test_tracing_jax_item16.py` + `tests/field/test_tracing_jax_item16_extended.py` + `tests/jax_core/test_tracing_jax_*.py` + tracing benchmark sanity.

### 7.8 — [ ] T3.8: Lazy-cache helper for `_make_traceable_*` family

- **Files:** `surfaceobjectives_traceable_jax.py` (**now-tracked, 3,543 LOC**) — the `_make_traceable_*` / `_ensure_traceable_runtime_*` family **relocated here** (`_ensure_traceable_runtime_*` family `:1565-1733`: reporting_metrics `:1565`, public_boundaries `:1625`, optimizer_compiled_bundle `:1660`, optimizer_value_and_grad `:1676`, seeded_value_and_grad `:1690`, host_wrappers `:1733`). (v3 cited `surfaceobjectives_jax.py:4357-4660`, now **dead**.)
- **Change:** Define a `_lazy(entry, key, builder)` helper or `@dataclass` cache record + `lazy_property`. ~250 LOC of ensure/make pairs collapse.
- **LOC saved:** ~80.
- **Risk:** Medium. Cache key contract load-bearing (CLAUDE.md "Traceable runtime bundle cache contract"). Must round-trip through `_traceable_runtime_cache_key` and confirm hash stability across solve generations.
- **Contracts:** Deterministic signatures of solved baseline state / objective kwargs / coil set spec; `is`-comparison for success-filter callables; no `id()` in any cache key.
- **Validation gate:** T3 + `test_single_stage_jax_cpu_reference.py::TestTraceableRuntimeBundleCache`.

### Tier 3 exit gate

All 8 items merged; net LOC reduction ≥ 4,500 (75% of upper bound 6,000 acceptable due to medium risk); full T3 suite green; parity matrix byte-identical; contract checklist re-affirmed; tag `bloat-reduction-T3-complete`.

---

## 8. Tier 4 — Decision Points

These items deliver significant LOC reduction (potentially 1,640+ LOC) but require contract clarifications the audit alone can't resolve.

### 8.1 — [ ] T4.1: Is `lbfgs-trace` host engine still needed?

- **Current-tree status:** Alive until proven otherwise. It is reachable through `reference_minimize(method="lbfgs-trace")`, the production single-stage example CLI, `benchmarks/single_stage_init_parity.py`, and tests that pin failure-callback / progress behavior.
- **Decision needed:** Are the failure-callback / invalid-step-log diagnostics from this engine still consumed by any active workflow (lab notebooks, autoresearch runs, perlmutter slurm jobs)? Or is on-device `invalid_step_log` + SciPy `lbfgs` sufficient with an explicit migration?
- **v4:** Plan A (HANDOFF.md, 2026-05-29) flipped the single-stage GPU *default* `ondevice`→`scipy-jax` but explicitly LEFT this host/reference engine in place ("Do not relitigate"), so **T4.1 remains OPEN**. `optimizer_host_lbfgs.py` is now **1,628 LOC** (grew since v3) — re-derive the deletion estimate.
- **If dead:** Delete `optimizer_host_lbfgs.py` + the `lbfgs-trace` method only after API-evolution gate, CLI/doc migration, and replacement of `record_optimizer_state_trace` / `invalid_step_log` diagnostics. ~1,628 LOC reduction (v3 said ~1,300).
- **If alive:** Keep as-is; consider deduplicating with `_line_search.py` JAX strong-Wolfe (~500 LOC duplicate).

### 8.2 — [x] T4.2: Reconcile `scipy-jax` and `scipy-jax-fullgraph` outer backends with CLAUDE.md spec — **DECISION MADE (v4): KEEP + document**

- **v7 status:** Plan A (HANDOFF.md, 2026-05-29) answered the open question, but v4/v5 misstated the default split. Live defaults route the JAX optimizer lane to `scipy-jax` on both CPU and CUDA (`tests/test_cli_defaults.py:36-41`, `:51-56`, `:133-145`; single-stage resolver `single_stage_banana_example.py:8346-8355`; Stage 2 resolver `banana_coil_solver.py:803-809`). `scipy-jax-fullgraph` remains an explicit stress/parity lane. This is the "If alive" branch — the only remaining work is the CLAUDE.md / user-doc update; **no removal**. Both lanes are live in `optimizer_jax.py` (mapping `:227-228`, dispatch `:703`).
- **Current-tree status:** Alive user-facing surfaces. `scipy-jax` and `scipy-jax-fullgraph` are exposed by Stage 2 and single-stage example CLIs, mapped in integration tests, and routed in benchmark helper tests.
- **Decision needed:** None on removal. CLAUDE.md and user docs need updating so they distinguish the default `scipy-jax` lane from the explicit `scipy-jax-fullgraph` stress/parity lane.
- **If dead:** ~150 LOC reduction only after API removal is complete.
- **If alive:** Update CLAUDE.md to document them.

### 8.3 — [ ] T4.3: Should `qfm_solver._bfgs_minimize` reuse `optimizer_jax_private._bfgs._minimize_bfgs_private`?

- **Surface:** `qfm_solver.py:497-634` (~138 LOC home-grown BFGS with Armijo line search; v4 said `:431-614` ~190 LOC). The private BFGS uses strong Wolfe; iteration counts will differ.
- **Decision needed:** Acceptable to replace Armijo-only inner BFGS with SciPy-style BFGS Armijo+curvature conditions and re-tune QFM `max_iter` defaults? Or is the Armijo behavior load-bearing for QFM convergence empirics?
- **If reuse approved:** ~138 LOC reduction; need QFM acceptance re-run covering natural-equality KKT success, feasible-nonstationary rejection, branch-stability invariants, host-SLSQP diagnostics, and infeasible/warm-start perturbation cases.
- **If reject:** Keep duplicate; document why in CLAUDE.md.

### 8.4 — [ ] T4.4: Rename or quarantine `minimize_qfm_exact_constraints_SLSQP` alias — **NOT obsolete (v5 correction); re-scoped to surface wrappers**

- **v5 status (CORRECTS v4's "likely obsolete"):** `SLSQP`/`slsqp` has 0 occurrences in `qfm_solver.py` (true — the solver-level exact path is augmented-Lagrangian / exact-KKT: `QfmAugmentedLagrangianInfo:61`, `QfmExactKktInfo:79`), but the **public alias still exists** in the surface wrappers — `qfmsurface_jax.py:281` and `qfmsurface.py:147` — with live test callers. So the item is **not** moot; it just moved up a layer. The rename-or-quarantine decision stands, now scoped to those two wrappers.
- **Current-tree status:** The docstring already says this is a compatibility alias for the JAX augmented-Lagrangian exact path.
- **Decision needed:** Keep compatibility alias as documented, or add a new clearer public name and deprecate the old alias through the API-evolution gate.
- **No LOC change either way; clarity-only.**

### 8.5 — [ ] T4.5: Surviving tautological tests

- **Surface:** Lane 7 listed 12–15 surviving tautologies. **v6 re-derivation:** `tests/field/test_trace_boozer_analytic_jax.py:139` + `:207` remain identity-routing assertions; `tests/integration/test_single_stage_jax_cpu_reference.py` is now 8,926 LOC with `weight_inv_modB` option propagation at `:2406`/`:2413-2414`, same-helper routing text at `:2706-2720`, cached-transform identity at `:2828-2866`, and adjoint/VJP health-only checks at `:2928-3031`. The related flux fast-path identity assertions are `tests/objectives/test_fluxobjective_jax_parity.py:463-464`; the large-point-cloud self-consistency test begins at `:681` and is not the fast-path identity check. NOTE: several A11 test-oracle tautologies cited here have since been **fixed** (A11-H2/H3/H4 — see §4.7); only the `test_curve_objectives_jax.py` cluster (A11-H1) and these routing/health tests remain.
- **Decision needed (per-test):** rewrite with independent oracle, or accept-and-document as Tier-4 routing tests per `tests/REVIEWER_ORACLE_LINT.md`?
- **LOC neutral or slightly positive (~40 from mock-invariant consolidation).**

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
4. **Tier 4 alongside Tier 1.** Open the Tier-4 questions early. The `lbfgs-trace` decision (T4.1) gates ~1,300 LOC and should be resolved before Tier 3 begins.
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

These need user answers before starting:

1. ☐ **Branch strategy.** Should this go on `gpu-purity-stage2-20260405` directly, or branch off as `bloat-reduction-20260520`? Recommendation: new branch.
2. ☐ **Time horizon.** Is this a 2-week sprint, a background project, or "as the spirit moves you"? Affects whether to open all Tier-4 decisions now or just T4.1 / T4.2.
3. ☐ **GPU proof venue.** Use the existing self-hosted GitHub CUDA runner, Perlmutter, or another CUDA host for Section 9.5 tier-exit proof?
4. ☐ **Tier 4 decisions.** Answer T4.1, T4.2, T4.3, T4.4, T4.5 now, or block on them until Tier 1 results land?
5. ☐ **CLAUDE.md updates.** This refactor will require CLAUDE.md edits in 3 places (the `scipy-jax` backend documentation, the new `_core/state_tokens.py` location, the SciPy 1.17.1-compatible-port disclosure). Draft those edits in this same effort, or separately?
6. ☐ **Memory/project note.** Record an OpenMemory/project note on completion if T1.4 changes the curve↔jax_core import-cycle story. Do not edit `MEMORY.md` directly unless explicitly requested.

---

## 13. Estimated Totals

| Tier | LOC reduction (est.) | Items | Effort | Risk |
|------|---------------------:|------:|--------|------|
| T1 — Mechanical Wins | ~630-670 guaranteed (v3 ~950; T1.1 cut ~620→~300 as file already shrank to 363 LOC) (+ up to ~1,245 probe-script decision-gated) | 11 (T1.11 verified) | 2 days | Low-Med |
| T2 — Factory Introductions | ~3,500 (v5 subtracted T2.2 incorrectly; v6 counts its ~400 LOC as remaining until the wrapper fold lands) | 9 (T2.2 partial/open) | 3–4 days | Low-Med |
| T3 — Structural Consolidations | ~4,000–5,500 (T3.2 field-eval dup **already folded** via `_BiotSavartFieldEvaluationMixin`, re-scope −~250; T3.3 / T3.8 partially done via the `surfaceobjectives_traceable_jax.py` split) | 8 | 1–2 weeks | Med |
| T4 — Decision Points | T4.2 resolved (doc-only); T4.1 ~1,628 + T4.3 ~138 + T4.4 (alias quarantine, re-scoped to the qfm **surface** wrappers — **not** obsolete) still decision-gated | 5 (T4.2 resolved) | varies | Decision-bound |
| **Aggregate** | **STALE until salvage splitting.** Historical estimate was ~7,900-9,800 guaranteed candidate LOC remaining pending T2.2 re-estimate, plus decision-gated deletions. The drift-checkpoint ledger is not net-shortening: effective `src/` is `+45` once untracked source helpers are included, and `src/`+`tests/`+`docs/` is `+2166`. Recalculate aggregate remaining LOC only after banked-shrink slices are isolated and foundation/not-banked slices are classified. | **33 items (T2.2 LOC-banking follow-up open; T3.2 partial; T4.2 resolved)** | **~3 weeks historical estimate; replan after salvage split** | **manageable only with the v8 drift gate** |

---

## 14. Appendix A — How to Use This Document

### As a checklist during execution

1. Work tiers sequentially: T1 → T2 → T3. Do not skip ahead.
2. Within a tier, items are mostly independent but ordered by risk (lowest first).
3. Each `- [ ]` item normally maps to one commit. v8 exception: the 2026-06-01 dirty-tree checkmarks record implemented/validated slices, not committed slices, until salvage splitting produces isolated commits.
4. After each item: run the per-item validation gate; do not proceed if it fails.
5. At tier exit: run the tier exit gate and the contract checklist in Section 4.1; tag the commit.

### As a status report

The current state of execution is encoded in the checkboxes, but under the v8 drift gate the checkboxes are implementation/validation status, not commit status. A `git diff` of this file against the merge-base or base branch shows progress only after the dirty tree is split into scoped commits.

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
| 2 | Optimizer JAX (`optimizer_jax`, `optimizer_jax_private/*`, `optimizer_jax_reference`, `optimizer_host_lbfgs`) | 800–1,000 (+1,300 if T4.1 dead) |
| 3 | jax_core kernels (`tracing`, `surface_fourier_kernels`, `biotsavart`, `specs`, `sharding`, etc.) | 2,600 |
| 4 | Field/backend layer (`backend/runtime`, `biotsavart_jax_backend`, `field/*`) | ~700 |
| 5 | PM / QFM / wireframe (`pm_optimization`, `pm_workflow`, `qfm_solver`, `boozer_radial_field`) | 3,100 |
| 6 | Benchmarks/parity infrastructure (`non_banana_*`, `single_stage_*`, `validation_ladder_*`) | 2,300–3,000 (+ ~1,245 probe scripts only if caller migration retires them) |
| 7 | JAX tests (`test_boozersurface_jax`, `test_single_stage_jax_cpu_reference`, etc.) | ~500 |
| 8 | Cross-cutting duplication (sibling-variant files, re-export shims, dtype helpers, state tokens) | ~850 |

Historical aggregate candidate estimate after lane overlap consolidation: **~8,300–9,800 LOC**, plus decision-gated optional deletions. Under the v8 drift gate, this is not a current banked-reduction claim; re-measure after salvage splitting.

Full audit transcripts available in the orchestrator session log (2026-05-20). The original bloat plan was generated from 8 bloat-reduction lanes; the separate code-smell artifact contains 11 per-lane reports + `SUMMARY.md` + 24 verification-round logs. In the `simsopt-jax-shared-jax` checkout reviewed at `b267b0d95`, `.artifacts/` is not present; the historical artifact tree was found at `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/code_smell_review_2026-05-20/`. Those logs are historical records with 2026-05-30 line-ref annotations originally refreshed against HEAD `21c3d517d`; use `SUMMARY.md` plus `verification/CORRECTIONS_round5.md` inside that artifact tree as the final status for retracted items, and re-grep current HEAD/dirty files before executing.

---

*End of plan v8. Crucible-reviewed; v5 basis + §4.1 / §5–§8 line refs were re-derived against clean HEAD `21c3d517d` on 2026-05-30, corrected against live HEAD `2bcaeff28` and a dirty worktree by a 3-agent doc-review pass, refreshed against clean HEAD `b267b0d95` on 2026-05-31, then updated with the 2026-06-01 `8b94c2bbd` execution-drift gate. Re-grep before executing — the repo is under active concurrent commits.*
