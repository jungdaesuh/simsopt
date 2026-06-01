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
- [ ] `_pre_newton_census_gate_failures` at `single_stage_init_parity.py:3379` (def; used `:3456`/`:3466`; same-candidate replay gate at `:3427`; `SystemExit` wiring `:4496`/`:4803`). Release blocker. (v4 cited def `:2877`/used `:2954`-`:2964`; the SUMMARY's `:2198`, the v3 `:3275-3279`, v6's `:2854` / `:2931` / `:2941` / `:4150-4153`, and the pre-T2.8 `:3344` / `:3421` / `:3431` / `:4515` / `:4822` refs are all stale.)
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
  - **T4.4** — **NOT obsolete.** `minimize_qfm_exact_constraints_SLSQP` has 0 occurrences in `qfm_solver.py` (true; the exact path is augmented-Lagrangian), but the public alias **still exists** in the surface wrappers `qfmsurface_jax.py:281` and `qfmsurface.py:147`, with live test callers. The clarity/quarantine decision is still open — just re-scoped to the surface wrappers.
  - **T2.8** — "16 trackers" undercounted the pre-helper inventory. The 2026-06-01 core helper now owns the two layer-decomposition drift families; remaining tracker cleanup should re-grep from `LayerDriftTracker` at `:3009` and `compare_same_candidate_objective_replay` at `:3471`, not v3's `:2300-2700`.
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

- **Current status:** the direct `boozer_radial_field.py` `_eval_*` Fourier/scalar formula duplication is folded. Fourier direct evaluators now route through the canonical `_eval_*_from_columns` family via typed direct-evaluator factories (`src/simsopt/jax_core/boozer_radial_field.py:1032`, `:1050`), and scalar direct evaluators use `_direct_scalar_evaluator(...)` (`:1067`, assignments at `:1138`) to preserve the original scalar spline path. `_eval_radial_columns` remains the public-wrapper points-cycle cache path (`:405`).
- **Hot-path guard:** the first cheap full-bundle wrapper version regressed the Boozer RHS benchmark, so the landed design adds RHS-specific radial columns (`_eval_radial_rhs_columns`, `:521`) and tracing dispatch through `_RADIAL_RHS_COLUMN_EVALUATORS` (`src/simsopt/jax_core/tracing.py:2711`) instead of calling each direct evaluator separately.
- **Bloat accounting:** do **not** bank the old `~400 LOC` estimate. The formula-dedup slice was not LOC-banked because benchmark-preserving subset builders offset the direct-formula deletion (`boozer_radial_field.py` diff was `262 insertions / 264 deletions`; `tracing.py` added the RHS dispatch path). The 2026-06-01 direct-wrapper follow-up banks 35 source LOC in `boozer_radial_field.py` (`99 insertions / 134 deletions`; 1,191 -> 1,156 LOC), and the later scalar-helper follow-up banks 12 more source LOC (`26 insertions / 38 deletions`) by factoring only scalar direct-evaluator ceremony into `_direct_scalar_evaluator(...)`. The full T2.2 estimate remains unbanked.

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
- **Validation evidence:** focused helper/trajectory summary proof passed (`4 passed`); forced CPU surface/seed sharding subprocess cases passed (`6 passed`); forced CPU points-coils subprocess cases passed (`2 passed`); scoped `ruff check`, `ruff format --check`, `py_compile`, and `git diff --check` passed. `mypy` remains blocked in `.conda/jax` with `No module named mypy`. This is CPU forced-device sharding proof, not CUDA/MPS proof.

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

All required T1 items merged; guaranteed net LOC reduction ≥ 600; probe-script deletion excluded unless T1.10 migration retires a script safely; full T1 suite green; contract checklist re-affirmed; tag `bloat-reduction-T1-complete`.

---

## 6. Tier 2 — Factory Introductions

**Target:** ~3,500 LOC reduction. **Effort:** ~3–4 days. **Risk:** Low-Medium.

Goal: convert repeated templates into data-driven factories. Each item proves the pattern at small scope before Tier 3 attempts larger folds.

### 6.1 — [ ] T2.1: Boozer result-dict factories — **REPORTING LOC-BANKED / FULL TARGET OPEN (2026-06-01)**

- **Files:** `boozersurface_jax.py` result-dict packaging sites after the 2026-06-01 follow-up are now at `:5520`, `:5652`, `:5756`, `:6057`, `:6222`, `:6312`, `:6430`, `:6530`, `:6598`, `:6804`, `:6900`, and `:7047`; schema/factory helpers live at `:533`, `:569`, `:593`, `:608`, `:663`, and the LS-Newton reporting packer now lives at `:3341`. The remaining solve-quality fields stay beside the owning solve paths.
- **Change:** Introduce small result-pack helpers such as `_boozer_traceable_result_core(...)`, `_boozer_public_result_core(...)`, `_boozer_public_linearized_result_core(...)`, `_boozer_ls_newton_result_core(...)`, and `_boozer_exact_newton_result_core(...)`; keep `_BOOZER_TRACEABLE_RESULT_KEYS` (`:312`, still exact) as the traceable schema SSOT. **v5 NOTE:** `_BOOZER_RESULT_SCHEMAS` is **confirmed ABSENT** from the tree (not merely renamed) — the only traceable result-dict schema SSOT is `_BOOZER_TRACEABLE_RESULT_KEYS:312`.
- **Historical LOC target:** ~130. **Current LOC banked:** 37 from the LS-Newton reporting helper follow-up (`src/simsopt/geo/boozersurface_jax.py` `31 insertions / 68 deletions`). Do not bank the full old target: solve-quality overrides and failure-path diagnostics remain intentionally local.
- **Risk:** Medium after the post-v2 committed drift. This item touches user-visible result dict schemas and must be semantically inventoried from the latest committed tree before implementation.
- **Contracts:** Result-dict required/forbidden keys; success-vs-failure `linear_solve_backend` strings; `adjoint_linear_solve_available` flag.
- **Validation gate:** T2 + result-dict schema tests in `test_boozersurface_jax.py`.
- **2026-06-01 partial slice:** Introduced schema-core helpers `_boozer_traceable_result_core(...)`, `_boozer_public_result_core(...)`, and `_boozer_public_linearized_result_core(...)`, then added `test_boozer_result_core_helpers_match_schema_sources` so the helpers stay tied to `_BOOZER_TRACEABLE_RESULT_KEYS`, `_BOOZER_SOLVER_RESULT_CORE_KEYS`, `_BOOZER_RUNTIME_RESULT_KEYS`, and `_BOOZER_LINEARIZED_RESULT_KEYS`.
- **2026-06-01 follow-up:** Added keyword-only public LS-Newton and exact-Newton envelope factories (`_boozer_ls_newton_result_core(...)`, `_boozer_exact_newton_result_core(...)`) and extended the same helper test to cover their key sets, fixed `"ls"`/`"exact"` type invariants, linearization-kind invariants, and forbidden-key contracts.
- **2026-06-01 LOC-banking follow-up:** Added `_ls_newton_reporting_fields(...)` for the repeated Hessian/Newton-polish reporting keys and reused it at the traceable LS success path plus public LS failure/success paths (`:5774`, `:6334`, `:6456`). The helper and public-result tests now tie the packed key set and distinct value forwarding to `_BOOZER_HESSIAN_REPORTING_RESULT_KEYS`.
- **Design-it-twice gate:** Option A, a generic record-builder keyed by record mode, was rejected because it would hide solve-specific payload fields and make exact/LS failure paths harder to audit. Option B, narrow schema-core helpers, was selected for the first slice because only duplicated core fields move while residuals, callbacks, dense factors, reporting fields, and failure metadata remain visible at each solve site. Option C, named LS-Newton/exact-Newton envelope factories, was selected for the follow-up after rejecting positional factories; the landed helpers are keyword-only so field mapping stays auditable at every call site.
- **Information-hiding test:** The schema constants remain the expected-key SSOT, while the helpers manually pack the matching values. A future public or traceable core key-set change still needs the schema constant and helper implementation updated together; the helper test fails if those drift or if a helper admits forbidden traceable/public keys. The envelope factories hide only the repeated public record envelope, and `_ls_newton_reporting_fields(...)` hides only repeated `result.get(...)` reporting extraction; solve-quality overrides, factorization backend strings, callbacks, and failure-path control flow remain local to the owning solve site.
- **Remaining target:** T2.1 is now LOC-banked but not fully closed against the old `~130` estimate. Further folds must still reduce source LOC and keep failure-path diagnostics readable.
- **Validation evidence:** focused schema/result selectors passed (`2 passed`, `13 passed`, and `20 passed, 2 skipped, 457 deselected`); `ruff check`, `ruff format --check`, `py_compile`, guardrail scan, and `git diff --check` passed for the touched source/test/doc files. `mypy` is now installed/runnable, but source-only `mypy src/simsopt/geo/boozersurface_jax.py` remains blocked by pre-existing errors at the runtime-state callable annotations, quadpoint signatures, local `guarded` redefinition, and grouped-field call arity.

### 6.2 — [ ] T2.2: `boozer_radial_field` 16×2 evaluator collapse — **FORMULA DEDUPED + DIRECT/SCALAR WRAPPERS LOC-BANKED / FULL TARGET OPEN (2026-06-01)**

- **2026-06-01 status:** The duplicated direct Fourier formulas are folded. `_eval_modB` / derivative siblings route through `_eval_*_from_columns` via direct or typed subset-column wrappers (`src/simsopt/jax_core/boozer_radial_field.py:545`, `:559`, `:1032`), and scalar direct evaluators use `_direct_scalar_evaluator(...)` (`:1067`, assignments at `:1138`) to keep the original scalar spline path. Follow-up slices now remove the repeated private Fourier/value direct-evaluator ceremony and scalar direct-evaluator ceremony without changing formula ownership or RHS column reuse.
- **Files:** `src/simsopt/jax_core/boozer_radial_field.py`; `src/simsopt/jax_core/tracing.py`; active tests in `tests/field/test_trace_boozer_analytic_jax.py`.
- **Change landed:** canonical formula ownership is now the `_eval_X_from_columns` family. Direct evaluators construct only the radial columns they need, and radial Boozer guiding-centre RHS calls evaluate one RHS column bundle per point rather than re-evaluating each field scalar separately. The direct-wrapper follow-up uses `_direct_radial_evaluator(...)` / `_direct_modB_value_evaluator(...)` to build the imported private direct evaluator objects and preserves `__name__`, `__qualname__`, and `__module__`. The scalar-helper follow-up uses `_direct_scalar_evaluator(...)` with typed profile selectors to build the seven scalar direct evaluator objects while preserving their scalar spline reads and metadata.
- **LOC saved:** 47 banked across the two T2.2 LOC follow-ups: 35 for the direct-wrapper follow-up (`99 insertions / 134 deletions`; `src/simsopt/jax_core/boozer_radial_field.py` 1,191 -> 1,156 LOC) plus 12 for the scalar-helper follow-up (`26 insertions / 38 deletions`). The old `~400` estimate is still not banked because the formula-dedup slice was effectively flat once benchmark-preserving scaffolding was included.
- **Risk:** Low-to-medium. The first simple full-bundle wrapper was simpler but regressed the RHS/direct microbenchmarks; the landed subset-column design preserves the benchmark gate at the cost of extra helper scaffolding.
- **Contracts:** `BoozerRadialColumnBundle` field ordering (pytree flattening); `state.stellsym` static branch; `inverse_fourier_transform_{even,odd}` switch.
- **Validation evidence:** `tests/field/test_trace_boozer_analytic_jax.py` passed (`28 passed`), including `stellsym=True/False` direct-vs-column routing coverage with distinct optional-mode profiles; focused routing/cache tests passed (`2 passed, 2 skipped, 45 deselected`); `tests/field/test_boozermagneticfield_jax_item33.py` remains skipped in this env (`21 skipped`); the direct-wrapper metadata preservation script passed for 19 generated private evaluators and the scalar-helper metadata/pickle probe passed for the 7 generated scalar evaluators; `ruff check`, `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/boozer_radial_field.py`, `pip check`, and `git diff --check` passed across the T2.2 follow-ups.
- **Benchmark evidence:** saved pre-change baseline was `direct_modB 0.000578229`, `direct_dmodBds 0.001090641`, `direct_G 0.000272629`, `rhs_vacuum 0.003945453`. Direct-wrapper final medians were `direct_modB 0.000612696`, `direct_dmodBds 0.001073811`, `direct_G 0.000250935`, `rhs_vacuum 0.003285109`. The scalar-helper follow-up reran the same `stellsym=True` synthetic non-JIT gate against the documented final medians and passed: `direct_modB 0.000637098`, `direct_dmodBds 0.001141614`, `direct_G 0.000267225`, `rhs_vacuum 0.003449284`, all below the +10% limit. Delta review reproduced the gate in the current checkout with 25-trial medians `direct_modB 0.000594125`, `direct_dmodBds 0.001056167`, `direct_G 0.000243000`, `rhs_vacuum 0.003481042`, also below the same limits.
- **Review clarification:** The direct-vs-column tests prove routing to the column SSOT, not an independent formula oracle; formula correctness remains covered by wrapper parity, closed-form analytic, and benchmark evidence.
- **Remaining LOC-banking follow-up:** Further T2.2 savings require a larger formula/subset-family redesign that remains readable and benchmark-safe. Do not reopen formula correctness unless the direct-vs-column parity tests fail.

### 6.3 — [ ] T2.3: Surface fourier `_from_dofs` / `_from_spec` factory — **FACADE + TENSOR + XYZ UNPACK + DCOEFF WRAPPER + XYZ ORDER-HATS + G1 PRODUCT-RULE LOC-BANKED / REMAINING PRODUCT-RULE FORMULAS EXPLICIT (2026-06-01)**

- **Files:** `surface_fourier_kernels.py` (pre tensor-kernel slice 3,139 LOC; current 2,749 LOC; the tensor simple `_*_from_dofs` wrappers spanned ~`:2208-2516`, the `SurfaceXYZFourier` analytic wrappers span ~`:1442-2172`, and the coefficient-derivative wrapper helpers now sit near `:2544-2735`) + `surface_fourier.py` (pre facade slice 978 LOC; current 813 LOC; the facade `_*_from_spec` / paired-linear `_*_from_dofs` wrappers spanned ~`:171-864`). (v4's `kernels:2200-2799` was stale but overlapped the tensor wrapper family.)
- **2026-06-01 facade slice:** `surface_fourier.py` now uses typed local factories for the `SurfaceXYZFourier` / `SurfaceXYZTensorFourier` kernel-backed spec wrappers and paired-linear dof wrappers. Public names remain exported symbols with assigned `__name__` / `__qualname__` / `__module__`, while composed geometry (`normal`, fundamental forms, curvatures, area, volume) stays explicit.
- **2026-06-01 tensor-kernel slice:** `surface_fourier_kernels.py` now uses `_eval_surface_tensor_from_dofs(...)` for the ten simple `SurfaceXYZTensorFourier` evaluator wrappers: `gamma`, paired `gamma_lin`, first/second coordinate derivatives, and `normal`. The public wrapper functions remain explicit so `__code__.co_name`, `inspect.getsource(...)`, signatures, and docs stay public-introspection compatible. The concrete evaluator functions, `_dofs_to_xyzc_any(...)`, stellsym scatter construction, `SurfaceXYZFourier` scatter/template wrappers, composed area/volume/unit-normal helpers, and coefficient-Jacobian wrappers stay explicit.
- **2026-06-01 `SurfaceXYZFourier` unpack slice:** the nine analytic `SurfaceXYZFourier` wrappers now call the existing `_scatter_surface_xyzfourier_dofs(...)` helper directly instead of repeating the six-line flat-DOF to `(xc, xs, yc, ys, zc, zs)` unpack block. Public wrappers, source introspection, analytic formulas, paired derivative helper, composed area/volume/unit-normal helpers, and coefficient-Jacobian factories stay explicit.
- **2026-06-01 coefficient-derivative wrapper slice:** `surface_fourier_kernels.py` now uses one `_surface_dof_transform(...)` helper for the repeated `jax.jacfwd` / explicit Hessian / `jax.grad` / `jax.hessian` wrapper ceremony across tensor and `SurfaceXYZFourier` coefficient-derivative families. The helper has two explicit inner signatures, preserving the tensor signature without `coeff_template` and the `SurfaceXYZFourier` signature with `coeff_template`. No derivative formula, public export name, scalar tolerance, CPU geometry code, CUDA/MPS path, or composed geometry formula was changed.
- **2026-06-01 `SurfaceXYZFourier` order-hat slice:** the six full-grid analytic `SurfaceXYZFourier` wrappers now share `_surface_xyzfourier_component_hat_derivatives(...)` for the repeated derivative-order to separable-basis to `(xhat, yhat, zhat)` component evaluation. Public wrappers, paired-linear wrappers, coefficient-Jacobian factories, rotation terms, and product-rule formulas stay explicit.
- **2026-06-01 `SurfaceXYZFourier.gammadash1` product-rule micro-slice:** the full-grid `gammadash1` wrapper now uses the same explicit radial/toroidal product-rule representation already used by the paired-linear path, then calls `_surface_xyzfourier_rotate(...)`. Public wrapper signature, source ownership, coefficient-Jacobian factories, paired-linear wrappers, and the other product-rule formulas are unchanged.
- **LOC saved:** 555 banked across completed T2.3 slices: 165 in `src/simsopt/jax_core/surface_fourier.py` for the facade slice (`281 insertions / 446 deletions`), 190 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the tensor-kernel slice (`58 insertions / 248 deletions`), 51 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the `SurfaceXYZFourier` unpack slice (`15 insertions / 66 deletions`), 109 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the coefficient-derivative wrapper slice (`52 insertions / 161 deletions`), 35 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the `SurfaceXYZFourier` order-hat slice (`73 insertions / 108 deletions`), plus 5 in `src/simsopt/jax_core/surface_fourier_kernels.py` for the `gammadash1` radial/toroidal product-rule micro-slice (`3 insertions / 8 deletions`). Do **not** close the old full-item estimate merely because the banked total now exceeds 550 actual LOC: the remaining `SurfaceXYZFourier` product-rule formulas are intentionally still explicit, and any further formula fold requires a separate readability and parity gate.
- **Risk:** Low-to-medium. `__all__` listings + cross-imports stay unchanged; factory-generated wrappers must keep public symbol names and JIT behavior. The tensor-kernel slice preserves public signatures, `__name__`, `__qualname__`, and `__module__` for all ten folded wrappers. The coefficient-derivative wrapper slice preserves public tensor and `SurfaceXYZFourier` derivative signatures, including the `coeff_template` distinction. The order-hat slice must preserve derivative-order tuple destructuring and full-grid versus paired-linear rotation paths. The `gammadash1` product-rule micro-slice must preserve the same `d/dphi` cylindrical rotation algebra while removing only the expanded cosine/sine spelling.
- **Contracts:** Stellsym scatter indices; every public symbol name; tensor conventions (`dgamma_by_dcoeff[i,j,l,k]`).
- **Validation gate:** T2 + `tests/geo/test_surface_fourier_jax.py`.
- **2026-06-01 tensor-kernel validation evidence:** signature-and-introspection preservation script for the ten folded wrappers passed, including `__code__.co_name`, `inspect.getsource(...)`, and public doc checks; `ruff check` passed for `surface_fourier_kernels.py` plus the two parity test files; `ruff format --check src/simsopt/jax_core/surface_fourier_kernels.py` passed; `py_compile` passed for the changed source plus parity tests; `mypy src/simsopt/jax_core/surface_fourier_kernels.py` passed; `tests/geo/test_surface_xyz_tensor_clamped_jax.py` passed (`38 passed`); the focused Fourier parity selector passed (`60 passed, 90 deselected`). `ruff format --check` over the two unmodified parity test files remains a pre-existing blocker (`Would reformat`) and is not part of this LOC-banked source slice.
- **2026-06-01 `SurfaceXYZFourier` unpack validation evidence:** `ruff check`, source-only `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/surface_fourier_kernels.py`, and public wrapper introspection passed. CPU/X64 tests passed: focused selector for geometry, dcoeff, paired-linear, and non-RZ fundamental form behavior (`34 passed, 116 deselected`) plus the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`).
- **2026-06-01 coefficient-derivative wrapper validation evidence:** derivative wrapper signature script passed for all 13 tensor derivative functions and all 13 `SurfaceXYZFourier` derivative functions, including the `coeff_template` arity distinction and `__code__.co_name == "wrapper"`; `ruff check`, source-only `ruff format --check`, `py_compile`, and `mypy src/simsopt/jax_core/surface_fourier_kernels.py` passed; focused CPU/X64 coefficient/normal derivative parity selector passed (`48 passed, 102 deselected`); scalar area/volume derivative wrappers are covered by the full CPU/X64 `tests/geo/test_surface_fourier_jax.py` file (`150 passed`).
- **2026-06-01 `SurfaceXYZFourier` order-hat validation evidence:** public wrapper introspection passed for the six touched full-grid analytic functions, including `__name__`, `__qualname__`, `__module__`, `__code__.co_name`, source prefix, and `coeff_template` signature preservation; `ruff check`, source-only `ruff format --check`, `py_compile`, `mypy src/simsopt/jax_core/surface_fourier_kernels.py`, `pip check`, and `git diff --check` passed. CPU/X64 tests passed: focused selector for geometry, tangents, second derivatives, paired-linear wrappers, and non-RZ fundamental-form behavior (`34 passed, 116 deselected`), the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`), and the full `tests/geo/test_surface_fourier_jax.py` file (`150 passed`).
- **2026-06-01 `SurfaceXYZFourier.gammadash1` product-rule validation evidence:** `ruff check src/simsopt/jax_core/surface_fourier_kernels.py tests/geo/test_surface_fourier_jax.py`, source-only `ruff format --check`, `py_compile`, and `mypy src/simsopt/jax_core/surface_fourier_kernels.py` passed. CPU/X64 tests passed: focused selector for geometry/tangent and adjacent derivative coverage (`26 passed, 124 deselected`) and the full `TestSurfaceXYZFourierJaxCppParity` class (`26 passed`).
- **Remaining LOC-banking follow-up:** Re-derive only the still-explicit `SurfaceXYZFourier` product-rule formulas as separate patches if they can stay more readable than the current local formulas. Do not merge those with the tensor wrapper fold; they use distinct scatter/template and product-rule terms.

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
- **Risk:** Low for CPU behavior after validation; CUDA/MPS-specific placement proof remains outside this CPU forced-device slice.
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

### 6.8 — [ ] T2.8: `LayerDriftTracker` dataclass for `single_stage_init_parity` — **CORE TRACKER + SCIPY CALLBACK + TARGET-NATIVE CACHE LOC-BANKED SMALL / FULL TARGET OPEN (2026-06-01)**

- **Files:** `benchmarks/single_stage_init_parity.py` — `_record_first_scipy_callback_split` at `:2746`, `LayerDriftTracker` at `:3009`, `_pre_newton_census_gate_failures` at `:3379`, and `compare_same_candidate_objective_replay` at `:3471` (target-native rejection cache around `:3599`).
- **Change:** Landed `LayerDriftTracker` for the two layer-decomposition families (`boozer_solve_decomposition` and `iota_penalty_decomposition`), a narrow `_record_first_scipy_callback_split(...)` helper for the repeated "first split wins" bookkeeping in the Boozer SciPy callback trace comparison, and a target-native rejection-predicate cache for the repeated per-pair contract checks. The helpers/cached predicates own repeated local state transitions while the final replay result dict remains explicit at the public schema boundary.
- **Remaining target:** The old full-item target remains open. These slices intentionally did **not** hide slice-owner, metadata, hardware, failure, candidate, or broader callback trackers behind a broad dynamic key-prefix helper.
- **LOC saved:** 19 net benchmark LOC across completed T2.8 micro-slices: 13 for the `LayerDriftTracker` slice (`79 insertions / 92 deletions`), 4 for the SciPy callback first-split slice (`34 insertions / 38 deletions`), plus 2 for the target-native predicate-cache slice (`9 insertions / 11 deletions`). Do not bank the old `~200 LOC` estimate until the remaining tracker families are folded without obscuring the replay schema.
- **Risk:** Low. Internal trackers; external dict shape preserved.
- **Contracts:** `_pre_newton_census_gate_failures` untouched; `parity_bug_census["divergent_layers"]` schema; all replay payload keys preserved.
- **Design-it-twice gate:** Option A, a generic key-prefix `.summary_dict(...)` emitter for every replay tracker, was rejected because it would hide the output schema in string construction. Option B, a typed state helper for only layer-decomposition drift plus explicit return-key mapping, was selected because it removes repeated state transitions while keeping the public replay payload auditable.
- **Validation gate:** focused same-candidate replay tests passed (`22 passed, 338 deselected`); the SciPy callback split tests passed (`2 passed, 358 deselected`); the target-native predicate selector passed (`3 passed, 357 deselected`); the pinned pre-Newton census failure-list tests passed (`9 passed, 351 deselected` for the focused replay/pre-Newton selector); scoped `ruff`, `ruff format --check`, `py_compile`, `pip check`, and `git diff --check` passed. `mypy benchmarks/single_stage_init_parity.py` remains blocked by pre-existing benchmark/example typing debt, including missing example modules and existing `slice_summary` type errors, not by the new tracker, callback helper, or target-native predicate cache.

### 6.9 — [x] T2.9: Quantity-aware tolerance helper in `validation_ladder_contract.py` — **COMPLETED / NOT LOC-BANKED (2026-06-01)**

- **Files:** `_tolerance_for` lives in **only one** file — `non_banana_example_cpp_jax_cpu_parity.py` (now a compatibility wrapper). **v5 CORRECTION:** v4's other two sites are wrong — `single_stage_parity_matrix.py` has **no** `_tolerance_for` (the cited `:292-302` does not exist) and `stage2_e2e_comparison.py` uses a **different** mechanism (`optimizer_drift_tolerances:71/74`). So this was a single-file local helper migration, not a 3-file consolidation.
- **Change:** Landed `QUANTITY_TOLERANCE_BUCKETS`, `quantity_parity_tolerance(...)`, and `quantity_uses_gradient_tolerance(...)` in `benchmarks/validation_ladder_contract.py`; `non_banana_example_cpp_jax_cpu_parity.py::_tolerance_for(quantity)` now delegates to the contract helper. The helper preserves `event_time_tracing` state-vector tolerances, `qfm_gradient` derivative-heavy tolerances, and float32 value/objective/gradient branches. It does not collapse to a generic `kind="rtol_atol"` bucket.
- **LOC saved:** 0 banked. `non_banana_example_cpp_jax_cpu_parity.py` shrank by 117 LOC, but `validation_ladder_contract.py` grew by 185 LOC and focused tests added 49 LOC, so this is an SSOT/contract-hardening slice, not bloat reduction.
- **Risk:** Low-Medium. Pure helper only if every quantity maps to the same bucket and pair as before.
- **Contracts:** `PARITY_LADDER_TOLERANCES` values frozen; `parity_ladder_tolerances(lane)` API unchanged.
- **Validation gate:** T2 + pre/post snapshot of every migrated quantity's `(bucket, rtol, atol)`.
- **2026-06-01 evidence:** 204-row pre/post snapshot over every migrated quantity and runtime tier was byte-identical. Focused tests passed: `tests/test_benchmark_helpers.py -k 'quantity_parity_tolerance or parity_ladder_tolerances'` (`6 passed, 354 deselected`) and `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py -k 'float32_smoke_tolerance_tier_routes_by_quantity or unknown_runtime_tolerance_tier_fails_closed or float32_smoke_keeps_gradient_as_diagnostic_failure'` (`3 passed, 59 deselected`). `ruff`, `ruff format --check`, `py_compile`, `mypy benchmarks/validation_ladder_contract.py`, `mypy tests/geo/test_surface_fourier_jax.py`, and `git diff --check` passed. Broader `mypy benchmarks/non_banana_example_cpp_jax_cpu_parity.py` still reports unrelated existing errors in neighboring benchmark modules, but the previous `validation_ladder_contract.py` blocker is closed.

### Tier 2 exit gate

Tier 2 is not closed yet. Recompute the net LOC target from live item estimates after the remaining full-open T2.1/T2.2/T2.3/T2.8 follow-ups are either banked, re-scoped, or explicitly deferred; then run the T2 suite, re-affirm the contract checklist, replay `_pre_newton_census_gate_failures` byte-identically, and only then tag `bloat-reduction-T2-complete`.

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

### 7.2 — [ ] T3.2: `SpecBackedBiotSavartJAX` ↔ `BiotSavartJAX` mixin extraction — **POINT HELPERS LOC-BANKED / COTANGENTS OPEN (2026-06-01)**

- **Current status:** the field-evaluation duplication was already folded through `_BiotSavartFieldEvaluationMixin` (`field/biotsavart_jax_backend.py:505`) inherited by both `SpecBackedBiotSavartJAX` (`:642`) and `BiotSavartJAX` (`:1281`). The 2026-06-01 follow-up now also folds the shared point-state mutation bodies through `_set_biot_savart_points(...)`, `_set_biot_savart_points_cyl(...)`, and `_get_biot_savart_points_cyl(...)` (`:485-499`).
- **Files:** `src/simsopt/field/biotsavart_jax_backend.py` (2,301 -> 2,299 LOC for this points-only slice). Public methods remain class-local so `__qualname__`, signatures, and `FieldEvalSpec` annotations stay on `SpecBackedBiotSavartJAX` / `BiotSavartJAX`; shared private helpers cover only the duplicated point-state mutation and cylindrical readback bodies. `BiotSavartJAX.clear_points()` remains live-class-only (`:1758`) for wireframe temporary mutation.
- **Change:** Hoisted only the duplicate mutable point-state helper bodies into private module-level helpers. The public wrappers preserve method metadata, `BiotSavartJAX.set_points(...)` preserves the live no-host-round-trip path for JAX arrays, and `clear_points()` stays out of `SpecBackedBiotSavartJAX`.
- **LOC saved:** 2 banked for the metadata-preserving points-only slice (`29 insertions / 31 deletions`). Do **not** bank the old `~250` estimate: `coil_cotangents_to_dofs_gradient` remains open because the spec-backed implementation delegates to the jitted extraction-spec helper (`:775`) while the live class still owns the fallback-compatible projection path (`:2118`).
- **Risk:** Medium. Token plumbing must stay live-class only; point-version increments must stay identical; `_coils` property, `_uses_uniform_curve_xyz_fourier_fastpath`, and `coil_dof_extraction_spec()` remain class-owned; cotangent reconciliation needs separate fallback coverage before any hoist.
- **Validation gate:** points-only gate completed with focused CPU/X64 cylindrical accessor parity for live/spec-backed adapters (`4 passed`), `FieldEvalSpec` round-trip and float64 promotion selectors (`2 passed`), a runtime probe confirming public method `__qualname__` / signatures / `FieldEvalSpec` annotations and `clear_points` absence on `SpecBackedBiotSavartJAX`, scoped `ruff`, `py_compile`, and `git diff --check`. Source-only `mypy` now runs in `.conda/jax` but remains blocked by pre-existing `SpecBackedBiotSavartJAX.x` / `save` override errors in this file.

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
| T2 — Factory Introductions | ~3,100-3,500 guaranteed pending re-estimate (T2.2 formula dedup landed but did **not** bank the old ~400 LOC estimate; T2.2 follow-ups bank 47 LOC total) | 9 (full T2.2 target still open) | 3–4 days | Low-Med |
| T3 — Structural Consolidations | ~4,000–5,500 (T3.2 field-eval dup **already folded** via `_BiotSavartFieldEvaluationMixin`, metadata-preserving points-helper follow-up banks 2 LOC, cotangent reconciliation remains open; T3.3 / T3.8 partially done via the `surfaceobjectives_traceable_jax.py` split) | 8 | 1–2 weeks | Med |
| T4 — Decision Points | T4.2 resolved (doc-only); T4.1 ~1,628 + T4.3 ~138 + T4.4 (alias quarantine, re-scoped to the qfm **surface** wrappers — **not** obsolete) still decision-gated | 5 (T4.2 resolved) | varies | Decision-bound |
| **Aggregate** | **STALE until salvage splitting.** Historical estimate was ~7,900-9,800 guaranteed candidate LOC remaining pending T2.2 re-estimate, plus decision-gated deletions. The drift-checkpoint ledger is not net-shortening: effective `src/` is `+45` once untracked source helpers are included, and `src/`+`tests/`+`docs/` is `+2166`. Recalculate aggregate remaining LOC only after banked-shrink slices are isolated and foundation/not-banked slices are classified. | **33 items (full T2.2 target still open; T3.2 partial; T4.2 resolved)** | **~3 weeks historical estimate; replan after salvage split** | **manageable only with the v8 drift gate** |

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
