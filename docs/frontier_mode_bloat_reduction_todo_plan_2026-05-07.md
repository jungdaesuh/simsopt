# Frontier Mode Bloat Reduction — TODO Plan

**Date:** 2026-05-07 (revised same day per Codex review)
**Branch:** surrogate-confinement-v2
**Status:** Implemented except certified-lane smoke and external-team NSGA3-use confirmation. Current dirty-tree validation includes the 467-test targeted frontier/workflow/ALM slice and the repo full suite (`./run_tests`: `Ran 1802 tests in 1408.073s`, `OK (skipped=94)`) after the current code updates; DC.13 remains open only for the certified-lane smoke blocker. Dry-run smoke, non-dry campaign JSON smoke, post-DC.5 focused sweep, algorithm-invariant backfill, all LOC gates, and removable-LOC audit budget are complete.
**Owner:** TBD
**Estimated effort:** ~3-5 focused days; can be split across PRs

**Blocking inputs before final closeout:**
- **D1.4:** Team confirmation that no retained production result requires NSGA3-specific Pareto coverage beyond `multilane_local` + ε-constraint.
- **1A.10 / CC.3 / DC.13:** A current strict Stage 2 artifact that is checksum-bound, signed-current, hardware-clean, non-init-only, and Boozer-bootable enough to produce at least one certified frontier lane. Local searches, probes, adjacent-artifact/doc sweeps, current-code smoke variants, and the r59 closeout audit did not find one; the strongest JSON-surface handoff smoke timed out with zero archive members.

**Blocker closure criteria:**
- **D1.4 closes only with external confirmation.** Local grep/artifact searches support the `DROP` decision, but they cannot prove that no teammate has a retained external NSGA3 campaign. Close this only after recording a team answer, or reopen the decision if a real NSGA3 artifact/benchmark is supplied.
- **Certified-lane smoke closes only with a certified archive member.** A passing wrapper, preserved JSON shape, dry-run smoke, or non-dry run with zero archive members is not enough. The closing artifact must be a `multilane_local` run with `dry_run=false`, `stage2_artifact_init_only=false`, `frontier_feasible_lane_count >= 1`, and `frontier_archive_size >= 1` from a strict Stage 2 seed.
- **The required Stage 2 seed contract is strict.** The seed must be checksum-bound, signed-current, hardware-clean, non-init-only, and Boozer-bootable. Legacy positive-current, unbound, init-only, missing-hardware-field, or self-intersecting/Boozer-failed seeds are rejection evidence, not partial closeout evidence.

**Next required input request:**
1. Send this exact D1.4 question to the team: "Does anyone have a retained production frontier result that requires NSGA3-specific Pareto coverage that is not reachable by the current `multilane_local` plus epsilon-constraint path? If yes, provide the campaign artifact path and benchmark evidence; if no, confirm NSGA3 deletion is acceptable."
2. Provide or generate a Stage 2 seed artifact whose adjacent metadata proves the strict contract above. After running the frontier smoke, the summary must satisfy:
   ```bash
   jq -e '
     .dry_run == false and
     .frontier_engine == "multilane_local" and
     .stage2_artifact_init_only == false and
     .frontier_feasible_lane_count >= 1 and
     .frontier_archive_size >= 1
   ' "$OUT/single_stage_frontier_campaign_summary.json"
   ```
3. Only after both checks pass should `D1.4`, `1A.10`, `CC.3`, and `DC.13` be marked complete.

**Completion audit snapshot (2026-05-09 r59):**

| Objective requirement | Evidence artifact | Status |
|---|---|---|
| Execute the bloat-reduction implementation plan | Implementation entries r7-r35 plus current closed checklist items; r49 closed DC.21 deferred follow-up via WONTFIX | Complete except rows below |
| Validate code/test/tooling coverage | r51 fresh 467-test frontier/workflow/ALM slice, repo `./run_tests` (`Ran 1802 tests in 1408.073s`, `OK (skipped=94)`), r54/r55 scoped checks (`ruff`, touched-scope `compileall`, `git diff --check`, stale-import grep, 467-test slice, dry-run shape probe), r56-r59 untracked/cache-artifact cross-checks, and audit subagents found no additional implementable plan item | Complete for code paths covered by this plan |
| Preserve current JSON/runtime-calibration shape | D3 recorded preserve-shape; dry-run and non-dry wrapper smoke produced expected manifest/summary/progress/archive/recommendation JSON; DC.21 WONTFIX retains 2-profile `FRONTIER_RUNTIME_CALIBRATION_PROFILES` to keep the JSON contract stable | Complete for wrapper shape |
| Remove/deprecate NSGA3 code path | Deleted engine/evaluator files, validators reject deleted engines/fields, local/adjacent artifact scan found no retained local NSGA3 artifacts; r56 inspected the local NSGA-III revival plan and found it dormant, with no trigger artifact or benchmark attached | Locally complete; external D1.4 confirmation still open |
| Produce final certified-lane smoke | Current strict smoke summaries have `frontier_feasible_lane_count=0` and `frontier_archive_size=0`; r34 circular probe reports `BOOZER_BOOTABLE=false`; r50 `--stage2-iota-mode=alm` decision-gate failed twice; r52 strict post-handoff smoke failed Boozer initialization; r53 strongest JSON-surface handoff smoke timed out after 300 s with zero archive members; r59 found no newer certified-lane evidence | Blocked at physics/fixture layer; close criteria narrowed to external action (new fixture / contract relaxation / iota target amendment) |
| Close all checklist items | Open boxes remain `D1.4`, `1A.10`, `CC.3`, `DC.13`; DC.21 deferred follow-up closed WONTFIX in r49; r59 closed no additional item | Blocked at the same 4 boxes |

**Revision log:**
- 2026-05-07 r1 — initial draft
- 2026-05-07 r2 — Codex review fixes: expanded NSGA3 test deletion list (Phase 1A); fixed runnable smoke command + correct summary filename (CC.3, CC.4); reframed hypervolume memoization acceptance (Phase 5.1); removed unsafe archive-metadata side-map task (Phase 6.2.3); dropped YAGNI schema-helper extraction (Phase 4.2); reworked lazy pymoo plan (Phase 5.3); split LOC done-criteria (DC.11); corrected importer counts in F1, F9
- 2026-05-07 r3 — Codex r2 review fixes: removed dropped Phase 4.2 reference from PR5 title; reworded Phase 1A.3 to "delete the optional NSGA3 summary payload validator entirely" (it's called unconditionally, not engine-gated); added PYTHONPATH/isolated-venv guidance to lazy-pymoo verification commands (Phase 5.3.1, 5.3.4) and removed destructive `pip uninstall pymoo` for local use; corrected `test_summary_validator_accepts_optional_nsga3_fields` LOC estimate from ~50 to ~127 (lines 189-315); reframed hypervolume acceptance to allow for prefix-archive sweeps in `build_frontier_hypervolume_history`
- 2026-05-07 r4 — Codex r3 review fixes: expanded Phase 1A.3 NSGA3 field list from 2 to all 6 optional fields (`frontier_generation_history`, `_path`, `frontier_engine_stats`, `frontier_evaluator_spec`, `_path`, `frontier_population_checkpoint_path`) per `frontier_contracts.py:446-489`; reframed Phase 5.1 wall-clock estimate from "~3× factor" to "up to ~3× on repeated final-archive calls; lower when prefix-history or leave-one-out calls dominate"
- 2026-05-07 r5 — Codex r4 review fix: synchronized D2.1 with Phase 1A.4 — D2.1 had stale "test_frontier_evaluator.py:111-275 / ~165 LOC" reference; now lists all 4 NSGA3-tied test blocks across 3 files totaling ~612 LOC, with cross-reference to Phase 1A.4 as canonical source
- 2026-05-07 r6 — cross-agent verification fixes (8 must-fix + 14 accuracy):
  - **E1** corrected stale `test_single_stage_workflow_helpers.py` line numbers (4986→5150, 5140→5304; off by 164) in 1A.4 and D2.1; updated per-block LOC (140→154, 180→192).
  - **E2** fixed §6.1.2 stale "all 6 importers" comment to "both importers" (F1 correctly says 2; only `run_single_stage_frontier_campaign.py:38` and `tests/geo/test_frontier_archive.py:25` import the module).
  - **E3** corrected §8.1.2 cache-key proposal: must canonicalize by lex-sort over the four-objective vectors (in `PARETO_OBJECTIVE_SPECS` order), NOT by `member_id` (which carries lane/campaign identity, not Pareto content).
  - **E4** dropped 9.1.5 "redundant except" task — the `except FrontierEvaluatorInitializationError: raise` arm at `frontier_evaluator.py:920` (NOT the runner — `run_single_stage_frontier_campaign.py:920` is unrelated NSGA3 artifact loading; r6.1 file-disambiguation correction) prevents the broader `except Exception` arm at `frontier_evaluator.py:922` from double-wrapping an already-typed init error; not redundant.
  - **E5** reworded 9.2.1 / 9.2.2 from "convert to frozen" to "add `slots=True`" — the four cited dataclasses are already `@dataclass(frozen=True)`; only `slots=True` is missing.
  - **E6** downgraded F12 / 8.5.1 / 8.5.2 from "duplicate resolution" to "conditional re-resolution + comment" — both runner pairs (762/827, 874/1035) operate on different inputs (lane-count reconciliation, certified-vs-archive members) and are not collapsible.
  - **E7** added `EpsilonThresholds.from_goal_config(config)` constructor — the proposed `from_rerun_contract(Mapping)` does not match the `single_stage_objectives.py:539-555` consumer which uses attribute-access on a frozen `frontier_goal_config` dataclass.
  - **E8** rewrote DC.12 — the original criterion pulled 320 LOC from `test_single_stage_workflow_helpers.py` toward a `tests/geo/test_frontier_*.py` budget (different glob, doesn't count); split into DC.12a (frontier_*.py budget) and DC.12b (workflow_helpers NSGA3 deletions, corrected to ~347 LOC).
  - **W1** F6 added 5th hypervolume call site at `frontier_runtime_calibration.py:231` (per-lane early-stop status); revised "≥3×" to "≥4×".
  - **W2** softened 8.1 wall-clock claim from "up to ~3×" to "sub-2× in typical campaigns" (when prefix-history dominates).
  - **W3** corrected 8.1.6: best-known leave-one-out exclusive-contribution algorithms (HSO/WFG) are O(N^{d−1}) for d≥4, not O(N) — the 4D regime here is non-trivial.
  - **W4** F3 / 2.1.1 expanded cluster boundary 255-558 → 255-588 (includes `_frontier_excess_penalty` body); also call out `augment_frontier_metric_state` (255) dependency on non-frontier `_objective_gradient` (114) which must remain in `single_stage_objectives` and be back-imported.
  - **W5** 2.2.2 expanded helper list from 2 to 4 (`_FINITE_SCALAR_FIELDS`, `_FINITE_VECTOR_FIELDS`, `_FINITE_VECTOR_LIST_FIELDS`, `_FINITE_EPS`); flagged `frontier_constraints.py:166` self-consumer.
  - **W6** 2.2.3 corrected "6+ import sites" to "2 source + 1 test" (the prior count conflated call sites with import sites).
  - **W7** 2.1.5 reframed JAX-purity warning as over-cautious — `_frontier_alm_base_total_grad` is pure NumPy with no closures.
  - **W8** F10 / 9.4 corrected boilerplate LOC from "~150" to "~294" (off by ~95% in r1-r5).
  - **W9** F7 added `calibration_basis` (tuple-of-strings) to the field-diff list; noted `profile_name` is tautological.
  - **W10** 7.3.1 acknowledged sort-key heterogeneity (scalar reduction, None-aware fields); revised signature to take `KeyFn` callable rather than `Sequence[tuple[str, Direction]]`.
  - **W11** F5 reframed "duplicated logic" as "shared magic-string keys with distinct downstream semantics"; added 2 missing CLI-parser sites (out of scope but flagged).
  - **W12** corrected Phase 1A bottom-line from "~600 source + ~612 test = ~1,212 LOC" to measured "~478 source + ~628 test = ~1,106 LOC" (includes helper at 34-35; r6.1 fixed an internal arithmetic transcription bug that had said "626").
  - **W13** 9.6.2 reframed subprocess-test deletion as "audit + justify or keep" — it adds a cross-process regression guard the in-process version does not.
  - **W14** corrected Wechsung citation: first author is Giuliani (Giuliani, Wechsung, Cerfon, Stadler, Landreman, JCP 459 (2022) 111147).
- 2026-05-07 r6.1 — second-pass validation fixes (5 issues caught by re-running checks against live tree):
  - **R1** disambiguated E4 file reference: the `except FrontierEvaluatorInitializationError: raise` arm is at `frontier_evaluator.py:920`, NOT `run_single_stage_frontier_campaign.py:920` (the runner line 920 is unrelated NSGA3 artifact loading). Fixed in revision log E4 entry and §9.1.5 body.
  - **R2** F5 hit count corrected: r6 said "9 hits across 5 files" (wrong on both axes). Live grep yields **8 string-literal hits across 4 files** (canonical scope) or **14 identifier hits across 6 files** (canonical scope); both counts surfaced in the F5 row.
  - **R3** F5 CLI-parser claim corrected: only `run_single_stage_goal_mode_comparison.py:363-364` defines `add_argument` for these flags. The frontier runner inherits via `parents=[goal_mode_comparison.build_parser(...)]` at `:119` — it does NOT have its own `add_argument`. The runner does, however, reference the keys at `:322-323` inside an attribute-copy loop over `lane_spec.scalarization_params` (a different concern).
  - **R4** Phase 1A test-LOC arithmetic harmonized to **628** (was inconsistently 626 vs 628 across D2.1, 1A.4 summary, Phase 1A bottom-line, and revision log W12). The +2 LOC for the helper at `test_frontier_evaluator.py:34-35` is now included consistently in all four totals.
  - Note (chat-only, not file): a side comment claiming "Document grew from 637 to 703 lines" was wrong — HEAD pre-r6 was 636 lines, not 637. Diff stat (+117 / −50) was correct.
- 2026-05-07 r6.2 — algorithm/computation audit (6 parallel agents covered hypervolume, dominance/normalization, scalarization/gradient, lane-spec generators, recommendation policies, NSGA-III + evaluator). **No correctness bugs producing wrong answers in nominal flow**, but several semantic mismatches and silent-pass paths. Added to plan:
  - **F17** (lead) `FRONTIER_REFERENCE_MODE_SHARED` (a.k.a. multilane_local) is mislabeled — `generate_multilane_local_specs` only varies iota↔volume shares clamped to [0.2, 0.8], **never touches qa_error/boozer_residual axes**, never reaches simplex vertices. It is a 1-parameter linear-scalarization sweep, not a 4-D Pareto frontier scan. Highest-impact finding for end-user interpretation.
  - **F18** ε-certifier silently passes when threshold key is missing from `scalarization_params` — `_evaluate_epsilon_constraint_status` (`frontier_archive.py:603-614`) reads via `Mapping.get` with `if limit is None: continue` fall-through. Risk: rerun_contracts built outside `_reference_scalarization_params`, typos, sidecar-JSON omissions silently certify members against absent thresholds.
  - **F19** ε-certification has zero floating-point slack — strict `excess > 0.0` at `frontier_archive.py:615-617` rejects `value == limit + 1e-16`. Members optimized to the boundary may fail certification without test coverage of the on-boundary case.
  - **F20** `dominates()` (`frontier_dominance.py:175-203`) does not guard NaN — IEEE 754 makes all comparisons return False, so `dominates(nan, b) == dominates(a, nan) == False`. Standard ingest filters NaN via `_as_finite_float`; hand-built members would slip past dominance entirely.
  - **F21** Hypervolume reference is not enforced as a true nadir — `_hypervolume_boxes` clips per-axis extent with `max(0, extent)` and silently drops boxes with all-zero extents; no warning when a member is worse-than-reference on some axis.
  - **F22** `frontier_conditioning.py` is misnamed: it's a **diagnostic max/min ratio gate** on objective magnitudes, NOT a preconditioner (no `D⁻¹` scaling, no Hessian approximation). Also computed in raw objective units (ignores `objective_metric_scale`), so the gate verdict doesn't reflect the metric-scale-normalized space the optimizer actually traverses.
  - **F23** Derived scalarization fields (`frontier_rank_total`, `frontier_base_total`, `frontier_trust_penalty`, `frontier_epsilon_penalty`, `frontier_goal_total`, `frontier_chebyshev_deltas`, `frontier_chebyshev_softmax_weights`) are NOT in `_FINITE_*` finiteness-check tables. Chebyshev softmax overflow or `epsilon_penalty + base_total` overflow would propagate as NaN undetected. *(Corrected r6.3: the Chebyshev softmax does NOT overflow under positive sharpness — the LSE-shift trick at `single_stage_objectives.py:500-501` stabilizes it. The genuine risk is in derived sums like `frontier_rank_total` that aren't re-checked after summing penalty terms; deltas/softmax_weights dropped from the F23 list.)*
  - **N1-N4 (naming/structural — math correct but names mislead):**
    - `evaluate_frontier_trust_penalty` is a **residual cap on `J_Boozer`**, not the canonical NLP $\|x-x_0\| \le \Delta$ trust region (no DOF reference $x_0$). Math C¹ at boundary; name misleads.
    - ε-mode (now implemented in `frontier_scalarization.py`) keeps `J_QS + res·J_Boozer + iotas·J_iota + volume·J_volume` as the base objective rather than reducing to pure $f_k$ as canonical Haimes ε-method does. Hybrid weighted-sum + ε-penalty.
    - `frontier_boozer_trust_threshold` is set equal to `epsilon_constraint_boozer_max` (`frontier_scalarization.py:649-652`) → **double penalization** of boozer residual (additive, not redundant): `(δ/trust_scale)² + epsilon_penalty_weight·(δ/boozer_reference)²`.
  - **P1-P5 (perf / hidden cost):**
    - NSGA-III `_ArchiveTrackingCallback` re-extracts `algorithm.pop` and re-evaluates the entire population every generation (`frontier_engine_nsga3.py:107-109`); correct due to caching but inflates lookups 2×.
    - `_select_reference_directions` decimation (`frontier_scalarization.py:362-378`) uses enumeration-order indices and dedupe-fills with the lowest unused index — biases extras toward the order-prefix (high-iota corners). Violates Das-Dennis uniform-coverage intent. *(Narrowed r6.3: rounded-index path is all-distinct for typical lane counts (verified for N=15 from H=3/20 — indices `[0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]`). The dedupe-fill loop is only exercised on collisions, which are rare. Bias claim narrowed accordingly.)*
    - `_WEIGHT_FLOOR=1e-12` clamp (`frontier_scalarization.py:537-540`) breaks the simplex unit-sum invariant: "pure (1,0,0,0)" becomes `(1, 1e-12, 1e-12, 1e-12)`. Necessary for downstream Chebyshev division but undocumented.
    - Pymoo internal RNG state is NOT serialized in `load_nsga3_frontier_campaign_artifacts` — resume after partial NSGA-III run cannot reproduce next-gen sample identically. *(Corrected r6.3: this is moot under the current binary resume design at `run_single_stage_frontier_campaign.py:918-924` — the resume path either loads a complete prior run's artifacts and skips `run_nsga3_frontier_campaign` entirely, or runs fresh; there is no "continue from generation N/M" path. RNG state being unserialized would matter only if a partial-run-continuation path were added.)*
    - `_eligible_members_for_policy` (`frontier_recommendation.py:333-336`) silently fell back to all members when no member passed the gate. Corrected 2026-05-09: the fallback branch is deleted; missing legacy Boozer-trust metadata remains eligible through `missing_is_eligible=True`, while explicit `False` stays ineligible.
  - **NSGA-III branch divergence in runner:** the NSGA-III dispatch (`run_single_stage_frontier_campaign.py:916-942`) does NOT populate `lane_records_by_id`. Final summary's `frontier_lane_records` for an NSGA-III run is whatever stale entries were already in `lane_records_by_id`. Falls away with Phase 1A engine deletion; if Phase 1B (KEEP) is chosen, must be fixed.
  - **NSGA-III sign-flip ✓ correct:** `objective_vector_for_minimization` (`frontier_evaluator.py:270-277`) negates max-axis values before packing into `out["F"]`. HV reference is shared across engines (resolved once in runner). Cache key covers full spec fingerprint + DOF bytes.
  - **Substantial test gaps** (consistent across all 6 agents): HV permutation invariance, HV monotonicity-under-dominance, HV reference-edge-cases, dominance reflexivity/symmetry/NaN, degenerate-axis (`ideal == nadir`), ε on-boundary/just-above/just-below, ε silent-pass on `scalarization_params={}`, recommendation empty-archive/single-member/all-fail-gate, synthetic flat-HV early-stop sequence, multilane share-sweep boundaries, decimation geometric-uniformity, achievement-scalarization sign/scaling invariance. *(Narrowed r6.3: two of these were over-claimed. NaN/Inf annotation IS tested at `tests/geo/test_frontier_constraints.py:27-49` — gap narrows to NaN propagation in `dominates`, `balanced_policy_score`, hypervolume box construction, and achievement scalarization. End-to-end early-stop stagnation IS tested at `tests/geo/test_single_stage_workflow_helpers.py:5059-5148` — gap narrows to a unit-level test on `update_frontier_early_stop_status` itself.)*
  - **No plan drift on F12 line citations** (false alarm during audit). The hypervolume agent searched `frontier_archive.py` for the cited lines (where the file is only 698 LOC) and concluded the citations were stale. In fact, F12 attributes the lines to `run_single_stage_frontier_campaign.py` correctly (`:874-878` is the initial `resolve_hypervolume_reference(members=archive_members)`; `:1035-1039` is the post-loop re-resolution with `members=certified_members`). Verified at HEAD.
- 2026-05-07 r6.3 — third-pass validation fixes (5 corrections from re-running r6.2 claims against the live tree):
  - **C1** P2 overstated. The `_select_reference_directions` decimation (`frontier_scalarization.py:362-380`) does use enumeration-index rounding (not geometric-distance), but for the cited example N=15 from H=3 (20 directions) the rounded indices are all distinct — the dedupe-fill loop at lines 373-378 is **not exercised** for that case. The "high-iota-corner bias" only applies when the rounded-index set has collisions (rare, only for very small or pathological N). Plan softened: the claim is narrowed to "no geometric uniformity guarantee, with a worst-case bias toward enumeration-prefix only when the rounded-index set collides".
  - **C2** P4 misframed. NSGA-III pymoo RNG state is genuinely not serialized, but the current resume path is **binary**: `load_nsga3_frontier_campaign_artifacts` (`run_single_stage_frontier_campaign.py:918-924`) either loads a complete prior run's artifacts and skips `run_nsga3_frontier_campaign` entirely, or runs fresh. There is no "continue from generation N/M" path. RNG state being unserialized is therefore moot for resume reproducibility — it would matter only if a partial-run continuation path were added. Plan reworded: P4 demoted from "blocker for KEEP" to "irrelevant given current resume design; would become relevant if Phase 1B added a partial-run-resume path".
  - **C3** F1 / F23 Chebyshev overflow example is weak. The Chebyshev `_frontier_chebyshev_goal` (`single_stage_objectives.py:498-508`) uses the standard LSE-shift trick (`max_delta` is subtracted before exponentiating at line 500-501), so the softmax is numerically stabilized for any positive `sharpness` × bounded `deltas`. Plan example replaced: the silent-NaN risk for derived scalarization fields is real (none of `frontier_rank_total`, `frontier_base_total`, etc. are in `_FINITE_*` tables) but a more honest example is "`frontier_base_total` = `length_weight·J_len + replacement_total + penalty_total` could go non-finite if any summand does", not the Chebyshev softmax.
  - **C4** Test-gap section had two overclaims:
    - "No NaN test anywhere" — wrong: `tests/geo/test_frontier_constraints.py:27-49` (`test_annotate_search_evaluation_finiteness_flags_nonfinite_fields`) covers NaN/Inf detection in raw evaluator fields. Plan narrowed: the genuine gaps are NaN in `dominates`, `balanced_policy_score` propagation, hypervolume box construction, and achievement-scalarization deltas — NOT in the finiteness annotation itself.
    - "No early-stop synthetic flat-HV sequence test" — wrong: `tests/geo/test_single_stage_workflow_helpers.py:5059-5148` (`test_frontier_campaign_early_stop_stops_after_archive_stagnation`) IS an end-to-end stagnation test. Plan narrowed: gap is a **unit-level** test on `update_frontier_early_stop_status` with hand-crafted HV sequences, not the integration test which exists.
  - **C5** Plan-document drift correction (already in r6.2) is confirmed correct after re-verification: HV reference resolution lines 874-878 and 1035-1039 are live and in `run_single_stage_frontier_campaign.py`, not `frontier_archive.py`. No further action.
- 2026-05-07 r6.4 — fourth-pass validation fixes (2 residual drift items caught by re-running r6.3 against the live tree):
  - **D1** DC.18 (line ~820) still said "a deliberate Chebyshev overflow is detected by `annotate_search_evaluation_finiteness`" — contradicts r6.3 C3 correction. DC.18 reworded: the test trigger is now "a deliberate non-finite penalty summand or `length_weight·J_len`" (which propagates into the summed derived fields `frontier_rank_total` / `frontier_base_total`), not Chebyshev internals (the LSE-shift trick prevents reaching that path).
  - **D2** Two r6.2 revision-log bullets carried stale phrasing without inline `*(Corrected r6.3: ...)*` annotations even though r6.3 had refuted them in narrative form. Inline annotations now added at:
    - The "Pymoo internal RNG state is NOT serialized" bullet (line ~60) — annotated to note this is moot under the current binary resume design.
    - The "Substantial test gaps" bullet (line ~64) — annotated to note that NaN/Inf annotation IS tested (`tests/geo/test_frontier_constraints.py:27-49`) and end-to-end early-stop stagnation IS tested (`tests/geo/test_single_stage_workflow_helpers.py:5059-5148`); only the narrower gaps remain.
  - **No further refactoring needed** — the actionable plan body (Phase 1B.C, Phase 7 sub-items, DC.18) had been corrected in r6.3; r6.4 only completes the inline-annotation discipline so the revision-log claims and the plan body are consistent at every site.
- 2026-05-07 r7 — execution record:
  - **D1 recorded as DROP.** Cross-repo artifact search found no completed `"frontier_engine": "nsga3"` campaign artifacts in `autoresearch`, this repo tmp/output trees, or run-history paths inspected during execution.
  - **D3 recorded as PRESERVE JSON SHAPE.** No external Python/notebook consumer of `frontier_runtime_calibration` was found, but the runtime-calibration payload shape was preserved while collapsing the profile registry to a factory.
  - Implemented Phase 1A, Phase 2, Phase 3.1/3.2, epsilon-threshold SSOT, hypervolume memoization, runtime-calibration registry collapse, re-resolution comments, slots on frontier dataclasses, and r6.2 algorithm hardening items F18-F23/P5/N1-N4/P3 plus the invariant-test backfill subset listed below.
- 2026-05-07 r8 — verification record:
  - Targeted frontier suite passed via direct `unittest` command: 205 tests across `test_frontier_*` plus frontier workflow-helper coverage after the r9/r10 regression tests were added.
  - Gradient-contract slice passed: `geo.test_single_stage_alm_integration` + `geo.test_frontier_evaluator` (84 tests).
  - Scoped syntax/lint/grep gates passed: `py_compile`, non-mutating scoped `ruff check`, NSGA3/deleted-engine greps, and `git diff --check`.
  - The repo `./run_tests` wrapper expanded the requested frontier pattern into the full unittest suite and was interrupted after frontier coverage had passed; a real full-suite completion is still open.
  - The repo `./run_ruff` wrapper auto-fixed unrelated pre-existing lint outside this plan's scope; those accidental hunks were reverted and the final lint gate is the scoped non-mutating `ruff check` on changed frontier files.
  - LOC budget criteria remain open: Phase 7 hardening/test backfill improved correctness but increased current frontier/test LOC.
- 2026-05-08 r9 — completion-audit follow-up:
  - Updated the cross-repo stale NSGA3 consumer note in `autoresearch/program_hbt_topology_surrogate_legacy_vmec.md:310`; it now directs users to `multilane_local` plus `achievement_chebyshev_full_simplex_v1` for the remaining solver path.
  - Added a hypervolume cache counter test that instruments the uncached computation and verifies repeated final-archive reporting paths reuse the cached result.
  - Dry-run frontier campaign smoke passed with a checksum-bound Stage 2 fixture, preserving summary/manifest runtime-calibration shape. A non-dry-run optimizer smoke and full-suite completion remain open.
- 2026-05-08 r10 — reviewer-agent fixes:
  - Restored valid partial epsilon-threshold lanes: archive certification now requires at least one epsilon threshold and enforces only thresholds present in the lane contract. Added regression coverage for QA-only and Boozer-only epsilon lanes.
  - Closed deleted-engine resume leakage: `FrontierCampaignProgress.from_json_dict` validates progress payloads on load, rejects unsupported `frontier_engine` values such as legacy `nsga3`, and the runner parser now uses the shared `SUPPORTED_FRONTIER_ENGINES` SSOT.
  - Fresh reviewer-agent pass returned PASS after these fixes.
- 2026-05-08 r11 — full-suite completion:
  - Repo `./run_tests` completed after the reviewer fixes: `Ran 1744 tests in 1303.562s` with `OK (skipped=94)`.
  - This closes the full-suite portion of 1A.9 / DC.13. The remaining open validation gap is the real non-dry-run optimizer smoke; the dry-run smoke already verified the summary/manifest runtime-calibration shape.
- 2026-05-08 r12 — real-smoke follow-up:
  - The first non-dry-run smoke attempt found a real wrapper/child CLI contract bug: `--allow-init-only-stage2-seed` was being forwarded into `single_stage_banana_example.py`, which does not define that wrapper-only validation flag. Removed that handoff from `append_single_stage_handoff_flags` and updated the two command-builder regressions to assert it is not forwarded.
  - Targeted regressions passed for the goal-mode and thresholded-physics command builders. `frontier_campaign_reporting.py` also completed Phase 9.5.1's remaining `getattr(args, ...)` cleanup; the manifest tests passed.
  - Retried non-dry-run `multilane_local` smoke after the CLI fix. The runner now reaches the single-stage subprocess, but no valid local Stage 2 smoke fixture was found: older init-only fixtures have positive TF-current metadata, signed-current artifacts either violate current hardware clearance or lack required hardware metrics, and fresh init-only Stage 2 generation failed geometry preflight. CC.3 / 1A.10 remain open pending a checksum-bound Stage 2 seed that satisfies the current signed-current and hardware contracts.
- 2026-05-08 r13 — selector-cleanup closeout:
  - Completed Phase 7.3's shared best-selector cleanup. Recommendation policies now use `select_best`, `lex_priority`, `scalar_score`, `none_aware_lex`, and a frozen `RECOMMENDATION_POLICIES` registry; `archive_best_by_metric` now reuses the same selector path and the duplicate `_metric_sort_key` was removed.
  - Completed the Phase 9.5.2 reporting cleanup by inlining the one-call lane-record sanitizer.
  - Added `test_archive_best_by_metric_uses_objective_directions`; `geo.test_frontier_archive` and `geo.test_frontier_recommendation` passed.
- 2026-05-08 r14 — frontier test cleanup:
  - Added `tests/geo/_frontier_test_helpers.py` and moved duplicated frontier module loader/path setup out of the six `test_frontier_*.py` files.
  - Collapsed the three frontier evaluator cache tests into one subtest-based cache behavior test covering file-cache reuse, LRU eviction, and parallel cache-state access.
  - Removed the `SingleStageFrontierEvaluator.from_spec` classmethod alias and switched the frontier evaluator tests to direct construction.
  - Audited the retained subprocess round-trip test, the only `patch.multiple(create=True, ...)` frontier-test site, and skip/xfail/marker usage. The subprocess test remains because it exercises a fresh-process runtime rebuild boundary; the patch site is the intentional inline-vs-shadow objective comparison; skip/xfail/marker grep returned zero hits.
- 2026-05-08 r15 — final narrow DRY cleanup:
  - Shared the repeated frontier archive-member test fixture through `tests/geo/_frontier_test_helpers.py`, removed duplicated local factories/raw constructors in archive, recommendation, and contract tests, and dropped redundant paired bare asserts after `assertIsNotNone`.
  - Added `resolve_frontier_runtime_defaults_from_args(...)` so the runner and manifest fallback no longer duplicate the same argparse-to-runtime-defaults argument wiring while preserving the explicit resolver API for direct tests.
  - Re-ran the focused frontier/workflow suite after this cleanup: 204 tests passed; touched-file `ruff check` and `py_compile` passed.
  - Source/test LOC moved in the right direction but did not close the original budget gates. A separate source audit found only ~115-160 LOC of additional conservative source cleanup, leaving roughly a 1,000 LOC gap to the current combined-source target; the budget criteria remain open rather than forcing contract churn.
- 2026-05-08 r16 — fixture-inventory completion audit:
  - Re-scanned Stage 2 seed candidates in both `/Users/suhjungdae/code/columbia/simsopt-surrogate` and sibling `/Users/suhjungdae/code/columbia/autoresearch`: 2,205 `biot_savart_opt.json` files inspected, 2,025 with adjacent `results.json`, 0 valid for the current non-dry frontier smoke contract.
  - Strict rejection buckets from the local checker: 1,911 missing `STAGE2_BS_SHA256`, 180 missing sidecar, 67 invalid/nonnegative TF current, 22 legacy upgrade failures from unsupported/mismatched metadata, 13 missing `CURVATURE_THRESHOLD`, 4 missing `POLOIDAL_EXTENT_THRESHOLD_RAD`, 3 invalid JSON sidecars, 3 LCFS-to-vessel clearance failures, and 2 init-only fixtures.
  - A second pass ignoring checksum still found 0 otherwise-contract-valid seeds, so this is not just a provenance backfill problem. Most legacy sidecars are missing/positive TF-current metadata, and the signed-current candidates fail current hard-contract fields such as clearance, required thresholds, HBT vessel radius, or init-only status.
  - Independent subagent inventory agreed: no smoke command exists until a new checksum-bound, non-init-only Stage 2 seed is generated under the current signed-current and hardware contracts. CC.3 / 1A.10 / DC.13 remain open for that external fixture gap.
- 2026-05-08 r17 — import-boundary closeout:
  - Closed DC.5 / Phase 2.2.4 by moving the search-contract helper implementation to `banana_opt/single_stage_search_contracts.py`; at this r17 checkpoint, `frontier_constraints.py` was a thin frontier-facing compatibility facade. r51 supersedes this note: the facade file is deleted.
  - Rechecked `rg "from \\.frontier_constraints|from banana_opt.frontier_constraints|import .*frontier_constraints" examples/single_stage_optimization -g '*.py'`: only `banana_opt/frontier_evaluator.py` imports the facade. The non-frontier single-stage entrypoint now imports `banana_opt.single_stage_search_contracts` directly.
- 2026-05-08 r18 — post-DC.5 validation:
  - Touched-file `py_compile` and `ruff check` passed after the search-contract move.
  - Focused post-move regressions passed: `geo.test_frontier_constraints`, `geo.test_frontier_evaluator`, and `geo.test_single_stage_alm_integration` ran 95 tests with `OK`.
  - Broader focused frontier/workflow suite passed after the facade split: 204 tests across `geo.test_frontier_archive`, `geo.test_frontier_contracts`, `geo.test_frontier_recommendation`, `geo.test_frontier_scalarization`, `geo.test_frontier_evaluator`, `geo.test_frontier_constraints`, and `geo.test_single_stage_workflow_helpers` ran in 6.760s with `OK`.
  - `git diff --check` passed in this repo, and `git -C /Users/suhjungdae/code/columbia/autoresearch diff --check -- program_hbt_topology_surrogate_legacy_vmec.md` passed for the sibling-doc update.
  - Current LOC after the facade split: frontier modules **6,113** LOC, runner **1,029** LOC, combined source **7,142** LOC, frontier tests/helper **3,415** LOC. DC.11/DC.12a/DC.15 remain open.
- 2026-05-08 r19 — LOC cleanup closeout:
  - Applied only behavior-preserving source simplifications after a source audit found the remaining conservative source reductions are tens of lines, not the 605 combined-source LOC still needed for DC.11c. Current measured source LOC: frontier modules **6,081** LOC, runner **1,024** LOC, combined source **7,105** LOC. DC.11a/b/c remain open rather than forcing contract churn.
  - Consolidated frontier test fixtures and table-driven cases while preserving algorithm-invariant coverage. Current measured frontier tests/helper LOC: **2,755**, so DC.12a is now closed.
  - Focused post-cleanup suite passed: 61 tests across `geo.test_frontier_archive`, `geo.test_frontier_contracts`, `geo.test_frontier_recommendation`, `geo.test_frontier_scalarization`, `geo.test_frontier_evaluator`, and `geo.test_frontier_constraints` ran with `OK`.
  - Broader post-cleanup frontier/workflow suite passed: 187 tests across the six frontier suites plus `geo.test_single_stage_workflow_helpers` ran with `OK`. Touched-file `py_compile`, scoped `ruff check`, and `git diff --check` also passed.
- 2026-05-08 r20 — final reviewer-loop validator fix:
  - A reviewer pass found one remaining stale-engine leak: manifest, summary, lane-contract, and lane-record validators still accepted the deleted `nsga3` engine even though progress payloads had already been hardened. The validators now reject unsupported frontier engines through `SUPPORTED_FRONTIER_ENGINES`, and resume-manifest loading validates before returning.
  - Added regression coverage for deleted-engine rejection in manifest, summary, lane-contract, lane-record, progress, and resume-manifest paths.
  - Targeted validator regressions passed: 11 tests across `geo.test_frontier_contracts` plus the two resume-engine workflow-helper tests ran with `OK`; touched-file `py_compile` and scoped `ruff check` passed.
  - Broader post-fix frontier/workflow suite passed: 189 tests across the six frontier suites plus `geo.test_single_stage_workflow_helpers` ran in 6.895s with `OK`.
  - Current measured source LOC after the validator hardening: frontier modules **6,108** LOC, runner **1,027** LOC, combined source **7,135** LOC. Current measured frontier tests/helper LOC: **2,795**. DC.11a/b/c remain open; DC.12a remains closed.
- 2026-05-08 r21 — completion-audit follow-up:
  - Re-ran the remaining local completion audit against the dirty tree rather than treating the focused green suite as full plan completion. Result: the implementation is substantial, but the plan is **not literally complete** because real non-dry smoke, source LOC, and final removable-LOC gates remain open.
  - Tried to generate a fresh non-init-only Stage 2 smoke seed with `run_stage2_alm.py --profile standard_80ka`. The first run failed because the default external equilibria path lacked `wout_nfp22ginsburg_000_014417_iota15.nc`; rerunning with `--equilibria-dir examples/single_stage_optimization/equilibria` reached the real preflight and failed the hard vessel-gap contract: best candidate `plasma_vessel_min_dist_m=0.003611` versus required `0.040000`.
  - Tried a stricter LCFS override (`--target-lcfs-max-major-radius-m 0.84 --target-lcfs-max-minor-radius-m 0.10`) without weakening validators. It also failed preflight: best candidate `plasma_vessel_min_dist_m=0.002431`, and the selected minor radius still exceeded the stricter minor-radius cap.
  - Probed several `/Users/suhjungdae/code/columbia/DATABASE/EQUILIBRIA` Ginsburg `wout` candidates at the same Stage 2 preflight contract; all sampled candidates failed `plasma_vessel_min_dist<0.040000`, with best gaps below `0.009 m`. This confirms the local smoke blocker is a hard geometry/fixture gap, not just a missing wrapper command.
  - Ran the optional type gate requested by 9.2.4. Plain scoped `mypy` failed before full checking on an untyped `simsopt` import plus duplicate module naming. Retrying with `MYPYPATH=examples/single_stage_optimization mypy --explicit-package-bases --ignore-missing-imports ...` found **291 errors in 25 files** across existing/transitive Stage 2, ALM, single-stage, and frontier modules; this repo slice is not mypy-clean today.
- 2026-05-08 r22 — strict non-dry smoke fixture follow-up:
  - Generated a real non-init-only Stage 2 smoke seed by keeping validators intact and shrinking only the LCFS target: `run_stage2_alm.py --profile standard_80ka --target-lcfs-max-major-radius-m 0.6 --equilibria-dir examples/single_stage_optimization/equilibria`. The artifact is checksum-bound (`STAGE2_BS_SHA256=f841898c71709f4af7f7d3706e5e06d7ee12ab19ec97d9a512ef1f503d6c9c75`), signed-current (`TF_CURRENT_A=-80000.0`), non-init-only, and hardware-clean (`coil_plasma_min_dist=0.073428 >= 0.015`, `plasma_vessel_min_dist=0.061179 >= 0.040`).
  - Ran the real non-dry `multilane_local` campaign smoke with that strict seed: `tmp/frontier_smoke_phase_check_r06_20260508_021909`. The runner exited 0 and emitted the expected non-dry summary/manifest/progress/archive/recommendation JSON shape (`dry_run=false`, `stage2_artifact_init_only=false`, `frontier_engine=multilane_local`, `frontier_num_lanes=3`, runtime defaults `lane_budget=10`, `total_budget=30`).
  - The strict seed is still not a passing single-stage frontier fixture: target baseline and all three lanes failed Boozer initialization (`self_intersecting=True`, solved iota about `-0.012338` from the default `0.15` guess), producing `frontier_feasible_lane_count=0` and `frontier_archive_size=0`.
  - A root-cause-preserving retry with the Stage 2 `surf_opt.vts` as `--stage2-seed-surf-path` failed because the warm-start loader only accepts simsopt/Boozer artifacts, not VTK `.vts` render outputs. A bounded donor-repair probe also failed before ranking because the seed surface went non-monotone during self-intersection checking. Therefore the wrapper/JSON smoke is now proven, but a certified-lane smoke remains open until a strict seed is also single-stage Boozer-bootable.
- 2026-05-08 r23 — completion-audit closeout:
  - Kept the certified-lane smoke gate open after root-cause-preserving fixture attempts failed. Stage 2 iota-aware ALM with the strict r06 geometry diverged during the Boozer/iota build, and probing old certified Boozer JSON plus the old surface-only JSON against the strict r06 Stage 2 seed failed with the same Newton/non-monotone self-intersection path. Older remembered 80 kA and autoresearch bridge/promoted seeds were rejected by the current strict contract (positive or missing TF-current metadata, missing checksum/hardware thresholds, or hard-constraint failures).
  - Closed the remaining algorithm-invariant test gaps: `test_achievement_scalarization_scaling_invariance` locks homogeneous goal/metric scaling preserving the selected member, and `test_epsilon_certifier_silent_pass_on_unknown_scalarization_type` locks the non-epsilon certification bypass as an explicit contract.
  - Focused invariant sweep passed: `geo.test_frontier_scalarization` + `geo.test_frontier_archive` ran 32 tests with `OK`; the broader frontier/workflow sweep ran 191 tests with `OK`; touched-file `py_compile`, scoped `ruff check`, and `git diff --check` passed. Current measured frontier tests/helper LOC is **2,799**, so DC.12a remains closed.
- 2026-05-08 r24 — reviewer-loop and Stage 2 handoff follow-up:
  - Reviewer loop found one remaining deleted-field acceptance path: summary validation rejected the deleted engine but still accepted the six deleted NSGA3-era summary fields if present. Added a `DELETED_FRONTIER_SUMMARY_FIELDS` rejection guard and regression coverage; these field names are now allowed only in the fail-fast guard/tests, not as accepted payload fields.
  - Removed the last Python `nsga3` test fixture strings by replacing legacy rejection examples with a generic unsupported-engine value. `rg -n "nsga3" examples/single_stage_optimization/banana_opt tests/geo -g '*.py'` now returns zero hits.
  - Stage 2 now writes a loadable `surf_opt.json` next to `surf_opt.vts`; the single-stage handoff fits the Stage 2 surface geometry when the seed surface Fourier order differs from the target order instead of assigning mismatched DOF arrays directly.
  - Regenerated the strict r06 Stage 2 seed with `surf_opt.json` and retried the real frontier smoke with `--stage2-seed-surf-path`. This removed the prior `DofLengthMismatchError` (`99` seed DOFs vs `331` target DOFs): target baseline and lane 01 reached BFGS (`iota=0.146755...`) before the Boozer Newton solve diverged (`iota=-33544...`) and `is_self_intersecting()` raised the non-monotone cross-section error. The run was stopped after the duplicate lane-01 failure path; no certified member or final summary/archive/recommendation JSON was produced, so the certified-lane smoke gate remains open.
  - Current measured LOC: frontier modules **6,123**, runner **1,027**, combined source **7,150**, frontier tests/helper **2,797**. DC.11 remains open; DC.12a remains closed. Independent LOC review found only ~115-160 lines of conservative additional source cleanup, not enough to meet the source gates without removing contract surface.
  - Validation after r24 changes: targeted handoff/validator/Stage 2 artifact regressions ran 5 tests with `OK`; post-trim contract regressions ran 2 tests with `OK`; touched-file `py_compile`, scoped `ruff check`, and the zero-NSGA3 grep passed; the broader frontier/Stage 2 regression slice ran 305 tests with `OK`.
- 2026-05-08 r25 — completion-audit pause update:
  - Closed CC.7 as not applicable to this checkout: no `STATE.md`, `state.json`, or lab-note file exists under the repo root, and the pause/blocker state is recorded in this plan.
  - Closed DC.15 for the behavior-preserving audit scope: the independent LOC review found only ~115-160 lines of conservative remaining cleanup, below 5% of the current **7,150** combined source LOC. This does **not** close DC.11 because the numeric source LOC targets still require approved contract/feature removal.
- 2026-05-08 r26 — orphaned evaluator retirement:
  - Removed unused `banana_opt/frontier_evaluator.py` and `tests/geo/test_frontier_evaluator.py` after live grep proved no production importer remained; only `frontier_contracts.py` keeps `frontier_evaluator_spec` / `_path` as deleted-field rejection keys.
  - Updated the superseded frontier global-pareto and gradient-contract docs so their evaluator/spec tasks are explicitly historical.
  - Current measured LOC after the evaluator retirement plus the CLI/docstring clarification: frontier modules **4,889**, runner **1,033**, combined source **5,922**, frontier tests/helper **2,173**. DC.11a, DC.11c, DC.12a, and DC.16 are closed; DC.11b remains open because the runner is still **83 LOC** over the 950-line threshold (**84** lines must be removed to satisfy the strict `< 950` gate), and the remaining large cuts require approved runner contract/feature removal.
  - Validation after r26: the broader frontier/Stage 2 regression slice ran **296 tests** with `OK`; scoped `ruff check`, touched-scope `compileall`, zero-NSGA3 grep, evaluator-deletion grep, and `git diff --check` passed. After the final help/docstring clarification, touched-file `ruff check` and `py_compile` passed. The evaluator grep now reports only the intended deleted-field guard keys in `frontier_contracts.py`.
- 2026-05-08 r27 — runner-extraction LOC closeout:
  - Moved behavior-preserving lane execution setup helpers from `run_single_stage_frontier_campaign.py` into `banana_opt/frontier_campaign_execution.py` while keeping runner-level compatibility re-exports for existing tests and callers.
  - Current measured LOC after the extraction: frontier modules **5,163**, runner **795**, combined source **5,958**, frontier tests/helper **2,173**. DC.11a, DC.11b, DC.11c, and DC.12a are all closed.
  - Validation after r27: the broader frontier/Stage 2 regression slice ran **296 tests** with `OK`; scoped `ruff check`, touched-scope `compileall`, zero-NSGA3 grep, evaluator-deletion grep, and `git diff --check` passed. The evaluator grep still reports only the intended deleted-field guard keys in `frontier_contracts.py`.
- 2026-05-08 r28 — certified-lane blocker refinement:
  - Rechecked the prior "self-intersection may be API mismatch" memory against the current tree. `src/simsopt/geo/surface.py` already calls `contour_self_intersects(contour, context=context)` and falls back to the older one-argument call on `TypeError`, and `tests/geo/test_surface.py` covers the installed `ground` / `bentley_ottmann` API seam.
  - Loaded the strict r06 `surf_opt.json` directly and verified `is_self_intersecting()` is `False` for default sampling, `thetas=32`, and `thetas=200`. Therefore the remaining certified-lane smoke blocker is not a raw saved-surface self-intersection or intersection-library API mismatch.
  - The open failure remains after Boozer initialization mutates/solves the surface: the saved campaign wrapper JSON still has `dry_run=false`, `frontier_feasible_lane_count=0`, and `frontier_archive_size=0`. A direct bootability probe against the same strict seed did not complete within the local bounded probe window and was stopped; this does not close CC.3 / 1A.10 / DC.13.
  - Checked the likely volume-mismatch lead: the strict r06 Stage 2 seed has `FINAL_VOLUME=0.02171447850018657`, while the default smoke command uses `--vol-target 0.1`. Re-running the bootability probe with `vol_target` set to the seed's `FINAL_VOLUME` still did not produce a bootable result within 180 seconds; the probe returned `BOOTABILITY_REASON=boozer_solve_failed`, `BOOTABILITY_ERROR_TYPE=TimeoutError`, and no solved iota.
- 2026-05-08 r29 — legacy certified-artifact rejection:
  - Inspected the two older certified-looking frontier v4 runs (`tmp/frontier_v4_canonical_014417_20260414_run3` and `tmp/frontier_v4_canonical_002084_20260414_run3`) against the current strict seed contract. Their Stage 2 sidecars are legacy and not checksum/schema-bound (`STAGE2_BS_SHA256=None`, `CONTRACT_SCHEMA_VERSION=None`) and omit current required fields such as `TF_CURRENT_A`, `TF_CURRENT_SUM_ABS_A`, `NUM_TF_COILS`, `POLOIDAL_EXTENT_RAD`, `POLOIDAL_EXTENT_THRESHOLD_RAD`, and final LCFS radii.
  - Loading their `biot_savart_opt.json` files showed the first 20 TF coil currents are `+100000.0 A`, violating the current signed CW convention (`TF_CURRENT_A` must be negative and `>= -80000 A`). The sidecars also carry off-contract geometry: `MAJOR_RADIUS=0.915` rather than the current HBT vessel radius `0.976`, `banana_surf_radius=0.22`, and `COIL_LENGTH=2.909678...` / `2.930551...` above the current `2.0 m` hard limit.
  - Therefore those old certified lanes cannot be promoted by checksum or metadata backfill without weakening provenance/hardware/current contracts. The remaining certified-lane blocker is still: generate or locate a current strict, checksum-bound, signed-current, hardware-clean Stage 2 seed that is also single-stage Boozer-bootable.
- 2026-05-08 r30 — harvested-seed and warm-start audit:
  - Rechecked the remembered non-init-only 80 kA Stage 2 seed at `tmp/full_sweep_runs/20260408-150301/repro_stage2/.../biot_savart_opt.json`. It is not usable for current smoke: `STAGE2_BS_SHA256=None`, `CONTRACT_SCHEMA_VERSION=None`, `TF_CURRENT_A=+80000.0`, loaded TF coils are `+80000.0 A`, `MAJOR_RADIUS=0.915`, `banana_surf_radius=0.22`, and required poloidal/LCFS/current hardware fields are missing.
  - Inspected `/Users/suhjungdae/code/columbia/autoresearch/harvested_seeds/R_nv2_iota305_hbtclean_2026-04-23`. It is a historically Boozer-bootable surface fixture, but its Stage 2 artifact is still not a current strict seed: no checksum/schema binding, sidecar and loaded TF coils are `+80000.0 A`, and required poloidal/LCFS fields are missing.
  - Ran a current-contract bootability probe using the strict r06 Biot-Savart artifact plus the R_nv2 warm-start surface (`surf_opt.json`) and Boozer-surface (`surf_opt_boozer_surface.json`) at native `mpol=10`, `ntor=10`, reduced `nphi=31`, `ntheta=16`, `vol_target=0.03992103666310159`, and `iota_target=0.3048386265857192`. Both probes timed out under the 180 s cap after the BFGS solve stalled at `iota=-0.0006158844048902357`, returning `BOOTABILITY_REASON=boozer_solve_failed`. This rules out using the bootable legacy surface as a warm start for the strict r06 field.
  - A parallel read-only scan of 74 immediate harvested-seed `biot_savart_opt.json` candidates found zero current strict candidates. First hard rejects: 31 missing checksum binding, 37 checksum-bound but positive TF current, and 6 checksum-bound/signed/R0-valid basinhop seeds missing required current/hardware fields; those 6 also fail loaded-current provenance because their serialized TF coils are actually `+80000.0 A`.
- 2026-05-08 r31 — strict LCFS-radius bootability sweep:
  - Tried generating looser Stage 2 seeds from the local `wout_nfp22ginsburg_000_014417_iota15.nc` equilibrium. `--target-lcfs-max-major-radius-m 0.65` failed plasma-vessel preflight with `plasma_vessel_min_dist_m=0.003975 < 0.040000`; `0.62` failed with `0.038110 < 0.040000`.
  - Generated the largest nearby passing strict seed found at `--target-lcfs-max-major-radius-m 0.618` under `tmp/stage2_frontier_bootability_sweep_m0618/...`: `STAGE2_BS_SHA256=b88a38e18e4fd0b1d2922ab106b9c1ec4b53807312904b6f925b286807d98486`, `TF_CURRENT_A=-80000.0`, `SURFACE_VESSEL_MIN_DIST=0.04041529711081896`, `CURVE_SURFACE_MIN_DIST=0.05288807323380542`, `COIL_LENGTH=0.5267124867788195`, `MAX_CURVATURE=92.14049365708102`, `FINAL_VOLUME=0.023727996948073433`, and `FINAL_LCFS_MAJOR_RADIUS_M=0.6180000000000105`.
  - A direct bootability probe with the matching seed volume and strict `surf_opt.json` timed out under the 180 s cap and returned `BOOTABILITY_REASON=boozer_solve_failed`, `BOOZER_BOOTABLE=false`, `BOOTABILITY_TARGET_IOTA=0.15`, and no solved iota. This narrows the local fixture gap: the strict LCFS window for this equilibrium is tiny (`0.618` passes the vessel margin by about `0.000415 m`; `0.620` fails by about `0.001890 m`), and the largest passing candidate found still is not single-stage Boozer-bootable. CC.3 / 1A.10 / DC.13 remain open pending a new strict seed, a different equilibrium, or upstream solver/fixture work.
- 2026-05-08 r32 — alternate local test-fixture sweep and reviewer PASS:
  - Reviewer-agent pass returned `PASS` on the current diff after checking the NSGA3/evaluator deletion, multilane-only execution, Stage 2 surface handoff, deleted summary-field guard, helper extraction, and tests. The reviewer also ran `PYTHONPATH=tests python -m unittest geo.test_frontier_contracts geo.test_single_stage_workflow_helpers geo.test_stage2_single_stage_handoff` (`Ran 176 tests ... OK`) plus a touched-scope compile check, and classified the certified-lane smoke gap and external-team confirmation as blockers rather than code regressions.
  - Coarse preflight over local `tests/test_files/wout*.nc` found several possible alternate VMEC fixtures, but production checks eliminated them as certified smoke inputs. `wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc` failed production geometry preflight at `--target-lcfs-max-major-radius-m 0.62` with `plasma_vessel_min_dist_m=0.001437 < 0.040000`.
  - `wout_circular_tokamak_aspect_100_reference.nc` generated a strict signed-current, checksum-bound, hardware-clean Stage 2 seed at `--target-lcfs-max-major-radius-m 0.918` (`STAGE2_BS_SHA256=77e79c235c621482b1cc5a76a9e87ad3562c5a23a93cbd603294f4d7c99a25dc`, `FINAL_VOLUME=0.00036651569153188333`, `SURFACE_VESSEL_MIN_DIST=0.15482000000003726`, `CURVE_SURFACE_MIN_DIST=0.1428249230109064`). It is not usable for the certified smoke: at the default `iota_target=0.15`, the probe solved near zero iota then timed out; at reduced quadrature with `iota_target=0.0`, the Boozer solve converged but the initialized surface was self-intersecting both with and without `surf_opt.json`.
  - `wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc` generated a strict signed-current, checksum-bound, hardware-clean Stage 2 seed at `--target-lcfs-max-major-radius-m 0.62` (`STAGE2_BS_SHA256=c3b224a0d69fd59cb8961aa93385f91c9bb42882c832f951af28572451c620f0`, `FINAL_VOLUME=0.010668713332171498`, `SURFACE_VESSEL_MIN_DIST=0.055705802745175115`, `CURVE_SURFACE_MIN_DIST=0.07537203181402635`), but its default-target bootability probe timed out with `BOOTABILITY_REASON=boozer_solve_failed`. This exhausts the productive local fixture attempts found in the repo without weakening strict current/hardware/provenance contracts.
- 2026-05-08 r33 — final local-dispatch cleanup:
  - Replaced `generate_frontier_lane_specs` mode if-chain with a frozen `MappingProxyType` dispatch registry and a typed immutable request object in `frontier_scalarization.py`.
  - Added a scalarization test that the dispatch registry covers exactly `SUPPORTED_FRONTIER_REFERENCE_MODES` and is immutable.
  - Closed remaining conditional DROP-path checklist items as not applicable rather than leaving them as active work. The profile-name CLI is retained because D3 preserved runtime-calibration JSON/profile shape; there is no `_resolve_calibration_profile` symbol left to delete.
  - Validation after r33: `PYTHONPATH=tests python -m unittest -q geo.test_frontier_scalarization geo.test_frontier_contracts` ran 22 tests with `OK`; the broader frontier/workflow slice ran 297 tests with `OK`; touched-file `compileall`, scoped `ruff check`, and `git diff --check` passed.
- 2026-05-08 r34 — expanded local equilibrium screen:
  - Ran the Stage 2 geometry preflight over 266 local `wout*.nc` candidates from `/Users/suhjungdae/code/columbia/DATABASE/EQUILIBRIA`, `examples/single_stage_optimization/DATABASE/EQUILIBRIA`, and `tests/test_files` at coarse quadrature before any optimizer/probe work. Only `tests/test_files/wout_circular_tokamak_aspect_100_reference.nc` passed the strict HBT LCFS-size and plasma-vessel clearance gate; representative rejected Ginsburg families topped out at `target_lcfs_major_radius_m=0.820000`, `lcfs_minor_radius_m=0.127228`, and `plasma_vessel_min_dist_m=0.009990 < 0.040000`.
  - Rechecked the sole passing circular fixture at production preflight quadrature (`nphi=91`, `ntheta=32`): `gap=0.156800000000`, `target_lcfs_major_radius_m=0.920000000000`, `lcfs_minor_radius_m=0.009200000000`, `s_working=0.240000000000`, `nfp=1`.
  - Ran `run_stage2_to_single_stage.py --probe-only` against the existing strict circular m0918 artifact under `tmp/stage2_frontier_bootability_probe_circular_r34`. Boozer solved, but the candidate remains unusable for certified smoke: `BOOZER_BOOTABLE=false`, `IOTA_FEASIBLE=false`, `BOOTABILITY_REASON=self_intersection`, `BOOTABILITY_SOLVED_IOTA=0.14285863522375472`, `BOOTABILITY_TARGET_IOTA=0.15`, `BOOTABILITY_ABS_IOTA_ERROR=0.007141364776245274`, and `BOOTABILITY_SELF_INTERSECTING=true`.
- 2026-05-08 r35 — completion-audit LOC refresh:
  - Refreshed the live `wc -l` evidence for the completion audit: frontier modules **5,235**, runner **795**, combined source **6,030**, frontier tests/helper **2,186**. All numeric LOC gates remain closed.
- 2026-05-08 r36 — blocked-input summary:
  - Added a top-level blocked-input summary so the remaining open checklist items are visible before the long audit trail. No implementation or validation status changed.
- 2026-05-08 r37 — final completion-audit refresh:
  - Rechecked the prompt-to-artifact closure state: **189** checklist items total, **185** closed, **4** open. The open boxes are `D1.4`, `1A.10`, `CC.3`, and `DC.13`; these collapse to the two top-level blockers listed above.
  - Fresh current-tree validation still passes: `git diff --check`; sibling autoresearch doc `diff --check`; `PYTHONPATH=tests python -m unittest -q geo.test_frontier_scalarization geo.test_frontier_contracts` (`22` tests, `OK`); broader frontier/workflow slice (`297` tests, `OK`).
  - Fresh artifact audit found no newer strict certified-lane evidence after this plan file. The latest real non-dry wrapper smoke at `tmp/frontier_smoke_phase_check_r06_20260508_021909/single_stage_frontier_campaign_summary.json` remains `dry_run=false`, `frontier_engine=multilane_local`, `stage2_artifact_init_only=false`, but `frontier_archive_size=0` and `frontier_feasible_lane_count=0`. The circular r34 bootability probe still reports `BOOZER_BOOTABLE=false`, `IOTA_FEASIBLE=false`, `BOOTABILITY_REASON=self_intersection`.
- 2026-05-08 r38 — blocker-closure clarification:
  - Added explicit closure criteria for the two remaining blockers so future audits do not treat wrapper JSON shape, tests, or failed seed probes as certified-lane evidence. No checklist item was closed.
- 2026-05-08 r39 — next-input handoff:
  - Added exact D1.4 team-confirmation wording and a `jq -e` acceptance check for the certified-lane smoke summary. No checklist item was closed.
- 2026-05-08 r40 — local smoke/archive sweep:
  - Rechecked every local `tmp/*/single_stage_frontier_campaign_summary.json`. All four current non-dry `multilane_local` summaries remain non-closing evidence: `tmp/frontier_bloat_reduction_smoke_20260508_real`, `tmp/frontier_smoke_phase_check_r06_20260508_021909`, `tmp/frontier_smoke_phase_check_r06_jsonseed_20260508`, and `tmp/frontier_smoke_phase_check_r06_seed_surf_20260508_022301` all report `frontier_feasible_lane_count=0` and `frontier_archive_size=0`.
  - Rechecked every local `tmp/*/frontier_archive.json`. The only archives with certified members are older `frontier_v4_canonical_*_20260414*` runs already rejected by the current strict seed contract; all current bloat-reduction smoke archives have zero members. No checklist item was closed.
- 2026-05-08 r41 — adjacent artifact-tree sweep:
  - Filtered **1,580** Stage 2 `outputs-*/*/results.json` files under `simsopt-surrogate` and adjacent `autoresearch` for the strict seed metadata gate (`CONTRACT_SCHEMA_VERSION`, `STAGE2_BS_SHA256`, `TF_CURRENT_A < 0`, `HARDWARE_CONSTRAINTS_OK`, `init_only != true`). Only five passed, all already documented local candidates: r06, r06 JSON seed, m0618, circular m0918, and W7-X m062. Zero adjacent `autoresearch` Stage 2 results passed this metadata gate.
  - A parallel explorer broadened the scan to **2,212** `results.json` files and found zero artifacts satisfying strict metadata plus Boozer/certified evidence. Its only adjacent `autoresearch` `multilane_local` archive tree, `/Users/suhjungdae/code/columbia/autoresearch/runs/goalmode_cmp_2026-04-22/frontier_lane_sweep`, rejects with `frontier_archive_size=0`, `frontier_feasible_lane_count=0`, three lanes, and zero certified lanes/archive members.
  - Closest adjacent `autoresearch` strict-current candidates reject before bootability: `runs/loop_2026-04-27/stage2_seed_clean_014417_TFcw/results.json` has checksum/schema/signed-current/non-init metadata but `HARDWARE_CONSTRAINTS_OK=false`, `SURFACE_VESSEL_MIN_DIST=0.001464069604103513 < 0.04`, and over-limit LCFS radii (`1.0421007071468398`, `0.1616883628523708`); `runs/loop_2026-04-27/laneA_stage2_pen_freshCW_014417_lt170_lw0p001/.../results.json` also has `CURVE_CURVE_MIN_DIST=0.04968765831843778 < 0.05`, `POLOIDAL_EXTENT_RAD=0.8961558319206485 > 0.7853981633974483`, and the same vessel/LCFS failures. No checklist item was closed.
- 2026-05-08 r42 — local D1 artifact/doc sweep:
  - Queried **56** local and adjacent campaign artifacts (`single_stage_frontier_campaign_summary.json`, `campaign_manifest.json`, `campaign_progress.json`, `frontier_archive.json`) for `frontier_engine`/`engine` values containing `nsga`; found zero artifact matches. This strengthens the local `DROP` evidence but does not replace D1.4 external confirmation.
  - The broader text scan found one adjacent stale consumer doc, `/Users/suhjungdae/code/columbia/autoresearch/program_cw_poloidal_legacy_vmec.md:310`, that still recommended `--frontier-engine nsga3`. Updated it to match the HBT program note: NSGA3 was dropped on 2026-05-07 after no validated production artifacts were found; use `multilane_local` plus `--frontier-reference-mode achievement_chebyshev_full_simplex_v1`. No checklist item was closed.
- 2026-05-08 r43 — historical lab-note classification:
  - Rechecked remaining adjacent `--frontier-engine nsga3` text hits after r42. The active program docs now describe the deleted flag only as historical/superseded guidance. The remaining recommending hit is an older generated lab journal entry (`lab_note.md` line 601, mirrored from `lab_notes.jsonl`), and `lab_note.md` explicitly says it is generated from the JSONL journal and must not be edited directly. Left the historical journal record unchanged rather than rewriting append-only provenance. No checklist item was closed.
- 2026-05-08 r44 — active-doc stale-command tightening:
  - Reworded the active adjacent `autoresearch` program notes (`program_hbt_topology_surrogate_legacy_vmec.md:310`, `program_cw_poloidal_legacy_vmec.md:310`) so they no longer contain the exact deleted invocation `--frontier-engine nsga3`; both now refer to the dropped NSGA3 engine path in prose and keep the current `multilane_local` / `achievement_chebyshev_full_simplex_v1` guidance. Remaining exact deleted-command hits are confined to historical lab-journal provenance. No checklist item was closed.
- 2026-05-08 r45 — active-plan wording drift cleanup:
  - Corrected the live scalarization total key in F23 / 9b.6 / DC.18 from stale `frontier_chebyshev_total` to the implemented `frontier_scalarization_total`.
  - Corrected the Phase 2 dependency note: `frontier_scalarization.py` imports the leaf `objective_gradients.py`; `single_stage_objectives.py` no longer imports frontier scalarization. No checklist item was closed.
- 2026-05-08 r46 — completion-audit snapshot:
  - Added the prompt-to-artifact audit table above. It records that code/test/doc gates are locally complete but final closeout remains blocked by external D1.4 confirmation and a strict certified-lane smoke artifact. No checklist item was closed.
- 2026-05-08 r47 — live completion-audit refresh:
  - Rechecked the prompt-to-artifact closure state against the current dirty tree: **189** checklist items total, **185** closed, **4** open (`D1.4`, `1A.10`, `CC.3`, `DC.13`). The open boxes still collapse to external NSGA3-use confirmation plus a strict certified-lane smoke artifact.
  - Current LOC evidence remains within all gates: frontier modules **5,261** LOC, runner **795** LOC, combined source **6,056** LOC, frontier tests/helper **2,264** LOC.
- 2026-05-08 r48 — plan-vs-reality reconciliation (Option B taken — doc-only fix, no code change):
  - **B1 (8.2.2 calibration registry collapse):** flipped from `[x]` to `[~]` (DEFERRED) and added DC.21 follow-up. Reason: the cross-agent r6.4 audit verified that `FRONTIER_RUNTIME_CALIBRATION_PROFILES` is still a 2-profile registry at `frontier_runtime_calibration.py:58-91`, with neither `FrontierRuntimeDefaults` nor `resolve_runtime_defaults(args)` present. Working-tree state is consistent with D3 = "preserve shape" (8.2.4 execution note already records this), but the `[x]` mark on 8.2.2 was incorrect because the rewrite-as-dataclass step never happened. r48 documents the deferral honestly rather than back-fitting code to a stale plan promise.
  - **B2 (9b.1.1 multilane rename):** flipped from `[x]` to `[~]` (DOWNGRADED to Option C). Reason: the audit verified the constant `FRONTIER_REFERENCE_MODE_SHARED = "shared_seed_relative_frontier_v2"` is unchanged at `frontier_scalarization.py:24`; only docstrings were updated. The plan task title was "Option A — preferred — rename"; the actual outcome is Option C (docstring-only), so the prior `[x]` claim that Option A was selected was wrong. r48 records the downgrade plus the schema-version compatibility rationale.
  - **No code change in r48.** All edits are inside `docs/frontier_mode_bloat_reduction_todo_plan_2026-05-07.md` only. The 4 original blockers (D1.4, 1A.10, CC.3, DC.13) are unchanged; the 185 closed items are unchanged except 8.2.2 → `[~]` and 9b.1.1 → `[~]`. Net checkbox count: **183 closed `[x]`, 2 deferred `[~]`, 5 open `[ ]`** (the 4 original blockers + new DC.21 deferred-collapse follow-up).
  - **Other minor narrative-drift items left in place** (B3-B7 from the validation report were structural mismatches where work landed slightly differently than planned but functionally complete — those carry no `[~]` flip because they don't represent un-done work). Plan-line-number drift (D1-D6 in the validation report) is cosmetic only and not corrected here.
  - Rechecked the prompt-to-artifact closure state against the current dirty tree: **189** checklist items total, **185** closed, **4** open (`D1.4`, `1A.10`, `CC.3`, `DC.13`). The open boxes still collapse to external NSGA3-use confirmation plus a strict certified-lane smoke artifact.
  - Current LOC evidence remains within all gates: frontier modules **5,261** LOC, runner **795** LOC, combined source **6,056** LOC, frontier tests/helper **2,264** LOC.
  - Fresh validation passed: `git diff --check`; touched-scope `compileall`; touched-scope `ruff check`; and `PYTHONPATH=tests python -m unittest -q geo.test_frontier_archive geo.test_frontier_contracts geo.test_frontier_recommendation geo.test_frontier_scalarization geo.test_frontier_constraints geo.test_single_stage_workflow_helpers geo.test_stage2_single_stage_handoff geo.test_single_stage_alm_integration` (`Ran 300 tests ... OK`).
  - Rechecked all local `tmp/*/single_stage_frontier_campaign_summary.json`: every current non-dry `multilane_local` summary still has `frontier_feasible_lane_count=0` and `frontier_archive_size=0`. No checklist item was closed.
- 2026-05-08 r49 — DC.21 WONTFIX closure:
  - Closed `DC.21` via option (b): D3 = "preserve shape" remains in force, so the registry is intentionally retained for JSON-schema stability. Added a one-line WONTFIX comment at `examples/single_stage_optimization/banana_opt/frontier_runtime_calibration.py:58` referencing DC.21 and the r48 deferral. The collapse remains a future-work hook tied to any future `_v3` schema bump.
  - **Net checkbox count after r49:** **184 closed `[x]`, 2 deferred `[~]`, 4 open `[ ]`**. The 4 open boxes are unchanged (`D1.4`, `1A.10`, `CC.3`, `DC.13`).
  - Fresh validation passed after the comment-only edit: targeted frontier/workflow regression slice ran **301** tests with `OK` (`PYTHONPATH=tests python -m unittest -q geo.test_frontier_archive geo.test_frontier_contracts geo.test_frontier_recommendation geo.test_frontier_scalarization geo.test_frontier_constraints geo.test_single_stage_workflow_helpers geo.test_stage2_single_stage_handoff geo.test_single_stage_alm_integration`); touched-scope `compileall` PASS; touched-scope `ruff check` PASS; `git diff --check` PASS; zero-NSGA3 grep PASS; evaluator-import grep returns only the intended deleted-field rejection keys in `frontier_contracts.py`. Superseded as current validation evidence by r51.
- 2026-05-08 r50 — `--stage2-iota-mode=alm` decision-gate probe:
  - Closed the only previously untried mitigation for `1A.10`/`CC.3`/`DC.13` by running the bootability decision-gate referenced in `autoresearch/program_cw_poloidal_legacy_vmec.md:137` ("Production default: keep `--stage2-iota-mode=off` on production lanes until a decision-gate benchmark says otherwise"). Two strict-contract `--stage2-iota-mode=alm` Stage 2 generations were attempted against the strict r06 frame (`--profile standard_80ka`, `wout_nfp22ginsburg_000_014417_iota15.nc`, `--target-lcfs-max-major-radius-m 0.6`, `--stage2-iota-target 0.15`):
    - **Probe A** at `tmp/stage2_alm_iota_probe_r49_20260508_224154/` with `--stage2-iota-vol-target 0.1` — Stage 2 inner subprocess raised `cross_section()` self-intersection during `attempt_initialize_boozer_surface` outer-surface bootstrap (`stage2_objectives.py:462` → `stage2_single_stage_handoff.py:727,588` → `simsopt/geo/surface.py:476,433`). Exit 1 before any iota-ALM optimization step ran.
    - **Probe B** at `tmp/stage2_alm_iota_probe_r49_lowvol_20260508_224514/` with `--stage2-iota-vol-target 0.025` (matched to r06's `FINAL_VOLUME=0.0217`) — same self-intersection at the same construction site. Exit 1 before any iota-ALM optimization step ran.
  - Both failures are at the **bootstrap Boozer outer-surface construction**, not in the iota optimizer. The Stage 2 basinhopping coil set produces an iota near zero, and lifting that to iota=0.15 at vol-target ≥ 0.025 with strict R0=0.976 + TF=-80kA forces a non-monotone-cross-section surface — i.e., the strict frame's available iota authority cannot sustain the requested target geometry. This matches the 2026-04-22 lab-note physics analysis (`lab_notes.jsonl` entry `ff82787639bb449684a35fbcf455fd6e`) cited by the bootability investigator.
  - **Outcome:** the decision-gate benchmark has now run and failed twice — the un-run `outputs_stage2_iota_decision_gate/` directory is implicitly populated with negative evidence. **`1A.10` / `CC.3` / `DC.13` remain open as documented**, but the close criteria are now narrowed to **explicit external action**:
    1. Commission a new equilibrium fixture matched to the strict R0=0.976 banana geometry whose physical iota at vol≈0.05-0.10 is close to 0.15 (so the bootstrap surface stays monotone), OR
    2. Approve relaxing the strict contract to allow R0=0.915 lineage (the only Stage 2 generation lineage on disk that has produced certified frontier members; rejected r29 because it violates the current signed-CW + radius contract), OR
    3. Approve a smaller frontier `iota-target` matched to what the strict frame can naturally sustain (~0 to ~0.05) — but this would invalidate the existing frontier-campaign goal definitions and require a frontier-contract amendment.
  - Each of the three close-criterion options is a **product/contract decision plus optionally upstream physics work**, not a frontier-bloat-reduction code change. The Phase 1A-7 implementation work is complete and the certified-lane smoke is gated on a fixture that does not exist locally and cannot be synthesized within this plan's scope.
  - **No checklist item closed in r50.** The 4 open blockers are unchanged. Plan-implementation status is unchanged. Validation evidence is unchanged.
- 2026-05-09 r51 — current-code validation refresh:
  - Fixed current-tree validation drift found during review: stale ALM contract citation ranges in `docs/alm_hybrid_signal_contract_2026-05-08.md` / `tests/geo/test_alm_utils.py`, and the runner/progress-state compatibility exports required by current tests.
  - Reconciled current-tense `frontier_constraints.py` wording after the facade file was deleted. The active implementation is now `single_stage_search_contracts.py`; the old test helper name remains only as a compatibility loader alias.
  - Fresh validation passed: `git diff --check`; touched-scope `compileall`; touched-scope `ruff check`; `PYTHONPATH=tests python -m unittest -q -b geo.test_frontier_archive geo.test_frontier_contracts geo.test_frontier_recommendation geo.test_frontier_scalarization geo.test_frontier_constraints geo.test_single_stage_workflow_helpers geo.test_stage2_single_stage_handoff geo.test_single_stage_alm_integration geo.test_alm_utils` (`Ran 467 tests ... OK`); and repo `./run_tests` (`Ran 1802 tests in 1408.073s`, `OK (skipped=94)`). No checklist item closed; the 4 open blockers are unchanged.
- 2026-05-09 r52 — post-handoff-fix strict smoke refresh:
  - Re-ran a bounded current-code non-dry `multilane_local` smoke after the committed `Surface.save()` / `gamma()` optional-handoff fix (`3372ba715`) to ensure the certified-lane blocker was not stale pre-fix evidence.
  - First rerun without `--equilibria-dir` failed at file discovery because `/Users/suhjungdae/code/columbia/DATABASE/EQUILIBRIA/wout_nfp22ginsburg_000_014417_iota15.nc` is absent; this was setup noise, not frontier evidence.
  - Corrected rerun with `--equilibria-dir examples/single_stage_optimization/equilibria`, `--skip-target`, `--frontier-num-lanes 1`, and `--frontier-lane-budget 10` reached Boozer initialization and failed with `self_intersecting=True`, `volume=0.09994554154987402`, `iota_guess=0.15`, and `iota_solved=-0.012338462725657728`. The wrapper exited 0 and wrote `tmp/frontier_smoke_r51_postfix_equilibria_20260509/single_stage_frontier_campaign_summary.json` with `dry_run=false`, `stage2_artifact_init_only=false`, `frontier_feasible_lane_count=0`, and `frontier_archive_size=0`.
  - No checklist item closed. This confirms the current-code post-handoff-fix blocker is still strict-seed Boozer bootability/certification, not a stale missing-surface handoff bug.
- 2026-05-09 r53 — strongest available JSON-surface smoke refresh:
  - Re-ran the strongest local strict path on current code: `--stage2-bs-path tmp/stage2_generation_for_frontier_smoke_r06_jsonseed/.../biot_savart_opt.json`, `--stage2-seed-surf-path tmp/stage2_generation_for_frontier_smoke_r06_jsonseed/.../surf_opt.json`, `--equilibria-dir examples/single_stage_optimization/equilibria`, `--skip-target`, one lane, lane budget 10, and `--single-stage-timeout-seconds 300`.
  - Lane 01 timed out after 300 seconds inside the single-stage subprocess. The wrapper exited 0 and wrote `tmp/frontier_smoke_r53_jsonseed_postfix_skiptarget_20260509/single_stage_frontier_campaign_summary.json` with `dry_run=false`, `stage2_artifact_init_only=false`, `frontier_engine=multilane_local`, `frontier_feasible_lane_count=0`, `frontier_archive_size=0`, no archive members, and lane `error_type=TimeoutExpired`.
  - Two parallel current-tree audits found no additional implementable plan item and no viable current certified-lane artifact. No checklist item closed; this is additional negative evidence for the same external fixture/contract blocker.
- 2026-05-09 r54 — latest frontier-code update check:
  - Re-audited the current frontier diff after the code update. The implementation remains aligned with the bloat-reduction plan: progress archive arrays are not persisted, recommendation gates no longer fall back to unsafe members, `frontier_constraints.py` has no production imports, and NSGA3 references are confined to runner help text plus deleted-field rejection keys.
  - Fresh scoped validation passed: `git diff --check`; scoped `ruff check`; touched-scope `compileall`; stale NSGA3/frontier-constraints/fallback grep; `PYTHONPATH=tests python -m unittest -q -b geo.test_frontier_archive geo.test_frontier_contracts geo.test_frontier_recommendation geo.test_frontier_scalarization geo.test_frontier_constraints geo.test_single_stage_workflow_helpers geo.test_stage2_single_stage_handoff geo.test_single_stage_alm_integration geo.test_alm_utils` (`Ran 467 tests in 5.699s`, `OK`); and a dry-run frontier shape probe confirmed progress JSON omits `archive_members` and `provisional_archive_members`.
  - No checklist item closed. Completion audit remains **184 closed `[x]`, 2 deferred `[~]`, 4 open `[ ]`**. The open boxes are still `D1.4`, `1A.10`, `CC.3`, and `DC.13`.
- 2026-05-09 r55 — completion-audit refresh after current-code check:
  - Recounted the checklist against the live file: **184 closed `[x]`, 2 deferred `[~]`, 4 open `[ ]`**. The open boxes remain `D1.4`, `1A.10`, `CC.3`, and `DC.13`.
  - Rechecked current smoke artifacts with `jq`: all seven current non-dry `single_stage_frontier_campaign_summary.json` files under `tmp/` report `frontier_engine=multilane_local`, `frontier_feasible_lane_count=0`, `frontier_archive_size=0`, and zero archive members. The older `frontier_v4_canonical_*` archives with members lack the current `frontier_archive_v1` schema and remain rejected as current certified-lane evidence.
  - Rechecked deleted-file state: `frontier_engine_nsga3.py`, `frontier_evaluator.py`, and `frontier_constraints.py` are absent from the working tree. Remaining `nsga3` text hits are docs/historical/revival-plan references, not active source imports.
  - No checklist item closed. The objective remains blocked by the same external inputs: D1.4 team confirmation and a strict certified-lane smoke artifact.
- 2026-05-09 r56 — NSGA-III revival-plan cross-check:
  - Inspected local `docs/nsga3_revival_plan_2026-05-09.md` because it is a new NSGA-III reference in the working tree. It is explicitly dormant: status is "Not started — gated on a trigger below" and says "If none of T1–T3 is met, do not revive. This plan is dormant. The deletion was YAGNI-correct."
  - Its T1 trigger is the same missing `D1.4` evidence: a retained NSGA-III campaign artifact plus a benchmark showing a Pareto member unreachable by `multilane_local + achievement_chebyshev_full_simplex_v1` under equal budget. No artifact path or benchmark result is attached in that plan.
  - Therefore the revival plan does not close `D1.4` or reopen Phase 1B. No checklist item closed; the 4 open blockers are unchanged.
- 2026-05-09 r57 — untracked validation/audit artifact cross-check:
  - Inspected local `validation_report.md`; it covers current-only polish, current Fourier modes, Gauss-Newton/QP predictor, and split line-search ideas. It contains no retained NSGA-III production artifact, no D1.4 team answer, and no certified `multilane_local` frontier smoke evidence.
  - Inspected `.alm_audit_v2/` report names and grepped them for `NSGA`, certified-lane, archive-size, feasible-lane, and `multilane_local` evidence. The reports discuss ALM/runner/numerics/physics review details but contain no artifact that satisfies the certified-lane closeout predicate.
  - No checklist item closed; the 4 open blockers are unchanged.
- 2026-05-09 r58 — remaining-untracked relevance scan:
  - Scanned the remaining untracked files (`consolidated_code.txt`, `docs/single_stage_modularization_tdd_plan_2026-04-27.md`, `docs/upstream_clean_merge_strategy_2026-04-24.md`, `docs/upstream_commits_banana_relevance_2026-05-07.md`, and `.claude/`) for `D1.4`, NSGA-III, certified-lane, archive-size, feasible-lane, and `multilane_local` evidence.
  - `consolidated_code.txt` is a generated "Source Code Consolidation" dump and contains stale embedded NSGA-III source/test snapshots; it is not a live source file, a retained production campaign artifact, or a benchmark. A live `find` still reports no `frontier_engine_nsga3.py`, `frontier_engine_multilane_local.py`, or `frontier_evaluator.py` files.
  - `docs/upstream_commits_banana_relevance_2026-05-07.md` contains historical wording that banana had an NSGA-3 + multilane-local engine when reviewing upstream PR #558. That is not a retained production result or D1.4 team answer, and it does not reopen Phase 1B.
  - No checklist item closed; the 4 open blockers are unchanged.
- 2026-05-09 r59 — stale bytecode cleanup:
  - User asked whether NSGA3 is still alive. Live Python source/test grep found no active NSGA3 implementation; `importlib.util.find_spec("banana_opt.frontier_engine_nsga3")` returned `None`.
  - Removed three ignored stale bytecode files left under `examples/single_stage_optimization/banana_opt/__pycache__/frontier_engine_nsga3.cpython-{310,311,313}.pyc`. A follow-up `find` for `frontier_engine_nsga3.py`, `frontier_engine_nsga3*.pyc`, and `*nsga3*.pyc` returned no live/cache file hits.
  - No checklist item closed; the 4 open blockers are unchanged.

---

## 1. Context & Motivation

### 1.1 Why this plan exists
At audit start, the `frontier` campaign mode under `examples/single_stage_optimization/banana_opt/frontier_*.py` (14 modules, **6,435 LOC**) plus its runner `run_single_stage_frontier_campaign.py` (**1,101 LOC**) and tests (**3,207 LOC**) had accumulated through four overlapping plan iterations:

- v1 plan (2026-04-12, 407 LOC): scalarized objective using existing ALM
- v4 plan (2026-04-13, 883 LOC): campaign optimizer + Pareto archive + recommendation policies
- Global Pareto plan (2026-04-22, 493 LOC): adds NSGA-III engine
- Gradient-contract plan (2026-04-26, 952 LOC): retroactive correction for gradient/value mismatch caused by NSGA-III layering

The audit (parallel multi-agent code review, 2026-05-07) and a second independent validation (Codex review, same day) agree: the subsystem is **moderately bloated**, with ~15-20% mechanically removable LOC, and the bloat is concentrated in **structural** issues rather than line-by-line padding.

### 1.2 What this plan is NOT
- It is not a rewrite. Frontier mode delivers real, currently-used Pareto-front exploration via the `multilane_local` engine.
- It is not a deletion sweep. The support contract layer (`frontier_contracts`, `frontier_conditioning`, `frontier_dominance`, `frontier_recommendation`) is largely well-scoped (~4% removable) and stays. The former `frontier_constraints.py` facade was later deleted after its live implementation moved to `single_stage_search_contracts.py`.
- It is not an algorithmic redesign. The Pareto-front method, ALM inner solver, and CLI surface stay functionally equivalent.

### 1.3 Guiding principles (apply to every task)
- **KISS** — prefer one obvious path over multiple parametric ones with one caller each
- **YAGNI** — drop code paths without evidence of production use; do not build for hypothetical future engines
- **DRY** — same numeric/schema contract written exactly once
- **SSOT** — frontier code lives in `frontier_*` modules; non-frontier code does not depend on `frontier_*`
- **SOLID** — name files for what they contain (no `engine_base.py` without an engine ABC); SRP at module level
- **Functional programming** — frozen dataclasses + pure transformations + map dispatch over if-chains
- **Immutable** — `frozen=True, slots=True` for all data classes; replace mutator methods with state-returning functions
- **Memory-safe / thread-safe** — preserve existing locks; remove unnecessary mutable shared state
- **Memory-efficient** — eliminate redundant copies and recomputations
- **Performant** — memoize expensive O(N²) recomputes; avoid eager imports of optional heavy deps

---

## 2. Verified Findings (cited evidence)

The following table records the original audit findings plus their current disposition. Rows marked closed may cite files that no longer exist because the fix was deletion or rename.

| # | Finding | Evidence | Severity |
|---|---|---|---|
| F1 | False engine layer — `frontier_engine_base.py` defined no ABC/Protocol; runner used hard-coded engine dispatch | Closed by dropping NSGA3 and renaming the remaining state container to `frontier_progress_state.py`. The runner now supports the single `multilane_local` engine literal via the shared `SUPPORTED_FRONTIER_ENGINES` tuple. | Closed (SOLID/SRP) |
| F2 | NSGA3 path was wired in CLI/runner/tests but no validated production artifact was found | Closed by deleting `frontier_engine_nsga3.py`, deleting NSGA3 dispatch/tests/summary fields, and updating the sibling autoresearch note to recommend `multilane_local` plus `achievement_chebyshev_full_simplex_v1`. | Closed (YAGNI) |
| F3 | Frontier scalarization leaked into central objective module — wrong dependency direction | Closed by moving frontier scalarization into `frontier_scalarization.py` and moving shared gradient access into the leaf `objective_gradients.py`. `single_stage_objectives.py` now has no frontier implementation/import surface; `frontier_scalarization.py` consumes `objective_gradients.py` directly. | Closed (SSOT) |
| F4 | `annotate_search_evaluation_finiteness` was moved out of frontier modules; the remaining search-contract implementation is now in a non-frontier module | defined in `search_evaluation.py`; consumed by `single_stage_objectives.py`, `frontier_scalarization.py`, and `single_stage_search_contracts.py`; the former `frontier_constraints.py` facade has been deleted, with only the test helper compatibility loader name retained | Closed (SSOT) |
| F5 | Epsilon-threshold magic-string keys had duplicated ad hoc access across frontier archive/scalarization/objective paths. | Closed by the `EpsilonThresholds` SSOT and the subsequent evaluator deletion. Current source no longer has the old evaluator-side attribute reads; tests keep literal fixture keys where they assert the JSON contract. | Closed (DRY/SSOT) |
| F6 | Hypervolume recomputed without memoization, called ≥4× per campaign, leave-one-out is N× | `frontier_archive.py:446-477` (`annotate_hypervolume_contributions`); also called at `frontier_archive.py:341` (serialize annotate) + `:355` (direct total), `frontier_campaign_reporting.py:303` (per-prefix in `build_frontier_hypervolume_history`), `:424` (final certified), and `frontier_runtime_calibration.py:231` (per-lane `update_frontier_early_stop_status`) | Medium (Performant) |
| F7 | Runtime calibration registry has 2 profiles differing by 1 int + 1 string + 1 tuple-of-strings (plus tautological `profile_name`) | `frontier_runtime_calibration.py:56-85` (registry; full file 272 LOC). Profiles `reduced_fixture_v1` vs `canonical_seed_v1` differ in: `default_early_stop_patience_lanes` (2 vs 3), `profile_name` (echoes registry key — tautological), `calibration_basis` (`("reduced_fixture_multilane_smoke", "deterministic_resume_smoke")` vs `("canonical_seed_bridge_smoke", "canonical_seed_resume_smoke")`) | Low (KISS) |
| F8 | `frontier_dominance.py` is misnamed (~85% Pareto normalization, ~15% dominance) | `frontier_dominance.py:8-72` (normalization rules); only `frontier_dominance.py:153-203` (~50 LOC) is dominance | Low (SOLID/naming) |
| F9 | `frontier_engine_multilane_local.py` was not an engine; it only carried the lane spec generator | Closed by folding the generator into `frontier_scalarization.py` and deleting `frontier_engine_multilane_local.py`. | Closed (SOLID/naming) |
| F10 | Hand-rolled JSON ser/de boilerplate existed across evaluator, progress-state, and archive dataclasses | Closed for the largest dead slice by deleting the unreferenced evaluator seam; `frontier_engine_base.py` was renamed to `frontier_progress_state.py`, and the remaining archive/progress serialization is kept because it is live contract code. | Mostly closed (DRY) |
| F11 | Defensive `getattr(args, ..., default)` against `argparse.Namespace` (always set by `add_argument(default=...)`) | `frontier_campaign_reporting.py:181-204, 246-255, 412-421` | Low (KISS) |
| F12 | ~~Duplicate~~ Conditional re-resolution of `runtime_defaults` and `hypervolume_reference` — both pairs operate on **different inputs** (lane-count reconciliation, certified-vs-archive members) and are not collapsible. Action: add a one-line comment at each site, do NOT collapse. | `run_single_stage_frontier_campaign.py:762-771 vs 827-837` (second is gated on `len(lane_specs) != runtime_defaults.num_lanes` — resumed lane count overrides requested); `:874-878 vs 1035-1039` (initial uses `members=archive_members`, second uses `members=certified_members`) | Trivial (documentation) |
| F13 | `# noqa: F401` import marked an unused runner-level symbol | Closed by removing the runner import; the real consumer is in `frontier_scalarization.py`. | Closed |
| F14 | `pymoo` is a soft optional runtime requirement, not a declared dep | `frontier_engine_nsga3.py:33-44` (try/except ImportError); not in `pyproject.toml` / `requirements.txt` / `setup.py` | Note (correction to prior claim) |
| F15 | `_require_*` schema helpers in `frontier_contracts.py` are NOT pre-existing in artifact_contracts; they are first-occurrence | `frontier_contracts.py:491-518`; no `_require_*` in `artifact_contracts.py` / `constraint_contract.py` / etc. | Note (correction to prior claim) |
| F16 | Test import bootstrap was more limited than the original audit claimed | Shared loader/path setup now lives in `tests/geo/_frontier_test_helpers.py`; the evaluator-specific bootstrap file was deleted with the unused evaluator tests. | Closed |
| F17 | `FRONTIER_REFERENCE_MODE_SHARED` (a.k.a. `multilane_local`) is a compatibility literal for the legacy iota/volume share sweep, not a generated 4-D Pareto simplex. | Closed by retaining the legacy literals for compatibility while clarifying behavior in `frontier_scalarization.py` module/generator docstrings, runner CLI help, and the superseded/current docs. For generated 4-D reference directions, users are directed to `FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX`. | Closed (compatibility-preserving clarification) |
| F18 | ε-certifier silently passes when threshold key is missing from `scalarization_params` | `frontier_archive.py:603-614` (`_evaluate_epsilon_constraint_status`) reads `scalarization_params.get("epsilon_constraint_qa_max")` with `if limit is None: continue`. A typo, ad-hoc rerun_contract built outside `_reference_scalarization_params`, or sidecar JSON missing keys silently certifies members against absent thresholds. No warning, no `KeyError`, no test exercises this path. | Medium (Silent-pass correctness) |
| F19 | ε-certification uses zero floating-point slack — strict `excess > 0.0` rejects `value == limit + 1e-16` | `frontier_archive.py:615-617`. Members optimized to the boundary via soft penalty pulls (asymptotic case) may fail certification within numerical precision. No on-boundary, just-above, or just-below test exists. | Low-Medium (Numerical robustness) |
| F20 | `dominates()` does not guard NaN — IEEE 754 comparisons return False, so NaN-bearing members are silently treated as non-dominating AND non-dominated (could persist in archive forever) | `frontier_dominance.py:175-203`. Standard ingest filters NaN via `_as_finite_float` upstream; hand-built `FrontierArchiveMember` instances with NaN floats slip past dominance entirely. Defensive responsibility delegated to ingest, not enforced at the dominance boundary. | Low (Ingest-path-dependent) |
| F21 | Hypervolume reference is not enforced as a true nadir | `frontier_archive.py:638-646` (`_hypervolume_boxes`): per-axis extent `max(0, extent)` clip + drop-if-all-zero filter silently treats members worse-than-reference as zero-contribution along the offending axis; no warning. `parse_hypervolume_reference` and `resolve_hypervolume_reference` accept user values verbatim with no nadir-domination check. | Low (Math safe; hides config errors) |
| F22 | `frontier_conditioning.py` is misnamed — it's a **diagnostic max/min ratio gate**, NOT a preconditioner | No diagonal scaling, no Hessian approximation, no `D⁻¹` applied to gradients anywhere. Computes max/min ratio of `\|J_QS\|, \|J_Boozer\|, \|J_iota\|, \|J_volume\|, \|trust_penalty\|, \|epsilon_penalty\|` against `FRONTIER_CONDITIONING_MAX_RATIO=1e3`. Computed in raw objective units (ignores `objective_metric_scale`) — gate verdict doesn't reflect the metric-scale-normalized space the optimizer actually traverses. Not consumed by the optimizer. | Low (Misleading filename + scale-mismatch) |
| F23 | Derived scalarization fields were added to the non-frontier finiteness-check tables | `search_evaluation.py` covers raw evaluator outputs plus scalarization-derived sums: `frontier_rank_total`, `frontier_base_total`, `frontier_trust_penalty`, `frontier_epsilon_penalty`, `frontier_goal_total`, `frontier_scalarization_total`, and the derived gradient fields. | Closed (Silent NaN propagation in derived sums) |

---

## 3. Decision Log (fill before executing Phase 1)

### Decision D1 — Keep or drop NSGA3?

**Question:** Is `--frontier-engine nsga3` exercised by any production campaign?

**Investigation TODO:**
- [x] **D1.1** Grep all completed JSONL/JSON artifacts in `autoresearch/` for `"frontier_engine": "nsga3"`
  ```bash
  rg -l '"frontier_engine":\s*"nsga3"' /Users/suhjungdae/code/columbia/autoresearch/ 2>/dev/null
  ```
- [x] **D1.2** Grep this repo's `tmp/` and `examples/single_stage_optimization/outputs_*` for the same string
  ```bash
  rg -l '"frontier_engine":\s*"nsga3"' tmp/ examples/single_stage_optimization/outputs_* 2>/dev/null
  ```
- [x] **D1.3** Check lab notes / Stellaris VM run history for `--frontier-engine nsga3` invocations
- [ ] **D1.4** Ask: does anyone on the team have a campaign result that requires NSGA3 (Pareto coverage NOT reachable by `multilane_local` + ε-constraint)? Not performed during this code execution; no KEEP evidence was supplied with the implementation request. r42 rechecked local and adjacent campaign artifacts and fixed one stale adjacent consumer doc, but this still does not prove that no teammate has an external retained NSGA3 campaign.

**Decision criteria:**
- [x] D1.A — Drop NSGA3 if zero validated production artifacts exist (most likely outcome based on prior audit)
- [x] D1.B — Keep NSGA3 only if (a) production artifact exists AND (b) there's a benchmark showing NSGA3 reaches Pareto regions multilane_local cannot. Not applicable because D1 recorded **DROP**.

**Decision recorded here:** **DROP** — no validated production artifacts were found; execute Phase 1A and delete the NSGA3 code path.

If **DROP** → execute Phase 1A
If **QUARANTINE/KEEP** → execute Phase 1B (define real Engine Protocol + benchmark gate)

### Decision D2 — Are NSGA3-targeted tests fixtures or production validation?

- [x] **D2.1** If D1 = DROP, plan to delete **all NSGA3-tied tests enumerated in Phase 1A.4** — 4 blocks + 1 helper across 3 test files totaling **~628 LOC** (153 + 154 + 192 + 127 + 2 helper; r6-corrected, r6.1 includes helper for arithmetic consistency; line numbers verified at HEAD):
  - `tests/geo/test_frontier_evaluator.py:111-263` (~153 LOC) + helper `load_frontier_engine_nsga3_module` at lines 34-35
  - `tests/geo/test_single_stage_workflow_helpers.py:5150-5303` (~154 LOC) — `test_frontier_campaign_nsga3_records_generation_summary`
  - `tests/geo/test_single_stage_workflow_helpers.py:5304-5495` (~192 LOC) — `test_frontier_campaign_nsga3_resume_reuses_saved_engine_artifacts`
  - `tests/geo/test_frontier_contracts.py:189-315` (~127 LOC) — `test_summary_validator_accepts_optional_nsga3_fields`
  - See Phase 1A.4 (Section 4.1) for the canonical enumeration; this D2.1 entry must stay in sync with it
- [x] **D2.2** If D1 = KEEP, plan to add a benchmark test that proves NSGA3 reaches at least one Pareto member that `multilane_local` cannot under equal evaluation budget. Not applicable because D1 recorded **DROP**.

### Decision D3 — Preserve frontier_runtime_calibration JSON shape?

- [x] **D3.1** Search consumed-by graph for any external tool that reads `frontier_runtime_calibration` block from a frontier summary JSON
  ```bash
  rg "frontier_runtime_calibration" --type json --type py 2>/dev/null
  ```
- [x] **D3.2** If consumers exist outside this repo (lab notebooks, dashboards), preserve schema_version + JSON keys when collapsing the registry. Execution found no external Python/notebook consumer, but preserved the on-disk JSON shape anyway.

---

## 4. Phase 1 — NSGA3 decision (gates everything else)

### 4.1 Phase 1A — DROP NSGA3 (preferred path)
**Acceptance criterion:** `grep -rn "nsga3\|NSGA3\|frontier_engine_nsga3" examples/ tests/` returns zero hits in source code (only string mentions in retired docs allowed).

**Bonus benefits of dropping NSGA3 (per r6.2 algorithm audit):** the following algorithm-audit issues fall away with the engine deletion and don't need separate fixes:
- **NSGA-III branch never writes per-lane records** — runner's `if args.frontier_engine == "nsga3"` branch (916-942) doesn't populate `lane_records_by_id`, so the final `frontier_lane_records` for an NSGA-III run is whatever stale entries were already there. Dropping the branch eliminates the divergent summary path entirely.
- **Double-evaluation in NSGA-III callback** — `_ArchiveTrackingCallback.notify` re-evaluates the entire population every generation (correct due to caching but inflates lookups 2×). Gone with the engine.
- **Pymoo internal RNG state not serialized** — `load_nsga3_frontier_campaign_artifacts` reconstructs spec/history/checkpoint but not pymoo RNG. Gone with the engine. (Note per r6.3: this is moot under the current binary resume design — see Phase 1B.C — but still becomes one less follow-up to track.)
- **6th hypervolume call site** — `_ArchiveTrackingCallback` (`frontier_engine_nsga3.py:153-156`) calls `frontier_archive_hypervolume` per generation. Gone with the engine.
- **Per-generation history fields** (already targeted by 1A.3) — the 6 optional NSGA3 summary fields and their write paths.

- [x] **1A.1** Delete `examples/single_stage_optimization/banana_opt/frontier_engine_nsga3.py` (372 LOC)
  - Pre-check: confirm no symbol from this module is imported anywhere except the runner
  ```bash
  rg "from .frontier_engine_nsga3|from banana_opt.frontier_engine_nsga3" --type py
  ```
- [x] **1A.2** Remove NSGA3 dispatch in runner
  - File: `examples/single_stage_optimization/run_single_stage_frontier_campaign.py`
  - Lines: 52-56 (imports), 130 (CLI choices), 916-942 (engine branch), 1072-1094 (summary post-processing)
  - Replace `--frontier-engine` choices with single literal or remove flag entirely if only one mode remains
- [x] **1A.3** Remove the optional NSGA3 summary payload validator and **all six** optional NSGA3 summary fields from accepted payloads
  - File: `examples/single_stage_optimization/banana_opt/frontier_contracts.py`
  - Function: `_validate_optional_nsga3_summary_payload` (defined line 446) — **delete the function**
  - Call site: `:416` calls it unconditionally (it's "validate optional fields if present", not gated on engine name) — **delete the call**
  - Remove **all six optional NSGA3 summary fields** the function validates from any payload contracts, JSON ser/de, runner/reporting writers that produce them, and any tests that expect them. The complete field list (per `frontier_contracts.py:446-489`):
    1. `frontier_generation_history` (list of per-generation telemetry mappings)
    2. `frontier_generation_history_path` (string path to history JSON)
    3. `frontier_engine_stats` (mapping)
    4. `frontier_evaluator_spec` (mapping with `schema_version`, `run_identity`)
    5. `frontier_evaluator_spec_path` (string path)
    6. `frontier_population_checkpoint_path` (string path)
  - Also clean up callers that *write* these fields:
    - `run_single_stage_frontier_campaign.py:1072-1094` (NSGA3 summary post-processing — already targeted by 1A.2)
    - `frontier_campaign_reporting.py` if it emits any of these (grep below)
    - Any per-engine artifact writer in `frontier_engine_nsga3.py` (already deleted in 1A.1)
  - Verify clean removal:
    ```bash
    rg -n "frontier_generation_history|frontier_engine_stats|frontier_evaluator_spec|frontier_population_checkpoint" \
       examples/single_stage_optimization/ tests/geo/
    ```
    Expected: zero accepted payload write/read paths. The six deleted field names may appear only in the fail-fast deleted-field rejection guard and its tests.
- [x] **1A.4** Delete ALL NSGA3-tied tests across the test suite
  - **Verification grep first:** `rg -n "nsga3|NSGA3" tests/geo/` — confirm full list before deletion
  - File: `tests/geo/test_frontier_evaluator.py`
    - Block: `test_nsga3_population_checkpoint_uses_final_population_arrays` (line 111-263, **~153 LOC** — measured to next `def test_` at 264; r6 correction from r5 estimate of ~165)
    - Helper: `load_frontier_engine_nsga3_module` (lines 34-35)
  - File: `tests/geo/test_single_stage_workflow_helpers.py`
    - Block: `test_frontier_campaign_nsga3_records_generation_summary` (line **5150**-5303, **~154 LOC** — r6 correction from r5's stale "line 4986, ~140 LOC"; off by 164)
    - Block: `test_frontier_campaign_nsga3_resume_reuses_saved_engine_artifacts` (line **5304**-5495, **~192 LOC** — r6 correction from r5's stale "line 5140, ~180 LOC"; off by 164)
  - File: `tests/geo/test_frontier_contracts.py`
    - Block: `test_summary_validator_accepts_optional_nsga3_fields` (lines 189-315, **~127 LOC** — measured by next-method offset; previously misstated as ~50)
  - **Estimated test LOC removed in 1A.4:** **628** (153 + 154 + 192 + 127 + 2 for the helper at lines 34-35), revised across r1→r6.1 from prior 535/612/626 estimates
- [x] **1A.5** Update or retire `docs/single_stage_frontier_global_pareto_plan_2026-04-22.md` (mark superseded; keep as historical record)
- [x] **1A.6** Update `docs/single_stage_frontier_gradient_contract_impl_plan_2026-04-26.md` — much of its motivation was NSGA3 gradient consistency; mark sections that no longer apply
- [x] **1A.7** Notify `autoresearch/program_hbt_topology_surrogate_legacy_vmec.md:310` consumers; remove or update reference. Updated on 2026-05-08: the note no longer recommends `--frontier-engine nsga3` and instead points users to `multilane_local` with `--frontier-reference-mode achievement_chebyshev_full_simplex_v1`.
- [x] **1A.8** Remove `# noqa: F401` import of `generate_multilane_local_specs` at `run_single_stage_frontier_campaign.py:50` (duplicate; real consumer is `frontier_scalarization.py:714`)
- [x] **1A.9** Run full test suite; expect green. Verified via repo `./run_tests`: latest current-tree run on 2026-05-09 was `Ran 1802 tests in 1408.073s` with `OK (skipped=94)`.
  ```bash
  ./run_tests tests/geo/test_frontier_*.py
  ```
- [ ] **1A.10** Confirm runner still passes a smoke run with `multilane_local` (the only remaining engine). Partial: dry-run smoke passed on 2026-05-08; r22 generated a strict checksum-bound/non-init-only/signed-current Stage 2 seed and the real non-dry campaign wrapper exited 0 with the expected JSON shape. r24 fixed the loadable-surface handoff (`surf_opt.json` + Fourier-order mismatch fitting), but the same strict seed still fails later in Boozer/self-intersection during target baseline and lane 01. r28 verified the saved `surf_opt.json` itself is not self-intersecting and the installed intersection-library API shim is already covered, so the remaining blocker is after Boozer mutates/solves the surface; matching `vol_target` to the seed's `FINAL_VOLUME` still timed out in bootability probing. r29 inspected older certified-looking frontier v4 artifacts and rejected them as current smoke evidence because their Stage 2 seeds are legacy/unbound, positive-current, off-radius, over-length, and missing required current/hardware fields. r30 rejected the remembered 80 kA non-init seed and all harvested-seed candidates; using the bootable R_nv2 harvested surface as a warm start with the strict r06 field still timed out at `iota=-0.0006158844048902357`. r31 found the local strict LCFS window tops out near `0.618` for this equilibrium (`0.620` fails vessel preflight), but the strict `0.618` seed also timed out in bootability probing with `BOOTABILITY_REASON=boozer_solve_failed`. r32 found alternate local test fixtures that can generate strict hardware-clean Stage 2 seeds, but none produced a bootable/certified surface: c09r00 failed production vessel preflight, circular aspect-100 self-intersected after Boozer initialization, and W7-X timed out in the default-target bootability probe. r34 expanded the local equilibrium preflight to 266 `wout*.nc` candidates; only circular aspect-100 passed strict geometry, and a production probe on its existing strict artifact still failed with `BOOTABILITY_REASON=self_intersection` and solved iota `0.14285863522375472` vs target `0.15`. r40 rechecked all local smoke summaries and archives; no current strict smoke has a certified archive member. r41 broadened the artifact-tree scan to adjacent `autoresearch` results and found no new strict metadata plus Boozer/certified candidate. r50 ran the previously un-executed `--stage2-iota-mode=alm` decision-gate benchmark referenced in `program_cw_poloidal_legacy_vmec.md:137` against the strict r06 frame at both `--stage2-iota-vol-target=0.1` and `=0.025`; both Stage 2 generations failed the bootstrap Boozer outer-surface construction with `cross_section()` non-monotone self-intersection (matches the 2026-04-22 lab-note physics analysis). r52 re-ran a current-code post-handoff-fix non-dry lane against the strict r06 seed and still produced `frontier_feasible_lane_count=0`, `frontier_archive_size=0` after Boozer initialization failed with `self_intersecting=True` and solved iota `-0.012338462725657728`. r53 re-ran the current strongest local JSON-surface handoff path (`surf_opt.json`, `--skip-target`, one lane, 300 s timeout), but lane 01 timed out and the wrapper summary still has `frontier_feasible_lane_count=0` and `frontier_archive_size=0`. Close criteria are now narrowed to explicit external action: new equilibrium fixture matched to R0=0.976 with iota authority near 0.15 at small volume, OR approved relaxation to R0=0.915 lineage with provenance, OR approved frontier-contract amendment to a smaller iota target. Keep open until external action provides the missing evidence.

**Estimated LOC removed in Phase 1A (r6 measurement, r6.1 arithmetic fix):** ~478 source + **~628 test** = **~1,106 LOC**.
- Source breakdown: `frontier_engine_nsga3.py` 372 LOC + runner edits ~60 LOC (imports 52-56, choices 130, dispatch 916-942, post-processing 1072-1094) + `frontier_contracts.py` validator + 6-field unwiring ~45 LOC + `# noqa: F401` 1 LOC ≈ **~478 LOC** (the prior r1-r5 estimate of ~600 LOC was high by ~25%).
- Test breakdown: `test_frontier_evaluator.py:111-263` 153 LOC + helper at 34-35 (~2 LOC) + `test_frontier_contracts.py:189-315` 127 LOC + `test_single_stage_workflow_helpers.py:5150-5303` 154 LOC + `:5304-5495` 192 LOC = **628 LOC** exact (153+2+127+154+192). r5 estimate of ~612 was within ~3%; r6 had a transcription bug that wrote "~626" alongside "≈ ~628" — corrected here.

### 4.2 Phase 1B — QUARANTINE NSGA3 (only if D1 = KEEP)
**Acceptance criterion:** there exists a real `Engine` Protocol with both `multilane_local` and `nsga3` as conforming implementations; runner dispatch is via the protocol, not `if/else`; a benchmark test proves NSGA3 utility.

**Required additional fixes if Phase 1B is chosen (per r6.2 algorithm audit; these are blockers for KEEP):**
- [x] **1B.A** Fix NSGA-III branch to populate `lane_records_by_id` so the final `frontier_lane_records` summary is engine-agnostic. Not applicable because D1 recorded **DROP** and the NSGA-III branch was deleted.
- [x] **1B.B** Eliminate or document the double-evaluation in `_ArchiveTrackingCallback.notify` (`frontier_engine_nsga3.py:107-109`). Not applicable because D1 recorded **DROP** and the callback was deleted.
- [x] **1B.C** ~~Serialize pymoo internal RNG state for resume reproducibility~~ — **r6.3 demoted from blocker.** Not applicable because D1 recorded **DROP** and the NSGA3 resume path was deleted.
- [x] **1B.D** Document or fix `frontier_reference_mode` constraint. Not applicable because D1 recorded **DROP** and the NSGA3 mode restriction no longer exists.

- [x] **1B.1** Define `frontier_engine_protocol.py` (new file, ~50 LOC). Not applicable because D1 recorded **DROP**.
  ```python
  from typing import Protocol, runtime_checkable

  @runtime_checkable
  class FrontierEngine(Protocol):
      def run(self, spec: SingleStageFrontierEvaluatorSpec, budget: int, ...) -> EngineArtifacts: ...
      def load_artifacts(self, path: Path) -> EngineArtifacts | None: ...
      def name(self) -> str: ...
  ```
- [x] **1B.2** Refactor `frontier_engine_nsga3.py` to expose a class implementing `FrontierEngine`. Not applicable because D1 recorded **DROP** and the file was deleted.
- [x] **1B.3** Wrap the inline `multilane_local` execution path (currently in runner) into a class implementing `FrontierEngine`. Not applicable because D1 recorded **DROP**; the single remaining engine path was simplified and lane execution helpers were extracted without adding a one-implementation protocol.
- [x] **1B.4** Replace runner `if/else` (lines 916-942) with a registry lookup. Not applicable because D1 recorded **DROP** and the runner no longer has multi-engine dispatch.
  ```python
  ENGINE_REGISTRY: Mapping[str, Callable[[], FrontierEngine]] = MappingProxyType({
      "multilane_local": MultilaneLocalEngine,
      "nsga3": NSGA3Engine,
  })
  engine = ENGINE_REGISTRY[args.frontier_engine]()
  ```
- [x] **1B.5** Add `tests/geo/test_frontier_engine_benchmark.py` — must show NSGA3 reaches a Pareto member that `multilane_local` cannot under equal budget. Not applicable because D1 recorded **DROP**.
- [x] **1B.6** Move `pymoo` from optional try/except to a documented optional extra (e.g., `pyproject.toml` `[project.optional-dependencies]`: `nsga3 = ["pymoo>=0.6"]`). Not applicable because D1 recorded **DROP** and pymoo is no longer imported by the frontier path.

---

## 5. Phase 2 — SSOT restoration (independent of Phase 1)

### 5.1 Move frontier scalarization back to `frontier_scalarization.py`

**Acceptance criterion:** `grep -n "frontier" examples/single_stage_optimization/banana_opt/single_stage_objectives.py` returns zero implementation hits; frontier scalarization lives in `frontier_scalarization.py`, and shared objective-gradient access lives in the leaf `objective_gradients.py`.

- [x] **2.1.1** Audit `single_stage_objectives.py:255-588` (the full frontier cluster, not the narrower 255-558 used in r1-r5) — list each function/symbol that mentions frontier
  ```bash
  rg -n "frontier" examples/single_stage_optimization/banana_opt/single_stage_objectives.py
  ```
- [x] **2.1.2** Move `apply_frontier_scalarization_override` (line 282) to `frontier_scalarization.py`
- [x] **2.1.3** Move `_frontier_chebyshev_goal` (line 473) to `frontier_scalarization.py` as private helper
- [x] **2.1.4** Move `_frontier_epsilon_penalties` (line 533-555) to `frontier_scalarization.py` (will be re-merged in Phase 4 with archive's epsilon code)
- [x] **2.1.5** Move `_frontier_alm_base_total_grad` (line 463). Note: the function is pure NumPy with no JAX, no closures, and no module-level mutable state — verified per r6 audit. Move is mechanically trivial; the prior r1-r5 "JAX function purity" warning was over-cautious.
- [x] **2.1.5b** ALSO move (added in r6 — these are part of the frontier cluster but were missed in r1-r5):
  - `_frontier_excess_penalty` (line 558-588) — body extends past r5's 255-558 range cap
  - `augment_frontier_metric_state` (line 255)
  - `_frontier_goal_component_total_grad` (line 404)
  - `_frontier_penalty_geometry_total_grad` (line 429)
  - **Gradient dependency constraint:** `augment_frontier_metric_state` uses the non-frontier `_objective_gradient` helper. Execution resolved this by moving the helper into the leaf `objective_gradients.py`; both `single_stage_objectives.py` and `frontier_scalarization.py` import that leaf helper, so no frontier import is needed from `single_stage_objectives.py`.
- [x] **2.1.6** Update import sites — `single_stage_objectives.py` now imports only `banana_opt.objective_gradients` and `banana_opt.search_evaluation`; frontier scalarization consumers import `frontier_scalarization.py` from the runner/single-stage entrypoint layer.
- [x] **2.1.7** Verify dependency direction is clean
  ```bash
  rg -n "from .frontier|from banana_opt.frontier" examples/single_stage_optimization/banana_opt/single_stage_*.py
  ```
- [x] **2.1.8** Run gradient-contract tests to confirm no regression. Verified during execution with `geo.test_single_stage_alm_integration` plus the then-live `geo.test_frontier_evaluator`; after r26, the evaluator test file is deleted because the evaluator seam has no production importer.
  ```bash
  python -m unittest -q geo.test_single_stage_alm_integration geo.test_frontier_scalarization
  ```

### 5.2 Move generic finiteness helper out of `frontier_constraints.py`

**Context:** `annotate_search_evaluation_finiteness` (`frontier_constraints.py:61`) is used by non-frontier code (`single_stage_objectives.py:8` and lines 690/740/794/818/1516). It is misnamed/misplaced.

- [x] **2.2.1** Create new module `examples/single_stage_optimization/banana_opt/search_evaluation.py`
- [x] **2.2.2** Move `annotate_search_evaluation_finiteness` + ALL supporting helpers (r6 expanded list — r5 missed two): `_FINITE_SCALAR_FIELDS`, `_FINITE_VECTOR_FIELDS`, `_FINITE_VECTOR_LIST_FIELDS`, `_FINITE_EPS` at `frontier_constraints.py:17-57` to `search_evaluation.py`. Also note the in-file self-consumer at `frontier_constraints.py:166` (inside `evaluate_frontier_trust_penalty`) — after the move, `frontier_constraints` becomes a consumer of the new module (back-import).
- [x] **2.2.3** Update import sites — actual count is **2 source files + 1 test file** (r6 correction; r1-r5 said "6+" which conflated call sites with import sites — `single_stage_objectives.py` has 5 call sites but only 1 import statement):
  - `single_stage_objectives.py:8` — change import path
  - `frontier_constraints.py:166` — add back-import after the helper moves out
  - `tests/geo/test_frontier_constraints.py:27,30` — update test imports
- [x] **2.2.4** Verify with grep that `frontier_constraints` no longer leaks into non-frontier code. Closed on 2026-05-08 by moving the implementation to `single_stage_search_contracts.py`; r51 reconciles the later deletion of the thin `frontier_constraints.py` facade. Current `rg -n "from \\.frontier_constraints|from banana_opt.frontier_constraints|import .*frontier_constraints" examples/single_stage_optimization -g '*.py'` returns zero hits.
  ```bash
  rg "from .frontier_constraints|from banana_opt.frontier_constraints" examples/ | grep -v frontier_
  ```
  Expected: zero hits.

**Estimated LOC moved (not deleted):** ~50 LOC. Net repo LOC change: 0. SSOT restoration: yes.

---

## 6. Phase 3 — Structural renames (low risk, high readability)

### 6.1 Rename `frontier_engine_base.py` → `frontier_progress_state.py`

**Context:** F1 — file contains zero engine abstraction; it holds JSON ser/de for `FrontierLaneContract`/`FrontierLaneRecord`/`FrontierCampaignProgress`.

- [x] **3.1.1** `git mv examples/single_stage_optimization/banana_opt/frontier_engine_base.py examples/single_stage_optimization/banana_opt/frontier_progress_state.py`
- [x] **3.1.2** Update **both file importers** (r6 correction — r1-r5 said "all 6 importers"; that count was the original audit's symbol-count, not file-count, as F1 itself flags). The two files: `examples/single_stage_optimization/run_single_stage_frontier_campaign.py:38` (production) and `tests/geo/test_frontier_archive.py:25` (test, via importlib).
  ```bash
  rg -l "frontier_engine_base" examples/ tests/ | xargs sed -i '' 's/frontier_engine_base/frontier_progress_state/g'
  ```
  (verify on macOS `sed -i ''` syntax; use a portable Python script if uncertain)
- [x] **3.1.3** Update test file `tests/geo/test_frontier_archive.py:25` reference (already covered by 3.1.2 sed but called out explicitly for clarity)
- [x] **3.1.4** Run frontier test suite to confirm imports resolve. Verified with direct `unittest` frontier suite: 205 tests passed.
  ```bash
  ./run_tests tests/geo/test_frontier_*.py
  ```

### 6.2 Fold `frontier_engine_multilane_local.py` into `frontier_scalarization.py`

**Context:** F9 — the file is 79 LOC, 1 dataclass + 1 weight-share generator, only consumer is `frontier_scalarization.py:714`. It is not an engine.

- [x] **3.2.1** Move `FrontierLaneSpec` dataclass and `generate_multilane_local_specs` function to top of `frontier_scalarization.py`
- [x] **3.2.2** Update importers — sites currently importing `FrontierLaneSpec` from `frontier_engine_multilane_local` should import from `frontier_scalarization`
  ```bash
  rg "from .frontier_engine_multilane_local|from banana_opt.frontier_engine_multilane_local" examples/ tests/
  ```
- [x] **3.2.3** Delete `frontier_engine_multilane_local.py`
- [x] **3.2.4** Remove the redundant `# noqa: F401` import in `run_single_stage_frontier_campaign.py:50` (already covered in 1A.8 if Phase 1A executes; otherwise do here)

### 6.3 Rename `frontier_dominance.py` → `frontier_pareto_normalization.py`

**Context:** F8 — only ~50 of 373 LOC are dominance logic; the rest is Pareto normalization.

- [x] **3.3.1** Decide whether to rename or keep — rename is honest but causes 6 importer updates and minor diff churn. Decision: keep name to avoid churn.
- [x] **3.3.2** If renaming: `git mv` + bulk replace; if not: add a one-line module docstring explaining the actual contents

**Phase 3 net LOC:** 0 deleted; ~3 import updates per renamed module. **Mental model improvement: significant.**

---

## 7. Phase 4 — DRY / SSOT consolidations

### 7.1 Centralize epsilon thresholds (F5)

**Acceptance criterion:** the strings `epsilon_constraint_qa_max`, `epsilon_constraint_boozer_max` (and similar threshold keys) appear exactly once in source — inside a frozen dataclass in `frontier_contracts.py`.

- [x] **7.1.1** Add to `frontier_contracts.py`. Two constructors are needed (per r6 audit) because the three call sites consume the thresholds differently — Mapping access on `rerun_contract.scalarization_params` versus attribute access on a frozen `frontier_goal_config` dataclass:
  ```python
  @dataclass(frozen=True, slots=True)
  class EpsilonThresholds:
      qa_max: float
      boozer_max: float

      @classmethod
      def from_rerun_contract(cls, contract: Mapping[str, object]) -> "EpsilonThresholds":
          # Used by `frontier_archive._evaluate_epsilon_constraint_status` — Mapping access.
          return cls(
              qa_max=float(contract["epsilon_constraint_qa_max"]),
              boozer_max=float(contract["epsilon_constraint_boozer_max"]),
          )

      @classmethod
      def from_goal_config(cls, config: object) -> "EpsilonThresholds":
          # Used by `single_stage_objectives._frontier_epsilon_penalties` (or its post-Phase-2
          # home in `frontier_scalarization`) — attribute access on a frozen `frontier_goal_config`
          # dataclass; NOT a Mapping.
          return cls(
              qa_max=float(config.epsilon_constraint_qa_max),
              boozer_max=float(config.epsilon_constraint_boozer_max),
          )

      def as_payload(self) -> dict[str, float]:
          return {
              "epsilon_constraint_qa_max": self.qa_max,
              "epsilon_constraint_boozer_max": self.boozer_max,
          }
  ```
- [x] **7.1.2** Replace 3 hand-rolled implementations:
  - `frontier_archive.py:596-621` (`_evaluate_epsilon_constraint_status`) — consume `EpsilonThresholds.from_rerun_contract`
  - `frontier_scalarization.py:633-647` — write via `EpsilonThresholds.as_payload`
  - `single_stage_objectives.py:533-555` (`_frontier_epsilon_penalties`) — consume `EpsilonThresholds.from_goal_config` (r6 correction: r1-r5 said `from_rerun_contract` but the call site uses attribute-access; line range corrected from 539-555 to 533-555)
  - Note: if Phase 2 already moved `_frontier_epsilon_penalties` into `frontier_scalarization.py`, only 2 sites remain in this step (the moved function still consumes `from_goal_config`)
- [x] **7.1.3** Add unit test: `test_epsilon_thresholds_round_trip` in `tests/geo/test_frontier_contracts.py`
- [x] **7.1.4** Verify no string-key duplication remains
  ```bash
  rg '"epsilon_constraint_(qa|boozer)_max"' examples/single_stage_optimization/banana_opt/
  ```
  Expected: matches only inside `frontier_contracts.py`.

**Estimated LOC removed:** ~60.

### 7.2 ~~Extract shared schema helpers~~ **DROPPED — YAGNI (per Codex r2 review)**

**Original rationale:** preemptively extract `_require_*` helpers from `frontier_contracts.py:491-518` so future contract modules can import them.

**Why dropped:**
- Original audit claimed these helpers duplicate logic in `artifact_contracts.py`. That was wrong (F15) — they're first-occurrence.
- No second consumer exists today. Extracting saves ~0 LOC.
- This plan's own guiding principle (YAGNI) prohibits building for hypothetical future modules.
- Action: **leave the helpers where they are** in `frontier_contracts.py:491-518`. If a second contract module ever needs the same idiom, extract at that point.

### 7.3 Collapse parallel "best-of" extractors (audit S3.4)

**Context:** 5 implementations of lex-max sort with different gate rules across `frontier_archive.archive_best_by_metric` (archive.py:312), `_metric_sort_key` (archive.py:584), and 4 policies in `frontier_recommendation.py:210-279`.

- [x] **7.3.1** Add a parameterized `select_best` to `frontier_recommendation.py`. Execution note: implemented with a `KeyFn` callable, `lex_priority`, `scalar_score`, and `none_aware_lex`; selection returns the best member plus eligible members and gate metadata. **Heterogeneity caveat (r6):** the 5 sites are not all reducible to `Sequence[tuple[str, Direction]]` — `_balanced_policy_sort_key` (`frontier_recommendation.py:210`) uses a scalar reduction (`balanced_policy_score`), and `_closest_to_seed_sort_key` (`:271`) uses None-aware `distance_from_seed`. The signature must therefore take a `KeyFn` callable rather than a tuple-priority spec:
  ```python
  KeyFn = Callable[[FrontierArchiveMember], tuple]

  def select_best(
      members: Sequence[FrontierArchiveMember],
      *,
      key: KeyFn,
      gate: Callable[[FrontierArchiveMember], bool],
      rationale: str,
  ) -> Recommendation: ...
  ```
  Build named `KeyFn` factories so each existing policy expresses cleanly:
  ```python
  def lex_priority(metrics: Sequence[tuple[str, Direction]]) -> KeyFn: ...
  def scalar_score(score_fn: Callable[[FrontierArchiveMember], float]) -> KeyFn: ...
  def none_aware_lex(field: str, *, fallback: KeyFn) -> KeyFn: ...
  ```
  Mapping of existing keys to factories:
  - `_metric_sort_key` → `lex_priority([(metric_name, "desc")])`
  - `_max_iota_under_safe_boozer_sort_key` → `lex_priority([("iota", "desc"), ("boozer_residual", "asc"), ("volume", "desc"), ("qa_error", "asc")])`
  - `_max_volume_under_safe_hardware_sort_key` → `lex_priority([("volume", "desc"), ...])` (analogous)
  - `_balanced_policy_sort_key` → `scalar_score(balanced_policy_score)`
  - `_closest_to_seed_sort_key` → `none_aware_lex("distance_from_seed", fallback=scalar_score(balanced_policy_score))`
- [x] **7.3.2** Build a frozen `RECOMMENDATION_POLICIES: Mapping[PolicyName, Callable] = MappingProxyType({...})` registry where each value is a `partial(select_best, key=..., gate=..., rationale=...)`. Execution note: the registry is frozen with `MappingProxyType` and stores each policy's key factory, gate rule, and rationale so normalization-aware policies can build keys from the runtime normalization payload.
- [x] **7.3.3** Delete the 4 per-policy sort-key functions
- [x] **7.3.4** Update `frontier_archive.archive_best_by_metric` to call `select_best(key=lex_priority([(metric, "desc")]), gate=lambda m: True, rationale=...)`. Execution note: the archive path now maps objective directions to `lex_priority` and calls `select_best(..., gate_rule=None)`; `_metric_sort_key` was removed.

**Estimated LOC removed:** ~60. **Functional discipline: higher.**

---

## 8. Phase 5 — Performance / Memory

### 8.1 Memoize hypervolume across reporting + serialization (F6, **revised per Codex r2**)

**Context (revised):** The current `annotate_hypervolume_contributions` (`frontier_archive.py:446-477`) does 1 base call + N leave-one-out calls per pass. Each leave-one-out call has a *different* member-set, so an `lru_cache` produces zero hits within a single annotation pass. The real win is across **repeated** invocations of the same hypervolume on the same archive (annotation, then serialization at line 341, then reporting at `frontier_campaign_reporting.py:303, 424`).

**Acceptance criterion (revised per r3):**
- Within a single campaign workflow (annotate → serialize → report → hypervolume_history), the total number of underlying hypervolume computations is **no more than the number of unique (archive-member-set, reference) inputs** seen across all call sites — not 3× that count as currently happens.
- Note: `build_frontier_hypervolume_history` (`frontier_campaign_reporting.py:281`) iterates lane records building a *running prefix archive* and calls hypervolume on each prefix, so each prefix is its own unique input — the cache cannot collapse those into a single call. The win is in collapsing **repeated** identical inputs (final-archive hypervolume called from annotate + serialize + report), not in shrinking the prefix sweep.
- Verified by an instrumentation counter in a benchmark test, not by `lru_cache` cache-info hits alone.
- Optional: redesign leave-one-out using inclusion-exclusion (incremental hypervolume) to drop the O(N) leave-one-out cost — but this is algorithmic redesign, mark as separate optional task.

- [x] **8.1.1** Identify the function `frontier_archive_hypervolume` and confirm its inputs are hashable
  - File: `frontier_archive.py` (find via `grep -n "def frontier_archive_hypervolume"`)
- [x] **8.1.2** Wrap with `lru_cache` keyed on tuple-of-tuples of objective vectors + reference point. The tuple MUST be canonicalized by **lex-sort over the four-objective vectors themselves** (in `PARETO_OBJECTIVE_SPECS` order: iota, volume, qa_error, boozer_residual), NOT by `member_id` (r6 correction; r1-r5 said "sorted by `member_id` or similar" — but `FrontierArchiveMember.member_id` at `frontier_archive.py:38` is built from `(campaign_id, lane_id, archive_state)` and carries lane/campaign identity, not Pareto content; two archives with identical hypervolume but different IDs would miss the cache). Hypervolume is invariant under permutation of S, so canonical ordering by the box-content tuples is the natural and correct equivalence class — and it matches what `_hypervolume_boxes` (`frontier_archive.py:624-648`) already operates on.
  ```python
  @functools.lru_cache(maxsize=128)
  def _hypervolume_cached(
      objective_vectors: tuple[tuple[float, ...], ...],  # canonical: sorted lex over the 4-tuples
      reference: tuple[float, ...],
  ) -> float: ...
  ```
- [x] **8.1.3** Refactor `annotate_hypervolume_contributions` to call the cached version
- [x] **8.1.4** Add a benchmark test using a per-call counter (not just `cache_info`) that asserts: across one annotation pass + one serialize (annotate + direct total) + one report + one history build + per-lane early-stop calls, the number of *underlying* `_hypervolume_cached` invocations equals the number of unique input tuples encountered — typically (1 final-archive call) + (N leave-one-outs) + (M unique prefix archives from history) + (L per-lane early-stop calls, typically distinct from prefix archives) — and that the same final-archive computation does not repeat across annotate/serialize/report. Note: `lru_cache.cache_info()` reports decorator-wrapper hits, which is fine as a sanity cross-check but does not by itself prove "no duplicated underlying compute"; a per-call counter on the underlying computation is the authoritative instrumentation. Implemented in `tests/geo/test_frontier_archive.py` by patching the uncached hypervolume function and counting unique underlying input tuples across annotation, serialization, reporting, and history paths.
- [x] **8.1.5** Verify reporting paths (`frontier_campaign_reporting.py:303, 424`) hit the cache (cache_hits ≥ 2 after annotation populates it). Covered by the same counter test; the final archive input is computed once even though annotation, serialization, direct total, report, and history paths all request it.
- [x] **8.1.6** (Optional, low priority — algorithmic redesign) Deferred to a separate future task. **Caveat (r6):** in this code's 4D objective space (`PARETO_OBJECTIVE_SPECS` has 4 entries: iota, volume, qa_error, boozer_residual; `frontier_engine_nsga3.py:65` sets `n_obj=4`), best-known leave-one-out exclusive-contribution algorithms (HSO, WFG-class) are O(N^{d−1}) worst case for d≥4 — not the routine O(N) optimization the r1-r5 wording implied. In 2D the contributions reduce to neighbor-rectangle differences in O(N log N), but that's not the regime here. If pursued, it must be a separate algorithmic-redesign task requiring (a) a literature review of recent multi-objective hypervolume-contribution algorithms in dimension 4, (b) a correctness benchmark against the naive O(N²) reference, and (c) a wall-clock benchmark to confirm the asymptotic actually wins for typical campaign archive sizes (often N < 50, where constant factors dominate).

**Estimated wall-clock improvement on long campaigns (revised per r6 audit):** **sub-2× in typical campaigns** where prefix-history sweeps (`build_frontier_hypervolume_history`) and per-lane early-stop hypervolume (`frontier_runtime_calibration.py:231`) dominate the call count — those have unique inputs and cannot be cache-deduplicated. The factor approaches "up to ~3×" only in degenerate cases where repeated-final-archive computations (annotate at `:424` + serialize annotate at `:341` + serialize direct at `:355`) dominate. The realized factor depends on the ratio (repeated-final-archive calls : unique-input calls) for a given campaign size. The cache is still worth landing — it's a few lines of code with no downside — but the user-facing benefit is more modest than r1-r5 claimed. Optional 8.1.6 is non-trivial in 4D (see caveat above).

### 8.2 Collapse runtime calibration registry (F7, with D3 guard)

**Acceptance criterion:** if D3 = preserve JSON shape, the on-disk `frontier_runtime_calibration` block in summary JSON remains identical for equivalent inputs.

- [x] **8.2.1** Confirm D3 outcome (D3.1, D3.2 above)
- [~] **8.2.2** ~~Replace `FRONTIER_RUNTIME_CALIBRATION_PROFILES` registry with single `FrontierRuntimeDefaults` dataclass + factory~~ — **DEFERRED (r48 reconciliation).** The registry remains at `frontier_runtime_calibration.py:58-91` with both profiles (`reduced_fixture_v1`, `canonical_seed_v1`). r6.4 audit verified the dataclass and factory symbols promised here are not present in the working tree. Reason for deferral: D3 preserved the profile-bearing JSON shape (per 8.2.4 execution note), and collapsing the registry would force a schema migration at every external lab-note consumer. The 2 profiles differ only in 1 int (`default_early_stop_patience_lanes`: 2 vs 3) and 1 tuple-of-strings (`calibration_basis`); future collapse remains tractable but is gated on either a no-shape-change confirmation OR explicit acceptance of a JSON-schema bump. **Tracker:** see DC.21.
- [x] **8.2.3** Preserve serialization shape via a `to_json_dict` method that emits the same keys profiles previously emitted (basis string, etc.)
- [x] **8.2.4** Delete `_resolve_calibration_profile` and the profile-name CLI argument if D3 allows. Execution note: no `_resolve_calibration_profile` symbol remains; the profile-name CLI argument is intentionally retained because D3 preserved profile-bearing runtime-calibration JSON shape and the explicit-profile test remains part of the public contract.
- [x] **8.2.5** Update the calibration-profile test to match new shape. **Test name (r6.4 corrected):** the canonical test asserting profile shape is `test_runtime_defaults_use_explicit_calibration_profile` at `tests/geo/test_frontier_contracts.py:315` (NOT line 407 — r6 plan-line stale by 92 lines after Phase 2 reorganization). The test asserts `lane_budget=300, total_budget=900, checkpoint_every=5, early_stop_patience_lanes=2`; shape unchanged because 8.2.2 was deferred (see above), so the test does not need to be rewritten.

**Estimated LOC removed:** ~70-90 (preserving JSON shape) or ~150 (if shape can change).

### 8.3 Lazy `pymoo` import (only if Phase 1B = KEEP NSGA3) — **revised per Codex r2**

**Context (revised):** `frontier_engine_nsga3.py` defines `class _FrontierNSGA3Problem(ElementwiseProblem)` and `class _ArchiveTrackingCallback(Callback)` at module level. These class declarations need pymoo symbols in scope at *import* time. Just moving the import statement into a function body breaks the subclass declarations.

Two viable approaches:

**Approach A — Function-local class definitions:**
- Move both classes inside `run_nsga3_frontier_campaign(...)` (and a similar function for the callback) so their definitions execute only when the function is called
- Pros: true lazy import, no module-level pymoo dependency
- Cons: classes redefined on each call (small overhead); test fixtures that import the classes directly need refactor

**Approach B — Keep current top-level optional import (`try/except ImportError`) + accept it as the lazy mechanism:**
- The try/except at `frontier_engine_nsga3.py:33-44` already returns sentinel `NSGA3 = None` etc. when pymoo absent
- The whole module import succeeds without pymoo; only call-time uses fail
- Pros: minimal change; classes remain top-level
- Cons: `import banana_opt.frontier_engine_nsga3` still attempts pymoo import (but doesn't fail) — RSS savings are smaller

- [x] **8.3.1** Decide between Approach A and Approach B based on RSS measurement. Not applicable because D1 recorded **DROP** and `frontier_engine_nsga3.py` was deleted.
  ```bash
  PYTHONPATH=examples/single_stage_optimization \
    /usr/bin/time -l python -c "import banana_opt.frontier_engine_nsga3" 2>&1 | tail -5
  ```
  Run with and without `pymoo` available in the same conda env to get RSS deltas. If cold-start cost (with pymoo present) is < 50 ms and < 30 MB RSS, prefer Approach B (smaller diff).
- [x] **8.3.2 (A)** Move `_FrontierNSGA3Problem` and `_ArchiveTrackingCallback` class definitions inside the functions that need them; ensure no other code references the classes at module level (check tests too). Not applicable because D1 recorded **DROP**.
- [x] **8.3.2 (B)** Document the existing `try/except ImportError` at module top as the intentional lazy mechanism with a comment. Not applicable because D1 recorded **DROP**.
- [x] **8.3.3** Document `pymoo` as optional extra in `pyproject.toml`. Not applicable because D1 recorded **DROP** and the pymoo-dependent frontier code was deleted.
  ```toml
  [project.optional-dependencies]
  nsga3 = ["pymoo>=0.6"]
  ```
- [x] **8.3.4** Add an isolated CI job (or pytest marker) that exercises the no-pymoo path. Not applicable because D1 recorded **DROP** and the no-pymoo path no longer exists.
  ```yaml
  # CI matrix snippet (illustrative — adapt to actual CI config)
  - name: nsga3-not-installed
    setup: pip install -e . --no-deps && pip install <minimal deps without pymoo>
    test: |
      PYTHONPATH=examples/single_stage_optimization \
        python -c "from banana_opt.frontier_engine_nsga3 import _PYMOO_IMPORT_ERROR; assert _PYMOO_IMPORT_ERROR is not None"
  ```
  Or, locally, use a throwaway env: `python -m venv /tmp/no_pymoo && /tmp/no_pymoo/bin/pip install <minimal deps> && /tmp/no_pymoo/bin/python -c "..."`

**Estimated cold-start RSS savings:** Approach A: ~150 MB on default `multilane_local` runs. Approach B: smaller (~0-30 MB depending on pymoo's lazy init). Measure before deciding.

### 8.4 Single-pass resume read

- [x] **8.4.1** Inspect the old `load_resume_lane_specs` / `load_resume_manifest` split. Execution note: the current runner no longer has a `load_resume_lane_specs` disk-reader; `load_resume_manifest` validates the manifest once and `resume_lane_specs_from_manifest` derives lane specs from that in-memory payload.
- [x] **8.4.2** Combine into one manifest read. Execution note: implemented as `load_resume_manifest(...)` plus `resume_lane_specs_from_manifest(...)`, preserving the validated manifest payload for both metadata and lane-spec recovery.
- [x] **8.4.3** Verify resume tests still pass. Covered by the focused post-fix frontier/workflow suite: 189 tests with `OK`.

### 8.5 Document conditional re-resolution sites (revised per r6 — F12 downgraded)

**Context (r6):** r1-r5 framed both runner re-resolutions as "duplicates that should be collapsed into a single pass." The r6 audit found this is wrong:

- Lines 762-771 vs 827-837: the second resolution at 828 is **gated on `len(lane_specs) != runtime_defaults.num_lanes`** — i.e., when resumed lane specs override the requested lane count, the runtime defaults must be re-resolved with the new lane count. This is a legitimate reconciliation, not a duplicate.
- Lines 874-878 vs 1035-1039: the second resolution uses `members=certified_members` (post-loop final certified set) versus the initial `members=archive_members` (possibly empty on fresh run / resumed list on resume). These are different inputs producing potentially different references. Not a duplicate.

Action: **document the why, do not collapse.**

- [x] **8.5.1** Add a one-line comment at runner line 827-828 explaining the re-resolution is gated on resumed-lane-count override of requested `num_lanes`. Do NOT collapse.
- [x] **8.5.2** Add a one-line comment at runner line 1035 explaining the re-resolution uses `certified_members` (post-loop) vs initial `archive_members`. Do NOT collapse.
- [x] **8.5.3** ~~If both resolutions genuinely depend on later state, document why with a one-line comment~~ — **subsumed by 8.5.1 / 8.5.2 above (r6: this is the verified state, not a maybe-condition).**

---

## 9. Phase 6 — Mechanical cleanup (do last)

### 9.1 Evaluator (`frontier_evaluator.py`) — **RETIRED in r26**

The live code no longer imports `banana_opt/frontier_evaluator.py`; the file and `tests/geo/test_frontier_evaluator.py` were deleted after source/test/import greps proved the evaluator seam was orphaned. The earlier micro-refactor tasks below are therefore closed by deletion rather than by editing dead code.

- [x] **9.1.1** ~~Split `build_single_stage_frontier_runtime`~~ — closed by deleting the unreferenced evaluator module.
- [x] **9.1.2** ~~Replace `constraint_violations` if-ladder~~ — closed by deleting the unreferenced evaluator module.
- [x] **9.1.3** ~~Extract `_results_payload_from(...)` helper~~ — closed by deleting the unreferenced evaluator module.
- [x] **9.1.4** Inline `from_spec` classmethod alias — completed before module retirement.
- [x] **9.1.5** ~~Drop redundant `except FrontierEvaluatorInitializationError: raise` arm~~ — historical only. The r6 audit correctly found it was not redundant while the evaluator existed; after r26 the whole module is gone.
- [x] **9.1.6** ~~Replace `_jsonable_value = single_stage._jsonable_value` re-export~~ — closed by deleting the unreferenced evaluator module.

### 9.2 Frozen dataclasses everywhere

- [x] **9.2.1** Add `slots=True` to `FrontierCampaignManifest` (`frontier_campaign_reporting.py:68-135`). r6 correction: r1-r5 said "convert to `frozen=True, slots=True`" — but the dataclass is already `@dataclass(frozen=True)`; only `slots=True` is missing.
- [x] **9.2.2** Add `slots=True` to `FrontierLaneContract`, `FrontierLaneRecord`, `FrontierCampaignProgress` (in renamed `frontier_progress_state.py`). r6 correction: all three are already `@dataclass(frozen=True)` (`FrontierLaneContract` line 27, `FrontierLaneRecord` line 84, `FrontierCampaignProgress` line 194); only `slots=True` is missing.
- [x] **9.2.3** ~~Move shadow-write fields off `FrontierArchiveMember`~~ **DROPPED — UNSAFE (per Codex r2)**. The original audit verified read-vs-write only inside `frontier_archive.py`. In reality `frontier_recommendation.py:96, 131, 165, 191, 205-206, 275-276` reads `recommendation_flags` and `distance_from_seed` heavily across all 4 policies. Moving these off the dataclass would break recommendation. If a contract migration is desired in the future, scope it as a separate, dedicated task with explicit consumer rewrites — not as part of this mechanical-cleanup phase.
- [x] **9.2.4** Run mypy or pyright if available. Executed on 2026-05-08. Plain scoped `mypy` failed before full checking due untyped `simsopt` imports and duplicate module naming; a second run with `MYPYPATH=examples/single_stage_optimization mypy --explicit-package-bases --ignore-missing-imports ...` found 291 errors in 25 files. This is recorded as type-gate evidence, not as a green release gate.

### 9.3 Functional dispatch

- [x] **9.3.1** Replace `generate_frontier_lane_specs` 5-arm if-chain with `MappingProxyType` registry. Implemented in `frontier_scalarization.py` using `_FrontierLaneSpecRequest`, `_FRONTIER_LANE_SPEC_GENERATORS`, and `test_generate_frontier_lane_specs_dispatch_registry_covers_supported_modes`.
- [x] **9.3.2** If Phase 1B = KEEP NSGA3, replace runner engine if/else (lines 916-942) with engine registry (covered in 1B.4). Not applicable because D1 recorded **DROP** and the runner no longer has a multi-engine branch.

### 9.4 Hand-rolled JSON ser/de helper (F10, optional — earn before invest)

**Context:** r6 measured ~294 LOC of similar `to_json_dict` / `from_json_dict` boilerplate across 3 modules. r26 removed the largest dead evaluator slice instead of adding a generic schema abstraction. The remaining live boilerplate in `frontier_progress_state.py` and `frontier_archive.py` is explicit contract code; introducing a decorator now would add abstraction for a smaller live surface.

- [x] **9.4.1** Decide: leave the remaining live boilerplate explicit after deleting the orphaned evaluator seam.
- [x] **9.4.2** Not selected for this plan after 9.4.1/9.4.4. If revived in the future, create `_schema_helpers.versioned_dataclass(version: str, *, coerce: dict[str, Callable] | None = None)`.
- [x] **9.4.3** Not selected for this plan after 9.4.1/9.4.4. If revived in the future, migrate one dataclass at a time; each migration must keep round-trip tests green.
- [x] **9.4.4** **Halt early** because the abstraction no longer pays for itself after r26.

### 9.5 Reporting cleanup

- [x] **9.5.1** Remove defensive `getattr(args, ..., default)` blocks at `frontier_campaign_reporting.py:181-204, 246-255, 412-421` — argparse always sets these. Execution note: current `rg "getattr\\(args" frontier_campaign_reporting.py` returns zero hits.
- [x] **9.5.2** Inline `_sanitize_lane_record_for_final_output` (lines 526-531) if it's a one-liner
- [x] **9.5.3** Decide if `FrontierCampaignManifest` dataclass adds value over a typed dict literal; if not, replace. Execution decision: keep it. The dataclass is the typed bridge between lower-case runtime inputs and the frozen uppercase manifest JSON contract, and replacing it with a single dict literal would remove the local field list without reducing downstream complexity.

### 9.6 Test cleanup

- [x] **9.6.1** Collapse/delete evaluator cache tests. Execution note: the cache tests were first collapsed while the evaluator existed, then removed with `tests/geo/test_frontier_evaluator.py` in r26 because no production evaluator importer remains.
- [x] **9.6.2** Audit subprocess-based evaluator round-trip test. Execution decision after r26: delete with the unused evaluator seam rather than retain a regression for dead production code.
- [x] **9.6.3** Extract shared `importlib.import_module("banana_opt.frontier_X")` wrapper into `tests/geo/_frontier_test_helpers.py` (helps 5 of 6 frontier test files; ~30-40 LOC saved)
- [x] **9.6.4** Audit each `patch.multiple(create=True, ...)` site for fragility — flag any test that mocks unrelated module globals. Execution note: the remaining evaluator-specific site was removed with `tests/geo/test_frontier_evaluator.py`.
- [x] **9.6.5** Confirm zero `xfail`, `skip`, marker-gated tests survive (the audit says there are none currently). Execution note: `rg "xfail|pytest\\.mark|@pytest\\.mark|@unittest\\.skip|skipTest|unittest\\.skip|pytest\\.skip|@.*skip" tests/geo/test_frontier_*.py tests/geo/_frontier_test_helpers.py` returned zero hits.

---

## 9b. Phase 7 — Algorithm / computation hardening (added in r6.2)

**Context:** the r6.2 audit (6 parallel agents, 2026-05-07) verified that the frontier algorithms produce mathematically-correct results in the nominal flow. No silent-wrong-answer bugs were found. However, **several silent-pass paths and semantic mismatches** can mask user errors or future regressions. This phase covers those.

### 9b.1 Rename / clarify `multilane_local` (F17 — highest-impact algorithm finding)

**Problem:** `generate_multilane_local_specs` (`frontier_engine_multilane_local.py:54-60`) sweeps only iota↔volume shares, clamped to [0.2, 0.8], and never touches qa_error or boozer_residual. The module name + the CLI flag `FRONTIER_REFERENCE_MODE_SHARED` suggest "multilane Pareto exploration" — users get a 1-parameter convex-combination sweep instead.

**Acceptance criterion:** either the name reflects the behavior, or the behavior matches the name. Pick one.

- [~] **9b.1.1** ~~(Option A — preferred — rename to match behavior)~~ — **DOWNGRADED to Option C (docstring-only mitigation; r48 reconciliation).** The constant `FRONTIER_REFERENCE_MODE_SHARED = "shared_seed_relative_frontier_v2"` is unchanged at `frontier_scalarization.py:24`. The function `generate_multilane_local_specs` retains its legacy name. The [0.2, 0.8] iota/volume share clamp at `frontier_scalarization.py:135-138` is unchanged. Behavior is unchanged. **Mitigation:** module-header docstring (`frontier_scalarization.py:3-8`) and function docstring (`:117-122`) honestly describe the legacy 2-axis sweep and direct users seeking 4-D Pareto coverage to `FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX`. The runner CLI help text reflects the same. **Reason for downgrade:** rename would invalidate the `schema_version` string embedded in every existing campaign-summary JSON and lab-note artifact pinned to `shared_seed_relative_frontier_v2`. Compatibility cost was judged higher than the residual mislabel risk now that docstrings are explicit. **Future-work:** if a `_v3` schema bump is taken for any other reason, fold the rename in then.
- [x] **9b.1.2 (Option B — match the name to the behavior):** not selected. Extending the legacy share sweep to all 4 axes would duplicate `_achievement_full_simplex_lane_specs`.
- [x] **9b.1.3** Update `docs/single_stage_frontier_*.md` plan series to clarify what each mode actually does.

### 9b.2 Surface ε-certifier missing-key path (F18)

**Acceptance criterion:** running `_evaluate_epsilon_constraint_status` on an epsilon lane with no threshold keys raises a clear error; one-threshold lanes remain valid and only enforce the configured threshold.

- [x] **9b.2.1** In `frontier_archive.py:603-614`, when `scalarization_type == "epsilon_constraint_sweep_v1"` and neither epsilon threshold key is present in `scalarization_params`, raise `ValueError("epsilon_constraint_sweep_v1 missing threshold key ...")`. QA-only and Boozer-only lanes are valid because the epsilon spec writer accepts each threshold independently.
- [x] **9b.2.2** Add tests `test_epsilon_certifier_raises_on_missing_threshold_keys` and `test_epsilon_certifier_allows_single_metric_threshold_contracts` in `tests/geo/test_frontier_archive.py`.

### 9b.3 Add slack tolerance to ε-certification (F19)

**Acceptance criterion:** `_evaluate_epsilon_constraint_status` accepts `value == limit + ε_machine` as feasible (or the strict-comparison choice is intentional and documented + tested).

- [x] **9b.3.1** Decide: introduce `epsilon_constraint_certification_slack` (default `1.0e-12` or `relative_tol * limit`) and use `excess > slack` instead of `excess > 0.0`; OR keep strict and document why (e.g., "all penalties are quadratic; on-boundary residual is structurally rare").
- [x] **9b.3.2** Add tests for on-boundary, just-above, just-below cases.

### 9b.4 Guard `dominates()` against NaN (F20)

- [x] **9b.4.1** In `frontier_dominance.py:175-203`, at function entry add `if any(value is not None and not math.isfinite(value) for value in (...))` over both maps; if NaN/Inf is detected, raise a clear error or short-circuit to `False` with a comment. Pick the choice and document it.
- [x] **9b.4.2** Add test `test_dominates_handles_nan` (covers `dominates(nan, b)`, `dominates(a, nan)`, `dominates(nan, nan)`).

### 9b.5 Enforce hypervolume reference as nadir (F21)

- [x] **9b.5.1** In `frontier_archive.py:resolve_hypervolume_reference` (currently 403-429), after resolution emit a warning (or assert) when any member has a metric value worse than the reference on its declared direction. Quiet acceptance hides config errors.
- [x] **9b.5.2** Document the silent zero-clip behavior of `_hypervolume_boxes` (currently 624-672) in a docstring so a future maintainer understands the `max(0, extent)` is intentional.

### 9b.6 Extend `_FINITE_*` tables to scalarization-derived fields (F23)

- [x] **9b.6.1** Add to `search_evaluation.py:_FINITE_SCALAR_FIELDS`: `frontier_rank_total`, `frontier_base_total`, `frontier_trust_penalty`, `frontier_epsilon_penalty`, `frontier_goal_total`, `frontier_scalarization_total`. Add to `_FINITE_VECTOR_FIELDS`: `frontier_goal_grad`, `frontier_scalarization_grad`. (r6.3: dropped `frontier_chebyshev_deltas` / `frontier_chebyshev_softmax_weights` from the original list — the Chebyshev computation uses the LSE-shift trick in `frontier_scalarization.py` so these are numerically stable under positive `sharpness`; the real risk is in derived sums like `frontier_rank_total` that aren't re-checked after summing penalty terms.)
- [x] **9b.6.2** Add test that an evaluation with a deliberately-NaN penalty summand (e.g., a non-finite `length_weight` or a synthetic `J_len = inf`) is caught by `annotate_search_evaluation_finiteness` after the derived fields are assembled (`finite_eval_ok=False, nonfinite_fields` includes the derived field name). r6.3 correction: prior text "deliberately-overflowing Chebyshev `sharpness * delta`" is wrong — the LSE shift prevents overflow there.

### 9b.7 Remove unsafe recommendation gate fallback (P5)

- [x] **9b.7.1** Delete the fallback-to-all branch in `frontier_recommendation.py`; keep the public `gate_fallback` boolean false under the current no-fallback contract.
- [x] **9b.7.2** Add tests that all explicit unsafe Boozer-trust members return no recommendation, while missing legacy Boozer metadata remains eligible without fallback.

### 9b.8 Documentation-only fixes (N1-N4, P3)

These are docstring/comment additions, no behavior change. Bundle them into one PR for tractability.

- [x] **9b.8.1 (N1) `frontier_conditioning.py`** — rename to `frontier_objective_balance_diagnostics.py` (or similar), OR keep the name and add a top-of-file comment: "This module is a diagnostic max/min-ratio gate on objective magnitudes. It is NOT a numerical preconditioner. The ratio uses raw objective units, not metric-scale-normalized space."
- [x] **9b.8.2 (N2) `evaluate_frontier_trust_penalty`** — add a docstring note: "Despite the 'trust penalty' name, this is a one-sided ε-constraint penalty on `J_Boozer` (a residual cap), not a canonical NLP trust region in DOF space (no `x_0` reference). Math is C¹ at the boundary."
- [x] **9b.8.3 (N3) ε-mode** — add a docstring on `apply_frontier_scalarization_override` ε-branch noting: "Hybrid ε-method — keeps `J_QS + res·J_Boozer + iotas·J_iota + volume·J_volume` as base objective rather than reducing to pure `f_k`. Penalty is ADDED (quadratic, fixed-coefficient — not augmented Lagrangian)."
- [x] **9b.8.4 (N4) double penalization** — add a comment at `frontier_scalarization.py:649-652` noting that `frontier_boozer_trust_threshold` and `epsilon_constraint_boozer_max` are intentionally tied (both penalize the same residual additively): `(δ/trust_scale)² + epsilon_penalty_weight·(δ/boozer_reference)²`.
- [x] **9b.8.5 (P3) `_WEIGHT_FLOOR=1e-12`** — add a comment at `frontier_scalarization.py:537-540` noting that the floor breaks the simplex unit-sum invariant intentionally to keep downstream Chebyshev divisions finite.

### 9b.9 Decimation geometric uniformity (P2 — optional, low priority per r6.3)

**r6.3 narrowing:** the original P2 framing claimed the dedupe-fill loop "biases extras toward the order-prefix." On re-verification, the rounded-index path at `frontier_scalarization.py:362-365` produces all-distinct indices for typical lane counts (verified for the cited N=15 from H=3/20 directions: indices `[0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]`, all distinct). The dedupe-fill loop at lines 373-378 is **only exercised** when the rounded-index set has collisions, which is rare. So the bias is not as broad as P2 originally claimed.

- [x] **9b.9.1** Deferred. Replacing `_select_reference_directions` with farthest-point sampling on the simplex is only warranted if strict geometric uniformity becomes a future product requirement; with the caveat above, this is a polish item, not a correctness fix. The sufficient remedy for this plan is 9b.9.2.
- [x] **9b.9.2** (Sufficient remedy) Add a docstring at `_select_reference_directions` noting that decimation is enumeration-order rounding, not geometric distance; and that the dedupe-fill fallback (lines 373-378) is only exercised on rounded-index collisions. Discourage non-canonical lane counts (e.g., N=L-1 from L Das-Dennis directions where collision is most likely).

### 9b.10 Algorithm-invariant test backfill

The r6.2 audit found that all 6 frontier subsystems are mathematically correct in nominal flow but lack tests that lock the **invariants** the math depends on. Backfill the following (most are short property-style tests). Tests already covered by Phase 7 individual sub-items above are not repeated here.

**r6.3 scope clarification:** existing test coverage that DOES exist (and r6.2 wrongly characterized as missing):
- `tests/geo/test_frontier_constraints.py:27-49` covers NaN/Inf detection in raw evaluator fields via `annotate_search_evaluation_finiteness` (`test_annotate_search_evaluation_finiteness_flags_nonfinite_fields`). NaN gaps below are specifically about NaN propagation in `dominates`, `balanced_policy_score`, hypervolume, and achievement scalarization — NOT in the finiteness annotation pipeline.
- `tests/geo/test_single_stage_workflow_helpers.py:5059-5148` covers end-to-end early-stop archive stagnation (`test_frontier_campaign_early_stop_stops_after_archive_stagnation`). Gap below is a unit-level test on `update_frontier_early_stop_status` (see 9b.10.10).

**Hypervolume invariants (`tests/geo/test_frontier_archive.py`):**
- [x] **9b.10.1** `test_hypervolume_permutation_invariance` — for the same set S and reference r, compute HV with members in original order and in reversed order; assert exact equality.
- [x] **9b.10.2** `test_hypervolume_monotone_under_dominance` — given non-dominated set S with `HV(S, r) = h`, add a member dominated by every existing member; assert `HV(S ∪ {dominated}, r) == h` (no change).
- [x] **9b.10.3** `test_hypervolume_reductions` — verify 1D and 2D analytical-value reductions against hand-computed values (sanity check the recursion).

**Dominance invariants (`tests/geo/test_frontier_archive.py`):**
- [x] **9b.10.4** `test_dominates_irreflexive` — for any member `a`, `dominates(a, a) == False`.
- [x] **9b.10.5** `test_dominates_asymmetric` — for any `a != b`, `not (dominates(a, b) and dominates(b, a))`.
- [x] **9b.10.6** `test_objective_metric_scale_handles_degenerate_axis` — when ideal == nadir for one axis, scale uses the floor (no division by zero).

**Recommendation invariants (`tests/geo/test_frontier_recommendation.py`):**
- [x] **9b.10.7** `test_recommend_empty_archive_returns_none` (already implicitly covered by `recommend_frontier_member` returning None at line 59-60, but no explicit test).
- [x] **9b.10.8** `test_recommend_single_member_archive_returns_that_member` (with and without gate pass).
- [x] **9b.10.9** `test_recommend_tiebreaker_is_member_id_lex_ordered` — two members with identical primary metrics return the lex-min `member_id`.

**Early-stop invariants (`tests/geo/test_frontier_contracts.py` or new test file for runtime calibration):**

**r6.3 narrowing:** an end-to-end stagnation test already exists at `tests/geo/test_single_stage_workflow_helpers.py:5059-5148` (`test_frontier_campaign_early_stop_stops_after_archive_stagnation`). The remaining gap is a **unit-level** test on `update_frontier_early_stop_status` that lets a maintainer reason about the state machine without spinning up a full campaign.

- [x] **9b.10.10** `test_update_frontier_early_stop_status_handles_synthetic_hv_sequence` — call the function directly with a fabricated sequence of `(certified_members_hv, archive_size)` tuples and `min_hypervolume_gain` / `patience_lanes` settings. Assert `triggered`, `reason`, `no_improvement_streak`, `previous_best_hypervolume` evolve correctly. Cases: (a) flat HV sequence → triggers after `patience` repeats, (b) marginal-but-below-threshold improvement → still triggers, (c) above-threshold improvement → resets the streak, (d) `previous_best_hypervolume is None` first-improvement-after-`None`-history path (the state-smell mentioned in the audit).
- [x] **9b.10.11** ~~`test_early_stop_does_not_trigger_on_marginal_improvement`~~ — subsumed by 9b.10.10 case (b).

**Multilane / lane-spec invariants (`tests/geo/test_frontier_scalarization.py`):**
- [x] **9b.10.12** `test_multilane_share_sweep_normalization` — for N ∈ {1, 2, 3, 5, 10}, every emitted lane has shares summing to 1.0 within tolerance, no negatives.
- [x] **9b.10.13** `test_multilane_share_sweep_boundary` — explicitly assert that for N=2, shares are (0.2, 0.8) and (0.8, 0.2) — endpoint-clamping is a documented behavior, not a bug.
- [x] **9b.10.14** `test_achievement_scalarization_scaling_invariance` — multiply the goal point by 2; assert lane choice (which member is selected) is invariant or changes in a documented way.

**ε-certifier (covered partly by 9b.2.2 and 9b.3.2):**
- [x] **9b.10.15** `test_epsilon_certifier_silent_pass_on_unknown_scalarization_type` — for `scalarization_type != "epsilon_constraint_sweep_v1"`, certifier should return `{"ok": True}` without consulting thresholds (this is the current behavior — lock it as a contract, not just an accident).

**Estimated total Phase 7 test LOC added:** ~250-350 LOC across the subsystems. Most are ≤20 LOC each (property tests with 1-3 fixtures).

**Estimated total LOC delta for Phase 7:** ~+60 to +120 LOC of guards/docstrings, ~+250-350 LOC of new tests, ~−30 LOC if 9b.1 (rename) consolidates. Net: a meaningful positive (~+300-450 LOC), but eliminates a class of silent-failure paths and makes the algorithm contracts explicit.

---

## 10. Cross-cutting: after each phase

These run after every phase commit, before opening a PR.

- [x] **CC.1** Run frontier-relevant tests. Verified with direct `unittest` targeted frontier/workflow suites, the r27 post-runner-extraction regression slice (`296 tests`, `OK`), the r51 targeted frontier/workflow/ALM slice (`467 tests`, `OK`), and current-tree repo `./run_tests` (`Ran 1802 tests in 1408.073s`, `OK (skipped=94)`).
  ```bash
  ./run_tests tests/geo/test_frontier_*.py tests/geo/test_single_stage_*.py
  ```
- [x] **CC.2** Run ruff/lint. Verified with non-mutating scoped `ruff check` on changed frontier/source/test files. Note: repo `./run_ruff` auto-fixes unrelated pre-existing lint, so those accidental hunks were reverted.
  ```bash
  ./run_ruff
  ```
- [ ] **CC.3** Smoke a frontier campaign run with `multilane_local`. **Required CLI args** (revised per Codex r2): `--plasma-surf-filename` and `--stage2-bs-path` are `required=True` in the parent parser at `run_single_stage_goal_mode_comparison.py:187-200`. Replace the placeholders below with actual paths from your local fixtures or lab notes. Partial: dry-run smoke passed on 2026-05-08 with `wout_nfp22ginsburg_000_014417_iota15.nc` and `tmp/e2e_independent_seed_smoke/biot_savart_opt.json` plus `--allow-init-only-stage2-seed`. A non-dry-run smoke attempt found and fixed invalid forwarding of that wrapper-only flag to `single_stage_banana_example.py`. r22 then generated a strict non-init-only, checksum-bound, signed-current Stage 2 seed with `--target-lcfs-max-major-radius-m 0.6` and ran a real non-dry campaign at `tmp/frontier_smoke_phase_check_r06_20260508_021909`; the campaign exited 0 and wrote summary/manifest/progress/archive/recommendation JSON, but target baseline plus all three lanes failed Boozer initialization, leaving `frontier_feasible_lane_count=0`. r24 fixed the JSON surface handoff and removed the DOF-mismatch crash, but target baseline and lane 01 still failed via Boozer Newton divergence plus non-monotone self-intersection; the duplicate lane loop was stopped before final summary/archive emission. r28 ruled out the saved surface artifact and installed intersection-library API shim as the remaining blocker: the strict `surf_opt.json` reports non-intersecting before Boozer initialization, and using the seed's own `FINAL_VOLUME` as `vol_target` still timed out during bootability probing. r29 ruled out the two older certified-looking frontier v4 runs as current smoke inputs because they use positive +100 kA TF currents, legacy/unbound sidecars, off-contract radius/length, and missing required hardware fields. r30 ruled out the remembered non-init 80 kA seed, the HBT-clean harvested Stage 2 artifact, all 74 immediate harvested seeds, and the R_nv2 surface-as-warm-start path under the strict r06 field. r31 generated a strict `0.618` LCFS-radius seed that passed current hardware/vessel checks, but its seed-volume-matched bootability probe timed out with `BOOTABILITY_REASON=boozer_solve_failed`; looser `0.620` and `0.650` candidates failed plasma-vessel preflight. r32 tried alternate local test fixtures without weakening strict contracts: c09r00 failed production vessel preflight, circular aspect-100 generated a hardware-clean seed but self-intersected after Boozer initialization, and W7-X generated a hardware-clean seed but timed out in the default-target bootability probe. r34 screened 266 local equilibria and found no additional geometry-valid fixture beyond circular aspect-100; re-probing its strict m0918 artifact still failed as self-intersecting after Boozer solve. r40 rechecked every local smoke summary and archive; all current bloat-reduction smoke archives have zero members. r41 found the only adjacent `autoresearch` `multilane_local` archive tree also has zero archive members/certified lanes. r50 ran the previously un-executed `--stage2-iota-mode=alm` decision-gate benchmark referenced in `program_cw_poloidal_legacy_vmec.md:137` against the strict r06 frame at both `--stage2-iota-vol-target=0.1` and `=0.025`; both Stage 2 generations failed the bootstrap Boozer outer-surface construction with `cross_section()` non-monotone self-intersection (matches the 2026-04-22 lab-note physics analysis). r52 re-ran the current-code strict smoke with `--equilibria-dir examples/single_stage_optimization/equilibria`, `--skip-target`, one lane, and lane budget 10; it exited 0 at the wrapper level but lane 01 failed during Boozer initialization (`self_intersecting=True`, solved iota `-0.012338462725657728`) and wrote zero archive members. r53 re-ran the JSON-surface handoff smoke with `--stage2-seed-surf-path .../surf_opt.json`, `--skip-target`, one lane, lane budget 10, and a 300 s timeout; it exited 0 at the wrapper level after lane timeout and still wrote zero archive members. Close criteria are now narrowed to explicit external action: new equilibrium fixture matched to R0=0.976 with iota authority near 0.15 at small volume, OR approved relaxation to R0=0.915 lineage with provenance, OR approved frontier-contract amendment to a smaller iota target. Keep open until external action provides the missing evidence.
  ```bash
  PLASMA_SURF=/path/to/wout_demo.nc           # required — VMEC wout fixture
  STAGE2_BS=/path/to/stage2/biot_savart_opt.json  # required — Stage 2 seed artifact
  OUT=tmp/frontier_smoke_phase_check_$(date +%Y%m%d_%H%M%S)

  python examples/single_stage_optimization/run_single_stage_frontier_campaign.py \
    --plasma-surf-filename "$PLASMA_SURF" \
    --stage2-bs-path "$STAGE2_BS" \
    --frontier-engine multilane_local \
    --frontier-num-lanes 3 \
    --frontier-lane-budget 10 \
    --output-root "$OUT"
  ```
- [x] **CC.4** Diff JSON artifact shape vs prior smoke run (regression guard for D3). The default summary file is **`single_stage_frontier_campaign_summary.json`** (`frontier_campaign_reporting.py:52` defines `DEFAULT_SUMMARY_JSON`); not `summary.json`. Dry-run smoke and the r22 non-dry wrapper smoke verified the summary and manifest preserve `frontier_runtime_calibration` / `FRONTIER_RUNTIME_CALIBRATION` with keys `profile`, `resolved_defaults`, and `schema_version`.
  ```bash
  jq -S '.frontier_runtime_calibration' "$OUT/single_stage_frontier_campaign_summary.json"
  ```
- [x] **CC.5** Verify dependency direction
  ```bash
  rg "from .frontier_|from banana_opt.frontier_" examples/single_stage_optimization/banana_opt/single_stage_*.py
  ```
  Expected after Phase 2: zero hits, OR only consumer-style references where `single_stage_objectives` calls into `frontier_scalarization.apply_*` rather than re-defining the body.
- [x] **CC.6** Confirm LOC delta matches plan. Measured current LOC after r47 audit: frontier modules **5,261**, runner **795**, combined source **6,056**, frontier tests/helper **2,264**. All numeric LOC gates are closed.
  ```bash
  wc -l examples/single_stage_optimization/banana_opt/frontier_*.py \
        examples/single_stage_optimization/run_single_stage_frontier_campaign.py
  ```
- [x] **CC.7** Update STATE / lab notes if work is paused mid-phase. No `STATE.md`, `state.json`, or lab-note file exists in this repo checkout; the pause/blocker state is captured in the r24 revision-log entry above.

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 2 (scalarization move) breaks gradient consistency | Medium | High | Run `test_frontier_scalarization.py`, `test_single_stage_alm_integration.py`, and the focused frontier/workflow slice after each function move; bisect on smallest reproduction if fail |
| Phase 1A premature deletion if NSGA3 has undocumented production use | Low (per audit) | Medium | D1 investigation must complete first; if any artifact found, escalate to Phase 1B (quarantine) |
| Phase 8.2 calibration JSON shape change breaks downstream tools | Medium | Medium | D3 audit before executing; preserve `to_json_dict` shape if external consumers exist |
| Phase 6.1 rename causes wide diff churn that conflicts with concurrent branches | Low | Low | Coordinate with team; do rename in dedicated commit; merge to base before continuing |
| Hypervolume memoization (8.1) wrong cache key → silent stale results | Low | High | Cache key must be tuple-of-tuples of objective vectors (immutable, hashable); add property test asserting cache invalidates when archive changes |
| Schema-versioned dataclass refactor (9.4) introduces decorator complexity > boilerplate it removes | Medium | Low | Halt early — see 9.4.4 stop condition |
| Test deletions hide future regressions | Low | Medium | For each deleted test, confirm coverage exists in another test file; document in commit message |

---

## 12. Done criteria (final review)

A reviewer should be able to confirm all of the following with grep + the test suite.

- [x] **DC.1** D1 decision is recorded in this file
- [x] **DC.2** If D1 = DROP: zero hits for `nsga3` in `examples/single_stage_optimization/banana_opt/` and `tests/geo/` source code (string mentions in retired docs allowed)
- [x] **DC.3** If D1 = QUARANTINE: a real `FrontierEngine` Protocol exists; runner dispatches via registry; benchmark test exists. Not applicable because D1 recorded **DROP**.
- [x] **DC.4** `single_stage_objectives.py` contains zero `_frontier_*` private functions; only consumer-style imports
- [x] **DC.5** `annotate_search_evaluation_finiteness` lives in `search_evaluation.py`; no `frontier_constraints` import remains in non-frontier code. `single_stage_banana_example.py` imports the implementation from `single_stage_search_contracts.py`, and `frontier_constraints.py` is only a frontier-facing facade.
- [x] **DC.6** `epsilon_constraint_qa_max` / `epsilon_constraint_boozer_max` string keys appear only inside `frontier_contracts.py` (one frozen dataclass) for source under `banana_opt/`; tests keep literal fixture keys.
- [x] **DC.7** `frontier_engine_base.py` does not exist (renamed to `frontier_progress_state.py`)
- [x] **DC.8** `frontier_engine_multilane_local.py` does not exist (folded into `frontier_scalarization.py`)
- [x] **DC.9** `frontier_archive_hypervolume` is memoized; benchmark test demonstrates cache hits. Covered by the uncached-function counter test in `tests/geo/test_frontier_archive.py`.
- [x] **DC.10** `pymoo` is either not imported (Phase 1A) or lazily imported (Phase 1B + 8.3)
- [x] **DC.11a** Frontier module LOC (sum of `examples/single_stage_optimization/banana_opt/frontier_*.py`) < **5,500** (started at 6,435). Current measured value: **5,261** after deleting the orphaned evaluator seam and extracting runner lane-execution helpers into `frontier_campaign_execution.py`.
- [x] **DC.11b** Runner LOC (`run_single_stage_frontier_campaign.py`) < **950** (started at 1,101). Current measured value: **795** after extracting lane execution setup helpers while preserving runner-level compatibility re-exports.
- [x] **DC.11c** Combined source LOC < **6,500** (started at 7,536). Current measured value: **6,056** after deleting the orphaned evaluator seam and extracting runner lane-execution helpers.
- [x] **DC.12a** Frontier test LOC (`tests/geo/test_frontier_*.py` plus shared helper `tests/geo/_frontier_test_helpers.py`) < **2,800** (started at 3,207). Current measured value: **2,264** after fixture/helper consolidation, table-driven cleanup, invariant-test preservation, and deletion of the unused evaluator test file. r6 correction: the prior target of <2,200 was unreachable from the listed phases until the dead evaluator seam was removed, and earlier wording conflated this glob with `test_single_stage_workflow_helpers.py` (a different file).
- [x] **DC.12b** NSGA3 blocks deleted from `tests/geo/test_single_stage_workflow_helpers.py`: **~347 LOC** removed (r6 measurement: 154 LOC at line 5150-5303 + 192 LOC at line 5304-5495 + ~1 LOC import cleanup). Earlier estimates of "~320 LOC" undercounted by ~27 LOC.
- [ ] **DC.13** All frontier tests green; ruff clean; smoke campaign run produces equivalent JSON shape (or shape change documented in D3). Partial: targeted frontier/workflow tests, current-tree full suite, touched-scope compileall, scoped ruff, and diff checks are green; dry-run smoke produced equivalent JSON shape; r22 real non-dry-run smoke produced equivalent wrapper JSON shape with a strict Stage 2 seed. This remains open only for certified-lane smoke: the strict r22/r24 seed fails single-stage Boozer/self-intersection before any certified archive member is produced, and r28 confirmed the saved strict `surf_opt.json` is non-intersecting before Boozer initialization; a seed-volume-matched bootability probe still timed out without solved iota. r29 rejected older certified-looking frontier v4 artifacts as current evidence because they fail checksum/schema/current/hardware provenance under the current contract. r30 rejected the remembered non-init 80 kA seed, harvested-seed inventory, and R_nv2 warm-start probe as current certified-lane evidence. r31 generated and rejected the largest nearby strict LCFS candidate found (`0.618`) as certified-lane evidence because its bootability probe timed out with `BOOTABILITY_REASON=boozer_solve_failed`; `0.620` and `0.650` failed geometry preflight. r32 found no alternate local test fixture that produced bootable/certified smoke evidence; reviewer-agent pass returned `PASS` and confirmed the remaining certified-lane smoke gap is not a code regression. Latest r33 broader frontier/workflow regression slice ran 297 tests with `OK`; r32 reviewer also ran 176 focused tests with `OK`. r34 screened 266 local equilibria and re-probed the only strict-geometry pass, confirming the remaining gap is fixture/bootability, not frontier-code regression. r40 rechecked every local smoke summary and archive, with no current strict certified-lane evidence found. r41 broadened the artifact scan through adjacent `autoresearch` and found no strict certified-lane evidence there either. r50 ran the previously un-executed `--stage2-iota-mode=alm` decision-gate benchmark referenced in `program_cw_poloidal_legacy_vmec.md:137` against the strict r06 frame at both `--stage2-iota-vol-target=0.1` and `=0.025`; both Stage 2 generations failed the bootstrap Boozer outer-surface construction with `cross_section()` non-monotone self-intersection (matches the 2026-04-22 lab-note physics analysis). r52 re-ran current-code post-handoff-fix non-dry smoke and confirmed the wrapper JSON shape still emits correctly, but lane certification still fails: `dry_run=false`, `stage2_artifact_init_only=false`, `frontier_feasible_lane_count=0`, `frontier_archive_size=0`, Boozer `self_intersecting=True`. r53 re-ran the strongest available current JSON-surface handoff smoke and confirmed the wrapper JSON shape still emits correctly, but lane 01 timed out after 300 s and the final archive still has zero members. The decision-gate is now implicitly populated with negative evidence; close criteria are narrowed to explicit external action: new equilibrium fixture matched to R0=0.976 with iota authority near 0.15, OR approved relaxation to R0=0.915 lineage, OR approved frontier-contract amendment to a smaller iota target. After r53, fresh validation evidence is: `git diff --check` PASS; touched-scope `compileall` PASS; touched-scope `ruff check` PASS; `PYTHONPATH=tests python -m unittest -q -b ...` ran **467** tests with `OK`; repo `./run_tests` ran **1802** tests in **1408.073s** with `OK (skipped=94)`; current-code strict smoke variants produced equivalent wrapper JSON shape but zero certified archive members.
- [x] **DC.14** `docs/single_stage_frontier_*.md` plan files updated: superseded sections marked; current state reflects this plan's outcomes
- [x] **DC.15** Updated audit re-run shows < 5% removable LOC (down from ~15-20%). Independent source audit found only ~115-160 lines of conservative behavior-preserving cleanup before r26; after deleting the orphaned evaluator seam and extracting runner lane-execution helpers, all numeric source LOC gates are closed without approved contract/feature removal.
- [x] **DC.16** (Phase 7 / r6.2) The legacy `multilane_local` / `shared_seed_relative_frontier_v2` literals remain for compatibility, but `frontier_scalarization.py` now has a module docstring and generator docstring stating that shared mode is an iota/volume share sweep, and the runner CLI help directs 4-objective generated reference directions to `achievement_chebyshev_full_simplex_v1`.
- [x] **DC.17** (Phase 7 / r6.2) Silent-pass paths closed: ε-certifier raises on missing threshold key (F18); `dominates()` guards NaN (F20); `resolve_hypervolume_reference` warns on non-nadir reference (F21)
- [x] **DC.18** (Phase 7 / r6.2, refined r6.4/r45) `_FINITE_*` tables cover scalarization-derived sums (`frontier_rank_total`, `frontier_base_total`, `frontier_trust_penalty`, `frontier_epsilon_penalty`, `frontier_goal_total`, `frontier_scalarization_total`); a deliberate non-finite penalty summand or `length_weight·J_len` is detected by `annotate_search_evaluation_finiteness` after derived fields are assembled. (r6.4 reword: r6.2's "deliberate Chebyshev overflow" example was wrong — LSE-shift trick in `frontier_scalarization.py` stabilizes the softmax under positive sharpness, so it can't be used as the test trigger; F23 fix is about detecting non-finiteness in the *summed* derived fields, not in Chebyshev internals.)
- [x] **DC.19** (Phase 7 / r6.2) Recommendation `gate_fallback` is surfaced in the public return signature, not just metadata (P5)
- [x] **DC.20** (Phase 7 / r6.2) Algorithm-invariant tests are present: HV permutation invariance, dominance reflexivity/asymmetry, ε on-boundary, recommendation empty-archive, multilane share-sweep normalization, early-stop synthetic flat-HV (subset of 9b.10)
- [x] **DC.21** (r48 — B1 deferred follow-up) Calibration registry collapse closure. **Closed WONTFIX 2026-05-08 (r49)** via option (b): D3 = "preserve shape" remains in force; the registry is intentionally retained for JSON-schema stability of the `FRONTIER_RUNTIME_CALIBRATION` payload pinned by external lab-note consumers. A one-line comment at `examples/single_stage_optimization/banana_opt/frontier_runtime_calibration.py:58` documents this decision. Future-work hook: if a `_v3` or later schema bump is taken for any other reason, the rewrite-as-`FrontierRuntimeDefaults` collapse from 8.2.2 (~70-150 LOC removed) becomes safe and should be folded in.

---

## 13. Out of scope (explicit non-goals)

- Algorithmic redesign of multi-objective optimization
- Replacing ALM with a different inner solver
- Adding new engines (e.g., MOEA/D, Bayesian/Kriging surrogates) — possible future work, separate plan
- Adding meta-level GA/LLM orchestration policies (cf. Kaptanoglu & Gil 2603.15240) — possible future work, separate plan
- Refactoring non-frontier `single_stage_*` runners (their `build_summary` / `build_parser` 4-way fork is a separate, larger problem)
- Touching `simsopt/` library code or JAX kernels

---

## 14. Suggested commit / PR structure

One PR per phase, with phase number in title. This keeps reviews tractable and makes any rollback localized. These checkboxes track implementation status of the suggested packaging slices, not creation of actual GitHub PRs; no commits or PRs were created in this execution.

- [x] **PR1** `frontier: drop NSGA3 engine path` (Phase 1A) — implemented by choosing D1=DROP; PR1' quarantine path is not applicable.
- [x] **PR2** `frontier: restore SSOT — move scalarization back to frontier_scalarization.py` (Phase 2.1)
- [x] **PR3** `frontier: extract search_evaluation helper out of frontier_constraints` (Phase 2.2)
- [x] **PR4** `frontier: rename engine_base → progress_state; fold multilane into scalarization` (Phase 3.1, 3.2)
- [x] **PR5** `frontier: centralize epsilon thresholds` (Phase 4.1) — Phase 4.2 dropped per r2 (YAGNI)
- [x] **PR6** `frontier: parameterize best-of selectors` (Phase 4.3)
- [x] **PR7** `frontier: memoize hypervolume; collapse calibration profile registry` (Phase 5.1, 5.2)
- [x] **PR8** `frontier: lazy pymoo + single-pass resume + single-pass resolution` (Phase 5.3, 5.4, 5.5) — pymoo path dropped with PR1; remaining resume/resolution comments are implemented.
- [x] **PR9** `frontier: retire unused evaluator; frozen dataclasses; functional dispatch` (Phase 6.1, 6.2, 6.3)
- [x] **PR10** `frontier: reporting + test cleanup` (Phase 6.5, 6.6)
- [x] **PR11** (optional) `frontier: schema_versioned helper` (Phase 6.4) — not selected; 9.4.1 decided against this helper.
- [x] **PR12** `frontier: clarify multilane_local compatibility semantics` (Phase 9b.1) — rename not selected; legacy literals remain for compatibility and docs/docstrings/CLI help now point 4-objective generated reference directions to `achievement_chebyshev_full_simplex_v1`.
- [x] **PR13** `frontier: silent-pass guards + finiteness-table extension` (Phase 9b.2-9b.6) — bundles ε-certifier missing-key raise, ε slack tolerance, dominates NaN guard, HV reference nadir warning, derived-fields finiteness check (F18-F23)
- [x] **PR14** `frontier: documentation-only naming clarifications` (Phase 9b.8) — docstring/comment additions for conditioning, trust-penalty, ε-mode, double-penalization, weight floor (N1-N4 + P3)
- [x] **PR15** `frontier: algorithm-invariant test backfill` (Phase 9b.10) — property-style tests lock the math contracts listed in DC.20

---

## 15. References

- Audit report (multi-agent, 2026-05-07) — see this conversation's record
- Codex independent validation (2026-05-07) — confirmed F1-F13; corrected F14, F15, F16
- Original plan series superseded by this work:
  - `docs/single_stage_frontier_impl_plan_2026-04-12.md` (v1)
  - `docs/single_stage_frontier_v4_requirements_and_impl_plan_2026-04-13.md` (v4)
  - `docs/single_stage_frontier_global_pareto_plan_2026-04-22.md` (NSGA3 introduction)
  - `docs/single_stage_frontier_gradient_contract_impl_plan_2026-04-26.md` (corrective)
- External literature:
  - Deb & Jain, NSGA-III Part I (IEEE TEVC 2014) — https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf
  - pymoo NSGA-III docs — https://pymoo.org/algorithms/moo/nsga3.html
  - Stellarator Optimization with Constraints (arXiv 2403.11033)
  - Kaptanoglu & Gil, "AI-driven stellarator coil optimization" (arXiv 2603.15240, March 2026) — supports gradient-based ALM + meta-level GA, not NSGA-III on DOFs
  - Giuliani, Wechsung, Cerfon, Stadler, Landreman, "Single-stage gradient-based stellarator coil design: Optimization for near-axis quasi-symmetry," J. Comput. Phys. 459 (2022) 111147 — supports gradient-based approach (r6 citation correction: first author is **Giuliani**, not Wechsung; r1-r5 cited as "Wechsung et al." which was misled)

---

**End of plan.** When picking up this work, start at section 3 (Decision Log). Do not skip ahead to mechanical cleanup — the structural cuts unblock and shrink the mechanical work.
