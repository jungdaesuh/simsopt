# Frontier Mode Bloat Reduction — TODO Plan

**Date:** 2026-05-07 (revised same day per Codex review)
**Branch:** surrogate-confinement-v2
**Status:** Proposed; awaiting NSGA3 production-evidence decision before execution
**Owner:** TBD
**Estimated effort:** ~3-5 focused days; can be split across PRs

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
    - ε-mode (`single_stage_objectives.py:533-555` + `frontier_scalarization.py:601-606`) keeps `J_QS + res·J_Boozer + iotas·J_iota + volume·J_volume` as the base objective rather than reducing to pure $f_k$ as canonical Haimes ε-method does. Hybrid weighted-sum + ε-penalty.
    - `frontier_boozer_trust_threshold` is set equal to `epsilon_constraint_boozer_max` (`frontier_scalarization.py:649-652`) → **double penalization** of boozer residual (additive, not redundant): `(δ/trust_scale)² + epsilon_penalty_weight·(δ/boozer_reference)²`.
  - **P1-P5 (perf / hidden cost):**
    - NSGA-III `_ArchiveTrackingCallback` re-extracts `algorithm.pop` and re-evaluates the entire population every generation (`frontier_engine_nsga3.py:107-109`); correct due to caching but inflates lookups 2×.
    - `_select_reference_directions` decimation (`frontier_scalarization.py:362-378`) uses enumeration-order indices and dedupe-fills with the lowest unused index — biases extras toward the order-prefix (high-iota corners). Violates Das-Dennis uniform-coverage intent. *(Narrowed r6.3: rounded-index path is all-distinct for typical lane counts (verified for N=15 from H=3/20 — indices `[0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]`). The dedupe-fill loop is only exercised on collisions, which are rare. Bias claim narrowed accordingly.)*
    - `_WEIGHT_FLOOR=1e-12` clamp (`frontier_scalarization.py:537-540`) breaks the simplex unit-sum invariant: "pure (1,0,0,0)" becomes `(1, 1e-12, 1e-12, 1e-12)`. Necessary for downstream Chebyshev division but undocumented.
    - Pymoo internal RNG state is NOT serialized in `load_nsga3_frontier_campaign_artifacts` — resume after partial NSGA-III run cannot reproduce next-gen sample identically. *(Corrected r6.3: this is moot under the current binary resume design at `run_single_stage_frontier_campaign.py:918-924` — the resume path either loads a complete prior run's artifacts and skips `run_nsga3_frontier_campaign` entirely, or runs fresh; there is no "continue from generation N/M" path. RNG state being unserialized would matter only if a partial-run-continuation path were added.)*
    - `_eligible_members_for_policy` (`frontier_recommendation.py:333-336`) silently falls back to all members when no member passes the gate; sets `gate_fallback_to_all_members=True` in metadata but doesn't surface in the public return signature.
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

---

## 1. Context & Motivation

### 1.1 Why this plan exists
The `frontier` campaign mode under `examples/single_stage_optimization/banana_opt/frontier_*.py` (14 modules, **6,435 LOC**) plus its runner `run_single_stage_frontier_campaign.py` (**1,101 LOC**) and tests (**3,207 LOC**) accumulated through four overlapping plan iterations:

- v1 plan (2026-04-12, 407 LOC): scalarized objective using existing ALM
- v4 plan (2026-04-13, 883 LOC): campaign optimizer + Pareto archive + recommendation policies
- Global Pareto plan (2026-04-22, 493 LOC): adds NSGA-III engine
- Gradient-contract plan (2026-04-26, 952 LOC): retroactive correction for gradient/value mismatch caused by NSGA-III layering

The audit (parallel multi-agent code review, 2026-05-07) and a second independent validation (Codex review, same day) agree: the subsystem is **moderately bloated**, with ~15-20% mechanically removable LOC, and the bloat is concentrated in **structural** issues rather than line-by-line padding.

### 1.2 What this plan is NOT
- It is not a rewrite. Frontier mode delivers real, currently-used Pareto-front exploration via the `multilane_local` engine.
- It is not a deletion sweep. The support contract layer (`frontier_contracts`, `frontier_constraints`, `frontier_conditioning`, `frontier_dominance`, `frontier_recommendation`) is largely well-scoped (~4% removable) and stays.
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

The following claims have been re-validated against the working tree at HEAD. Each file:line reference resolves to the current code.

| # | Finding | Evidence | Severity |
|---|---|---|---|
| F1 | False engine layer — `frontier_engine_base.py` defines no ABC/Protocol; runner uses hard-coded `if/else` dispatch. Module has 2 importers (1 production + 1 test) — original audit's "6 importers" was symbol-count, not file-count | `frontier_engine_base.py:28` (state dataclasses, no ABC); `run_single_stage_frontier_campaign.py:38, 916-942`; `tests/geo/test_frontier_archive.py:25` | High (SOLID/SRP) |
| F2 | NSGA3 path is structurally suspicious — wired in CLI/runner/tests but no validated production artifact found | `frontier_engine_nsga3.py:33-44, 161`; `run_single_stage_frontier_campaign.py:124, 916-942, 1072-1094`; cited in `autoresearch/program_hbt_topology_surrogate_legacy_vmec.md:310` but no completed `frontier_engine: nsga3` JSON/JSONL artifact | High (YAGNI) |
| F3 | Frontier scalarization leaked into central objective module — wrong dependency direction | `single_stage_objectives.py:255-588` (full frontier cluster). Contains `apply_frontier_scalarization_override` (282), `_frontier_chebyshev_goal` (473), `_frontier_epsilon_penalties` (533-555), `_frontier_alm_base_total_grad` (463), `_frontier_excess_penalty` (558-588), `augment_frontier_metric_state` (255), `_frontier_goal_component_total_grad` (404), `_frontier_penalty_geometry_total_grad` (429). Note: `augment_frontier_metric_state` consumes non-frontier `_objective_gradient` (line 114) — that must remain in `single_stage_objectives` and be back-imported into `frontier_scalarization` after the move. | High (SSOT) |
| F4 | `annotate_search_evaluation_finiteness` defined in frontier module is consumed by non-frontier code | defined `frontier_constraints.py:61`; consumed `single_stage_objectives.py:8` and lines 690/740/794/818/1516 | Medium (SSOT) |
| F5 | Epsilon-threshold magic-string keys duplicated across 3 modules with distinct semantics — the violation is shared keys (`epsilon_constraint_qa_max`, `epsilon_constraint_boozer_max`), not duplicated logic. Plus the CLI argparse site (out of scope for Phase 4.1) and a runner attribute-shuffle. | Three semantically-distinct sites: `frontier_archive.py:596-621` (post-hoc certification — reads via Mapping); `frontier_scalarization.py:633-647` (lane spec writer — writes to dict); `single_stage_objectives.py:539-555` (penalty enforcement — reads via attribute access on a frozen `frontier_goal_config`). Out-of-scope satellites: argparse `add_argument` definitions exist only in `run_single_stage_goal_mode_comparison.py:363-364` (the frontier runner inherits this parent at `run_single_stage_frontier_campaign.py:119` via `parents=[...]`, so it does NOT have its own `add_argument`); the runner does, however, reference the keys at `:322-323` inside an attribute-copy loop over `lane_spec.scalarization_params`. Counts (r6.1 corrected; r6 said "9 hits across 5 files"): **string-literal hits in canonical scope** (`banana_opt/` + `run_single_stage_*.py`): 8 across 4 files; **identifier hits in canonical scope**: 14 across 6 files (adds `frontier_evaluator.py:771-772` attribute access and `single_stage_objectives.py` attribute reads). | Medium (DRY/SSOT) |
| F6 | Hypervolume recomputed without memoization, called ≥4× per campaign, leave-one-out is N× | `frontier_archive.py:446-477` (`annotate_hypervolume_contributions`); also called at `frontier_archive.py:341` (serialize annotate) + `:355` (direct total), `frontier_campaign_reporting.py:303` (per-prefix in `build_frontier_hypervolume_history`), `:424` (final certified), and `frontier_runtime_calibration.py:231` (per-lane `update_frontier_early_stop_status`) | Medium (Performant) |
| F7 | Runtime calibration registry has 2 profiles differing by 1 int + 1 string + 1 tuple-of-strings (plus tautological `profile_name`) | `frontier_runtime_calibration.py:56-85` (registry; full file 272 LOC). Profiles `reduced_fixture_v1` vs `canonical_seed_v1` differ in: `default_early_stop_patience_lanes` (2 vs 3), `profile_name` (echoes registry key — tautological), `calibration_basis` (`("reduced_fixture_multilane_smoke", "deterministic_resume_smoke")` vs `("canonical_seed_bridge_smoke", "canonical_seed_resume_smoke")`) | Low (KISS) |
| F8 | `frontier_dominance.py` is misnamed (~85% Pareto normalization, ~15% dominance) | `frontier_dominance.py:8-72` (normalization rules); only `frontier_dominance.py:153-203` (~50 LOC) is dominance | Low (SOLID/naming) |
| F9 | `frontier_engine_multilane_local.py` is 79 LOC — 1 dataclass + 1 weight-share generator, not an engine. Has 3 importers (runner, reporting, scalarization) | `frontier_engine_multilane_local.py`; importers `run_single_stage_frontier_campaign.py:48`, `frontier_campaign_reporting.py:38`, `frontier_scalarization.py:8` | Low (SOLID/naming) |
| F10 | Hand-rolled JSON ser/de boilerplate (~294 LOC, corrected from r1-r5 estimate of ~150) across multiple dataclasses | `frontier_evaluator.py:70-86 (~17 LOC), 106-156 (~51 LOC), 207-235 (~29 LOC)`; `frontier_engine_base.py:28-191 (~164 LOC)`; `frontier_archive.py:173-205 (~33 LOC)` | Low (DRY) |
| F11 | Defensive `getattr(args, ..., default)` against `argparse.Namespace` (always set by `add_argument(default=...)`) | `frontier_campaign_reporting.py:181-204, 246-255, 412-421` | Low (KISS) |
| F12 | ~~Duplicate~~ Conditional re-resolution of `runtime_defaults` and `hypervolume_reference` — both pairs operate on **different inputs** (lane-count reconciliation, certified-vs-archive members) and are not collapsible. Action: add a one-line comment at each site, do NOT collapse. | `run_single_stage_frontier_campaign.py:762-771 vs 827-837` (second is gated on `len(lane_specs) != runtime_defaults.num_lanes` — resumed lane count overrides requested); `:874-878 vs 1035-1039` (initial uses `members=archive_members`, second uses `members=certified_members`) | Trivial (documentation) |
| F13 | `# noqa: F401` import marks unused symbol that is in fact used elsewhere | `run_single_stage_frontier_campaign.py:50` imports `generate_multilane_local_specs` (unused at runner level; actually used via `frontier_scalarization.py:714`) | Trivial |
| F14 | `pymoo` is a soft optional runtime requirement, not a declared dep | `frontier_engine_nsga3.py:33-44` (try/except ImportError); not in `pyproject.toml` / `requirements.txt` / `setup.py` | Note (correction to prior claim) |
| F15 | `_require_*` schema helpers in `frontier_contracts.py` are NOT pre-existing in artifact_contracts; they are first-occurrence | `frontier_contracts.py:491-518`; no `_require_*` in `artifact_contracts.py` / `constraint_contract.py` / etc. | Note (correction to prior claim) |
| F16 | `_load_banana_opt_module`/`EXAMPLES_ROOT` is in 1 test file, not 6; but importlib bootstrap pattern repeats across 5 files | only `test_frontier_evaluator.py:16, 25`; other 5 use small `importlib.import_module(...)` wrappers | Note (correction to prior claim) |
| F17 | **`FRONTIER_REFERENCE_MODE_SHARED` (a.k.a. multilane_local) is mislabeled as Pareto exploration but is a 1-parameter linear-scalarization sweep over 2 of 4 objectives, with non-extremal endpoints.** Highest-impact algorithm finding for end-user interpretation. | `frontier_engine_multilane_local.py:54-60` clamps `iota_share` ↔ `volume_share` to [0.2, 0.8]; never reaches simplex vertices `(1,0)` / `(0,1)`; `qa_error` and `boozer_residual` axes are not parameters at all. For true 4-D Das-Dennis coverage, users must use `FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX`. | High (Naming/correctness-of-claim) |
| F18 | ε-certifier silently passes when threshold key is missing from `scalarization_params` | `frontier_archive.py:603-614` (`_evaluate_epsilon_constraint_status`) reads `scalarization_params.get("epsilon_constraint_qa_max")` with `if limit is None: continue`. A typo, ad-hoc rerun_contract built outside `_reference_scalarization_params`, or sidecar JSON missing keys silently certifies members against absent thresholds. No warning, no `KeyError`, no test exercises this path. | Medium (Silent-pass correctness) |
| F19 | ε-certification uses zero floating-point slack — strict `excess > 0.0` rejects `value == limit + 1e-16` | `frontier_archive.py:615-617`. Members optimized to the boundary via soft penalty pulls (asymptotic case) may fail certification within numerical precision. No on-boundary, just-above, or just-below test exists. | Low-Medium (Numerical robustness) |
| F20 | `dominates()` does not guard NaN — IEEE 754 comparisons return False, so NaN-bearing members are silently treated as non-dominating AND non-dominated (could persist in archive forever) | `frontier_dominance.py:175-203`. Standard ingest filters NaN via `_as_finite_float` upstream; hand-built `FrontierArchiveMember` instances with NaN floats slip past dominance entirely. Defensive responsibility delegated to ingest, not enforced at the dominance boundary. | Low (Ingest-path-dependent) |
| F21 | Hypervolume reference is not enforced as a true nadir | `frontier_archive.py:638-646` (`_hypervolume_boxes`): per-axis extent `max(0, extent)` clip + drop-if-all-zero filter silently treats members worse-than-reference as zero-contribution along the offending axis; no warning. `parse_hypervolume_reference` and `resolve_hypervolume_reference` accept user values verbatim with no nadir-domination check. | Low (Math safe; hides config errors) |
| F22 | `frontier_conditioning.py` is misnamed — it's a **diagnostic max/min ratio gate**, NOT a preconditioner | No diagonal scaling, no Hessian approximation, no `D⁻¹` applied to gradients anywhere. Computes max/min ratio of `\|J_QS\|, \|J_Boozer\|, \|J_iota\|, \|J_volume\|, \|trust_penalty\|, \|epsilon_penalty\|` against `FRONTIER_CONDITIONING_MAX_RATIO=1e3`. Computed in raw objective units (ignores `objective_metric_scale`) — gate verdict doesn't reflect the metric-scale-normalized space the optimizer actually traverses. Not consumed by the optimizer. | Low (Misleading filename + scale-mismatch) |
| F23 | Derived scalarization fields are NOT in `_FINITE_*` finiteness-check tables | `frontier_constraints.py:17-57` covers raw evaluator outputs but misses every field the scalarization layer produces: `frontier_rank_total`, `frontier_base_total`, `frontier_trust_penalty`, `frontier_epsilon_penalty`, `frontier_goal_total`, `frontier_chebyshev_total` (deltas are stabilized by the LSE-shift trick at `single_stage_objectives.py:500-501`, so they don't overflow under positive sharpness — corrected r6.3). Risk: a non-finite `length_weight`, `J_len`, or any penalty summand propagates into `frontier_base_total` / `frontier_rank_total` undetected, since these derived fields aren't re-checked after assembly. | Low (Silent NaN propagation in derived sums; not the Chebyshev softmax) |

---

## 3. Decision Log (fill before executing Phase 1)

### Decision D1 — Keep or drop NSGA3?

**Question:** Is `--frontier-engine nsga3` exercised by any production campaign?

**Investigation TODO:**
- [ ] **D1.1** Grep all completed JSONL/JSON artifacts in `autoresearch/` for `"frontier_engine": "nsga3"`
  ```bash
  rg -l '"frontier_engine":\s*"nsga3"' /Users/suhjungdae/code/columbia/autoresearch/ 2>/dev/null
  ```
- [ ] **D1.2** Grep this repo's `tmp/` and `examples/single_stage_optimization/outputs_*` for the same string
  ```bash
  rg -l '"frontier_engine":\s*"nsga3"' tmp/ examples/single_stage_optimization/outputs_* 2>/dev/null
  ```
- [ ] **D1.3** Check lab notes / Stellaris VM run history for `--frontier-engine nsga3` invocations
- [ ] **D1.4** Ask: does anyone on the team have a campaign result that requires NSGA3 (Pareto coverage NOT reachable by `multilane_local` + ε-constraint)?

**Decision criteria:**
- [ ] D1.A — Drop NSGA3 if zero validated production artifacts exist (most likely outcome based on prior audit)
- [ ] D1.B — Keep NSGA3 only if (a) production artifact exists AND (b) there's a benchmark showing NSGA3 reaches Pareto regions multilane_local cannot

**Decision recorded here:** _TBD — fill in: DROP / QUARANTINE / KEEP_

If **DROP** → execute Phase 1A
If **QUARANTINE/KEEP** → execute Phase 1B (define real Engine Protocol + benchmark gate)

### Decision D2 — Are NSGA3-targeted tests fixtures or production validation?

- [ ] **D2.1** If D1 = DROP, plan to delete **all NSGA3-tied tests enumerated in Phase 1A.4** — 4 blocks + 1 helper across 3 test files totaling **~628 LOC** (153 + 154 + 192 + 127 + 2 helper; r6-corrected, r6.1 includes helper for arithmetic consistency; line numbers verified at HEAD):
  - `tests/geo/test_frontier_evaluator.py:111-263` (~153 LOC) + helper `load_frontier_engine_nsga3_module` at lines 34-35
  - `tests/geo/test_single_stage_workflow_helpers.py:5150-5303` (~154 LOC) — `test_frontier_campaign_nsga3_records_generation_summary`
  - `tests/geo/test_single_stage_workflow_helpers.py:5304-5495` (~192 LOC) — `test_frontier_campaign_nsga3_resume_reuses_saved_engine_artifacts`
  - `tests/geo/test_frontier_contracts.py:189-315` (~127 LOC) — `test_summary_validator_accepts_optional_nsga3_fields`
  - See Phase 1A.4 (Section 4.1) for the canonical enumeration; this D2.1 entry must stay in sync with it
- [ ] **D2.2** If D1 = KEEP, plan to add a benchmark test that proves NSGA3 reaches at least one Pareto member that `multilane_local` cannot under equal evaluation budget

### Decision D3 — Preserve frontier_runtime_calibration JSON shape?

- [ ] **D3.1** Search consumed-by graph for any external tool that reads `frontier_runtime_calibration` block from a frontier summary JSON
  ```bash
  rg "frontier_runtime_calibration" --type json --type py 2>/dev/null
  ```
- [ ] **D3.2** If consumers exist outside this repo (lab notebooks, dashboards), preserve schema_version + JSON keys when collapsing the registry

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

- [ ] **1A.1** Delete `examples/single_stage_optimization/banana_opt/frontier_engine_nsga3.py` (372 LOC)
  - Pre-check: confirm no symbol from this module is imported anywhere except the runner
  ```bash
  rg "from .frontier_engine_nsga3|from banana_opt.frontier_engine_nsga3" --type py
  ```
- [ ] **1A.2** Remove NSGA3 dispatch in runner
  - File: `examples/single_stage_optimization/run_single_stage_frontier_campaign.py`
  - Lines: 52-56 (imports), 130 (CLI choices), 916-942 (engine branch), 1072-1094 (summary post-processing)
  - Replace `--frontier-engine` choices with single literal or remove flag entirely if only one mode remains
- [ ] **1A.3** Remove the optional NSGA3 summary payload validator and **all six** optional NSGA3 summary fields entirely
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
    Expected: zero hits in source (mentions in retired `docs/single_stage_frontier_global_pareto_plan_*` are fine).
- [ ] **1A.4** Delete ALL NSGA3-tied tests across the test suite
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
- [ ] **1A.5** Update or retire `docs/single_stage_frontier_global_pareto_plan_2026-04-22.md` (mark superseded; keep as historical record)
- [ ] **1A.6** Update `docs/single_stage_frontier_gradient_contract_impl_plan_2026-04-26.md` — much of its motivation was NSGA3 gradient consistency; mark sections that no longer apply
- [ ] **1A.7** Notify `autoresearch/program_hbt_topology_surrogate_legacy_vmec.md:310` consumers; remove or update reference
- [ ] **1A.8** Remove `# noqa: F401` import of `generate_multilane_local_specs` at `run_single_stage_frontier_campaign.py:50` (duplicate; real consumer is `frontier_scalarization.py:714`)
- [ ] **1A.9** Run full test suite; expect green
  ```bash
  ./run_tests tests/geo/test_frontier_*.py
  ```
- [ ] **1A.10** Confirm runner still passes a smoke run with `multilane_local` (the only remaining engine)

**Estimated LOC removed in Phase 1A (r6 measurement, r6.1 arithmetic fix):** ~478 source + **~628 test** = **~1,106 LOC**.
- Source breakdown: `frontier_engine_nsga3.py` 372 LOC + runner edits ~60 LOC (imports 52-56, choices 130, dispatch 916-942, post-processing 1072-1094) + `frontier_contracts.py` validator + 6-field unwiring ~45 LOC + `# noqa: F401` 1 LOC ≈ **~478 LOC** (the prior r1-r5 estimate of ~600 LOC was high by ~25%).
- Test breakdown: `test_frontier_evaluator.py:111-263` 153 LOC + helper at 34-35 (~2 LOC) + `test_frontier_contracts.py:189-315` 127 LOC + `test_single_stage_workflow_helpers.py:5150-5303` 154 LOC + `:5304-5495` 192 LOC = **628 LOC** exact (153+2+127+154+192). r5 estimate of ~612 was within ~3%; r6 had a transcription bug that wrote "~626" alongside "≈ ~628" — corrected here.

### 4.2 Phase 1B — QUARANTINE NSGA3 (only if D1 = KEEP)
**Acceptance criterion:** there exists a real `Engine` Protocol with both `multilane_local` and `nsga3` as conforming implementations; runner dispatch is via the protocol, not `if/else`; a benchmark test proves NSGA3 utility.

**Required additional fixes if Phase 1B is chosen (per r6.2 algorithm audit; these are blockers for KEEP):**
- [ ] **1B.A** Fix NSGA-III branch to populate `lane_records_by_id` so the final `frontier_lane_records` summary is engine-agnostic. Without this, downstream consumers must branch on engine name and currently produce stale records for NSGA-III runs.
- [ ] **1B.B** Eliminate or document the double-evaluation in `_ArchiveTrackingCallback.notify` (`frontier_engine_nsga3.py:107-109`). Either consume `algorithm.opt`/`algorithm.off` (the offspring set actually changed each generation) or attach the callback to pymoo's `_advance` hook so the population is read once per generation, not re-evaluated.
- [ ] **1B.C** ~~Serialize pymoo internal RNG state for resume reproducibility~~ — **r6.3 demoted from blocker.** On re-verification, the current resume path is binary: `load_nsga3_frontier_campaign_artifacts` (`run_single_stage_frontier_campaign.py:918-924`) either loads complete prior-run artifacts and skips `run_nsga3_frontier_campaign` entirely, or runs fresh. There is no "continue from generation N/M" path, so pymoo's internal RNG state being unserialized is moot for the existing resume design. **This becomes a real concern only if 1B grows a partial-run-continuation path** — flag for that hypothetical future work, do not block KEEP on it.
- [ ] **1B.D** Document or fix `frontier_reference_mode` constraint: NSGA-III currently silently restricts to `ACHIEVEMENT_FULL_SIMPLEX` at `frontier_engine_nsga3.py:178-184`. If multilane modes should also work with NSGA-III, generalize; otherwise document the restriction in the Engine Protocol's docstring.

- [ ] **1B.1** Define `frontier_engine_protocol.py` (new file, ~50 LOC)
  ```python
  from typing import Protocol, runtime_checkable

  @runtime_checkable
  class FrontierEngine(Protocol):
      def run(self, spec: SingleStageFrontierEvaluatorSpec, budget: int, ...) -> EngineArtifacts: ...
      def load_artifacts(self, path: Path) -> EngineArtifacts | None: ...
      def name(self) -> str: ...
  ```
- [ ] **1B.2** Refactor `frontier_engine_nsga3.py` to expose a class implementing `FrontierEngine`
- [ ] **1B.3** Wrap the inline `multilane_local` execution path (currently in runner) into a class implementing `FrontierEngine`
- [ ] **1B.4** Replace runner `if/else` (lines 916-942) with a registry lookup
  ```python
  ENGINE_REGISTRY: Mapping[str, Callable[[], FrontierEngine]] = MappingProxyType({
      "multilane_local": MultilaneLocalEngine,
      "nsga3": NSGA3Engine,
  })
  engine = ENGINE_REGISTRY[args.frontier_engine]()
  ```
- [ ] **1B.5** Add `tests/geo/test_frontier_engine_benchmark.py` — must show NSGA3 reaches a Pareto member that `multilane_local` cannot under equal budget
- [ ] **1B.6** Move `pymoo` from optional try/except to a documented optional extra (e.g., `pyproject.toml` `[project.optional-dependencies]`: `nsga3 = ["pymoo>=0.6"]`)

---

## 5. Phase 2 — SSOT restoration (independent of Phase 1)

### 5.1 Move frontier scalarization back to `frontier_scalarization.py`

**Acceptance criterion:** `grep -n "frontier" examples/single_stage_optimization/banana_opt/single_stage_objectives.py` returns matches only for *consumer* references (e.g., `if args.use_frontier_scalarization:` calls into `frontier_scalarization.apply_frontier_scalarization_override(...)`).

- [ ] **2.1.1** Audit `single_stage_objectives.py:255-588` (the full frontier cluster, not the narrower 255-558 used in r1-r5) — list each function/symbol that mentions frontier
  ```bash
  rg -n "frontier" examples/single_stage_optimization/banana_opt/single_stage_objectives.py
  ```
- [ ] **2.1.2** Move `apply_frontier_scalarization_override` (line 282) to `frontier_scalarization.py`
- [ ] **2.1.3** Move `_frontier_chebyshev_goal` (line 473) to `frontier_scalarization.py` as private helper
- [ ] **2.1.4** Move `_frontier_epsilon_penalties` (line 533-555) to `frontier_scalarization.py` (will be re-merged in Phase 4 with archive's epsilon code)
- [ ] **2.1.5** Move `_frontier_alm_base_total_grad` (line 463). Note: the function is pure NumPy with no JAX, no closures, and no module-level mutable state — verified per r6 audit. Move is mechanically trivial; the prior r1-r5 "JAX function purity" warning was over-cautious.
- [ ] **2.1.5b** ALSO move (added in r6 — these are part of the frontier cluster but were missed in r1-r5):
  - `_frontier_excess_penalty` (line 558-588) — body extends past r5's 255-558 range cap
  - `augment_frontier_metric_state` (line 255)
  - `_frontier_goal_component_total_grad` (line 404)
  - `_frontier_penalty_geometry_total_grad` (line 429)
  - **Back-import constraint:** `augment_frontier_metric_state` calls `_objective_gradient` (line 114), which is non-frontier and must remain in `single_stage_objectives.py`. After the move, `frontier_scalarization.py` imports `_objective_gradient` consumer-style from `single_stage_objectives` — this is a legitimate consumer-direction import, not a cycle.
- [ ] **2.1.6** Update import sites — `single_stage_objectives.py` should `from .frontier_scalarization import apply_frontier_scalarization_override` (one-line import) where it now defines the body
- [ ] **2.1.7** Verify dependency direction is clean
  ```bash
  rg -n "from .frontier|from banana_opt.frontier" examples/single_stage_optimization/banana_opt/single_stage_*.py
  ```
- [ ] **2.1.8** Run gradient-contract tests to confirm no regression
  ```bash
  ./run_tests tests/geo/test_single_stage_alm_integration.py tests/geo/test_frontier_evaluator.py
  ```

### 5.2 Move generic finiteness helper out of `frontier_constraints.py`

**Context:** `annotate_search_evaluation_finiteness` (`frontier_constraints.py:61`) is used by non-frontier code (`single_stage_objectives.py:8` and lines 690/740/794/818/1516). It is misnamed/misplaced.

- [ ] **2.2.1** Create new module `examples/single_stage_optimization/banana_opt/search_evaluation.py`
- [ ] **2.2.2** Move `annotate_search_evaluation_finiteness` + ALL supporting helpers (r6 expanded list — r5 missed two): `_FINITE_SCALAR_FIELDS`, `_FINITE_VECTOR_FIELDS`, `_FINITE_VECTOR_LIST_FIELDS`, `_FINITE_EPS` at `frontier_constraints.py:17-57` to `search_evaluation.py`. Also note the in-file self-consumer at `frontier_constraints.py:166` (inside `evaluate_frontier_trust_penalty`) — after the move, `frontier_constraints` becomes a consumer of the new module (back-import).
- [ ] **2.2.3** Update import sites — actual count is **2 source files + 1 test file** (r6 correction; r1-r5 said "6+" which conflated call sites with import sites — `single_stage_objectives.py` has 5 call sites but only 1 import statement):
  - `single_stage_objectives.py:8` — change import path
  - `frontier_constraints.py:166` — add back-import after the helper moves out
  - `tests/geo/test_frontier_constraints.py:27,30` — update test imports
- [ ] **2.2.4** Verify with grep that `frontier_constraints` no longer leaks into non-frontier code
  ```bash
  rg "from .frontier_constraints|from banana_opt.frontier_constraints" examples/ | grep -v frontier_
  ```
  Expected: zero hits.

**Estimated LOC moved (not deleted):** ~50 LOC. Net repo LOC change: 0. SSOT restoration: yes.

---

## 6. Phase 3 — Structural renames (low risk, high readability)

### 6.1 Rename `frontier_engine_base.py` → `frontier_progress_state.py`

**Context:** F1 — file contains zero engine abstraction; it holds JSON ser/de for `FrontierLaneContract`/`FrontierLaneRecord`/`FrontierCampaignProgress`.

- [ ] **3.1.1** `git mv examples/single_stage_optimization/banana_opt/frontier_engine_base.py examples/single_stage_optimization/banana_opt/frontier_progress_state.py`
- [ ] **3.1.2** Update **both file importers** (r6 correction — r1-r5 said "all 6 importers"; that count was the original audit's symbol-count, not file-count, as F1 itself flags). The two files: `examples/single_stage_optimization/run_single_stage_frontier_campaign.py:38` (production) and `tests/geo/test_frontier_archive.py:25` (test, via importlib).
  ```bash
  rg -l "frontier_engine_base" examples/ tests/ | xargs sed -i '' 's/frontier_engine_base/frontier_progress_state/g'
  ```
  (verify on macOS `sed -i ''` syntax; use a portable Python script if uncertain)
- [ ] **3.1.3** Update test file `tests/geo/test_frontier_archive.py:25` reference (already covered by 3.1.2 sed but called out explicitly for clarity)
- [ ] **3.1.4** Run frontier test suite to confirm imports resolve
  ```bash
  ./run_tests tests/geo/test_frontier_*.py
  ```

### 6.2 Fold `frontier_engine_multilane_local.py` into `frontier_scalarization.py`

**Context:** F9 — the file is 79 LOC, 1 dataclass + 1 weight-share generator, only consumer is `frontier_scalarization.py:714`. It is not an engine.

- [ ] **3.2.1** Move `FrontierLaneSpec` dataclass and `generate_multilane_local_specs` function to top of `frontier_scalarization.py`
- [ ] **3.2.2** Update importers — sites currently importing `FrontierLaneSpec` from `frontier_engine_multilane_local` should import from `frontier_scalarization`
  ```bash
  rg "from .frontier_engine_multilane_local|from banana_opt.frontier_engine_multilane_local" examples/ tests/
  ```
- [ ] **3.2.3** Delete `frontier_engine_multilane_local.py`
- [ ] **3.2.4** Remove the redundant `# noqa: F401` import in `run_single_stage_frontier_campaign.py:50` (already covered in 1A.8 if Phase 1A executes; otherwise do here)

### 6.3 Rename `frontier_dominance.py` → `frontier_pareto_normalization.py`

**Context:** F8 — only ~50 of 373 LOC are dominance logic; the rest is Pareto normalization.

- [ ] **3.3.1** Decide whether to rename or keep — rename is honest but causes 6 importer updates and minor diff churn
- [ ] **3.3.2** If renaming: `git mv` + bulk replace; if not: add a one-line module docstring explaining the actual contents

**Phase 3 net LOC:** 0 deleted; ~3 import updates per renamed module. **Mental model improvement: significant.**

---

## 7. Phase 4 — DRY / SSOT consolidations

### 7.1 Centralize epsilon thresholds (F5)

**Acceptance criterion:** the strings `epsilon_constraint_qa_max`, `epsilon_constraint_boozer_max` (and similar threshold keys) appear exactly once in source — inside a frozen dataclass in `frontier_contracts.py`.

- [ ] **7.1.1** Add to `frontier_contracts.py`. Two constructors are needed (per r6 audit) because the three call sites consume the thresholds differently — Mapping access on `rerun_contract.scalarization_params` versus attribute access on a frozen `frontier_goal_config` dataclass:
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
- [ ] **7.1.2** Replace 3 hand-rolled implementations:
  - `frontier_archive.py:596-621` (`_evaluate_epsilon_constraint_status`) — consume `EpsilonThresholds.from_rerun_contract`
  - `frontier_scalarization.py:633-647` — write via `EpsilonThresholds.as_payload`
  - `single_stage_objectives.py:533-555` (`_frontier_epsilon_penalties`) — consume `EpsilonThresholds.from_goal_config` (r6 correction: r1-r5 said `from_rerun_contract` but the call site uses attribute-access; line range corrected from 539-555 to 533-555)
  - Note: if Phase 2 already moved `_frontier_epsilon_penalties` into `frontier_scalarization.py`, only 2 sites remain in this step (the moved function still consumes `from_goal_config`)
- [ ] **7.1.3** Add unit test: `test_epsilon_thresholds_round_trip` in `tests/geo/test_frontier_contracts.py`
- [ ] **7.1.4** Verify no string-key duplication remains
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

- [ ] **7.3.1** Add a parameterized `select_best` to `frontier_recommendation.py`. **Heterogeneity caveat (r6):** the 5 sites are not all reducible to `Sequence[tuple[str, Direction]]` — `_balanced_policy_sort_key` (`frontier_recommendation.py:210`) uses a scalar reduction (`balanced_policy_score`), and `_closest_to_seed_sort_key` (`:271`) uses None-aware `distance_from_seed`. The signature must therefore take a `KeyFn` callable rather than a tuple-priority spec:
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
- [ ] **7.3.2** Build a frozen `RECOMMENDATION_POLICIES: Mapping[PolicyName, Callable] = MappingProxyType({...})` registry where each value is a `partial(select_best, key=..., gate=..., rationale=...)`
- [ ] **7.3.3** Delete the 4 per-policy sort-key functions
- [ ] **7.3.4** Update `frontier_archive.archive_best_by_metric` to call `select_best(key=lex_priority([(metric, "desc")]), gate=lambda m: True, rationale=...)`

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

- [ ] **8.1.1** Identify the function `frontier_archive_hypervolume` and confirm its inputs are hashable
  - File: `frontier_archive.py` (find via `grep -n "def frontier_archive_hypervolume"`)
- [ ] **8.1.2** Wrap with `lru_cache` keyed on tuple-of-tuples of objective vectors + reference point. The tuple MUST be canonicalized by **lex-sort over the four-objective vectors themselves** (in `PARETO_OBJECTIVE_SPECS` order: iota, volume, qa_error, boozer_residual), NOT by `member_id` (r6 correction; r1-r5 said "sorted by `member_id` or similar" — but `FrontierArchiveMember.member_id` at `frontier_archive.py:38` is built from `(campaign_id, lane_id, archive_state)` and carries lane/campaign identity, not Pareto content; two archives with identical hypervolume but different IDs would miss the cache). Hypervolume is invariant under permutation of S, so canonical ordering by the box-content tuples is the natural and correct equivalence class — and it matches what `_hypervolume_boxes` (`frontier_archive.py:624-648`) already operates on.
  ```python
  @functools.lru_cache(maxsize=128)
  def _hypervolume_cached(
      objective_vectors: tuple[tuple[float, ...], ...],  # canonical: sorted lex over the 4-tuples
      reference: tuple[float, ...],
  ) -> float: ...
  ```
- [ ] **8.1.3** Refactor `annotate_hypervolume_contributions` to call the cached version
- [ ] **8.1.4** Add a benchmark test using a per-call counter (not just `cache_info`) that asserts: across one annotation pass + one serialize (annotate + direct total) + one report + one history build + per-lane early-stop calls, the number of *underlying* `_hypervolume_cached` invocations equals the number of unique input tuples encountered — typically (1 final-archive call) + (N leave-one-outs) + (M unique prefix archives from history) + (L per-lane early-stop calls, typically distinct from prefix archives) — and that the same final-archive computation does not repeat across annotate/serialize/report. Note: `lru_cache.cache_info()` reports decorator-wrapper hits, which is fine as a sanity cross-check but does not by itself prove "no duplicated underlying compute"; a per-call counter on the underlying computation is the authoritative instrumentation.
- [ ] **8.1.5** Verify reporting paths (`frontier_campaign_reporting.py:303, 424`) hit the cache (cache_hits ≥ 2 after annotation populates it)
- [ ] **8.1.6** (Optional, low priority — algorithmic redesign) Investigate incremental leave-one-out hypervolume. **Caveat (r6):** in this code's 4D objective space (`PARETO_OBJECTIVE_SPECS` has 4 entries: iota, volume, qa_error, boozer_residual; `frontier_engine_nsga3.py:65` sets `n_obj=4`), best-known leave-one-out exclusive-contribution algorithms (HSO, WFG-class) are O(N^{d−1}) worst case for d≥4 — not the routine O(N) optimization the r1-r5 wording implied. In 2D the contributions reduce to neighbor-rectangle differences in O(N log N), but that's not the regime here. If pursued, becomes a separate algorithmic-redesign task requiring (a) a literature review of recent multi-objective hypervolume-contribution algorithms in dimension 4, (b) a correctness benchmark against the naive O(N²) reference, and (c) a wall-clock benchmark to confirm the asymptotic actually wins for typical campaign archive sizes (often N < 50, where constant factors dominate).

**Estimated wall-clock improvement on long campaigns (revised per r6 audit):** **sub-2× in typical campaigns** where prefix-history sweeps (`build_frontier_hypervolume_history`) and per-lane early-stop hypervolume (`frontier_runtime_calibration.py:231`) dominate the call count — those have unique inputs and cannot be cache-deduplicated. The factor approaches "up to ~3×" only in degenerate cases where repeated-final-archive computations (annotate at `:424` + serialize annotate at `:341` + serialize direct at `:355`) dominate. The realized factor depends on the ratio (repeated-final-archive calls : unique-input calls) for a given campaign size. The cache is still worth landing — it's a few lines of code with no downside — but the user-facing benefit is more modest than r1-r5 claimed. Optional 8.1.6 is non-trivial in 4D (see caveat above).

### 8.2 Collapse runtime calibration registry (F7, with D3 guard)

**Acceptance criterion:** if D3 = preserve JSON shape, the on-disk `frontier_runtime_calibration` block in summary JSON remains identical for equivalent inputs.

- [ ] **8.2.1** Confirm D3 outcome (D3.1, D3.2 above)
- [ ] **8.2.2** Replace `FRONTIER_RUNTIME_CALIBRATION_PROFILES` registry at `frontier_runtime_calibration.py:56-85` (r6 line correction; r1-r5 said `:11-272`, but line 11 is the dataclass decorator on `FrontierRuntimeCalibrationProfile`, not the registry — the registry occupies lines 56-85, and the rest of the 272-line file is helpers) with a single `FrontierRuntimeDefaults` frozen dataclass + `resolve_runtime_defaults(args)` factory. Cover three differing fields: `default_early_stop_patience_lanes`, `profile_name`, and `calibration_basis` (per F7 r6 expansion).
- [ ] **8.2.3** Preserve serialization shape via a `to_json_dict` method that emits the same keys profiles previously emitted (basis string, etc.)
- [ ] **8.2.4** Delete `_resolve_calibration_profile` and the profile-name CLI argument if D3 allows
- [ ] **8.2.5** Update the calibration-profile test to match new shape. **Test name (r6):** the only test currently asserting profile shape is `test_runtime_defaults_use_explicit_calibration_profile` at `tests/geo/test_frontier_contracts.py:407` (asserts `lane_budget=300, total_budget=900, checkpoint_every=5, early_stop_patience_lanes=2`). r1-r5 referred vaguely to "test `test_frontier_runtime_calibration`" — there is no test file or function by that name; the canonical test lives in the contracts test file.

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

- [ ] **8.3.1** Decide between Approach A and Approach B based on RSS measurement. The runner imports the engine module via `examples/single_stage_optimization` on the path, so the measurement command must set `PYTHONPATH` (or `cd` first):
  ```bash
  PYTHONPATH=examples/single_stage_optimization \
    /usr/bin/time -l python -c "import banana_opt.frontier_engine_nsga3" 2>&1 | tail -5
  ```
  Run with and without `pymoo` available in the same conda env to get RSS deltas. If cold-start cost (with pymoo present) is < 50 ms and < 30 MB RSS, prefer Approach B (smaller diff).
- [ ] **8.3.2 (A)** Move `_FrontierNSGA3Problem` and `_ArchiveTrackingCallback` class definitions inside the functions that need them; ensure no other code references the classes at module level (check tests too)
- [ ] **8.3.2 (B)** Document the existing `try/except ImportError` at module top as the intentional lazy mechanism with a comment
- [ ] **8.3.3** Document `pymoo` as optional extra in `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  nsga3 = ["pymoo>=0.6"]
  ```
- [ ] **8.3.4** Add an isolated CI job (or pytest marker) that exercises the no-pymoo path. **Do NOT run `pip uninstall pymoo` in a developer's local env.** Use a fresh venv or CI matrix entry:
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

- [ ] **8.4.1** Inspect `load_resume_lane_specs` (runner lines 635-647) and `load_resume_manifest` (lines 650-653)
- [ ] **8.4.2** Combine into one read returning `(lane_specs, manifest)`
- [ ] **8.4.3** Verify resume tests still pass

### 8.5 Document conditional re-resolution sites (revised per r6 — F12 downgraded)

**Context (r6):** r1-r5 framed both runner re-resolutions as "duplicates that should be collapsed into a single pass." The r6 audit found this is wrong:

- Lines 762-771 vs 827-837: the second resolution at 828 is **gated on `len(lane_specs) != runtime_defaults.num_lanes`** — i.e., when resumed lane specs override the requested lane count, the runtime defaults must be re-resolved with the new lane count. This is a legitimate reconciliation, not a duplicate.
- Lines 874-878 vs 1035-1039: the second resolution uses `members=certified_members` (post-loop final certified set) versus the initial `members=archive_members` (possibly empty on fresh run / resumed list on resume). These are different inputs producing potentially different references. Not a duplicate.

Action: **document the why, do not collapse.**

- [ ] **8.5.1** Add a one-line comment at runner line 827-828 explaining the re-resolution is gated on resumed-lane-count override of requested `num_lanes`. Do NOT collapse.
- [ ] **8.5.2** Add a one-line comment at runner line 1035 explaining the re-resolution uses `certified_members` (post-loop) vs initial `archive_members`. Do NOT collapse.
- [ ] **8.5.3** ~~If both resolutions genuinely depend on later state, document why with a one-line comment~~ — **subsumed by 8.5.1 / 8.5.2 above (r6: this is the verified state, not a maybe-condition).**

---

## 9. Phase 6 — Mechanical cleanup (do last)

### 9.1 Evaluator (F10, audit findings on `frontier_evaluator.py`)

- [ ] **9.1.1** Split `build_single_stage_frontier_runtime` (lines 621-925, 305 LOC) into 3 helpers: `_resolve_runtime_inputs`, `_build_objective_bundle`, `_synthesize_spec`
- [ ] **9.1.2** Replace `constraint_violations` if-ladder (lines 535-569) with dict comprehension + 2 special-case overrides
- [ ] **9.1.3** Extract `_results_payload_from(...)` helper consumed by both valid path (lines 579-606) and invalid path (lines 1191-1206)
- [ ] **9.1.4** Inline `from_spec` classmethod alias (lines 301-309)
- [ ] **9.1.5** ~~Drop redundant `except FrontierEvaluatorInitializationError: raise` arm (line 920); add a single comment if needed~~ **DROPPED — NOT REDUNDANT (per r6 audit).** The `except FrontierEvaluatorInitializationError: raise` arm at `frontier_evaluator.py:920` (this section is about `frontier_evaluator.py`; the runner line 920 is unrelated NSGA3 artifact loading) prevents the next arm (`except Exception as error: raise FrontierEvaluatorInitializationError(...) from error` at `frontier_evaluator.py:922`) from **double-wrapping** an already-typed init error. Removing the arm changes runtime behavior (a `FrontierEvaluatorInitializationError` would be caught by the broader `Exception` handler and re-wrapped, losing the original `__cause__` chain and stamping a new `from error` cause). **Action: leave the arm in place; add a one-line comment** if needed: `# preserve already-typed init error; the broader handler below would otherwise double-wrap`.
- [ ] **9.1.6** Replace `_jsonable_value = single_stage._jsonable_value` re-export (line 1251) with direct `from ... import _jsonable_value`

### 9.2 Frozen dataclasses everywhere

- [ ] **9.2.1** Add `slots=True` to `FrontierCampaignManifest` (`frontier_campaign_reporting.py:68-135`). r6 correction: r1-r5 said "convert to `frozen=True, slots=True`" — but the dataclass is already `@dataclass(frozen=True)`; only `slots=True` is missing.
- [ ] **9.2.2** Add `slots=True` to `FrontierLaneContract`, `FrontierLaneRecord`, `FrontierCampaignProgress` (in renamed `frontier_progress_state.py`). r6 correction: all three are already `@dataclass(frozen=True)` (`FrontierLaneContract` line 27, `FrontierLaneRecord` line 84, `FrontierCampaignProgress` line 194); only `slots=True` is missing.
- [ ] **9.2.3** ~~Move shadow-write fields off `FrontierArchiveMember`~~ **DROPPED — UNSAFE (per Codex r2)**. The original audit verified read-vs-write only inside `frontier_archive.py`. In reality `frontier_recommendation.py:96, 131, 165, 191, 205-206, 275-276` reads `recommendation_flags` and `distance_from_seed` heavily across all 4 policies. Moving these off the dataclass would break recommendation. If a contract migration is desired in the future, scope it as a separate, dedicated task with explicit consumer rewrites — not as part of this mechanical-cleanup phase.
- [ ] **9.2.4** Run mypy or pyright if available

### 9.3 Functional dispatch

- [ ] **9.3.1** Replace `generate_frontier_lane_specs` 5-arm if-chain (`frontier_scalarization.py:700-771`) with `MappingProxyType` registry
- [ ] **9.3.2** If Phase 1B = KEEP NSGA3, replace runner engine if/else (lines 916-942) with engine registry (covered in 1B.4)

### 9.4 Hand-rolled JSON ser/de helper (F10, optional — earn before invest)

**Context:** ~294 LOC of similar `to_json_dict` / `from_json_dict` boilerplate across 3 modules (r6 measurement; r1-r5 estimate of "~150 LOC" was off by ~95% — actual breakdown: `frontier_evaluator.py` 17+51+29 = 97 LOC, `frontier_engine_base.py` 164 LOC, `frontier_archive.py` 33 LOC). A `schema_versioned` decorator could collapse them, but the decorator itself is ~50 LOC and adds an abstraction layer. **Decide whether the benefit warrants the new abstraction.**

- [ ] **9.4.1** Decide: implement `schema_versioned` decorator, OR leave boilerplate as-is (KISS argument)
- [ ] **9.4.2** If implementing: create `_schema_helpers.versioned_dataclass(version: str, *, coerce: dict[str, Callable] | None = None)`
- [ ] **9.4.3** Migrate one dataclass at a time; each migration must keep round-trip tests green
- [ ] **9.4.4** **Halt early** if migration cost exceeds the boilerplate it removes — that's a sign the abstraction isn't paying for itself

### 9.5 Reporting cleanup

- [ ] **9.5.1** Remove defensive `getattr(args, ..., default)` blocks at `frontier_campaign_reporting.py:181-204, 246-255, 412-421` — argparse always sets these
- [ ] **9.5.2** Inline `_sanitize_lane_record_for_final_output` (lines 526-531) if it's a one-liner
- [ ] **9.5.3** Decide if `FrontierCampaignManifest` dataclass adds value over a typed dict literal; if not, replace

### 9.6 Test cleanup

- [ ] **9.6.1** Collapse 3 cache tests in `test_frontier_evaluator.py:395-536` into 1 parameterized test
- [ ] **9.6.2** Audit subprocess-based round-trip test (`test_frontier_evaluator.py:326-393`, ~68 LOC). r6 caveat: it tests `build_single_stage_frontier_runtime` rebuild + evaluator instantiation across **process boundary** (clean module state). The in-process version at `:278` only tests JSON round-trip + fingerprint, NOT process isolation. The subprocess version is **not strictly covered** by the in-process one. Two options: (a) keep it as a cross-process regression guard, or (b) document a justification for dropping (e.g., "module-state cleanliness covered elsewhere in CI matrix"). Do not silently delete.
- [ ] **9.6.3** Extract shared `importlib.import_module("banana_opt.frontier_X")` wrapper into `tests/geo/_frontier_test_helpers.py` (helps 5 of 6 frontier test files; ~30-40 LOC saved)
- [ ] **9.6.4** Audit each `patch.multiple(create=True, ...)` site for fragility — flag any test that mocks unrelated module globals
- [ ] **9.6.5** Confirm zero `xfail`, `skip`, marker-gated tests survive (the audit says there are none currently)

---

## 9b. Phase 7 — Algorithm / computation hardening (added in r6.2)

**Context:** the r6.2 audit (6 parallel agents, 2026-05-07) verified that the frontier algorithms produce mathematically-correct results in the nominal flow. No silent-wrong-answer bugs were found. However, **several silent-pass paths and semantic mismatches** can mask user errors or future regressions. This phase covers those.

### 9b.1 Rename / clarify `multilane_local` (F17 — highest-impact algorithm finding)

**Problem:** `generate_multilane_local_specs` (`frontier_engine_multilane_local.py:54-60`) sweeps only iota↔volume shares, clamped to [0.2, 0.8], and never touches qa_error or boozer_residual. The module name + the CLI flag `FRONTIER_REFERENCE_MODE_SHARED` suggest "multilane Pareto exploration" — users get a 1-parameter convex-combination sweep instead.

**Acceptance criterion:** either the name reflects the behavior, or the behavior matches the name. Pick one.

- [ ] **9b.1.1 (Option A — preferred — rename to match behavior):** rename `FRONTIER_REFERENCE_MODE_SHARED` to `FRONTIER_REFERENCE_MODE_IOTA_VOLUME_SHARE_SWEEP`; rename `frontier_engine_multilane_local.py` → `frontier_iota_volume_share_sweep.py`; rename `generate_multilane_local_specs` → `generate_iota_volume_share_specs`. Update CLI, docstrings, tests. Add a docstring note: "Sweeps the iota/volume reward-share simplex axis; QS and Boozer residuals are NOT swept and use the seed-derived references. For 4-D Pareto coverage, use `FRONTIER_REFERENCE_MODE_ACHIEVEMENT_FULL_SIMPLEX`."
- [ ] **9b.1.2 (Option B — match the name to the behavior):** extend the share sweep to all 4 axes (full Das-Dennis simplex). Then the name is honest. **Caveat:** this duplicates `_achievement_full_simplex_lane_specs`. Probably not worth it.
- [ ] **9b.1.3** Update `docs/single_stage_frontier_*.md` plan series to clarify what each mode actually does.

### 9b.2 Surface ε-certifier missing-key path (F18)

**Acceptance criterion:** running `_evaluate_epsilon_constraint_status` on a `scalarization_params` dict missing one of the threshold keys raises a clear error (or at minimum logs a warning).

- [ ] **9b.2.1** In `frontier_archive.py:603-614`, when `scalarization_type == "epsilon_constraint_sweep_v1"` and a configured constraint metric has no threshold key in `scalarization_params`, raise `ValueError("epsilon_constraint_sweep_v1 missing threshold key '<name>'")`. The current `if limit is None: continue` silently treats it as unconstrained.
- [ ] **9b.2.2** Add test `test_epsilon_certifier_raises_on_missing_threshold_key` in `tests/geo/test_frontier_archive.py`.

### 9b.3 Add slack tolerance to ε-certification (F19)

**Acceptance criterion:** `_evaluate_epsilon_constraint_status` accepts `value == limit + ε_machine` as feasible (or the strict-comparison choice is intentional and documented + tested).

- [ ] **9b.3.1** Decide: introduce `epsilon_constraint_certification_slack` (default `1.0e-12` or `relative_tol * limit`) and use `excess > slack` instead of `excess > 0.0`; OR keep strict and document why (e.g., "all penalties are quadratic; on-boundary residual is structurally rare").
- [ ] **9b.3.2** Add tests for on-boundary, just-above, just-below cases.

### 9b.4 Guard `dominates()` against NaN (F20)

- [ ] **9b.4.1** In `frontier_dominance.py:175-203`, at function entry add `if any(value is not None and not math.isfinite(value) for value in (...))` over both maps; if NaN/Inf is detected, raise a clear error or short-circuit to `False` with a comment. Pick the choice and document it.
- [ ] **9b.4.2** Add test `test_dominates_handles_nan` (covers `dominates(nan, b)`, `dominates(a, nan)`, `dominates(nan, nan)`).

### 9b.5 Enforce hypervolume reference as nadir (F21)

- [ ] **9b.5.1** In `frontier_archive.py:resolve_hypervolume_reference` (currently 403-429), after resolution emit a warning (or assert) when any member has a metric value worse than the reference on its declared direction. Quiet acceptance hides config errors.
- [ ] **9b.5.2** Document the silent zero-clip behavior of `_hypervolume_boxes` (currently 624-672) in a docstring so a future maintainer understands the `max(0, extent)` is intentional.

### 9b.6 Extend `_FINITE_*` tables to scalarization-derived fields (F23)

- [ ] **9b.6.1** Add to `frontier_constraints.py:_FINITE_SCALAR_FIELDS`: `frontier_rank_total`, `frontier_base_total`, `frontier_trust_penalty`, `frontier_epsilon_penalty`, `frontier_goal_total`, `frontier_chebyshev_total`. Add to `_FINITE_VECTOR_FIELDS`: `frontier_goal_grad`, `frontier_scalarization_grad`. (r6.3: dropped `frontier_chebyshev_deltas` / `frontier_chebyshev_softmax_weights` from the original list — the Chebyshev computation uses the LSE-shift trick at `single_stage_objectives.py:500-501` so these are numerically stable under positive `sharpness`; the real risk is in derived sums like `frontier_rank_total` that aren't re-checked after summing penalty terms.)
- [ ] **9b.6.2** Add test that an evaluation with a deliberately-NaN penalty summand (e.g., a non-finite `length_weight` or a synthetic `J_len = inf`) is caught by `annotate_search_evaluation_finiteness` after the derived fields are assembled (`finite_eval_ok=False, nonfinite_fields` includes the derived field name). r6.3 correction: prior text "deliberately-overflowing Chebyshev `sharpness * delta`" is wrong — the LSE shift prevents overflow there.

### 9b.7 Surface recommendation gate-fallback in return signature (P5)

- [ ] **9b.7.1** In `frontier_recommendation.py:333-336`, when `gate_fallback_to_all_members` is set, also surface it in the public `Recommendation` dataclass / return mapping (not just metadata). E.g., add a `gate_fallback: bool = False` field on the recommendation. Callers can then decide whether to label the recommendation "unsafe" downstream.
- [ ] **9b.7.2** Add test `test_recommend_with_no_safe_members_surfaces_gate_fallback`.

### 9b.8 Documentation-only fixes (N1-N4, P3)

These are docstring/comment additions, no behavior change. Bundle them into one PR for tractability.

- [ ] **9b.8.1 (N1) `frontier_conditioning.py`** — rename to `frontier_objective_balance_diagnostics.py` (or similar), OR keep the name and add a top-of-file comment: "This module is a diagnostic max/min-ratio gate on objective magnitudes. It is NOT a numerical preconditioner. The ratio uses raw objective units, not metric-scale-normalized space."
- [ ] **9b.8.2 (N2) `evaluate_frontier_trust_penalty`** — add a docstring note: "Despite the 'trust penalty' name, this is a one-sided ε-constraint penalty on `J_Boozer` (a residual cap), not a canonical NLP trust region in DOF space (no `x_0` reference). Math is C¹ at the boundary."
- [ ] **9b.8.3 (N3) ε-mode** — add a docstring on `apply_frontier_scalarization_override` ε-branch noting: "Hybrid ε-method — keeps `J_QS + res·J_Boozer + iotas·J_iota + volume·J_volume` as base objective rather than reducing to pure `f_k`. Penalty is ADDED (quadratic, fixed-coefficient — not augmented Lagrangian)."
- [ ] **9b.8.4 (N4) double penalization** — add a comment at `frontier_scalarization.py:649-652` noting that `frontier_boozer_trust_threshold` and `epsilon_constraint_boozer_max` are intentionally tied (both penalize the same residual additively): `(δ/trust_scale)² + epsilon_penalty_weight·(δ/boozer_reference)²`.
- [ ] **9b.8.5 (P3) `_WEIGHT_FLOOR=1e-12`** — add a comment at `frontier_scalarization.py:537-540` noting that the floor breaks the simplex unit-sum invariant intentionally to keep downstream Chebyshev divisions finite.

### 9b.9 Decimation geometric uniformity (P2 — optional, low priority per r6.3)

**r6.3 narrowing:** the original P2 framing claimed the dedupe-fill loop "biases extras toward the order-prefix." On re-verification, the rounded-index path at `frontier_scalarization.py:362-365` produces all-distinct indices for typical lane counts (verified for the cited N=15 from H=3/20 directions: indices `[0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19]`, all distinct). The dedupe-fill loop at lines 373-378 is **only exercised** when the rounded-index set has collisions, which is rare. So the bias is not as broad as P2 originally claimed.

- [ ] **9b.9.1** (Optional, low priority) Replace `_select_reference_directions` with farthest-point sampling on the simplex if pursuit of strict geometric uniformity is desired — but with the caveat above, this is a polish item, not a correctness fix.
- [ ] **9b.9.2** (Sufficient remedy) Add a docstring at `_select_reference_directions` noting that decimation is enumeration-order rounding, not geometric distance; and that the dedupe-fill fallback (lines 373-378) is only exercised on rounded-index collisions. Discourage non-canonical lane counts (e.g., N=L-1 from L Das-Dennis directions where collision is most likely).

### 9b.10 Algorithm-invariant test backfill

The r6.2 audit found that all 6 frontier subsystems are mathematically correct in nominal flow but lack tests that lock the **invariants** the math depends on. Backfill the following (most are short property-style tests). Tests already covered by Phase 7 individual sub-items above are not repeated here.

**r6.3 scope clarification:** existing test coverage that DOES exist (and r6.2 wrongly characterized as missing):
- `tests/geo/test_frontier_constraints.py:27-49` covers NaN/Inf detection in raw evaluator fields via `annotate_search_evaluation_finiteness` (`test_annotate_search_evaluation_finiteness_flags_nonfinite_fields`). NaN gaps below are specifically about NaN propagation in `dominates`, `balanced_policy_score`, hypervolume, and achievement scalarization — NOT in the finiteness annotation pipeline.
- `tests/geo/test_single_stage_workflow_helpers.py:5059-5148` covers end-to-end early-stop archive stagnation (`test_frontier_campaign_early_stop_stops_after_archive_stagnation`). Gap below is a unit-level test on `update_frontier_early_stop_status` (see 9b.10.10).

**Hypervolume invariants (`tests/geo/test_frontier_archive.py`):**
- [ ] **9b.10.1** `test_hypervolume_permutation_invariance` — for the same set S and reference r, compute HV with members in original order and in reversed order; assert exact equality.
- [ ] **9b.10.2** `test_hypervolume_monotone_under_dominance` — given non-dominated set S with `HV(S, r) = h`, add a member dominated by every existing member; assert `HV(S ∪ {dominated}, r) == h` (no change).
- [ ] **9b.10.3** `test_hypervolume_reductions` — verify 1D and 2D analytical-value reductions against hand-computed values (sanity check the recursion).

**Dominance invariants (`tests/geo/test_frontier_archive.py`):**
- [ ] **9b.10.4** `test_dominates_irreflexive` — for any member `a`, `dominates(a, a) == False`.
- [ ] **9b.10.5** `test_dominates_asymmetric` — for any `a != b`, `not (dominates(a, b) and dominates(b, a))`.
- [ ] **9b.10.6** `test_objective_metric_scale_handles_degenerate_axis` — when ideal == nadir for one axis, scale uses the floor (no division by zero).

**Recommendation invariants (`tests/geo/test_frontier_recommendation.py`):**
- [ ] **9b.10.7** `test_recommend_empty_archive_returns_none` (already implicitly covered by `recommend_frontier_member` returning None at line 59-60, but no explicit test).
- [ ] **9b.10.8** `test_recommend_single_member_archive_returns_that_member` (with and without gate pass).
- [ ] **9b.10.9** `test_recommend_tiebreaker_is_member_id_lex_ordered` — two members with identical primary metrics return the lex-min `member_id`.

**Early-stop invariants (`tests/geo/test_frontier_contracts.py` or new test file for runtime calibration):**

**r6.3 narrowing:** an end-to-end stagnation test already exists at `tests/geo/test_single_stage_workflow_helpers.py:5059-5148` (`test_frontier_campaign_early_stop_stops_after_archive_stagnation`). The remaining gap is a **unit-level** test on `update_frontier_early_stop_status` that lets a maintainer reason about the state machine without spinning up a full campaign.

- [ ] **9b.10.10** `test_update_frontier_early_stop_status_handles_synthetic_hv_sequence` — call the function directly with a fabricated sequence of `(certified_members_hv, archive_size)` tuples and `min_hypervolume_gain` / `patience_lanes` settings. Assert `triggered`, `reason`, `no_improvement_streak`, `previous_best_hypervolume` evolve correctly. Cases: (a) flat HV sequence → triggers after `patience` repeats, (b) marginal-but-below-threshold improvement → still triggers, (c) above-threshold improvement → resets the streak, (d) `previous_best_hypervolume is None` first-improvement-after-`None`-history path (the state-smell mentioned in the audit).
- [ ] **9b.10.11** ~~`test_early_stop_does_not_trigger_on_marginal_improvement`~~ — subsumed by 9b.10.10 case (b).

**Multilane / lane-spec invariants (`tests/geo/test_frontier_scalarization.py`):**
- [ ] **9b.10.12** `test_multilane_share_sweep_normalization` — for N ∈ {1, 2, 3, 5, 10}, every emitted lane has shares summing to 1.0 within tolerance, no negatives.
- [ ] **9b.10.13** `test_multilane_share_sweep_boundary` — explicitly assert that for N=2, shares are (0.2, 0.8) and (0.8, 0.2) — endpoint-clamping is a documented behavior, not a bug.
- [ ] **9b.10.14** `test_achievement_scalarization_scaling_invariance` — multiply the goal point by 2; assert lane choice (which member is selected) is invariant or changes in a documented way.

**ε-certifier (covered partly by 9b.2.2 and 9b.3.2):**
- [ ] **9b.10.15** `test_epsilon_certifier_silent_pass_on_unknown_scalarization_type` — for `scalarization_type != "epsilon_constraint_sweep_v1"`, certifier should return `{"ok": True}` without consulting thresholds (this is the current behavior — lock it as a contract, not just an accident).

**Estimated total Phase 7 test LOC added:** ~250-350 LOC across the subsystems. Most are ≤20 LOC each (property tests with 1-3 fixtures).

**Estimated total LOC delta for Phase 7:** ~+60 to +120 LOC of guards/docstrings, ~+250-350 LOC of new tests, ~−30 LOC if 9b.1 (rename) consolidates. Net: a meaningful positive (~+300-450 LOC), but eliminates a class of silent-failure paths and makes the algorithm contracts explicit.

---

## 10. Cross-cutting: after each phase

These run after every phase commit, before opening a PR.

- [ ] **CC.1** Run frontier-relevant tests
  ```bash
  ./run_tests tests/geo/test_frontier_*.py tests/geo/test_single_stage_*.py
  ```
- [ ] **CC.2** Run ruff/lint
  ```bash
  ./run_ruff
  ```
- [ ] **CC.3** Smoke a frontier campaign run with `multilane_local`. **Required CLI args** (revised per Codex r2): `--plasma-surf-filename` and `--stage2-bs-path` are `required=True` in the parent parser at `run_single_stage_goal_mode_comparison.py:187-200`. Replace the placeholders below with actual paths from your local fixtures or lab notes.
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
- [ ] **CC.4** Diff JSON artifact shape vs prior smoke run (regression guard for D3). The default summary file is **`single_stage_frontier_campaign_summary.json`** (`frontier_campaign_reporting.py:52` defines `DEFAULT_SUMMARY_JSON`); not `summary.json`.
  ```bash
  jq -S '.frontier_runtime_calibration' "$OUT/single_stage_frontier_campaign_summary.json"
  ```
- [ ] **CC.5** Verify dependency direction
  ```bash
  rg "from .frontier_|from banana_opt.frontier_" examples/single_stage_optimization/banana_opt/single_stage_*.py
  ```
  Expected after Phase 2: zero hits, OR only consumer-style references where `single_stage_objectives` calls into `frontier_scalarization.apply_*` rather than re-defining the body.
- [ ] **CC.6** Confirm LOC delta matches plan
  ```bash
  wc -l examples/single_stage_optimization/banana_opt/frontier_*.py \
        examples/single_stage_optimization/run_single_stage_frontier_campaign.py
  ```
- [ ] **CC.7** Update STATE / lab notes if work is paused mid-phase

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 2 (scalarization move) breaks gradient consistency | Medium | High | Run full `test_frontier_evaluator.py` and `test_single_stage_alm_integration.py` after each function move; bisect on smallest reproduction if fail |
| Phase 1A premature deletion if NSGA3 has undocumented production use | Low (per audit) | Medium | D1 investigation must complete first; if any artifact found, escalate to Phase 1B (quarantine) |
| Phase 8.2 calibration JSON shape change breaks downstream tools | Medium | Medium | D3 audit before executing; preserve `to_json_dict` shape if external consumers exist |
| Phase 6.1 rename causes wide diff churn that conflicts with concurrent branches | Low | Low | Coordinate with team; do rename in dedicated commit; merge to base before continuing |
| Hypervolume memoization (8.1) wrong cache key → silent stale results | Low | High | Cache key must be tuple-of-tuples of objective vectors (immutable, hashable); add property test asserting cache invalidates when archive changes |
| Schema-versioned dataclass refactor (9.4) introduces decorator complexity > boilerplate it removes | Medium | Low | Halt early — see 9.4.4 stop condition |
| Test deletions hide future regressions | Low | Medium | For each deleted test, confirm coverage exists in another test file; document in commit message |

---

## 12. Done criteria (final review)

A reviewer should be able to confirm all of the following with grep + the test suite.

- [ ] **DC.1** D1 decision is recorded in this file
- [ ] **DC.2** If D1 = DROP: zero hits for `nsga3` in `examples/single_stage_optimization/banana_opt/` and `tests/geo/` source code (string mentions in retired docs allowed)
- [ ] **DC.3** If D1 = QUARANTINE: a real `FrontierEngine` Protocol exists; runner dispatches via registry; benchmark test exists
- [ ] **DC.4** `single_stage_objectives.py` contains zero `_frontier_*` private functions; only consumer-style imports
- [ ] **DC.5** `annotate_search_evaluation_finiteness` lives in `search_evaluation.py` (or equivalent non-frontier module); no `frontier_constraints` import in non-frontier code
- [ ] **DC.6** `epsilon_constraint_qa_max` / `epsilon_constraint_boozer_max` string keys appear only inside `frontier_contracts.py` (one frozen dataclass)
- [ ] **DC.7** `frontier_engine_base.py` does not exist (renamed to `frontier_progress_state.py`)
- [ ] **DC.8** `frontier_engine_multilane_local.py` does not exist (folded into `frontier_scalarization.py`)
- [ ] **DC.9** `frontier_archive_hypervolume` is memoized; benchmark test demonstrates cache hits
- [ ] **DC.10** `pymoo` is either not imported (Phase 1A) or lazily imported (Phase 1B + 8.3)
- [ ] **DC.11a** Frontier module LOC (sum of `examples/single_stage_optimization/banana_opt/frontier_*.py`) < **5,500** (started at 6,435)
- [ ] **DC.11b** Runner LOC (`run_single_stage_frontier_campaign.py`) < **950** (started at 1,101)
- [ ] **DC.11c** Combined source LOC < **6,500** (started at 7,536)
- [ ] **DC.12a** Frontier test LOC (`tests/geo/test_frontier_*.py`) < **2,800** (started at 3,207). Reachable from Phase 1A (~280 LOC: 153 from `test_frontier_evaluator.py` + 127 from `test_frontier_contracts.py`) + Phase 6.6 cleanups (~100-150 LOC from cache-test parameterization, importlib helper extraction, optional subprocess-test removal pending 9.6.2 decision). r6 correction: the prior target of <2,200 was unreachable from the listed phases (~565 LOC gap) and conflated this glob with `test_single_stage_workflow_helpers.py` (a different file).
- [ ] **DC.12b** NSGA3 blocks deleted from `tests/geo/test_single_stage_workflow_helpers.py`: **~347 LOC** removed (r6 measurement: 154 LOC at line 5150-5303 + 192 LOC at line 5304-5495 + ~1 LOC import cleanup). Earlier estimates of "~320 LOC" undercounted by ~27 LOC.
- [ ] **DC.13** All frontier tests green; ruff clean; smoke campaign run produces equivalent JSON shape (or shape change documented in D3)
- [ ] **DC.14** `docs/single_stage_frontier_*.md` plan files updated: superseded sections marked; current state reflects this plan's outcomes
- [ ] **DC.15** Updated audit re-run shows < 5% removable LOC (down from ~15-20%)
- [ ] **DC.16** (Phase 7 / r6.2) `multilane_local` is renamed OR has a top-of-module docstring honestly describing the 2-axis sweep behavior; CLI flag matches; users can no longer mistake it for 4-D Pareto exploration (F17)
- [ ] **DC.17** (Phase 7 / r6.2) Silent-pass paths closed: ε-certifier raises on missing threshold key (F18); `dominates()` guards NaN (F20); `resolve_hypervolume_reference` warns on non-nadir reference (F21)
- [ ] **DC.18** (Phase 7 / r6.2, refined r6.4) `_FINITE_*` tables cover scalarization-derived sums (`frontier_rank_total`, `frontier_base_total`, `frontier_trust_penalty`, `frontier_epsilon_penalty`, `frontier_goal_total`, `frontier_chebyshev_total`); a deliberate non-finite penalty summand or `length_weight·J_len` is detected by `annotate_search_evaluation_finiteness` after derived fields are assembled. (r6.4 reword: r6.2's "deliberate Chebyshev overflow" example was wrong — LSE-shift trick at `single_stage_objectives.py:500-501` stabilizes the softmax under positive sharpness, so it can't be used as the test trigger; F23 fix is about detecting non-finiteness in the *summed* derived fields, not in Chebyshev internals.)
- [ ] **DC.19** (Phase 7 / r6.2) Recommendation `gate_fallback` is surfaced in the public return signature, not just metadata (P5)
- [ ] **DC.20** (Phase 7 / r6.2) Algorithm-invariant tests are present: HV permutation invariance, dominance reflexivity/asymmetry, ε on-boundary, recommendation empty-archive, multilane share-sweep normalization, early-stop synthetic flat-HV (subset of 9b.10)

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

One PR per phase, with phase number in title. This keeps reviews tractable and makes any rollback localized.

- [ ] **PR1** `frontier: drop NSGA3 engine path` (Phase 1A) — or **PR1'** `frontier: quarantine NSGA3 behind Engine Protocol` (Phase 1B)
- [ ] **PR2** `frontier: restore SSOT — move scalarization back to frontier_scalarization.py` (Phase 2.1)
- [ ] **PR3** `frontier: extract search_evaluation helper out of frontier_constraints` (Phase 2.2)
- [ ] **PR4** `frontier: rename engine_base → progress_state; fold multilane into scalarization` (Phase 3.1, 3.2)
- [ ] **PR5** `frontier: centralize epsilon thresholds` (Phase 4.1) — Phase 4.2 dropped per r2 (YAGNI)
- [ ] **PR6** `frontier: parameterize best-of selectors` (Phase 4.3)
- [ ] **PR7** `frontier: memoize hypervolume; collapse calibration profile registry` (Phase 5.1, 5.2)
- [ ] **PR8** `frontier: lazy pymoo + single-pass resume + single-pass resolution` (Phase 5.3, 5.4, 5.5) — combine if PR1' chosen, otherwise drop the pymoo bit
- [ ] **PR9** `frontier: split build_runtime monolith; frozen dataclasses; functional dispatch` (Phase 6.1, 6.2, 6.3)
- [ ] **PR10** `frontier: reporting + test cleanup` (Phase 6.5, 6.6)
- [ ] **PR11** (optional) `frontier: schema_versioned helper` (Phase 6.4) — only if 9.4.1 decides yes
- [ ] **PR12** `frontier: rename multilane_local → iota_volume_share_sweep` (Phase 9b.1) — single-file rename + CLI flag rename + test update; high-impact for end-user clarity (F17)
- [ ] **PR13** `frontier: silent-pass guards + finiteness-table extension` (Phase 9b.2-9b.6) — bundles ε-certifier missing-key raise, ε slack tolerance, dominates NaN guard, HV reference nadir warning, derived-fields finiteness check (F18-F23)
- [ ] **PR14** `frontier: documentation-only naming clarifications` (Phase 9b.8) — docstring/comment additions for conditioning, trust-penalty, ε-mode, double-penalization, weight floor (N1-N4 + P3)
- [ ] **PR15** `frontier: algorithm-invariant test backfill` (Phase 9b.10) — ~250-350 LOC of property-style tests locking the math contracts

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
