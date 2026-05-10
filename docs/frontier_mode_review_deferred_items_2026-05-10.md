# Frontier Mode Review — Deferred Items

**Date:** 2026-05-10
**Branch:** `surrogate-confinement-v2`
**HEAD at validation:** `6fda93ab8`
**Source review:** 9-agent parallel audit run on 2026-05-09 (raw reports at ephemeral `/tmp/frontier_review/01_math_physics.md` … `09_perf_thread.md`).
**Scope:** `examples/single_stage_optimization/banana_opt/frontier_*.py`, `examples/single_stage_optimization/run_single_stage_frontier_campaign.py`, plus the corresponding test files under `tests/geo/`.

---

## Status snapshot

### Landed in this review cycle (commits `c137aea85`, `6fda93ab8`)

- SSOT literal cleanup (`frontier_campaign_progress_v1` → constant; `"certified"` → `FRONTIER_ARCHIVE_STATE_CERTIFIED`).
- Strict `schema_version` access in `LaneRecord.from_json_dict` and `CampaignProgress.from_json_dict`.
- Dropped dual `schema_version`/`SCHEMA_VERSION` casing in spec loaders.
- `f.flush() + os.fsync()` before `os.replace` in three atomic-write paths (progress, JSON helper, solver checkpoint).
- Sorted `dominance_signature` lists for byte-equal determinism.
- `>=` → `>` in `update_frontier_early_stop_status` (min_gain=0 no longer disables early stop).
- Hardware-safe gate no longer falls back to all members; `_eligible_members_for_gate_rule` returns empty on no-pass; `select_best` returns `None`; `recommend_frontier_member` returns `None`.
- Trim `lane_payload["results"]` → `results_summary` in lane records (~75% size reduction in progress.json).
- Threaded `pareto_objective_normalization` through `replay_archive_*` and `from_json_dict`.
- Strip stale `hypervolume_contribution` on replay.
- Dropped persisted `archive_members`/`provisional_archive_members` from progress.json (replay is SSOT).
- Worker-thread Namespace via dict-spread instead of in-place mutation of `lane_args.resume_solver_checkpoint`.
- Moved `annotate_archive_members` out of insert hot path (O(N²) per insert eliminated; deferred to `serialize_frontier_archive`).
- Replaced lazy import in `archive_best_by_metric` with inline implementation; circular dep gone.
- Deleted stale `frontier_constraints.py` re-export shim; tests load `single_stage_search_contracts` directly.
- Dropped unused `SUPPORTED_PARETO_NORMALIZATION_KINDS` and `FrontierArchiveMember.from_json_dict` wrapper.
- Dropped dishonest `__all__` in runner.
- `_positive_int`/`_non_negative_int` on five CLI args.
- Restored `_select_reference_directions` early-return for `partitions is not None` (reverted an auto-formatter regression).

### Coverage downgrades (claims that no longer apply)

- Multi-worker lane parallelism is now tested (`test_single_stage_workflow_helpers.py:3721`, `:3874`).
- Lane-grouping predicate (`build_frontier_lane_execution_groups`) has direct tests (lines 3874-3900).
- Resume flow has reuse / salvage / deterministic-match coverage (lines 4236, 4650, 4930).
- The `recommend_frontier_member` silent-bypass on strict gates is closed.
- Fresh campaign IDs are non-deterministic per fresh run again (`uuid.uuid4().hex[:12]`).

### Total post-cycle test count

192 tests pass on the frontier suite + workflow helpers.

---

## Deferred items

Severity scale follows the original review reports:
- **CRITICAL** — silent wrong, race, data loss, divide-by-zero, sign flip.
- **HIGH** — likely-bug-at-scale, ship-blocker.
- **MEDIUM** — smell, semantic risk.
- **LOW** — cosmetic.

Each item shows the cited location, current HEAD state confirmation (verified 2026-05-10), and a proposed root-cause fix consistent with the repo guardrails (no defensive try/except, no fallbacks; SSOT/DRY/IMMUTABLE).

### Math / physics

#### F1.1 — CRITICAL — Chebyshev/achievement scalarization uses `qs_reference` / `boozer_reference` as both target and scale denominator
- **Location:** `frontier_scalarization.py:374-445` (lines 395-396, 403-404 in `_frontier_chebyshev_goal`).
- **Current state:** `(J_QS - qs_reference) / qs_reference` and `(J_Boozer - boozer_reference) / boozer_reference`. Iota / volume use separate `iota_scale` / `volume_scale` — asymmetric.
- **Fix:** add `qs_scale`, `boozer_scale` fields to `FrontierGoalConfig` and require them explicitly (no defaults — repo guardrails forbid fallbacks). Bump `frontier_achievement_spec_v1` to a new schema version that mandates the scales; update the spec generators (`_achievement_chebyshev_lane_specs` at `frontier_scalarization.py:883` and `_achievement_full_simplex_lane_specs` at `:951`) to emit the new fields. Old specs become a hard schema-version-mismatch error at load. Use the new fields as denominators in both the delta and the gradient. Same denominator threading goes into the epsilon-penalty path (F1.3).

#### F1.2 — CRITICAL — `physics_total` semantics differ between ALM and non-ALM branches
- **Location:** `frontier_scalarization.py:267-301`. Non-ALM branch sums all six geometry penalties (length, cc, cs, curvature, surf_dist, poloidal_extent — see `_frontier_penalty_geometry_total_grad` at lines 330-360); ALM branch (`_frontier_alm_base_total_grad`, lines 364-371) sums only `length_weight*J_len`.
- **Current state:** Same physical configuration produces different `physics_total` / `base_total` values across formulations; cross-run comparability broken.
- **Fix:** decide one definition. SSOT-compliant choice is to include all geometry penalties in both paths (delete `_frontier_alm_base_total_grad`, call `_frontier_penalty_geometry_total_grad` directly in both arms). Alternative: rename the ALM-side key to `frontier_alm_base_total` and update downstream consumers explicitly.

#### F1.3 — HIGH — Epsilon penalty divides by `max(qs_reference, 1e-6)` (same scale-vs-target conflation as F1.1)
- **Location:** `frontier_scalarization.py:458-470` (`_frontier_excess_penalty` callers).
- **Fix:** folded into F1.1; once `qs_scale` exists, pass it as the `scale=` argument and drop the redundant `max(..., 1e-6)`.

#### F1.7 — HIGH — Das–Dennis simplex weight floor breaks unit-sum invariant without renormalization
- **Location:** `frontier_scalarization.py:43-45, 985-998`. `max(direction[i], _WEIGHT_FLOOR)` for each component, no normalization back to the unit simplex.
- **Current state:** Manifest-level `frontier_chebyshev_weight_*` values sum to `1 + 4·_WEIGHT_FLOOR ≈ 1 + 4e-12`; readers comparing across lanes assume `Σwᵢ = 1`.
- **Fix:** after flooring, renormalize:
  ```python
  weights = tuple(max(direction[i], _WEIGHT_FLOOR) for i in range(4))
  total = sum(weights)
  weights = tuple(w / total for w in weights)
  ```

#### F1.8 — HIGH — `_select_reference_directions` selection-by-rounding produces non-uniform geometric coverage
- **Location:** `frontier_scalarization.py:792-833`.
- **Current state:** Enumeration-order rounding clusters at simplex boundaries; user expectation of "evenly spaced" is violated.
- **Fix:** force `partitions` explicit when `requested_num_directions != C(p+n-1, n-1)` and document the constraint. Or replace the rounding with a greedy farthest-point selection on the unit simplex.

#### F2.1 — CRITICAL — `dominates()` returns `False` on missing metrics; partial members never pruned
- **Location:** `frontier_dominance.py:185-186`.
- **Current state:** `if candidate_value is None or incumbent_value is None: return False`. A complete candidate cannot dominate a partial incumbent → archive bloat.
- **Fix:** raise on missing keys, treat completeness as the upstream invariant. Replace `.get(metric_name)` with `[metric_name]`. Tests need updating.
- **Caveat:** behavioral API change with possible external blast radius. Land alongside a release-notes note.

#### F2.2 — HIGH — `_strictly_better` and `_better_or_equal` use the same τ; ε-tight Pareto behavior is undocumented
- **Location:** `frontier_dominance.py:157-176`.
- **Fix:** docstring on `dominates()` calling out Laumanns-2002 ε-tight semantics and the 2τ buffer band. No code change.

#### F2.3 — MEDIUM — `objective_metric_scale` ideal-nadir branch absorbs spec inversions silently via `abs(...)`
- **Location:** `frontier_dominance.py:254-255`.
- **Current state:** `abs(float(ideal_metrics[metric_name]) - float(nadir_metrics[metric_name]))` corrects magnitude but downstream `(value - reference) / scale` flips delta sign.
- **Fix:** at spec-load time (`_coerce_defined_normalization_metrics`), assert direction-aware ordering: `direction == "max" → ideal > nadir`, `direction == "min" → ideal < nadir`. Raise on inversion.

#### F2.4 — MEDIUM — `objective_metric_scale` seed-relative branch uses `abs(reference)`; future-fragile for signed metrics
- **Location:** `frontier_dominance.py:259`.
- **Current state:** All four current Pareto metrics are non-negative, so this is a no-op today. Adding any signed metric (curl, twist, etc.) would silently zero the scale near the origin.
- **Fix:** docstring explicitly noting the non-negativity assumption. Code change deferred until a signed metric is added.

#### F3.1 — MEDIUM — Conditioning report mixes bounded `J_iota` with unbounded `J_QS_objective` in `value_ratio`
- **Location:** `frontier_conditioning.py:19-32`.
- **Current state:** `J_iota` is `-tanh((m-ref)/scale) ∈ (-1, 1)`; `J_QS_objective = J_QS / qs_reference` is unbounded. Their ratio is meaningless.
- **Fix:** scale the `iota_objective` and `volume_objective` entries by `effective_iotas_weight` and `effective_volume_weight` so all four terms are at the optimizer-total contribution scale; or strip them from the report (since bounded tanh terms are never the conditioning issue).

#### F3.3 — MEDIUM — `_usable_ratio` filters zero-valued terms; gate is blind to missing-term cases
- **Location:** `frontier_conditioning.py:103-111`.
- **Current state:** `> 0.0` filter drops `J_volume = 0` (no surface volume term) entirely; `usable_scale_ok` evaluates a 3-term ratio and reports "OK".
- **Fix:** require all four terms present; return `None` if any is `<= 0` or non-finite. Optionally surface a separate `incomplete_terms` flag.

#### F4.2 — MEDIUM — `_normalized_delta` returns `0.0` on missing reference, conflating "no data" with "at-reference"
- **Location:** `frontier_recommendation.py:319-333`.
- **Fix:** raise on missing reference (treat completeness as a certified-archive invariant). Touches `_balanced_policy_score` callers; archive-member loaders enforce reference completeness.

#### F4.4 — LOW — `closest_to_seed` policy uses cached `member.distance_from_seed`; stale under normalization changes
- **Location:** `frontier_recommendation.py:185-191` (`_closest_to_seed_key` → `none_aware_lex("distance_from_seed", ...)`).
- **Fix:** recompute distance at recommendation time using `normalized_objective_distance(...)` with the current `pareto_objective_normalization`. Drop the cached `distance_from_seed` field, or version-stamp it.

#### Cross-cutting (architectural) — three implementations of `(value − reference) / scale`
- **Locations:** `_frontier_chebyshev_goal` (frontier_scalarization), `_normalized_delta` (frontier_recommendation), `normalized_objective_distance` (frontier_dominance).
- **Fix:** introduce a single `normalized_metric_delta(metric_name, value, reference, *, pareto_objective_normalization)` helper in `frontier_dominance` (or a new `frontier_normalization` module). Funnel all three call sites through it.

### Algorithm / state

#### A-1 — HIGH — Duplicate detection picks first match, not closest
- **Location:** `frontier_archive.py:549-565` (`_find_duplicate_member_index`).
- **Current state:** Returns the first member within `duplicate_distance_threshold`; iteration order changes after `annotate_archive_members` re-sort and dominated drops. Different runs that arrive at the same set in different orders pick different "duplicate of" anchors → archive shape flips.
- **Fix:** scan all members, return the closest, ties broken by `member_id`:
  ```python
  best_index, best_distance = None, math.inf
  for index, member in enumerate(members):
      distance = ...
      if distance is None or distance > duplicate_distance_threshold:
          continue
      if (
          distance < best_distance
          or (distance == best_distance and members[index].member_id < members[best_index].member_id)
      ):
          best_index, best_distance = index, distance
  return best_index
  ```

#### A-3 — MEDIUM — CERTIFIED and REJECTED `member_id` collide
- **Location:** `frontier_archive.py:749-758` (`_build_member_id`).
- **Current state:** Both states yield `f"{campaign_id}:{lane_id}"`. A REJECTED record persisted in `lane_record.archive_member` collides with a same-lane CERTIFIED record across reruns.
- **Fix:** add `:rejected` suffix:
  ```python
  if archive_state == FRONTIER_ARCHIVE_STATE_PROVISIONAL:
      return f"{base_member_id}:provisional"
  if archive_state == FRONTIER_ARCHIVE_STATE_REJECTED:
      return f"{base_member_id}:rejected"
  return base_member_id
  ```
  Cross-cuts `member_id` parsing in tests.

#### A-4 — MEDIUM — Non-nadir hypervolume reference only emits `RuntimeWarning`
- **Location:** `frontier_archive.py:695-718` (`_warn_if_hypervolume_reference_not_nadir`).
- **Current state:** `_hypervolume_boxes` silently zeroes the offending axis extent; downstream hypervolume number is wrong without a hard signal.
- **Fix:** in `serialize_frontier_archive`, raise when any member axis violates the nadir condition. Keep the soft warning at the upstream `resolve_hypervolume_reference` call for diagnostic logs.

### Resume / orchestration

#### Resume #1 — CRITICAL — Manifest is not authoritative on resume
- **Location:** `run_single_stage_frontier_campaign.py:527-639`.
- **Current state:** Only `frontier_version`, `frontier_engine`, `stage2_bs_path` are restored from persisted state. `--frontier-runtime-calibration-profile`, `--frontier-rng-seed`, `--frontier-lane-budget`, `--frontier-total-budget`, `--frontier-normalization-kind`, `--frontier-hypervolume-reference`, `--frontier-recommendation-policy`, `--frontier-lane-warm-start-mode`, `--frontier-lane-workers`, `--frontier-early-stop-*` all silently take whatever the user typed on resume.
- **Fix:** drive resume from the persisted manifest. Restore `args.frontier_runtime_calibration_profile`, `args.frontier_lane_budget`, `args.frontier_total_budget`, the early-stop knobs, `args.frontier_rng_seed`, `args.frontier_hypervolume_reference`, `args.frontier_normalization_kind`, `args.frontier_recommendation_policy`, `args.frontier_lane_warm_start_mode` from the manifest's `FRONTIER_RUNTIME_CALIBRATION`, `FRONTIER_EARLY_STOP_POLICY`, `RNG_SEED`, `FRONTIER_HYPERVOLUME_REFERENCE`, `PARETO_OBJECTIVE_NORMALIZATION`, `FRONTIER_RECOMMENDATION_POLICY` blocks. Recompute `runtime_defaults` and `pareto_objective_normalization` from the restored args. Add `--allow-resume-arg-drift` if the user wants to override.

#### Resume #5 — HIGH — `min_hypervolume_gain == 0.0` validator still permits the policy
- **Location:** `frontier_runtime_calibration.py:169-176`.
- **Current state:** `>=` was already changed to `>` (so equality no longer counts as improvement), but the validator still allows `0.0`. Now a policy choice rather than a correctness bug.
- **Fix (optional):** tighten validator to require strictly positive gain — `early_stop_min_hypervolume_gain > 0.0`.

#### Resume — HIGH — Salvage shadows solver checkpoint
- **Location:** `run_single_stage_frontier_campaign.py:359-385` (`_load_resumed_results`) and `:494-522` (`resume_or_run_goal_mode_case`).
- **Current state:** `_load_resumed_results` returns salvage payload (when no final results.json exists) before checking for a solver checkpoint. A partially-completed lane that has both a partial `results_best_*.partial.json` and a `solver_state_checkpoint.json` cannot continue from the checkpoint — salvage short-circuits.
- **Fix:** prefer checkpoint over salvage when both are present. Either inline a `prefer_checkpoint_over_salvage` policy flag, or unconditionally check `discover_single_solver_checkpoint_path` first and only fall through to salvage if the checkpoint is absent.

#### E-1 — MEDIUM — `--frontier-lane-workers > 1` silently overridden when patience > 0
- **Location:** `frontier_campaign_execution.py:174-184` (`frontier_lanes_require_ordered_execution`).
- **Current state:** Both shipped calibration profiles have `early_stop_patience_lanes > 0`, so the predicate forces sequential groups even when the user passes `--frontier-lane-workers 8`. No log line surfaces the override.
- **Fix:** in `main()`, after computing `lane_execution_groups`, emit a structured log when `args.frontier_lane_workers > 1` but groups have size 1, naming the reason (`warm_start_mode`, `early_stop_patience_lanes`, or `lane_workers`).

#### Resume F02 — MEDIUM — Resume silently accepts arg drift relative to manifest
- Folded into Resume #1.

#### Resume F06 — MEDIUM — Dry-run creates target/lane subdirectories
- **Location:** `run_single_stage_goal_mode_comparison.run_goal_mode_case` (the `case_output_root.mkdir(parents=True, exist_ok=True)` call).
- **Current state:** Dry-run is non-hermetic; creates empty subdirectories that are presumed not to exist for downstream tooling.
- **Fix:** gate the `case_output_root.mkdir(...)` on `not args.dry_run`. Or in the orchestrator, only mkdir lane subdirectories when `not args.dry_run`.

### Reporting / output

#### P5 — HIGH — Manifest not refreshed on resume even when CLI args drifted
- **Location:** `run_single_stage_frontier_campaign.py:633-634` (`if not args.resume or not paths.manifest_path.exists(): write_json(...)`).
- **Current state:** Manifest is only rewritten if missing. Combined with Resume #1, runtime args drift away from the persisted manifest silently.
- **Fix:** under the Resume #1 fix, the in-memory state is forced to match the manifest, making this a no-op. Alternative without #1: assert `runtime_defaults` matches the manifest's `FRONTIER_RUNTIME_CALIBRATION` block on resume; abort loudly on mismatch.

#### P6 — MEDIUM — Early-stopped lanes silently absent from `frontier_lanes`
- **Location:** `frontier_campaign_reporting.build_frontier_campaign_summary` (lane-record assembly).
- **Current state:** Lanes that exist in `lane_specs` but were skipped due to early stop have no entry in `lane_records_by_id`; the summary emits `frontier_lanes` shorter than `frontier_num_lanes`. Downstream consumers can't tell an early-stopped run from a 5-lane run.
- **Fix:** emit `frontier_lanes_skipped`:
  ```python
  ran_lane_ids = {lane.get("lane_id") for lane in lane_records}
  summary["frontier_lanes_skipped"] = [
      spec.lane_id for spec in lane_specs if spec.lane_id not in ran_lane_ids
  ]
  ```
  Add `frontier_lanes_skipped` to `validate_frontier_campaign_summary_payload`'s `_require_keys`.

#### P8 — MEDIUM — Summary validator missing several emitted keys
- **Location:** `frontier_contracts.validate_frontier_campaign_summary_payload`.
- **Current state:** Emits `dry_run`, `stage2_bs_path`, `stage2_results_path`, `stage2_artifact_init_only`, plus `stopped_after_lane_id` inside `frontier_early_stop`, none of which are in the validator's required-keys list.
- **Fix:** extend `_require_keys` to lock the schema. Validate `frontier_early_stop` substructure too (`policy`, `triggered`, `reason`, `stopped_after_lane_id`).

#### P10 — LOW — Archive lacks campaign provenance
- **Location:** `frontier_archive.serialize_frontier_archive`.
- **Current state:** No `frontier_campaign_id` or `created_at` in the archive payload; standalone archives can't be linked to their campaign.
- **Fix:** thread `campaign_id` into `serialize_frontier_archive`; emit `frontier_campaign_id` and `created_at` (UTC ISO 8601). Update both callers in `run_single_stage_frontier_campaign.py` and `frontier_campaign_reporting.py`.

### Orchestrator

#### F01 — HIGH — `main()` returns 0 unconditionally
- **Location:** `run_single_stage_frontier_campaign.py:803`.
- **Current state:** No way for batch tooling to detect a campaign that produced no certified members.
- **Fix:**
  ```python
  if args.dry_run:
      return 0
  return 0 if certified_members else 1
  ```
  Or stricter: also return non-zero if `target_payload` failed and target was not skipped.

#### F03 — MEDIUM — Mode-specific CLI flags silently ignored
- **Location:** `parse_args` post-validation in `run_single_stage_frontier_campaign.py`.
- **Current state:** `--frontier-full-simplex-partitions`, `--frontier-reference-points-file`, `--frontier-epsilon-spec-file` are accepted unconditionally; only the matching reference mode reads them. Users can pass them in the wrong mode and they're silently dropped.
- **Fix:** post-parse check that raises `ArgumentTypeError` when a mode-specific flag is supplied for a non-matching `--frontier-reference-mode`.

#### F05 — MEDIUM — `output_root` summary path missing `expanduser`
- **Location:** `frontier_campaign_reporting.build_frontier_campaign_summary` (the line that records `"output_root"` via `Path(args.output_root).resolve()`).
- **Current state:** For `--output-root ~/work/frontier`, runtime resolves and writes to `/Users/me/work/frontier`, but the summary records the literal-tilde path resolved against `cwd`.
- **Fix:** use `resolved_path(args.output_root)` from `workflow_runner_common`, or pass the already-resolved `output_root` from the orchestrator.

### Performance

#### F1 (perf) — HIGH — Hypervolume "leave-one-out" annotation in history loop
- **Location:** `frontier_campaign_reporting.build_frontier_hypervolume_history` calls `annotate_hypervolume_contributions` per lane while walking lane records. With L lanes and N members per step, total work is `O(L · N · cost_HV)`.
- **Fix:** the history only needs `frontier_archive_hypervolume`, not per-member contributions. Replace the per-step annotation with a bare hypervolume call.

#### F2 (perf) — HIGH — `_union_hypervolume` recursive sweep allocates per frame
- **Location:** `frontier_archive.py:721-742`.
- **Fix:** replace the tuple-slice recursion with a NumPy-array view-based recursion (no copy on slice). Same algorithmic complexity, lower allocation overhead.

#### F8 (perf) — MEDIUM — Redundant `np.stack` allocation in `_frontier_chebyshev_goal`
- **Location:** `frontier_scalarization.py:424-439`.
- **Fix:** replace `np.stack([...]).sum(axis=0)` with explicit weighted sum (`coeffs[0] * (-w_iota * dJ_iota_metric / iota_scale) + …`).

#### F9 (perf) — MEDIUM — `to_json_dict` re-validates schema on every persist
- **Locations:** `frontier_archive.FrontierArchiveMember.to_json_dict`, `frontier_progress_state.FrontierLaneRecord.to_json_dict`, `frontier_progress_state.FrontierCampaignProgress.to_json_dict`.
- **Current state:** Each `persist_progress()` walks the entire `lane_records` list, calling validators per record. With 50 lanes × 2 persists/lane that's ~100 redundant validate walks per campaign.
- **Fix:** split each `to_json_dict` into validating (`to_validated_json_dict`) and non-validating variants. Validate at load and at the final write boundary; skip on intermediate persists.

### Tests

The full coverage map and 30 proposed new tests live in `/tmp/frontier_review/06_tests.md`. Highlights still-deferred:

- **Solver checkpoint disk round-trip:** the existing `test_solver_checkpoint_round_trip_preserves_conditioning_reports` is misnamed — it round-trips through an in-memory dict. Add a test that calls `write_solver_checkpoint` to a tempfile and `load_solver_checkpoint` back.
- **Multi-worker lane parallelism:** present (`lane_workers=2` exists) but does not assert order preservation across the parallel path.
- **Resume regressions:** present at the high level, but missing arg-drift coverage (Resume #1) and salvage-vs-checkpoint precedence (Resume).
- **`apply_frontier_scalarization_override`:** entry point that wires Chebyshev / epsilon / ALM into the live solver — has no integration test.
- **Tautological assertion** at `tests/geo/test_frontier_archive.py:150`: `assertFalse(dominates(better, worse) and dominates(worse, better))` reduces to `assertFalse(False)`. Replace with `assertTrue(dominates(better, worse) ^ dominates(worse, better))` or remove.
- **Weak `assertAlmostEqual`** at `tests/geo/test_frontier_archive.py:531, 533, 534`: default `places=7` on values around `1e-9` is `5e-8` tolerance ≈ 14× the value. Use `delta=1e-15` or `places=15`.
- **`build_pareto_objective_normalization` ideal/nadir branch:** only the seed-relative kind is exercised through the manifest builder; the ideal/nadir branch is constructed by hand in tests rather than routed through this builder.

---

## Recommended fix ordering

A sensible landing order (non-binding, ordered by risk × value):

1. **F2.1** — small patch, eliminates archive-bloat foot-gun. Behavioral API change; land alongside test updates.
2. **F1.1 + F1.3** together — add `qs_scale` / `boozer_scale` fields with no defaults, bump `frontier_achievement_spec_v1` to a new schema version, regenerate all in-tree spec fixtures so they emit the new keys (no backward-compat shim — old specs error out per the no-fallbacks guardrail).
3. **F1.2** — decide `physics_total` semantics; either include all geometry penalties in ALM or rename the key.
4. **Resume #1** — manifest-authoritative resume. Largest single change; folds F02 and P5 (manifest-vs-args drift, manifest refresh-on-resume) in one shot. Resume #5 is independent — it's a one-line validator change in `frontier_runtime_calibration.py:165-176` and should be landed alongside but not assumed to be covered.
5. **F1.7** — simplex-floor renormalization. Two-line patch.
6. **A-1**, **A-3**, **A-4** — closest-match duplicate detection, REJECTED suffix, fail-closed on non-nadir.
7. **F01**, **F03**, **F05** — small orchestrator hardening.
8. **P6**, **P8**, **P10** — summary completeness.
9. **F1 (perf)**, **F9 (perf)** — drop hypervolume re-annotation in history; defer schema validation in `to_json_dict`.
10. Test gaps — solver-checkpoint disk round-trip, integration test for `apply_frontier_scalarization_override`, tautological-assertion cleanup.

The remaining MEDIUM/LOW math findings (F2.2, F2.3, F2.4, F3.1, F3.3, F4.2, F4.4) and perf items (F2 perf, F8 perf) follow as housekeeping.

---

## Source reports

The 9 raw agent reports live at `/tmp/frontier_review/01_math_physics.md` … `09_perf_thread.md` (ephemeral). Each finding above carries that report's ID prefix (`F1.x`, `A-x`, `P-x`, etc.) so the precise patch sketch and reasoning can be cross-checked.
