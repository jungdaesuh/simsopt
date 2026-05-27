# Standalone Donor Repair Retirement Plan

## Purpose

This plan defines the end-to-end removal of the standalone
`run_single_stage_donor_repair.py` wrapper. The goal is to reduce workflow
confusion while preserving the shared Stage 2 to single-stage Boozer probe and
recovery machinery that is still useful and aligned with the SIMSOPT
`BoozerSurface.run_code(iota, G=...)` solve path.

## Goals

- Remove `examples/single_stage_optimization/run_single_stage_donor_repair.py`
  as a user-facing and test-loaded entrypoint.
- Remove active README, CLI, and test references that advertise or depend on
  the standalone donor-repair lane.
- Keep `run_stage2_to_single_stage.py` as the single Stage 2.5 handoff entrypoint.
- Keep shared bootability and recovery helpers in
  `banana_opt/stage2_single_stage_handoff.py` because they are used by the
  unified runner and Stage 2 reporting probes.
- Make the Stage 2 iota decision gate independent of deleted donor-repair
  summary artifacts.

## Non-Goals

- Do not remove `run_stage2_to_single_stage.py`.
- Do not remove `probe_stage2_seed_bootability`, `bootability_passes`,
  `run_recovery_stage`, or the shared `BOOTABILITY_*` / `RECOVERY_*` schema.
- Do not change SIMSOPT Boozer solve behavior or finite-current handling.
- Do not modify jhalpern30 replay/exporter behavior except to ensure docs do
  not route replay through the retired wrapper.
- Do not rewrite historical dated plans as if the old wrapper never existed;
  mark retirement explicitly where needed.

## Current Context

- `examples/single_stage_optimization/run_single_stage_donor_repair.py` imports
  `run_stage2_to_single_stage.py` as `unified_runner` and reuses its parser,
  validation, Stage 2 input resolution, Boozer probe, and recovery stage.
- The wrapper is brittle as a public workflow: its "best donor" selection is a
  repo-local policy, not a SIMSOPT or literature-backed target-iota repair
  algorithm.
- The Boozer primitive itself is still valid: the probe eventually initializes
  a Boozer surface and calls `boozer_surface.run_code(iota, G)`.
- `run_stage2_iota_decision_gate.py` currently accepts
  `--donor-repair-summary`, loads `single_stage_donor_repair` summary JSON, and
  can emit `prefer_unified_runner_donor_repair`.
- `tests/geo/test_stage2_track_b_wrappers.py` directly loads
  `run_single_stage_donor_repair.py` and tests the decision-gate donor-repair
  branch.
- `examples/single_stage_optimization/README.md` currently advertises the
  standalone donor-repair wrapper and its `best_repaired_donor.json` output.
- Dated planning docs reference the wrapper as a landed artifact. These are
  historical, but active command guidance inside them should not contradict the
  retirement decision.

Verified source anchors:

- `examples/single_stage_optimization/run_single_stage_donor_repair.py`
- `examples/single_stage_optimization/run_stage2_to_single_stage.py`
- `examples/single_stage_optimization/run_stage2_iota_decision_gate.py`
- `tests/geo/test_stage2_track_b_wrappers.py`
- `examples/single_stage_optimization/README.md`
- upstream SIMSOPT source at
  `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/boozersurface.py`

## Rationale

The repository already has clearer owner boundaries:

- Stage 2 artifact generation and reporting belong to `run_stage2_alm.py` and
  `STAGE_2/banana_coil_solver.py`.
- Stage 2 to single-stage handoff, bootability probing, bounded recovery, and
  full single-stage handoff belong to `run_stage2_to_single_stage.py`.
- Direct single-stage optimization belongs to
  `SINGLE_STAGE/single_stage_banana_example.py` and its focused wrappers.

Keeping a second standalone donor-repair lane adds cognitive load and duplicate
workflow vocabulary without adding a distinct physics primitive. It also makes
the decision gate depend on a side-channel summary file from a brittle wrapper.

Design-it-twice outcome:

- Option A: tighten `run_single_stage_donor_repair.py` so "best donor" requires
  `IOTA_FEASIBLE=True` and make target-coupled recovery the default. This
  preserves the CLI, but keeps another public lane and another artifact
  contract to explain.
- Option B: retire the standalone wrapper and keep the shared probe/recovery
  helpers under `run_stage2_to_single_stage.py`. This removes the confusing
  public surface while preserving the reusable implementation. Choose Option B.

## Assumptions

- No production automation depends on invoking
  `examples/single_stage_optimization/run_single_stage_donor_repair.py`
  directly. Repo-tracked automation will be checked by grep; external scripts
  are out of scope unless surfaced before implementation.
- Existing users can use `run_stage2_to_single_stage.py --probe-only`,
  `--recovery-only`, or full mode for supported Stage 2.5 handoff workflows.
- Historical docs may retain references only when the text clearly marks the
  standalone wrapper as retired or historical.
- The decision gate does not need to ingest old donor-repair summary JSON after
  this cleanup.

## Implementation Plan

1. Remove the standalone wrapper entrypoint.
   - [ ] Delete `examples/single_stage_optimization/run_single_stage_donor_repair.py`.
   - [ ] Confirm no remaining executable import of `run_single_stage_donor_repair`.
   - [ ] Keep `run_stage2_to_single_stage.py` and shared handoff helpers unchanged
         unless tests expose a real dependency cleanup issue.

2. Remove the decision-gate donor-repair side channel.
   - [ ] In `examples/single_stage_optimization/run_stage2_iota_decision_gate.py`,
         remove the `--donor-repair-summary` CLI argument.
   - [ ] Remove `_load_donor_repair_signal(...)`.
   - [ ] Remove `donor_repair_signal` from `build_summary(...)`,
         `_recommendation_payload(...)`, and `_decision_summary(...)`.
   - [ ] Remove recommendation value `prefer_unified_runner_donor_repair`.
   - [ ] Make the no-runtime-data and normal recommendation paths depend only on
         Stage 2 mode payloads and the existing runtime/iota metrics.

3. Update active tests.
   - [ ] In `tests/geo/test_stage2_track_b_wrappers.py`, remove
         `DONOR_REPAIR_PATH` and `load_donor_repair_module()`.
   - [ ] Remove `DonorRepairWrapperTests`.
   - [ ] Remove
         `test_decision_gate_prefers_donor_repair_when_report_is_not_bootable`.
   - [ ] Keep `UnifiedRunnerStage2InputTests` and Stage 2 decision-gate tests
         that do not load the deleted wrapper.
   - [ ] Add or update a decision-gate assertion showing that a nonbootable
         report path recommends the remaining supported seam, not donor repair.

4. Update active user documentation.
   - [ ] In `examples/single_stage_optimization/README.md`, remove the top-level
         bullet for `run_single_stage_donor_repair.py`.
   - [ ] Remove the numbered "Standalone Donor Repair" entry from the current
         workflow model and renumber following entries.
   - [ ] Remove the directory-layout line for `run_single_stage_donor_repair.py`.
   - [ ] Remove the "Standalone Donor Repair" section and command example.
   - [ ] Remove the `best_repaired_donor.json` and `--donor-repair-summary`
         references.
   - [ ] Ensure the README points Stage 2.5 users to
         `run_stage2_to_single_stage.py` only.

5. Update historical docs without rewriting history.
   - [ ] In `docs/stage2_single_stage_unified_runner_plan_2026-04-16.md`, add a
         dated note that `run_single_stage_donor_repair.py` was retired on
         2026-05-27 after review; the unified runner remains the supported lane.
   - [ ] In `docs/stage2_iota_root_fix_and_ishw_deliverables_plan_2026-04-16.md`,
         mark standalone donor repair as retired where it is listed as a landed
         artifact.
   - [ ] In `docs/single_stage_surface_mode_split_impl_plan_2026-04-19.md`,
         mark references historical or remove them if they are active file lists.
   - [ ] Leave purely historical discussion intact when it is clearly dated and
         not presented as current runnable guidance.

6. Run reference cleanup checks.
   - [ ] Run:
         `rg "run_single_stage_donor_repair|single_stage_donor_repair|best_repaired_donor|prefer_unified_runner_donor_repair|donor-repair-summary|donor_repair" examples tests docs --glob '!docs/single_stage_donor_repair_retirement_plan_2026-05-27.md'`
   - [ ] Confirm no active code, test, or README references remain.
   - [ ] Confirm any remaining dated-doc references include explicit
         "retired" or "historical" wording.

## Validation Plan

- [ ] Run focused wrapper and decision-gate tests:
      `PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_stage2_track_b_wrappers.py tests/geo/test_stage2_single_stage_handoff.py -q`
- [ ] Run the single-stage example regression tests if touched imports ripple:
      `PYTHONPATH=build/cp313-cp313-macosx_26_0_arm64:src:examples/single_stage_optimization python3.13 -m pytest tests/geo/test_single_stage_example.py -q`
- [ ] Run lint on edited Python files:
      `ruff check examples/single_stage_optimization/run_stage2_iota_decision_gate.py tests/geo/test_stage2_track_b_wrappers.py`
- [ ] Run docs/reference grep:
      `rg "run_single_stage_donor_repair|single_stage_donor_repair|best_repaired_donor|prefer_unified_runner_donor_repair|donor-repair-summary|donor_repair" examples tests docs --glob '!docs/single_stage_donor_repair_retirement_plan_2026-05-27.md'`
- [ ] Re-check the upstream/local Boozer solve contract:
      `rg -n "def run_code\\(|class BoozerSurface" /Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/boozersurface.py examples/single_stage_optimization/banana_opt/boozer_finite_current.py`
- [ ] Run whitespace check before commit:
      `git diff --check`

## Risks and Mitigations

- Risk: A user or script still invokes `run_single_stage_donor_repair.py`.
  Mitigation: Search tracked files first; if an external dependency appears,
  replace it with documented `run_stage2_to_single_stage.py` usage rather than
  keeping a compatibility shim.

- Risk: Removing `donor_repair_signal` changes decision-gate summary schema.
  Mitigation: Treat this as intentional public-surface reduction and update
  tests/docs to assert the new schema.

- Risk: Historical docs become misleading after file deletion.
  Mitigation: Add explicit retirement notes to dated plans that mention the
  deleted wrapper as landed or supported.

- Risk: Shared probe/recovery helper removal is accidentally pulled into scope.
  Mitigation: Keep changes scoped to the standalone wrapper, decision-gate
  side-channel, tests, and docs; verify `run_stage2_to_single_stage.py` tests
  still pass.

## Completion Criteria

- [ ] `examples/single_stage_optimization/run_single_stage_donor_repair.py` is
      deleted.
- [ ] No active README, Python, or test reference advertises or imports the
      deleted wrapper.
- [ ] `run_stage2_iota_decision_gate.py` no longer accepts
      `--donor-repair-summary` or emits `prefer_unified_runner_donor_repair`.
- [ ] `run_stage2_to_single_stage.py` and shared Boozer probe/recovery helpers
      remain tested and available.
- [ ] Focused tests, lint, reference grep, and `git diff --check` pass.
- [ ] Historical dated docs either omit the wrapper from active runnable lists
      or clearly mark it retired.

## Open Questions

- None. The implementation should remove the wrapper, clean active references,
  mark dated historical references as retired where they remain visible, and
  route supported Stage 2.5 workflows through `run_stage2_to_single_stage.py`.
