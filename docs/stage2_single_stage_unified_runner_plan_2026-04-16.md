# Stage 2 + Single-Stage Unified Runner Plan

Date: 2026-04-16
Status: Partially implemented. The unified runner, donor-repair sibling wrapper, shared handoff schema, and Stage 2 iota decision-gate wrapper have landed; remaining items are rollout/measurement follow-ups.
Scope: `examples/single_stage_optimization/` user-facing runner orchestration, donor-handoff contract, and shared Boozer probe seam.

## Implementation Status

The core workflow described here is now live in the working tree:

- `examples/single_stage_optimization/run_stage2_to_single_stage.py`
- `examples/single_stage_optimization/run_single_stage_donor_repair.py`
- `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py`
- `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py`

The later Stage 2-native benchmark / recommendation layer that this plan feeds is also
implemented in `examples/single_stage_optimization/run_stage2_iota_decision_gate.py`.
What still remains open from this document is empirical execution: collecting runtime and
success-rate measurements and deciding whether any deeper merge is justified.

## Verdict

Use a **unified workflow**, not a fully unified hot-loop objective.

- Keep current Stage 2 as the cheap coil-shaping phase.
- Insert a Stage 2.5 donor-bootability gate and (optional) bootstrap-recovery step.
- Enter full single-stage optimization only after the donor passes the gate.

This gives one user-facing command and one artifact handoff contract, while avoiding the main failure mode of a true merge: putting Boozer solves inside the Stage 2 L-BFGS-B inner loop.

## Relationship To The Broader Stage 2 Iota Plan

This document is the **first implementation slice** of the broader root-fix roadmap in
`docs/stage2_iota_root_fix_and_ishw_deliverables_plan_2026-04-16.md`.

- The broader document remains the umbrella strategy and decision log.
- This document is the concrete execution plan for the sibling plan's **Phase B2** ("Implement The Unified Runner And Reporting-Only Probe"), which explicitly names this file at sibling-plan lines 423–424.
- The probe and recovery helpers defined here must be reused by any later Stage 2-native reporting, soft, or hard `iota` implementation (sibling plan Phases B3, B4, and B6).
- The sibling plan's **Phase B7** ("Optional Standalone Donor-Repair Entrypoint") is a later batch-workflow sibling of this runner and must share the same helpers and status schema.
- Contract open questions frozen in sibling plan Phase B0 (reference surface, rational-surface blocklist, acceptance tolerances) are inputs here; this document does not re-decide them.
- This document does **not** compete with the broader plan; it operationalizes the "fix the handoff contract first" path before modifying the Stage 2 hot loop.

## Why Not A True Merge

Current code is split for a real performance reason:

- Stage 2 objective is cheap and geometry-dominant.
- Single-stage objective is expensive and Boozer-coupled.
- `Iotas.compute()` triggers `BoozerSurface.run_code(...)` whenever the Boozer state is dirty (`src/simsopt/geo/surfaceobjectives.py:954`, dirty-flag set by `BoozerSurface.recompute_bell(...)` at `src/simsopt/geo/boozersurface.py:253`). Stage 2 changes coil DOFs every objective and line-search evaluation, so any in-loop `Iotas` call re-solves the Boozer surface every step.
- Stage 2.5, by contrast, probes Boozer **once per donor** (outside any L-BFGS-B inner loop) and only re-solves inside the bounded bootstrap-recovery optimizer when recovery is actually invoked.

A true merge would make every Stage 2 trial step much more expensive and much less numerically predictable; the proposed Stage 2.5 gate pays Boozer cost only at handoff boundaries.

## Recommended Target Architecture

Single entrypoint with three internal stages. Named "Stage-S2/S2.5/S3" so the labels do not collide with the numbered implementation workstreams below:

1. **Stage-S2: Stage 2 shaping**
   - Optimize current Stage 2 objective (SquaredFlux + hardware penalties/ALM).
   - Output a standard Stage 2 artifact (`biot_savart_opt.json` + `results.json`).
2. **Stage-S2.5: donor bootability and (optional) bootstrap recovery**
   - Load the Stage 2 artifact via `workflow_runner_common.load_stage2_artifact_results(...)` and `banana_opt.artifact_contracts.upgrade_legacy_stage2_artifact_results(...)`.
   - Build a Boozer surface via the shared probe seam.
   - Check self-intersection, solved `iota`, and bootability.
   - If needed and not disabled by CLI, run a small-budget bootstrap-recovery optimization using single-stage physics-aware terms.
3. **Stage-S3: full single-stage**
   - Run the existing single-stage entrypoint only if the seed passes the bootability contract.

## Terminology Disambiguation

To avoid collision with existing single-stage vocabulary in the code:

- Existing `_SEED_REGIME_REPAIR_FIRST` / `phase1_repair_mode_active` in `banana_opt/single_stage_phase1.py:47` = bounded local feasibility recovery **inside** single-stage Phase 1.
- Existing `_SEED_REGIME_BRIDGE_ONLY` in `banana_opt/single_stage_phase1.py:48` = short local bridge solve from a clean initializer inside single-stage.
- **This plan's "Stage 2.5 bootstrap recovery"** = bounded cross-regime optimization that takes a Stage 2 donor and repairs Boozer/`iota` bootability *before* single-stage is invoked.

Do not call the new step "repair mode" or "bridge" unqualified in code, docstrings, or artifact keys. Use the fully qualified "Stage 2.5 bootstrap recovery" label and prefix new keys / helpers with `RECOVERY_` or `BOOTABILITY_`.

## Implementation Goals

- [x] Ship one user-facing runner for the combined workflow. → `examples/single_stage_optimization/run_stage2_to_single_stage.py`.
- [x] Preserve existing standalone Stage 2 and standalone single-stage entrypoints with exact unchanged CLI behavior when the new runner is not invoked. → Standalone `run_stage2_alm.py` and `SINGLE_STAGE/single_stage_banana_example.py` continue to run unchanged; the unified wrapper imports them rather than rewriting them.
- [x] Reuse the existing Stage 2 artifact contract and single-stage artifact upgrade path (`artifact_contracts.upgrade_legacy_stage2_artifact_results`). → `workflow_runner_common.load_validated_stage2_seed_results(...)` drives both sides.
- [x] Add an explicit seed bootability contract instead of relying on "single-stage will recover". → `probe_stage2_seed_bootability(...)` + `BOOTABILITY_*` keys.
- [x] Make recovery results observable in `results.json` via the existing hardware-schema payload helper convention. → `build_bootability_recovery_payload_fields(...)` emits `RECOVERY_ATTEMPTED/SUCCEEDED/ITERS/TERMINATION_REASON`.
- [x] Avoid duplicating large objective-construction logic across files. → Recovery delegates to `run_single_stage_thresholded_physics_alm.build_single_stage_thresholded_physics_command(...)` rather than rebuilding single-stage objective machinery.

## Non-Goals (verified preserved)

- [x] Do not rewrite Stage 2 and single-stage into one giant objective first.
- [x] Do not put full Boozer/`iota`/QS terms directly into the Stage 2 L-BFGS-B hot loop. → Stage 2 hot loop still geometry-only when `--stage2-iota-mode=off` (default); reporting-only probe runs after Stage 2 convergence.
- [x] Do not replace existing Stage 2 or single-stage result schemas wholesale in the first pass.
- [x] Do not require ALM parity between Stage 2 and single-stage before the workflow seam is proven.
- [x] Do not overload the single-stage-only `alm_state.partial.json` file; Stage 2.5 recovery state, if persisted, gets a distinct filename. → Recovery runs the existing single-stage `thresholded_physics` ALM under its own `<output>/recovery/` root; no Stage 2 partial-state artifact introduced.

## Workstream 1: Unified Runner Skeleton

### Deliverable

- [x] Add a new wrapper entrypoint, `examples/single_stage_optimization/run_stage2_to_single_stage.py`.
- [x] Before landing, decide whether to unify with the sibling plan's Phase B7 alias `run_single_stage_donor_repair.py`. → Decision: **coexist**. Both landed (`run_stage2_to_single_stage.py`, `run_single_stage_donor_repair.py`). Both import the same `banana_opt/stage2_single_stage_handoff.py` helpers and emit the same `BOOTABILITY_*`/`RECOVERY_*` payload via `build_bootability_recovery_payload_fields(...)`; no second repair stack was created.

### Responsibilities

- [x] Parse a combined CLI:
  - [x] Stage 2 generation or `--stage2-bs-path` → `--stage2-bs-path` / `--stage2-profile` / `--stage2-spec-json` (mutually-exclusive parse-time check).
  - [x] bootability probe options → `--bootability-iota-tolerance`, `--equilibrium-path`, `--num-tf-coils`.
  - [x] recovery-budget options → `--recovery-maxiter`, `--recovery-ftol`, `--recovery-gtol`, `--recovery-stage`, `--recovery-output-root`.
  - [x] full single-stage options → parent parser reused from `run_single_stage_goal_mode_comparison.build_parser(...)` via argparse `parents=[...]`.
- [x] Run Stage 2 directly or load an existing Stage 2 donor. → `resolve_stage2_input(...)` dispatches on `--stage2-bs-path`.
- [x] Pass a normalized Stage 2 artifact into Stage-S2.5 using the existing `load_stage2_artifact_results(...)` / `upgrade_legacy_stage2_artifact_results(...)` seams. → `load_validated_stage2_seed_results(...)` wraps both.

### Constraints

- [x] Keep existing `STAGE_2/banana_coil_solver.py` and `SINGLE_STAGE/single_stage_banana_example.py` usable as standalone scripts; acceptance = their existing smoke tests plus a `--help` diff still pass after any refactor. → No CLI changes to standalone entrypoints; `initialize_boozer_surface` extraction is a pure move into `banana_opt/stage2_single_stage_handoff.py`.
- [x] Minimize new business logic in the wrapper; prefer calling shared helpers under `banana_opt/`. → Runner orchestrates; probe/classify/payload logic lives in `banana_opt/stage2_single_stage_handoff.py` + `banana_opt/hardware_constraint_schema.py`.

## Workstream 2: Stage 2.5 Bootability Contract

### Contract Definition

Add a seed-level bootability contract with explicit statuses. These keys should be treated as the SSOT status vocabulary referenced by the broader Stage 2 `iota` plan, not a parallel schema. All new keys are UPPERCASE to match the existing hardware schema (see `build_hardware_constraint_artifact_payload_fields(...)` at `banana_opt/hardware_constraint_schema.py:165`, which emits `HARDWARE_CONSTRAINTS_OK` / `HARDWARE_CONSTRAINT_VIOLATIONS`):

- [x] `BOOZER_BOOTABLE`
- [x] `IOTA_FEASIBLE`
- [x] `BOOTABILITY_REASON`
- [x] `BOOTABILITY_STAGE`
- [x] `BOOTABILITY_TARGET_IOTA`
- [x] `BOOTABILITY_SOLVED_IOTA`
- [x] `BOOTABILITY_SELF_INTERSECTING`

All seven keys emitted through the SSOT companion helper `build_bootability_recovery_payload_fields(...)` at `banana_opt/hardware_constraint_schema.py:249` (enumerable via `bootability_recovery_payload_field_names(...)`). The helper also emits four diagnostic siblings (`BOOTABILITY_SOLVE_SUCCESS`, `BOOTABILITY_ABS_IOTA_ERROR`, `BOOTABILITY_ERROR_TYPE`, `BOOTABILITY_ERROR_MESSAGE`) that are populated by `classify_bootability_result(...)` but are not part of the required schema.

### Minimum Acceptance Rule

- [x] Boozer solve converges or reaches accepted residual threshold.
- [x] Surface is not self-intersecting.
- [x] `|iota_solved - iota_target| <= tolerance`, where `iota_target` is the CLI `--iota-target` value used as a bootstrap check.
- [x] Note: in single-stage `frontier` mode, `--iota-target` (defined at `SINGLE_STAGE/single_stage_banana_example.py:1064`) is only a Boozer initialization guess, **not** the outer-objective direction. Stage 2.5 uses it for gate-keeping only and should not rely on it being the same value the frontier lane optimizes against.
- [ ] Optional: `iota_solved` must lie outside a rational-surface blocklist — blocklist contents are frozen in sibling plan Phase B0; this plan consumes it rather than defining it. → **not implemented**; deferred until sibling Phase B0 publishes a concrete blocklist.

### Suggested Tolerances

- [x] Start with one CLI tolerance for `abs(iota error)`. → `--bootability-iota-tolerance` (default 5.0e-3).
- [x] Keep it separate from full single-stage ALM thresholds. → not reused by `--alm-iota-penalty-threshold`.
- [x] Treat bootability as a binary handoff gate, not a scalar score. → `bootability_passes(...)` returns `bool`.

## Workstream 3: Bootability Probe Implementation

### Reuse Existing Seams

- [x] Reuse Stage 2 artifact loading and upgrade helpers (`workflow_runner_common.load_stage2_artifact_results`, `banana_opt.artifact_contracts.upgrade_legacy_stage2_artifact_results`). → Upgrader now backfills `BOOTABILITY_*` defaults too, so older artifacts cleanly pass through.
- [x] Reuse single-stage coil loading from the Stage 2 `biot_savart_opt.json` (already driven by `--stage2-bs-path`). → `simsopt._core.optimizable.load` invoked inside `probe_stage2_seed_bootability(...)`.
- [x] Reuse `initialize_boozer_surface(...)` — extraction completed. → Function now lives in `banana_opt/stage2_single_stage_handoff.py:271`; the single-stage script re-imports from there. The 7110-line script's module-level `sys.path` manipulation no longer leaks into callers of the seam.

### New Helper Functions

- [x] Add a shared Boozer seam under `banana_opt/` (e.g. `banana_opt/boozer_probe.py`) with:
  - [x] `build_reference_boozer_surface(...)` equivalent. → `attempt_initialize_boozer_surface(...)` at `banana_opt/stage2_single_stage_handoff.py:180`. Also exposes the legacy `initialize_boozer_surface(...)` wrapper for call-sites that want the raising behavior.
  - [x] `evaluate_bootability(...)` equivalent. → `classify_bootability_result(...)` at `banana_opt/stage2_single_stage_handoff.py:352`, backed by the `BoozerInitializationResult` dataclass. Deviation: the seam was named `stage2_single_stage_handoff.py` rather than `boozer_probe.py` so handoff/contract validation and Boozer-probing live in one module.
- [x] Add a helper that builds the minimal Boozer boot context from a Stage 2 artifact. → `probe_stage2_seed_bootability(...)` at `banana_opt/stage2_single_stage_handoff.py:430` consumes the upgraded Stage 2 `results.json`.
- [x] Add a helper that writes bootability metadata into a JSON artifact payload via the UPPERCASE payload-builder convention. → `build_bootability_recovery_payload_fields(...)` / `bootability_recovery_payload_field_names(...)` at `banana_opt/hardware_constraint_schema.py:229–288`.

### Output Requirements

- [x] On success, emit a structured bootability payload. → `classify_bootability_result` returns `BOOTABILITY_REASON="ok"` payload with all expected fields populated.
- [x] On failure, emit diagnostics explaining whether failure came from:
  - [x] self-intersection → `BOOTABILITY_REASON_SELF_INTERSECTION`.
  - [x] bad `iota` → `BOOTABILITY_REASON_IOTA_MISMATCH`.
  - [x] Boozer solve failure → `BOOTABILITY_REASON_BOOZER_SOLVE_FAILED`.
  - [x] missing artifact metadata → `BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA`.
- [x] Probe runs **once per donor**, not per objective evaluation. → Called once before recovery and once on the recovery artifact; never inside the L-BFGS-B inner loop.

## Workstream 4: Stage 2.5 Bootstrap Recovery

### Purpose

Bootstrap recovery is the bridge between a geometry-good donor and a single-stage-usable donor. It is deliberately **not** named "repair" (to avoid colliding with single-stage's `_SEED_REGIME_REPAIR_FIRST`) and **not** named "bridge" (to avoid colliding with `_SEED_REGIME_BRIDGE_ONLY`).

### Design

- [x] Start from the loaded Stage 2 coil state.
- [x] Build Boozer surfaces once via the shared seam.
- [x] Run a small-budget optimization with single-stage physics-aware terms. → Delegates to `run_single_stage_thresholded_physics_alm` under `<output>/recovery/`.
- [x] Keep hardware checks active throughout (reuse `hardware_constraint_schema.py` bounds). → Thresholded-physics ALM runner already applies hardware bounds; nothing is disabled in the recovery command builder.
- [~] Terminate via whichever fires first:
  - [ ] bootability callback that re-probes every `--recovery-probe-every` inner iterations and stops once the acceptance rule is met → **not implemented**; deferred pending benchmarking data that justifies the orchestration complexity.
  - [x] the `--recovery-maxiter` bounded iteration budget. → Bootability is re-checked **once** on the recovery artifact after L-BFGS-B exits (either via `--recovery-ftol`/`--recovery-gtol` or the max-iter budget).

### Initial Bootstrap-Recovery Objective

Prefer a narrow objective over the full single-stage stack. Align term choice with what single-stage already builds at `banana_opt/single_stage_objectives.py:422–444`, which wraps physics terms in thresholded ALM form via `_objective_upper_bound_constraint(...)`:

- [x] `iota_penalty` — thresholded wrap of `Jiota` matching single-stage `thresholded_physics` ALM formulation. → Forwarded via `--alm-iota-penalty-threshold` default.
- [x] `boozer_residual` — same thresholded wrap style as single-stage. → Forwarded via `--alm-boozer-threshold` default.
- [x] existing hardware penalties / bounds. → Inherited from the thresholded-physics ALM stack.
- [x] optional light field-error (`SquaredFlux`) term to prevent geometry drift. → Included whenever the thresholded-physics ALM objective includes the field-error term; not gated off by the unified runner.

Avoid starting with:

- [x] full topology scoring (see `docs/single_stage_topology_fidelity_contract_2026-04-15.md`; Stage 2.5 recovery runs below the cheap-tier natural use case) → not wired into recovery.
- [x] full frontier goal-mode search → recovery uses the target-style thresholded-physics runner, not the frontier engine.
- [x] multi-surface complexity unless the single-surface recovery proves insufficient → probe and recovery operate on the outermost surface configuration only.

### Bootstrap-Recovery CLI Controls

- [x] `--recovery-maxiter` (positive int; bounded inner budget). → Default `80`.
- [x] `--recovery-ftol` (positive float; scipy-style objective tolerance). → Default `1.0e-15`.
- [x] `--recovery-gtol` (positive float; scipy-style gradient tolerance). → Default `1.0e-15`.
- [ ] `--recovery-probe-every N` (int; bootability callback cadence inside the inner loop). → **not implemented** (see Design block).
- [x] `--recovery-only` (run Stage-S2.5 and stop; full single-stage suppressed). → Mutually exclusive with `--probe-only`.
- [x] `--skip-recovery` (probe only; if bootability fails, workflow stops unless the force flag is also set).
- [x] `--force-full-single-stage-after-recovery-fail` (only meaningful when recovery is available to attempt).

Flag-combination rules (reject contradictions at CLI parse time):

| `--skip-recovery` | `--recovery-only` | `--force-full-single-stage-after-recovery-fail` | behavior |
|---|---|---|---|
| no | no | no (default) | probe → recover if needed → single-stage if bootable (happy path) |
| no | yes | — | probe → recover → stop (single-stage suppressed) |
| yes | no | no | probe only → stop at first failure |
| yes | no | yes | probe only → single-stage even if probe failed (escape hatch) |
| yes | yes | — | reject at parse time (contradictory) |

## Workstream 5: Handoff Into Full Single-Stage

### Rules

- [x] If bootability passes immediately, go straight to full single-stage.
- [x] If recovery succeeds (and `--recovery-only` is not set), promote the recovered coils as the single-stage seed. → Handoff promotes `<recovery-output>/biot_savart_opt.json` and sets `UNIFIED_SEED_SOURCE=recovered_stage2_donor`.
- [x] If recovery fails, stop cleanly unless `--force-full-single-stage-after-recovery-fail` is passed.

### Artifact Requirements

- [x] Record whether the final run came from:
  - [x] direct Stage 2 donor → `UNIFIED_SEED_SOURCE=direct_stage2_donor`.
  - [x] recovered Stage 2 donor → `UNIFIED_SEED_SOURCE=recovered_stage2_donor`.
  - [ ] fully standalone single-stage seed → **N/A for this runner**; the unified runner always starts from a Stage 2 donor (generated or loaded). This seed-source option belongs to the legacy direct single-stage entrypoint, not to `run_stage2_to_single_stage.py`.
- [x] Record the recovery summary in final `results.json`. → `RECOVERY_ATTEMPTED`/`RECOVERY_SUCCEEDED`/`RECOVERY_ITERS`/`RECOVERY_TERMINATION_REASON` + `BOOTABILITY_*` written by `update_results_json(full_results_path, handoff_results_payload(...))`.

## File-Level Touch Plan

### New

- [x] `examples/single_stage_optimization/run_stage2_to_single_stage.py`
- [x] `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py` — deliberately named "handoff", not "bridge", to avoid colliding with `_SEED_REGIME_BRIDGE_ONLY`.
- [~] `examples/single_stage_optimization/banana_opt/boozer_probe.py` — **consolidated**, not created. The shared Boozer probe seam lives in `banana_opt/stage2_single_stage_handoff.py` alongside the handoff contract validators (`attempt_initialize_boozer_surface`, `initialize_boozer_surface`, `classify_bootability_result`, `probe_stage2_seed_bootability`). No separate `boozer_probe.py` file was created.

### Likely Edits

- [x] `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  - [x] factor `initialize_boozer_surface` (currently at line 1411) into `banana_opt/stage2_single_stage_handoff.py` and re-import it.
  - [x] expose a minimal bootability probe seam → re-exported via the handoff module's public surface.
  - [ ] acceptance: existing `--help` output and a canonical standalone smoke run produce equivalent artifacts. → Verified manually during extraction; no automated `--help`/artifact regression test was added. **Remaining follow-up** if desired.
- [x] `examples/single_stage_optimization/banana_opt/artifact_contracts.py`
  - [x] extend `upgrade_legacy_stage2_artifact_results(...)` with a no-op passthrough for the new `BOOTABILITY_*` / `RECOVERY_*` keys (so older artifacts still upgrade cleanly). → Backfills `BOOTABILITY_STAGE2_BS_PATH` / `BOOTABILITY_STAGE2_RESULTS_PATH` defaults.
- [x] `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py`
  - [x] add one shared helper for bootability/recovery status payload fields alongside `build_hardware_constraint_artifact_payload_fields(...)`. → `build_bootability_recovery_payload_fields(...)`.
  - [x] keep the helper reusable by later Stage 2-native `iota` reporting. → Already consumed by `STAGE_2/banana_coil_solver.py` for Phase B3 reporting.

### Avoid Editing Unless Needed

- [~] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  - Stage 2 *did* gain a reporting-only probe here after all (sibling plan's Phase B3 landed alongside this workstream). `banana_coil_solver.py` now calls `probe_stage2_seed_bootability(...)` when `--stage2-iota-mode` is not `off`, and emits the same `BOOTABILITY_*` payload via `build_bootability_recovery_payload_fields(...)`. This is additive and does not change default behavior (`--stage2-iota-mode=off`).

## Schema / Metadata Todos

- [x] Define bootability/recovery payload keys in one SSOT helper shared with any later Stage 2-native `iota` reporting path.
- [x] Keep top-level hardware artifact status untouched. → `HARDWARE_CONSTRAINTS_OK` / `HARDWARE_CONSTRAINT_VIOLATIONS` remain independent of the bootability keys.
- [x] Avoid overloading single-stage-only files like `alm_state.partial.json`; use a distinct filename (e.g. `bootstrap_recovery_state.partial.json`) if recovery inner state is ever persisted. → Not applicable: recovery state reuses the single-stage ALM runner's own output directory; no cross-stage partial-state aliasing was introduced.

### Provenance Field Inventory

Distinguish existing fields from newly proposed fields — two fields that earlier drafts listed as "new" are **already** emitted by today's code:

**Already emitted (reuse, do not redefine):**

- `STAGE2_BS_PATH` — written into Stage 2 `results.json` at `banana_opt/stage2_objectives.py:194`. Consumed by single-stage via the `--stage2-bs-path` CLI argparse seam at `SINGLE_STAGE/single_stage_banana_example.py:1167` (env-var default `STAGE2_BS_PATH`) and resolved into a path at `SINGLE_STAGE/single_stage_banana_example.py:5350` (`build_stage2_bs_path(args)`). Also re-emitted into single-stage's own payloads at `SINGLE_STAGE/single_stage_banana_example.py:4091` and `:6734` as provenance.
- `STAGE2_RESULTS_PATH` — written into single-stage payloads at `SINGLE_STAGE/single_stage_banana_example.py:4092` and `:6735`; the value itself is derived by `workflow_runner_common.load_stage2_artifact_results(...)` (invoked at `single_stage_banana_example.py:5351`).

**Genuinely new (add via the SSOT payload helper):**

- `RECOVERY_ATTEMPTED`
- `RECOVERY_SUCCEEDED`
- `RECOVERY_ITERS`
- `RECOVERY_TERMINATION_REASON`
- `BOOTABILITY_REASON` (and the `BOOTABILITY_*` / `IOTA_FEASIBLE` / `BOOZER_BOOTABLE` set from Workstream 2)

## Validation Plan

### Unit / Targeted Tests

- [x] Add a handoff-helper test for Stage 2 artifact metadata upgrade. → `tests/geo/test_stage2_single_stage_handoff.py::test_upgrade_legacy_stage2_artifact_results_backfills_handoff_defaults` (line 138).
- [x] Add a bootability-result classification test. → `test_classify_bootability_result_rejects_iota_mismatch` (line 157).
- [ ] Add a test that a known good seed is marked bootable — reuse an existing fixture under `examples/single_stage_optimization/equilibria/` such as `wout_nfp22ginsburg_000_014417_iota15.nc`. → **remaining follow-up**. Current probe coverage is metadata-shape and mismatch-classification only.
- [ ] Add a test that a known bad near-zero-`iota` donor is marked not bootable — if no existing fixture covers this, construct one deterministically by zeroing banana current on a known bootable donor. → **remaining follow-up**.
- [x] Add a test that recovery refusal is clean and structured when Boozer init fails. → Partial: `test_probe_stage2_seed_bootability_reports_missing_metadata_without_loading_bs` (line 179) covers missing-metadata structured refusal. Full Boozer-init-failure fixture is still part of the follow-ups above.
- [x] Add a CLI parse-time rejection test for contradictory flag combinations (`--skip-recovery` + `--recovery-only`). → `test_recovery_only_conflict_with_skip_recovery_is_rejected` (line 583).

### Integration Tests

- [x] Add a smoke test for:
  - [x] load Stage 2 donor → `test_load_stage2_seed_metadata_for_handoff_backfills_legacy_tf_current_from_cli` (line 269).
  - [x] bootability probe only → `test_probe_only_writes_summary_with_bootability_status` (line 221).
  - [x] recovery-only mode → `test_recovery_only_updates_recovery_results_with_handoff_metadata` (line 352).
  - [x] full handoff mode → `test_full_mode_augments_final_results_with_recovered_handoff_metadata` (line 455).
- [x] Verify that output artifacts are written and contain bootability metadata.
- [ ] Verify that standalone `banana_coil_solver.py` and `single_stage_banana_example.py` continue to run with no behavior change. → **remaining follow-up**; no automated regression test for standalone-entrypoint equivalence exists yet.

### Manual Validation

- [ ] Confirm a strict Stage 2 donor that currently fails direct single-stage bootstrap is classified correctly.
- [ ] Confirm at least one known bootable donor passes the probe without recovery.
- [ ] Confirm recovered donors hand off into single-stage without schema drift.

Manual validation items remain human acceptance steps; they are not tracked in the test suite.

## Rollout Order

### Step 1

- [x] Extract `initialize_boozer_surface` into a shared library module. → Moved to `banana_opt/stage2_single_stage_handoff.py` (not a separate `banana_opt/boozer_probe.py`, see File-Level Touch Plan above).
- [x] Implement the unified wrapper with probe-only mode. → `run_stage2_to_single_stage.py --probe-only`.
- [x] No recovery optimization yet.
- [x] Goal: make bootability visible and measurable.

### Step 2

- [x] Add recovery-only mode with a small-budget `iota_penalty`-style thresholded objective.
- [x] Goal: recover some bad donors without touching Stage 2 itself.

### Step 3

- [x] Add full pipeline mode: Stage 2 → probe → recover → single-stage.
- [x] Goal: one-command workflow.

### Step 4

- [ ] Benchmark recovery cost and success rate. → **pending**; scaffolding exists via `run_stage2_iota_decision_gate.py`, but the benchmark data collection/summary is empirical work that still needs to be executed on canonical cases.
- [ ] Feed those measurements into the broader Stage 2 `iota` decision gate (sibling plan Phase B4). → **pending**.
- [ ] Decide whether Stage 2 itself needs an `iota`-aware soft term. → **partial**: the soft term is now *available* behind `--stage2-iota-mode=soft` in `banana_coil_solver.py`, but the measurement-driven decision to promote it to default has not been made.

### Step 5

- [~] Only after the handoff works, reconsider Stage 2-native soft or hard `iota`. → Implementation ladder now present (`off → report → soft → alm` via `--stage2-iota-mode`); promotion / default-mode choice is still pending measurement data.
- [ ] Only after that, reconsider a deeper objective-level merge. → Not undertaken; see Decision Gate below.

## Decision Gate For A True Merge Later

Only revisit a full Stage 2 + single-stage objective merge if all are true:

- [ ] recovery mode is still too expensive or too unreliable → not yet measured on canonical cases.
- [ ] Stage 2 donors remain systematically unbootable → not yet measured across the sweep.
- [ ] Boozer-aware evaluation cost is benchmarked and acceptable → not yet benchmarked.
- [ ] a shared objective builder exists so the merge is mostly orchestration, not duplication → partial groundwork via shared payload helpers; no unified objective builder.

## Cross-Repo State Verified (2026-04-16)

Anchors this plan depends on, cross-checked against the working tree:

- Stage 2 objective at `STAGE_2/banana_coil_solver.py` (1157 lines); objective terms verified; no `BoozerSurface`/`Iotas` in Stage 2 today.
- Stage 2 ALM evaluation at `banana_opt/stage2_objectives.py` (977 lines); already emits `STAGE2_BS_PATH` at line 194.
- Single-stage at `SINGLE_STAGE/single_stage_banana_example.py` (7110 lines); `initialize_boozer_surface` at line 1411; `--iota-target` at line 1064; `STAGE2_BS_PATH`/`STAGE2_RESULTS_PATH` emitted at lines 4091–4092 and 6734–6735.
- Single-stage objective builder at `banana_opt/single_stage_objectives.py` (530 lines) wraps `Jiota`, `raw_J_QS_obj`, `raw_J_Boozer_obj` as `iota_penalty`, `qs_error`, `boozer_residual` via `_objective_upper_bound_constraint(...)` for the `thresholded_physics` ALM formulation (lines 422–444).
- `Iotas.compute()` at `src/simsopt/geo/surfaceobjectives.py:954` calls `BoozerSurface.run_code(...)` when `boozer_surface.need_to_run_code` is True; flag set by `BoozerSurface.recompute_bell(...)` at `src/simsopt/geo/boozersurface.py:253` via the Optimizable dependency chain.
- Hardware schema: `build_hardware_constraint_artifact_payload_fields(...)` at `banana_opt/hardware_constraint_schema.py:165` emits `HARDWARE_CONSTRAINTS_OK` / `HARDWARE_CONSTRAINT_VIOLATIONS` as UPPERCASE keys.
- Existing seed-regime vocabulary: `_SEED_REGIME_REPAIR_FIRST` (line 47) and `_SEED_REGIME_BRIDGE_ONLY` (line 48) in `banana_opt/single_stage_phase1.py`, exposed through `SINGLE_STAGE/single_stage_banana_example.py:1175–1186`.

Any downstream edits should recheck these anchors before landing.

## Bottom Line

- [x] Build **one workflow**. → `run_stage2_to_single_stage.py` (plus Phase B7 batch alias `run_single_stage_donor_repair.py` sharing the same helpers).
- [x] Keep **two optimization regimes** (Stage 2 and single-stage). → Stage 2 hot loop remains geometry-only by default; single-stage physics terms live in the recovery and full-run stages.
- [x] Insert **one explicit Stage 2.5 bootability / bootstrap-recovery contract** between them, named to avoid collision with single-stage's existing `REPAIR_FIRST` / `BRIDGE_ONLY` vocabulary. → `BOOTABILITY_*` / `RECOVERY_*` payload + `UNIFIED_SEED_SOURCE` labels.

That was the lowest-risk root-level fix for the current codebase. The follow-up empirical work listed under Rollout Step 4 is what is still open.
