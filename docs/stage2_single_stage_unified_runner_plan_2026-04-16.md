# Stage 2 + Single-Stage Unified Runner Plan

## Verdict

Use a **unified workflow**, not a fully unified hot-loop objective.

- Keep current Stage 2 as the cheap coil-shaping phase.
- Insert a Stage 2.5 Boozer/iota bootability and repair phase.
- Enter full single-stage optimization only after the donor is bootable.

This gives one user-facing command and one artifact contract, while avoiding the main failure mode of a true merge: putting Boozer solves inside the Stage 2 L-BFGS-B inner loop.

## Why Not a True Merge

Current code is split for a real performance reason:

- Stage 2 objective is cheap and geometry-dominant.
- Single-stage objective is expensive and Boozer-coupled.
- `Iotas` can trigger `BoozerSurface.run_code(...)` when the Boozer state is dirty.
- Stage 2 evaluates objective and gradient repeatedly inside L-BFGS-B and basin/ALM flows.

So a true merge would make every Stage 2 trial step much more expensive and much less numerically predictable.

## Recommended Target Architecture

Single entrypoint with three internal phases:

1. **Phase A: Stage 2 shaping**
   - Optimize current Stage 2 objective.
   - Output a standard Stage 2 artifact.
2. **Phase B: Stage 2.5 donor bootability / repair**
   - Load the Stage 2 artifact.
   - Build Boozer surfaces.
   - Check self-intersection, solved `iota`, and bootability.
   - If needed, run a small-budget repair optimization using single-stage physics terms.
3. **Phase C: Full single-stage**
   - Run full single-stage only if the seed passes the bootability contract.

## Implementation Goals

- [ ] Ship one user-facing runner for the combined workflow.
- [ ] Preserve existing standalone Stage 2 and standalone single-stage entrypoints.
- [ ] Reuse the existing Stage 2 artifact contract and single-stage artifact upgrade path.
- [ ] Add an explicit seed bootability contract instead of relying on “single-stage will recover.”
- [ ] Make repair results observable in `results.json`.
- [ ] Avoid duplicating large objective-construction logic across files.

## Non-Goals

- [ ] Do not rewrite Stage 2 and single-stage into one giant objective first.
- [ ] Do not put full Boozer/iota/QS terms directly into the Stage 2 L-BFGS-B hot loop.
- [ ] Do not replace existing Stage 2 or single-stage result schemas wholesale in the first pass.
- [ ] Do not require ALM parity between Stage 2 and single-stage before the workflow seam is proven.

## Phase A: Unified Runner Skeleton

### Deliverable

- [ ] Add a new wrapper entrypoint, e.g. `examples/single_stage_optimization/run_stage2_to_single_stage.py`.

### Responsibilities

- [ ] Parse a combined CLI:
  - [ ] Stage 2 generation or `--stage2-bs-path`
  - [ ] bootability probe options
  - [ ] repair-budget options
  - [ ] full single-stage options
- [ ] Run Stage 2 directly or load an existing Stage 2 donor.
- [ ] Pass a normalized Stage 2 artifact into the next phase.

### Constraints

- [ ] Keep existing `banana_coil_solver.py` and `single_stage_banana_example.py` usable as standalone scripts.
- [ ] Minimize new business logic in the wrapper; prefer calling shared helpers.

## Phase B: Stage 2.5 Bootability Contract

### Contract Definition

Add a seed-level bootability contract with explicit statuses:

- [ ] `BOOZER_BOOTABLE`
- [ ] `IOTA_FEASIBLE`
- [ ] `BOOTABILITY_REASON`
- [ ] `BOOTABILITY_STAGE`
- [ ] `BOOTABILITY_TARGET_IOTA`
- [ ] `BOOTABILITY_SOLVED_IOTA`
- [ ] `BOOTABILITY_SELF_INTERSECTING`

### Minimum Acceptance Rule

- [ ] Boozer solve converges or reaches accepted residual threshold.
- [ ] Surface is not self-intersecting.
- [ ] `|iota_solved - iota_target| <= tolerance`.

### Suggested Tolerances

- [ ] Start with one CLI tolerance for `abs(iota error)`.
- [ ] Keep it separate from full single-stage ALM thresholds.
- [ ] Treat bootability as a binary handoff gate, not a scalar score.

## Phase C: Bootability Probe Implementation

### Reuse Existing Seams

- [ ] Reuse Stage 2 artifact loading and upgrade helpers.
- [ ] Reuse single-stage coil loading from the Stage 2 `biot_savart_opt.json`.
- [ ] Reuse `initialize_boozer_surface(...)` to avoid inventing a second Boozer init path.

### New Helper Functions

- [ ] Add a helper that builds the minimal single-stage boot context from a Stage 2 artifact.
- [ ] Add a helper that attempts Boozer initialization and returns a structured bootability result.
- [ ] Add a helper that writes bootability metadata into a JSON artifact payload.

### Output Requirements

- [ ] On success, emit a structured bootability payload.
- [ ] On failure, emit diagnostics explaining whether failure came from:
  - [ ] self-intersection
  - [ ] bad `iota`
  - [ ] Boozer solve failure
  - [ ] missing artifact metadata

## Phase D: Repair Mode

### Purpose

Repair mode is the bridge between a geometry-good donor and a single-stage-usable donor.

### Design

- [ ] Start from the loaded Stage 2 coil state.
- [ ] Build Boozer surfaces once.
- [ ] Run a small-budget optimization with single-stage physics-aware terms.
- [ ] Keep hardware checks active throughout.
- [ ] Stop as soon as bootability passes.

### Initial Repair Objective

Prefer a narrow objective over the full single-stage stack:

- [ ] `Jiota`
- [ ] Boozer residual term
- [ ] existing hardware penalties / bounds
- [ ] optional light field-error term to avoid geometry drift

Avoid starting with:

- [ ] full topology scoring
- [ ] full frontier goal-mode search
- [ ] multi-surface complexity unless the single-surface repair proves insufficient

### Repair Budget Controls

- [ ] `--repair-maxiter`
- [ ] `--repair-ftol`
- [ ] `--repair-gtol`
- [ ] `--repair-stage`
- [ ] `--repair-only`
- [ ] `--skip-repair`

## Phase E: Handoff into Full Single-Stage

### Rules

- [ ] If bootability passes immediately, go straight to full single-stage.
- [ ] If repair succeeds, promote the repaired coils as the single-stage seed.
- [ ] If repair fails, stop cleanly unless an explicit `--force-full-single-stage-after-repair-fail` flag is passed.

### Artifact Requirements

- [ ] Record whether the final run came from:
  - [ ] direct Stage 2 donor
  - [ ] repaired Stage 2 donor
  - [ ] fully standalone single-stage seed
- [ ] Record the repair summary in final `results.json`.

## File-Level Touch Plan

### New

- [ ] `examples/single_stage_optimization/run_stage2_to_single_stage.py`
- [ ] `examples/single_stage_optimization/banana_opt/stage2_single_stage_bridge.py`

### Likely edits

- [ ] `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  - [ ] factor reusable Stage 2-seed loading / Boozer-init helpers
  - [ ] expose a minimal bootability probe seam
- [ ] `examples/single_stage_optimization/banana_opt/artifact_contracts.py`
  - [ ] extend Stage 2 artifact upgrade / validation for bridge metadata
- [ ] `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py`
  - [ ] optional shared helper for adding bridge status payload fields

### Avoid editing unless needed

- [ ] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  - [ ] only touch if a tiny helper extraction materially reduces duplication

## Schema / Metadata Todos

- [ ] Define bridge-status payload keys in one SSOT helper.
- [ ] Keep top-level hardware artifact status untouched.
- [ ] Avoid overloading single-stage-only files like `alm_state.partial.json`.
- [ ] Add provenance fields:
  - [ ] `STAGE2_BS_PATH`
  - [ ] `STAGE2_RESULTS_PATH`
  - [ ] `REPAIR_ATTEMPTED`
  - [ ] `REPAIR_SUCCEEDED`
  - [ ] `REPAIR_ITERS`
  - [ ] `BOOTABILITY_REASON`

## Validation Plan

### Unit / targeted tests

- [ ] Add a bridge helper test for Stage 2 artifact metadata upgrade.
- [ ] Add a bootability-result classification test.
- [ ] Add a test that a known good seed is marked bootable.
- [ ] Add a test that a known bad near-zero-`iota` donor is marked not bootable.
- [ ] Add a test that repair-mode refusal is clean and structured when Boozer init fails.

### Integration tests

- [ ] Add a smoke test for:
  - [ ] load Stage 2 donor
  - [ ] bootability probe only
  - [ ] repair-only mode
  - [ ] full bridge mode
- [ ] Verify that output artifacts are written and contain bridge metadata.

### Manual validation

- [ ] Confirm a strict Stage 2 donor that currently fails direct single-stage bootstrap is classified correctly.
- [ ] Confirm at least one known bootable donor passes probe without repair.
- [ ] Confirm repaired donors hand off into single-stage without schema drift.

## Rollout Order

### Step 1

- [ ] Implement the unified wrapper with probe-only mode.
- [ ] No repair optimization yet.
- [ ] Goal: make bootability visible and measurable.

### Step 2

- [ ] Add repair-only mode with a small-budget objective.
- [ ] Goal: recover some bad donors without touching Stage 2 itself.

### Step 3

- [ ] Add full pipeline mode: Stage 2 -> probe -> repair -> single-stage.
- [ ] Goal: one-command workflow.

### Step 4

- [ ] Benchmark repair cost and success rate.
- [ ] Decide whether Stage 2 itself needs an `iota`-aware soft term.

### Step 5

- [ ] Only after the bridge works, reconsider a deeper objective-level merge.

## Decision Gate for a True Merge Later

Only revisit a full Stage 2 + single-stage objective merge if all are true:

- [ ] repair mode is still too expensive or too unreliable
- [ ] Stage 2 donors remain systematically unbootable
- [ ] Boozer-aware evaluation cost is benchmarked and acceptable
- [ ] a shared objective builder exists so the merge is mostly orchestration, not duplication

## Bottom Line

- [ ] Build **one workflow**
- [ ] Keep **two optimization regimes**
- [ ] Insert **one explicit bootability/repair contract** between them

That is the lowest-risk root-level fix for the current codebase.
