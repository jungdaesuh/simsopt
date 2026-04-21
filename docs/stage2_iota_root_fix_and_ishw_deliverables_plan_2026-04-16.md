# Stage 2 Iota Root Fix And ISHW Deliverables Plan

Date: 2026-04-16
Status: Partially implemented. The Track A deliverable wrappers and the Track B Stage 2 iota/reporting/decision-gate code paths have landed; remaining items are empirical benchmark runs, decision outcomes, and any follow-on polish.
Scope: `examples/single_stage_optimization/` Stage 2 donor contract, single-stage runner workflows, scan/plot generation, and Poincare deliverables for the ISHW talk.

## Implementation Status

The codebase now contains the planned wrapper and Stage 2 iota seams referenced by this
document:

- Track A wrappers:
  - `examples/single_stage_optimization/run_single_stage_iota_target_sweep.py`
  - `examples/single_stage_optimization/run_banana_current_scan.py`
  - `examples/single_stage_optimization/plot_ishw_tradeoffs.py`
- Track B handoff / donor-repair wrappers:
  - `examples/single_stage_optimization/run_stage2_to_single_stage.py`
  - `examples/single_stage_optimization/run_single_stage_donor_repair.py`
- Track B Stage 2-native iota and decision-gate paths:
  - `examples/single_stage_optimization/run_stage2_alm.py`
  - `examples/single_stage_optimization/run_stage2_iota_decision_gate.py`
  - `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  - `examples/single_stage_optimization/banana_opt/stage2_objectives.py`

The open items in this plan are now mostly execution questions rather than missing entry
points: benchmark a canonical case, compare `report`/`soft`/`alm` runtime and bootability,
and decide whether the hard Stage 2 ALM path is worth carrying further for a given campaign.

## Executive Summary

The quoted request set is really two different projects with very different risk profiles:

1. Near-term analysis and plotting deliverables for Carlos's talk.
2. A root-level Stage 2 fix so Stage 2 donors are not hardware-valid but single-stage-invalid.

They should not be handled as one monolithic implementation.

The near-term talk asks are mostly runner and artifact work:

- scan over `iota` targets and report tradeoffs
- produce a cleaned field-error vs coil-length view
- generate a current Poincare plot
- scan banana-coil current from zero to optimized current and report diagnostics

Those are achievable mostly by reusing the existing single-stage wrappers and Poincare tooling, plus one or two new wrapper scripts.

The Stage 2 root fix is a solver-contract change:

- Stage 2 currently optimizes boundary field error plus hardware penalties/constraints.
- single-stage expects a Stage 2 donor that can initialize a usable Boozer surface and a non-degenerate `iota`.
- Those contracts do not currently match.

The right implementation strategy is:

- do the talk deliverables first because they are lower risk and time sensitive
- implement the Stage 2 root fix in phases
- start with a measured soft `iota` prototype before committing to a full ALM-level Stage 2 `iota` constraint

## Current Repo Facts

These facts are the basis for the plan.

- Stage 2 objective assembly lives in `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`.
- Stage 2 ALM evaluation lives in `examples/single_stage_optimization/banana_opt/stage2_objectives.py`.
- Stage 2 currently includes:
  - `SquaredFlux`
  - coil length
  - coil-coil spacing
  - coil-surface spacing
  - curvature
  - banana current upper bound in ALM mode
- Stage 2 currently does not include:
  - `BoozerSurface`
  - `Iotas`
  - an `iota` loss
  - an `iota` feasibility gate
- single-stage owns the current Boozer and `iota` path in `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`.
- single-stage already builds `Iotas(...)`, `Jiota`, and thresholded-physics ALM handling for `iota_penalty` in `examples/single_stage_optimization/banana_opt/single_stage_objectives.py`.
- `src/simsopt/geo/surfaceobjectives.py` shows that `Iotas.J()` is the user-facing entry point (lazy); it delegates to `Iotas.compute()` (at `surfaceobjectives.py:954`), which re-solves the Boozer surface whenever `self.boozer_surface.need_to_run_code` is `True`.
- `src/simsopt/geo/boozersurface.py` shows that `BoozerSurface.recompute_bell(...)` (at `boozersurface.py:253`) sets `need_to_run_code = True` whenever a parent (e.g. `biotsavart`) state changes, via the `Optimizable` dependency bell.
- Because Stage 2 changes coil DOFs every objective evaluation, any Stage 2 `Iotas` integration sits directly on the hot path. On the current code path that means repeated Boozer resolves on repeated objective and line-search evaluations; any finite-difference prototype would amplify that cost further, but finite differences are not part of the verified current Stage 2 path.
- `examples/single_stage_optimization/run_finite_current_smoke.py` already exists, but it scans plasma current (CLI flag `--currents-A` is a CSV of plasma currents, default `"0,8000,-35200"`, forwarded into the single-stage script as `--plasma-current-A`); it does not vary banana-coil current.
- `examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py` already produces the strict and diagnostic Poincare artifacts, and it is already automatable non-interactively via the `POINCARE_OUT_DIR` environment variable. Its current limitation is that the entry point is `__main__`-only with hardcoded `nfieldlines=50`, `tmax_fl=7000`, etc., so a small CLI or helper seam would improve configurability and reuse but is not a strict prerequisite for wrapper automation.
- `alm_state.partial.json` currently belongs to single-stage only. Stage 2 does not currently write that file; its analogous artifacts are `biot_savart_opt.json` and `results.json`.
- Stage 2 already exposes banana-coil current controls via `--banana-init-current-A` and `--banana-current-max-A` in `STAGE_2/banana_coil_solver.py`, but `--banana-init-current-A` is only used for fresh initialization. When `--stage2-bs-path` is used, the saved banana current is loaded from the artifact instead, so a banana-current scan over an existing optimized donor needs either direct mutation of the loaded `Current` DOF or a new explicit override seam.
- The existing hardware-status vocabulary lives in `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py` and emits `HARDWARE_CONSTRAINTS_OK` and `HARDWARE_CONSTRAINT_VIOLATIONS` fields into artifact payloads. Any new Stage 2 status fields must extend this schema rather than shadow it with parallel names.

## Requirement Inventory

### A. Root Fix Requirements

- [x] Stage 2 donor success must align with what single-stage actually needs to consume. → Stage 2 now emits `BOOTABILITY_*` / `IOTA_FEASIBLE` / `BOOZER_BOOTABLE` / `STAGE2_ROOT_FIX_ENABLED` through the shared payload helper; single-stage and the unified runner consume the same vocabulary.
- [x] Stage 2 must distinguish hardware feasibility from Boozer and `iota` bootability. → `HARDWARE_CONSTRAINTS_OK` stays separate from `BOOZER_BOOTABLE` / `IOTA_FEASIBLE`; they can fail independently.
- [x] A Stage 2 donor that is marked as the best rescue or incumbent must remain exactly hardware-valid under the current Stage 2 contract. → Stage 2 incumbent/rescue logic is unchanged; the bootability probe runs after incumbent selection and does not rewrite it.
- [x] The new Stage 2 path must expose a clear `iota` target and a clear `iota` tolerance. → `--stage2-iota-target` and `--stage2-iota-tolerance` CLI flags (plus env-var defaults).
- [x] The Stage 2 `iota` path should reuse single-stage Boozer and `iota` machinery where practical instead of reimplementing the math. → Stage 2 calls `probe_stage2_seed_bootability(...)` in `banana_opt/stage2_single_stage_handoff.py`; the probe wraps the same `BoozerSurface` / `Iotas` objects single-stage uses.
- [x] The new behavior must be feature-flagged so current Stage 2 semantics remain unchanged when the feature is disabled. → `--stage2-iota-mode` defaults to `off`; `STAGE2_ROOT_FIX_ENABLED` is emitted so downstream consumers can branch on the extended schema.
- [~] Runtime cost must be measured explicitly because every added Boozer solve can significantly raise per-iteration cost. → Per-run cost is now split across `STAGE2_IOTA_PROBE_SECONDS`, `STAGE2_IOTA_BOOTSTRAP_SECONDS`, and `STAGE2_IOTA_RUNTIME_SECONDS`; aggregated benchmark across canonical cases is still outstanding (see Phase B4 runtime checks).
- [x] Results artifacts must report whether a candidate is:
  - [x] hardware-feasible → `HARDWARE_CONSTRAINTS_OK`.
  - [x] Boozer-bootable → `BOOZER_BOOTABLE`.
  - [x] `iota`-feasible → `IOTA_FEASIBLE`.
- [x] Stage 2 final artifact preservation must not regress the recently landed exact-hardware-pass salvage behavior. → Salvage path is untouched; bootability probe emits only additive keys.
- [x] The plan must avoid pretending Stage 2 already has the same partial-state or ALM artifact schema as single-stage. → Stage 2 does not emit `alm_state.partial.json`; bootability payload lives alongside the existing Stage 2 `results.json` keys rather than duplicating single-stage partial-state artifacts.

### B. Near-Term ISHW Deliverable Requirements

- [x] Produce updated tradeoff data for increasing `iota` target versus engineering and physics metrics. → `run_single_stage_iota_target_sweep.py` scaffolding shipped; actual data collection on canonical surfaces is an execution step, not a code gap.
- [x] Include at minimum:
  - [x] coil length
  - [x] curvature
  - [x] field error or QS proxy
  - [x] banana current when relevant
- [x] Produce a cleaned plot for field error versus coil length, ideally using existing JD scan outputs if they already cover the needed range. → `plot_ishw_tradeoffs.py` emits the cleaned figure; which JD-scan inputs to use is a data-selection step.
- [x] Produce a Poincare plot for the current reference configuration. → Driven via `POINCARE_PLOTTING/poincare_surfaces.py` + `POINCARE_OUT_DIR`; wrappers invoke it non-interactively.
- [x] Produce a banana-coil current scan from zero to the optimized current value. → `run_banana_current_scan.py`.
- [x] For each banana-current scan point, attempt:
  - [x] Boozer initialization
  - [x] single-stage init-only metrics
  - [x] Poincare diagnostics
- [x] If Boozer initialization fails at some current values, still emit a usable fallback artifact set rather than failing the entire sweep. → Sweep classifies each point as `success`/`Boozer-failed`/`Poincare-only fallback` and continues.
- [x] The output format must be presentation-friendly, not just raw JSON dumps. → `plot_ishw_tradeoffs.py` emits PNG/PDF; summary JSON + slide-friendly CSV written by the sweep wrappers.

### C. Non-Goals (verified preserved)

- [x] Do not block talk plots on a full Stage 2 solver rewrite. → Talk wrappers ship without a Stage 2 rewrite.
- [x] Do not claim self-consistent finite-current equilibrium physics from the existing `--plasma-current-A` surrogate workflows.
- [x] Do not move Stage 2 into JAX or rewrite the whole optimization graph as part of the first pass.
- [x] Do not overload single-stage-only artifact names like `alm_state.partial.json` into Stage 2 without intentionally introducing a Stage 2 schema.
- [x] Do not silently redefine Stage 2 "hardware pass" to include physics constraints. → `HARDWARE_CONSTRAINTS_OK` still reflects the hardware contract only.
- [x] Do not implement a generic framework for arbitrary extra Stage 2 physics constraints before the `iota` use case is proven. → Only `iota` terms were added; no generic physics-extension framework introduced.

## Root Cause In Software Terms

For a lay software engineer, the bug is a producer-consumer contract mismatch.

Stage 2 currently produces something equivalent to:

```text
{
  "field_error_ok": true,
  "hardware_ok": true
}
```

single-stage actually needs something closer to:

```text
{
  "field_error_ok": true,
  "hardware_ok": true,
  "boozer_bootable": true,
  "iota_near_target": true
}
```

Stage 2 never checks the last two fields. That means it can truthfully say "success" and still hand single-stage a donor that is unusable for the next step.

The root fix is therefore not "retry more" or "scan more seeds". The root fix is to fix the contract boundary.

## Design Options

### Option 1. Add A Soft Stage 2 Iota Loss

Definition:

- compute `iota` inside the Stage 2 objective
- add a weighted penalty such as `W_iota * (iota_computed - iota_target)^2`

Pros:

- easiest path to prototype
- easiest path to benchmark
- minimal ALM schema churn
- lets us learn whether Stage 2 naturally moves away from near-axisymmetric donors

Cons:

- does not guarantee `iota` feasibility
- weight tuning can be awkward
- can still preserve a hardware-valid but `iota`-bad donor if the weight is weak

Recommendation:

- use this as the first experimental implementation, not the final contract

### Option 2. Add A Hard Stage 2 Iota ALM Constraint

Definition:

- compute a Stage 2 `iota` or `iota_penalty`
- treat it like a real ALM inequality constraint alongside the existing Stage 2 hard constraints

Pros:

- matches the current ALM contract style better
- supports a crisp tolerance
- better matches the "Stage 2 donor is single-stage bootable" end-state

Cons:

- more plumbing
- higher testing burden
- more state bookkeeping
- still pays the Boozer solve cost inside the hot loop

Recommendation:

- do this only after the soft prototype proves the signal is useful and the runtime is acceptable

### Option 3. Do Donor Repair Outside Stage 2

Definition:

- keep Stage 2 as geometry and hardware only
- add a mandatory single-stage-side "Boozer and `iota` acquisition" repair step before full optimization

Pros:

- lowest-risk root-level alternative
- reuses more of the existing single-stage code
- decouples expensive Boozer work from the Stage 2 hot loop

Cons:

- does not make Stage 2 itself semantically complete
- keeps the system multi-stage rather than truly unifying the contract
- can still be expensive, just in a different location

Recommendation:

- treat this as the fallback if Stage 2 hot-loop Boozer cost is too high

## Recommended Architecture Decision

- [x] Split the work into two tracks:
  - [x] Track A: talk deliverables → three wrappers landed.
  - [x] Track B: Stage 2 root fix → Phases B0–B3, B6, B7 landed; B4 soft mode wired pending benchmark; B5 decision-gate runner landed but the decision itself awaits measurements.
- [x] Use a phased Stage 2 plan:
  - [x] Phase 0: contract freeze → status vocabulary locked in (`HARDWARE_CONSTRAINTS_OK`, `BOOZER_BOOTABLE`, `IOTA_FEASIBLE`, `STAGE2_ROOT_FIX_ENABLED`).
  - [x] Phase 1: shared Boozer and `iota` probe seam → `banana_opt/stage2_single_stage_handoff.py`.
  - [x] Phase 2: unified runner with bootability probe and donor-repair path → `run_stage2_to_single_stage.py` + `run_single_stage_donor_repair.py`.
  - [~] Phase 3: soft Stage 2 `iota` prototype → implemented behind `--stage2-iota-mode=soft`; benchmark verdict still pending.
  - [~] Phase 4: hard Stage 2 ALM `iota` constraint → implemented behind `--stage2-iota-mode=alm`; promotion to default still pending the decision-gate measurement.
- [x] Treat the unified-runner / donor-repair path in `docs/stage2_single_stage_unified_runner_plan_2026-04-16.md` as the first implementation slice of Track B, not as a competing plan. → That sibling plan is now marked **Implemented**.
- [x] Keep Stage 2-native `iota` work behind a later decision gate, informed by bridge runtime and success-rate measurements. → `run_stage2_iota_decision_gate.py` exists to drive the measurement; the decision outcome itself is still pending.
- [x] Keep hardware feasibility and `iota` bootability as separate reported statuses even if both eventually gate incumbent selection. → Independent keys in the payload helper.

## Proper Implementation Plan

### Track A. ISHW Talk Deliverables

#### Phase A0. Freeze Scope And Inputs

Status: scope/input decisions are human acceptance steps carried by the talk
owner (Carlos); the plan-level gating criteria remain open even though the
code-level wrappers that consume those decisions have already shipped.

- [ ] Confirm which plasma surfaces are in scope for the talk.
- [ ] Confirm which existing Stage 2 artifact is the reference donor for all near-term scans.
- [ ] Confirm whether Carlos wants full single-stage reruns for the `iota` target sweep or init-only comparison is acceptable for the first draft.
- [ ] Confirm whether JD's existing scan data is authoritative enough to reuse for the field-error versus coil-length cleanup plot.
- [ ] Freeze the presentation metric list:
  - `iota`
  - QS error or proxy
  - Boozer residual if available
  - coil length
  - max curvature
  - banana current
  - field error where directly available

#### Phase A1. Build The Iota-Target Sweep Runner

Implemented artifact:

- `examples/single_stage_optimization/run_single_stage_iota_target_sweep.py` (landed).

Responsibilities:

- [x] reuse one explicit Stage 2 artifact across all sweep points
- [x] sweep `--iota-target` values over a user-provided list
- [x] keep other single-stage knobs fixed
- [x] collect a compact summary JSON and a slide-friendly CSV
- [x] emit per-run paths so plots can be regenerated later

Minimum implementation details:

- [x] accept `--stage2-bs-path`
- [x] accept `--plasma-surf-filename`
- [x] accept `--iota-targets`
- [x] accept `--constraint-method`
- [x] forward the same geometry and Boozer discretization flags currently used by the single-stage wrappers
- [x] record failure modes without aborting the entire sweep

Actual file touches:

- [x] `examples/single_stage_optimization/run_single_stage_iota_target_sweep.py`
- [x] `examples/single_stage_optimization/workflow_runner_common.py` — minimal shared helpers reused, no new helpers introduced.
- [x] `examples/single_stage_optimization/README.md` — wrapper documented alongside the other Track A entrypoints.

#### Phase A2. Build The Banana-Current Scan Runner

Recommended new artifact:

- `examples/single_stage_optimization/run_banana_current_scan.py`

Reason this should be separate from `run_finite_current_smoke.py`:

- `run_finite_current_smoke.py` varies plasma current (via `--currents-A`, forwarded as `--plasma-current-A`), not banana-coil current.
- Carlos explicitly asked for banana-coil current variation with TF current fixed.
- The output and fallback semantics are different (banana-current sweeps must tolerate Boozer-init failure at low currents and still emit Poincare artifacts).

Responsibilities:

- [x] load one optimized coil set
- [x] scale the banana current from zero to the optimized value
- [x] for reused optimized Stage 2 donors, mutate the loaded banana `Current` DOF after deserialization or add one explicit override seam; do not assume `--banana-init-current-A` can drive this path when `--stage2-bs-path` is present
- [x] use `--banana-init-current-A` and `--banana-current-max-A` only for fresh-artifact generation paths
- [x] attempt single-stage init-only Boozer surface solve at each scale
- [x] run Poincare diagnostics for each case using `POINCARE_OUT_DIR`; add a CLI or helper seam to `poincare_surfaces.py` only if the scan needs configurable tracing parameters beyond the current defaults
- [x] collect metrics per current value
- [x] preserve partial outputs when some current values fail

Required behavior:

- [x] do not stop the whole sweep because one low-current case produces a self-intersecting or invalid Boozer surface
- [x] classify each current point as:
  - success
  - Boozer-failed
  - Poincare-only fallback
- [x] emit one summary table suitable for slide plotting

Actual file touches:

- [x] `examples/single_stage_optimization/run_banana_current_scan.py`
- [~] `examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py` — `POINCARE_OUT_DIR` remained sufficient; no CLI seam added.
- [x] `examples/single_stage_optimization/README.md`

#### Phase A3. Plotting And Slide Artifacts

Recommended new artifact:

- `examples/single_stage_optimization/plot_ishw_tradeoffs.py`

Responsibilities:

- [x] read the `iota` sweep summary
- [x] read the banana-current scan summary
- [x] generate presentation-ready PNG or PDF plots
- [x] support one cleaned field-error versus coil-length plot
- [x] support one current-configuration Poincare export step

Plot set:

- [x] `iota_target` versus coil length
- [x] `iota_target` versus max curvature
- [x] `iota_target` versus QS error or field error
- [x] banana current scale versus QS error
- [x] banana current scale versus `iota`
- [x] banana current scale versus Boozer success status
- [x] cleaned field-error versus coil-length figure

Actual file touches:

- [x] `examples/single_stage_optimization/plot_ishw_tradeoffs.py`
- [x] `examples/single_stage_optimization/README.md`

#### Phase A4. Validate And Package Talk Outputs

Status: manual packaging steps owned by the talk owner; wrappers emit the
required artifacts but the human-in-the-loop acceptance is still open.

- [ ] rerun the current reference configuration through `POINCARE_PLOTTING/poincare_surfaces.py`
- [ ] verify the chosen Poincare artifact is from the intended run directory
- [ ] verify plot labels use physical units and consistent naming
- [ ] check that all summary tables include explicit source run paths
- [ ] archive one slide-ready output directory with:
  - plots
  - summaries
  - the exact commands used

### Track B. Stage 2 Root Fix

#### Phase B0. Freeze The Contract

Status: frozen. The decisions below are captured in code and propagate through
the shared payload helper. Remaining open questions are scientific rather than
schema-level.

- [x] Define the user-visible Stage 2 contract in one sentence. → "Stage 2 is geometry-only by default (`--stage2-iota-mode=off`); when the feature flag is on, Stage 2 additionally certifies Boozer bootability and `iota` feasibility using the shared handoff probe."
- [x] Decide whether Stage 2 will certify:
  - hardware only
  - hardware plus `iota` bootability
  - or hardware plus `iota` bootability only when the feature flag is enabled → **flag-gated**. `STAGE2_ROOT_FIX_ENABLED` is written whenever `--stage2-iota-mode` is not `off`.
- [x] Define the status vocabulary. The existing schema in `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py` already emits `HARDWARE_CONSTRAINTS_OK` and `HARDWARE_CONSTRAINT_VIOLATIONS` — new fields are layered into the companion helper `build_bootability_recovery_payload_fields(...)` (at `hardware_constraint_schema.py:256`) rather than introduced as parallel lower-case keys:
  - reuse: `HARDWARE_CONSTRAINTS_OK` (already means: hardware-feasible under current Stage 2 contract)
  - add: `BOOZER_BOOTABLE` (Boozer surface initialization succeeded on the reference surface)
  - add: `IOTA_FEASIBLE` (measured reference-surface `iota` sits within `±tol` of the target and outside any rational-neighborhood blocklist)
  - add: `STAGE2_ROOT_FIX_ENABLED` (feature-flag marker so stale consumers can detect the extended schema)
- [x] Reuse the same bootability-status payload helper for:
  - unified runner probe and repair artifacts → via `build_bootability_recovery_payload_fields(...)` in `run_stage2_to_single_stage.py` and `run_single_stage_donor_repair.py`.
  - Stage 2 reporting-only mode → same helper invoked from `STAGE_2/banana_coil_solver.py` when `--stage2-iota-mode != off`.
  - any later Stage 2-native soft or hard `iota` path → same helper; `soft`/`alm` runs also emit `STAGE2_IOTA_HOT_LOOP_ENABLED`, `STAGE2_IOTA_RUNTIME_SECONDS`, `STAGE2_IOTA_RUNTIME_CALLS`.
- [x] Decide whether incumbent selection will require both exact hardware pass and `iota` feasibility. → **ALM-mode only**. In `alm` mode the promoted incumbent must be hardware-exact and `iota`-feasible; otherwise the secondary hardware-pass-but-iota-fail candidate is preserved under `STAGE2_SECONDARY_ARTIFACT_*`.
- [x] Decide whether the reported `iota` target applies on:
  - the plasma boundary
  - an outer Boozer surface
  - or a dedicated reference surface ratio → **outer Boozer surface**, consistent with single-stage.
- [x] Decide whether the Stage 2 feature should target:
  - raw `iota`
  - `Jiota`
  - or a thresholded `iota_penalty` → **`Jiota` in `soft` mode, thresholded `iota_penalty` in `alm` mode** (see `evaluate_stage2_alm_problem(...)` at `banana_opt/stage2_objectives.py:957`).
- [x] Decide on initial acceptance tolerances. → `--stage2-iota-tolerance` default `5.0e-3`; CLI-overridable.

Open questions that remain scientific (not schema-level):

- [~] Which reference surface is physically meaningful and numerically stable for Stage 2? → defaulted to the outer Boozer surface built by `initialize_boozer_surface(...)`; revisit after benchmark data.
- [ ] Should rational-surface neighborhoods be explicitly avoided in the default target list? → blocklist item deferred; see unified-runner plan Workstream 2 ("Optional: rational-surface blocklist").
- [x] Is a donor that is hardware-exact but `iota`-bad still worth preserving as a secondary artifact? → **yes**; see `build_stage2_secondary_artifact_metadata(...)` at `STAGE_2/banana_coil_solver.py:649` and the `STAGE2_SECONDARY_ARTIFACT_*` keys.

#### Phase B1. Factor A Shared Boozer And Iota Probe Seam

Status: implemented. The seam lives in
`examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py`
(see `attempt_initialize_boozer_surface`, `initialize_boozer_surface`,
`classify_bootability_result`, `probe_stage2_seed_bootability`).

Implementation (landed):

- [x] extract the minimum shared helper seam from single-stage Boozer initialization into a shared `banana_opt/` helper module → `banana_opt/stage2_single_stage_handoff.py`.
- [x] define one small helper that can:
  - build a reference Boozer surface → `attempt_initialize_boozer_surface(...)`.
  - evaluate `Iotas` → `probe_stage2_seed_bootability(...)` wraps `Iotas` via the same `BoozerSurface` single-stage uses.
  - return both scalar value and gradient-ready object handles → `BoozerInitializationResult` dataclass exposes the scalar value plus the Boozer-surface object.
- [x] keep the helper solver-owned, not runner-owned → lives under `examples/single_stage_optimization/banana_opt/`, imported by both the Stage 2 solver and the runner wrappers.
- [x] keep Stage 2-specific policy outside the helper → the helper returns the probe result; gate/incumbent policy stays in `STAGE_2/banana_coil_solver.py`.

Actual file touches:

- [x] `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` — re-imports the extracted helper instead of owning `initialize_boozer_surface` locally.
- [x] `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py` — new helper module.
- [x] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` — invokes the shared probe when `--stage2-iota-mode != off`.

Acceptance criteria:

- [x] single-stage behavior is unchanged after the helper extraction → `single_stage_banana_example.py` public CLI surface unchanged; helper is a pure relocation.
- [x] the helper can be invoked from a Stage 2-only smoke path → exercised by `tests/geo/test_stage2_single_stage_handoff.py`.
- [x] the helper can also be invoked from the unified runner's probe-only mode → `run_stage2_to_single_stage.py --probe-only` goes through the same seam.

#### Phase B2. Implement The Unified Runner And Reporting-Only Probe

This phase is the concrete execution plan captured in
`docs/stage2_single_stage_unified_runner_plan_2026-04-16.md`.

Goal:

- make the producer-consumer mismatch observable before changing the Stage 2 hot loop
- avoid building a second, separate donor-repair stack later

Implementation:

- [x] add the unified runner wrapper → `run_stage2_to_single_stage.py`.
- [x] implement probe-only mode first → `--probe-only` mode in the unified runner.
- [x] reuse the shared Boozer and `iota` helper from Phase B1 → direct import from `banana_opt/stage2_single_stage_handoff.py`.
- [x] report bootability and `iota` statuses through the shared payload helper → `build_bootability_recovery_payload_fields(...)`.
- [x] preserve existing standalone Stage 2 and standalone single-stage entrypoints → standalone CLIs unchanged.

Acceptance criteria:

- [x] one command can load or generate a Stage 2 donor and classify its bootability → `run_stage2_to_single_stage.py --probe-only`.
- [x] results use the same SSOT status vocabulary planned for later Stage 2-native work → all downstream Stage 2-native modes route through the same payload helper.
- [x] no second probe or repair schema is introduced outside the shared helper path → verified by `run_single_stage_donor_repair.py` sharing the same helper stack.

#### Phase B3. Add Reporting Before Optimization Gating

Goal:

- make Stage 2 able to compute and report `iota` before making it part of the optimization objective

Implementation:

- [x] add optional Stage 2 CLI flags for the `iota` probe
- [x] compute reference-surface `iota` for the initial donor
- [x] report the value in `results.json`
- [x] report Boozer success or failure explicitly
- [x] record elapsed time for the probe

Actual file touches:

- [x] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` — `--stage2-iota-mode report` probe + timing fields.
- [x] Stage 2 results or artifact helper modules under `banana_opt/` — `banana_opt/stage2_single_stage_handoff.py` + `banana_opt/hardware_constraint_schema.py` payload helpers.
- [x] `examples/single_stage_optimization/README.md`

Acceptance criteria:

- [x] Stage 2 can run in reporting-only mode with `iota` probe enabled → `--stage2-iota-mode=report` emits `STAGE2_IOTA_PROBE_SECONDS`, `BOOZER_BOOTABLE`, `IOTA_FEASIBLE`.
- [x] results clearly show whether the donor is hardware-valid but `iota`-bad → `HARDWARE_CONSTRAINTS_OK` remains independent of `IOTA_FEASIBLE`.

#### Phase B4. Implement The Soft Stage 2 Iota Prototype

Goal:

- test whether Stage 2 can be nudged toward single-stage-usable donors without full ALM constraint plumbing

Implementation:

- [x] add Stage 2 config for:
  - `--stage2-iota-mode {off,report,soft,alm}` → `DEFAULT_STAGE2_IOTA_MODE = "off"` at `banana_coil_solver.py:92`.
  - `--stage2-iota-target`
  - `--stage2-iota-weight`
  - `--stage2-iota-tol` → shipped as `--stage2-iota-tolerance`.
- [x] create an optional `Jiota` term for Stage 2 → guarded by `args.stage2_iota_mode in {"soft","alm"}` at `banana_coil_solver.py:1134`.
- [x] wire the scalar term into the Stage 2 penalty objective path → soft mode adds the weighted `Jiota`; ALM mode adds thresholded `iota_penalty` via `evaluate_stage2_alm_problem(...)`.
- [x] preserve current behavior when `--stage2-iota-mode=off` → default remains `off`; no behavior change unless the flag is enabled.
- [x] record final and initial `iota` → `BOOTABILITY_SOLVED_IOTA` from the probe, plus `STAGE2_IOTA_RUNTIME_*` counters.
- [x] record whether the best exact-hardware artifact also meets the `iota` tolerance → driven by `IOTA_FEASIBLE` and the secondary-artifact keys.

Actual file touches:

- [x] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- [x] `examples/single_stage_optimization/banana_opt/stage2_objectives.py` — extended with `include_iota_penalty` and `stage2_iota_penalty_threshold(...)`.
- [x] Stage 2 artifact summary helpers — `build_bootability_recovery_payload_fields(...)` extended; timing/hot-loop keys added to results payload.

Runtime and robustness checks:

- [ ] benchmark one canonical Stage 2 run with and without the soft `iota` term → **pending**; driven by `run_stage2_iota_decision_gate.py`.
- [ ] measure objective evaluation slowdown → **pending** (timing fields are emitted, but aggregate benchmark is not yet collected).
- [x] log Boozer failures separately from ordinary constraint failures → `classify_bootability_result` routes through `BOOTABILITY_REASON_BOOZER_SOLVE_FAILED`.
- [ ] check for unstable sign flips or noisy gradients near rational targets → **pending** (needs benchmark data).

Acceptance criteria:

- [ ] at least one canonical case improves donor `iota` materially without losing exact hardware feasibility → **pending measurement**.
- [ ] slowdown is measured and documented → **pending measurement**.
- [x] feature-off behavior matches current Stage 2 semantics → verified: when `--stage2-iota-mode=off` no probe, no `STAGE2_ROOT_FIX_ENABLED`, no new constraint terms.

#### Phase B5. Decide Whether Hard Stage 2 ALM Iota Is Worth It

This phase is a decision gate, not an automatic implementation step. The
decision-gate runner has landed (`run_stage2_iota_decision_gate.py`), but the
measurement itself and the resulting verdict are still outstanding.

- [ ] If the soft prototype clearly improves donor quality and runtime is acceptable, proceed to the hard ALM design.
- [ ] If the soft prototype is unstable or too expensive, stop here and keep the unified-runner / donor-repair path as the practical root fix.

Decision inputs:

- [ ] runtime multiplier → emitted via `STAGE2_IOTA_RUNTIME_SECONDS`/`_CALLS`, not yet aggregated.
- [ ] donor bootability improvement → emitted via `IOTA_FEASIBLE` / `BOOZER_BOOTABLE`, not yet aggregated.
- [ ] implementation complexity still remaining → hard ALM path already shipped; remaining work is scientific evaluation.
- [ ] whether the unified-runner probe and repair path already solves the practical workflow problem → pending the benchmark comparison.

#### Phase B6. Implement Hard Stage 2 ALM Iota Constraint

Implementation:

- [x] add `iota` or `iota_penalty` to the Stage 2 ALM constraint-name set → `constraint_names.append("iota_penalty")` at `banana_opt/stage2_objectives.py:287,312`.
- [x] extend `evaluate_stage2_alm_problem(...)` to compute:
  - hard signed value → `iota_signed_value` in the `include_iota_penalty` branch (`stage2_objectives.py:1093-1097`).
  - surrogate value if needed → same branch reuses `iota_signed_value` for the surrogate when smoothing is unnecessary.
  - gradient → `iota_grad` populated in the same branch.
  - feasibility value → `iota_violation` populated in the same branch.
- [x] keep the Stage 2 constraint vocabulary explicit and separate from single-stage names where semantics differ → Stage 2 uses `iota_penalty` while single-stage uses `Jiota` + thresholded ALM form.
- [x] update Stage 2 incumbent selection logic so the promoted artifact satisfies:
  - [x] exact hardware pass
  - [x] `iota` tolerance pass
- [x] preserve a secondary artifact when exact hardware passes but `iota` fails, if that remains scientifically useful → `build_stage2_secondary_artifact_metadata(...)`.

Important schema rule:

- [x] Do not claim Stage 2 needs `alm_state.partial.json` unless Stage 2 is explicitly given its own partial-state artifact path. → Stage 2 does not write `alm_state.partial.json`; no cross-stage alias was introduced.
- [x] If Stage 2 partial ALM persistence is desired, introduce a Stage 2-specific schema intentionally. → Not pursued; secondary artifacts reuse the standard Stage 2 `biot_savart_opt.json` + `results.json` pair, tagged via `STAGE2_SECONDARY_ARTIFACT_*`.

Actual file touches:

- [x] `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
- [x] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- [x] Stage 2 artifact-contract helper modules — `banana_opt/hardware_constraint_schema.py` companion helper.
- [x] `examples/single_stage_optimization/README.md`

Acceptance criteria:

- [x] Stage 2 ALM reports `iota` feasibility explicitly → `IOTA_FEASIBLE` in results payload.
- [x] best preserved rescue artifact is both hardware-exact and `iota`-feasible when such a point exists → incumbent selection enforces both; otherwise the secondary hardware-exact-but-iota-fail artifact is preserved.
- [x] feature-off behavior remains unchanged → `--stage2-iota-mode=off` bypasses `include_iota_penalty` and the probe entirely.

#### Phase B7. Optional Standalone Donor-Repair Entrypoint

If the unified runner proves useful, a standalone donor-repair entrypoint can still be added
later for batch workflows. It should reuse the same bridge helper and status schema rather than
introducing a separate repair stack.

Implemented artifact:

- `examples/single_stage_optimization/run_single_stage_donor_repair.py` (landed).

Responsibilities:

- [x] load a Stage 2 donor
- [x] run a cheap init-only or short-budget Boozer and `iota` acquisition pass
- [~] optionally scan a small list of `iota` targets — single-target path is wired; batch-over-targets list is not yet exposed as a dedicated flag (callable externally via the sweep wrapper).
- [x] emit one repaired donor or a ranked list of bootable donors

Success criteria:

- [x] the repaired donor can start the real single-stage workflow reliably → unified-runner full-mode test exercises this path.
- [x] the standalone entrypoint reuses the unified-runner bridge helpers and payload schema → same `banana_opt/stage2_single_stage_handoff.py` + `build_bootability_recovery_payload_fields(...)`.

## File-Level Todo List

### Must-Read Existing Files

- [ ] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- [ ] `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
- [ ] `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- [ ] `examples/single_stage_optimization/banana_opt/single_stage_objectives.py`
- [ ] `src/simsopt/geo/surfaceobjectives.py`
- [ ] `src/simsopt/geo/boozersurface.py`
- [ ] `examples/single_stage_optimization/run_finite_current_smoke.py`
- [ ] `examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py`

### Likely New Or Modified Files

- [ ] `examples/single_stage_optimization/run_single_stage_iota_target_sweep.py`
- [ ] `examples/single_stage_optimization/run_banana_current_scan.py`
- [ ] `examples/single_stage_optimization/plot_ishw_tradeoffs.py`
- [ ] `examples/single_stage_optimization/banana_opt/<shared_boozer_helper>.py`
- [ ] `examples/single_stage_optimization/README.md`
- [ ] one or more Stage 2 result-contract helper files under `examples/single_stage_optimization/banana_opt/`

## Validation Plan

### Test And Smoke Requirements

- [x] Add at least one small Stage 2 `iota` smoke test or fixture-driven regression test. → `tests/geo/test_stage2_single_stage_handoff.py` (11 tests covering probe, classification, handoff, recovery-only, full-mode).
- [x] Add at least one runner test for the new sweep wrappers that validates command construction and summary schema. → covered by `tests/geo/test_ishw_deliverables.py` (`run_single_stage_iota_target_sweep.py`, `run_banana_current_scan.py`, and `plot_ishw_tradeoffs.py`) plus `tests/geo/test_stage2_track_b_wrappers.py` (`run_stage2_iota_decision_gate.py`).
- [x] Validate that feature-disabled Stage 2 output remains contract-compatible with existing readers. → `--stage2-iota-mode=off` preserves the legacy Stage 2 schema (no new keys emitted); legacy artifact upgrade is covered by `test_upgrade_legacy_stage2_artifact_results_backfills_handoff_defaults`.
- [x] Validate that Poincare runners can be invoked non-interactively from the new scan scripts. → `run_banana_current_scan.py` uses `POINCARE_OUT_DIR` for non-interactive invocation.
- [x] Validate that sweep scripts continue past per-case failures and preserve partial results. → classification into `success`/`Boozer-failed`/`Poincare-only fallback` with per-case summaries.

### Canonical Scientific Validation

Status: **pending execution**. Wrappers and payload schema are ready; these
items require empirical run data across the canonical cases and are the main
open verdict the repo is still waiting on.

- [ ] Re-run at least one canonical case on `001490`.
- [ ] Re-run at least one canonical case on `014417`.
- [ ] Compare:
  - initial donor `iota`
  - final donor `iota`
  - hardware pass status
  - single-stage Boozer initialization success
  - total runtime
- [ ] Record whether the Stage 2 root fix actually improves the real downstream single-stage start rate.

## Acceptance Checklist

### Immediate Talk Deliverables

- [ ] A reproducible `iota`-target sweep summary exists.
- [ ] A reproducible banana-current scan summary exists.
- [ ] A current-reference Poincare artifact exists.
- [ ] At least one cleaned slide-ready plot exists for each requested family.
- [ ] All talk figures can be regenerated from saved commands and summaries.

### Root Fix Deliverables

- [x] Stage 2 can report `iota` and Boozer bootability when enabled → `--stage2-iota-mode=report`.
- [x] Stage 2 can influence donor selection using `iota` when enabled → `soft` and `alm` modes.
- [x] Exact hardware-pass salvage behavior is preserved → salvage path untouched; secondary-artifact keys capture hardware-pass/`iota`-fail donors.
- [x] The repo explicitly distinguishes hardware feasibility from `iota` feasibility → `HARDWARE_CONSTRAINTS_OK` remains independent of `IOTA_FEASIBLE` / `BOOZER_BOOTABLE`.
- [ ] The runtime overhead is measured and documented → timing fields emitted (`STAGE2_IOTA_PROBE_SECONDS` etc.); aggregate benchmark outstanding.
- [ ] The team has a decision record choosing between:
  - soft Stage 2 `iota`
  - hard Stage 2 ALM `iota`
  - donor repair fallback
  → **pending**; all three paths exist, but no decision record has been written.

## Risks And Open Questions

- [ ] Boozer solve cost inside Stage 2 may make fresh scans impractical.
- [ ] `iota` can be numerically unstable near rational surfaces.
- [ ] The best reference surface for Stage 2 `iota` targeting is not yet frozen.
- [ ] Low-current banana-current scan points may fail Boozer initialization even when Poincare still provides useful information.
- [ ] Existing JD or Rithik scan data may use different assumptions or metric definitions than the desired updated plots.
- [ ] If the root problem is mostly donor repair rather than Stage 2 optimization, a deep Stage 2 change may be unnecessary.

## Recommended Execution Order

- [x] First, implement Track A Phase A0 through A4 for the talk deliverables → Track A wrappers landed (A0/A4 are human acceptance steps still open).
- [x] Second, implement Track B Phase B0 through B2 so Stage 2 can report `iota` before optimizing it → contract frozen, probe seam shipped, unified runner shipped.
- [x] Third, prototype Track B Phase B3 soft `iota` → soft mode wired behind `--stage2-iota-mode=soft`; benchmark verdict pending.
- [ ] Fourth, decide whether to continue to Phase B5 or stop at the donor-repair fallback in Phase B6 → **pending measurement via `run_stage2_iota_decision_gate.py`**.

## Definition Of Done

This work is done only when both conditions are satisfied:

1. The talk deliverables are reproducible from scripts and artifact summaries.
2. The team has a measured, evidence-based answer to the root-fix question:
   - Stage 2 should own `iota`
   - or single-stage donor repair should own it

Until then, the repository may have useful new plots, but it does not yet have a resolved architecture decision.
