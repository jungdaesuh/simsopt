# Stage 2 Iota Root Fix And ISHW Deliverables Plan

Date: 2026-04-16
Status: Proposed implementation plan. No code landed yet from this document.
Scope: `examples/single_stage_optimization/` Stage 2 donor contract, single-stage runner workflows, scan/plot generation, and Poincare deliverables for the ISHW talk.

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

- [ ] Stage 2 donor success must align with what single-stage actually needs to consume.
- [ ] Stage 2 must distinguish hardware feasibility from Boozer and `iota` bootability.
- [ ] A Stage 2 donor that is marked as the best rescue or incumbent must remain exactly hardware-valid under the current Stage 2 contract.
- [ ] The new Stage 2 path must expose a clear `iota` target and a clear `iota` tolerance.
- [ ] The Stage 2 `iota` path should reuse single-stage Boozer and `iota` machinery where practical instead of reimplementing the math.
- [ ] The new behavior must be feature-flagged so current Stage 2 semantics remain unchanged when the feature is disabled.
- [ ] Runtime cost must be measured explicitly because every added Boozer solve can significantly raise per-iteration cost.
- [ ] Results artifacts must report whether a candidate is:
  - hardware-feasible
  - Boozer-bootable
  - `iota`-feasible
- [ ] Stage 2 final artifact preservation must not regress the recently landed exact-hardware-pass salvage behavior.
- [ ] The plan must avoid pretending Stage 2 already has the same partial-state or ALM artifact schema as single-stage.

### B. Near-Term ISHW Deliverable Requirements

- [ ] Produce updated tradeoff data for increasing `iota` target versus engineering and physics metrics.
- [ ] Include at minimum:
  - coil length
  - curvature
  - field error or QS proxy
  - banana current when relevant
- [ ] Produce a cleaned plot for field error versus coil length, ideally using existing JD scan outputs if they already cover the needed range.
- [ ] Produce a Poincare plot for the current reference configuration.
- [ ] Produce a banana-coil current scan from zero to the optimized current value.
- [ ] For each banana-current scan point, attempt:
  - Boozer initialization
  - single-stage init-only metrics
  - Poincare diagnostics
- [ ] If Boozer initialization fails at some current values, still emit a usable fallback artifact set rather than failing the entire sweep.
- [ ] The output format must be presentation-friendly, not just raw JSON dumps.

### C. Non-Goals

- [ ] Do not block talk plots on a full Stage 2 solver rewrite.
- [ ] Do not claim self-consistent finite-current equilibrium physics from the existing `--plasma-current-A` surrogate workflows.
- [ ] Do not move Stage 2 into JAX or rewrite the whole optimization graph as part of the first pass.
- [ ] Do not overload single-stage-only artifact names like `alm_state.partial.json` into Stage 2 without intentionally introducing a Stage 2 schema.
- [ ] Do not silently redefine Stage 2 "hardware pass" to include physics constraints.
- [ ] Do not implement a generic framework for arbitrary extra Stage 2 physics constraints before the `iota` use case is proven.

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

- [ ] Split the work into two tracks:
  - Track A: talk deliverables
  - Track B: Stage 2 root fix
- [ ] Use a phased Stage 2 plan:
  - Phase 0: contract freeze
  - Phase 1: shared Boozer and `iota` probe seam
  - Phase 2: unified runner with bootability probe and donor-repair path
  - Phase 3: soft Stage 2 `iota` prototype if still justified
  - Phase 4: hard Stage 2 ALM `iota` constraint if still justified
- [ ] Treat the unified-runner / donor-repair path in
  `docs/stage2_single_stage_unified_runner_plan_2026-04-16.md`
  as the first implementation slice of Track B, not as a competing plan.
- [ ] Keep Stage 2-native `iota` work behind a later decision gate, informed by bridge
  runtime and success-rate measurements.
- [ ] Keep hardware feasibility and `iota` bootability as separate reported statuses even if both eventually gate incumbent selection.

## Proper Implementation Plan

### Track A. ISHW Talk Deliverables

#### Phase A0. Freeze Scope And Inputs

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

Recommended new artifact:

- `examples/single_stage_optimization/run_single_stage_iota_target_sweep.py`

Responsibilities:

- [ ] reuse one explicit Stage 2 artifact across all sweep points
- [ ] sweep `--iota-target` values over a user-provided list
- [ ] keep other single-stage knobs fixed
- [ ] collect a compact summary JSON and a slide-friendly CSV
- [ ] emit per-run paths so plots can be regenerated later

Minimum implementation details:

- [ ] accept `--stage2-bs-path`
- [ ] accept `--plasma-surf-filename`
- [ ] accept `--iota-targets`
- [ ] accept `--constraint-method`
- [ ] forward the same geometry and Boozer discretization flags currently used by the single-stage wrappers
- [ ] record failure modes without aborting the entire sweep

Expected file touches:

- [ ] `examples/single_stage_optimization/run_single_stage_iota_target_sweep.py`
- [ ] `examples/single_stage_optimization/workflow_runner_common.py` if small shared helpers are needed
- [ ] `examples/single_stage_optimization/README.md`

#### Phase A2. Build The Banana-Current Scan Runner

Recommended new artifact:

- `examples/single_stage_optimization/run_banana_current_scan.py`

Reason this should be separate from `run_finite_current_smoke.py`:

- `run_finite_current_smoke.py` varies plasma current (via `--currents-A`, forwarded as `--plasma-current-A`), not banana-coil current.
- Carlos explicitly asked for banana-coil current variation with TF current fixed.
- The output and fallback semantics are different (banana-current sweeps must tolerate Boozer-init failure at low currents and still emit Poincare artifacts).

Responsibilities:

- [ ] load one optimized coil set
- [ ] scale the banana current from zero to the optimized value
- [ ] for reused optimized Stage 2 donors, mutate the loaded banana `Current` DOF after deserialization or add one explicit override seam; do not assume `--banana-init-current-A` can drive this path when `--stage2-bs-path` is present
- [ ] use `--banana-init-current-A` and `--banana-current-max-A` only for fresh-artifact generation paths
- [ ] attempt single-stage init-only Boozer surface solve at each scale
- [ ] run Poincare diagnostics for each case using `POINCARE_OUT_DIR`; add a CLI or helper seam to `poincare_surfaces.py` only if the scan needs configurable tracing parameters beyond the current defaults
- [ ] collect metrics per current value
- [ ] preserve partial outputs when some current values fail

Required behavior:

- [ ] do not stop the whole sweep because one low-current case produces a self-intersecting or invalid Boozer surface
- [ ] classify each current point as:
  - success
  - Boozer-failed
  - Poincare-only fallback
- [ ] emit one summary table suitable for slide plotting

Expected file touches:

- [ ] `examples/single_stage_optimization/run_banana_current_scan.py`
- [ ] `examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py` only if the scan needs a runner-friendly CLI or helper seam beyond `POINCARE_OUT_DIR`
- [ ] `examples/single_stage_optimization/README.md`

#### Phase A3. Plotting And Slide Artifacts

Recommended new artifact:

- `examples/single_stage_optimization/plot_ishw_tradeoffs.py`

Responsibilities:

- [ ] read the `iota` sweep summary
- [ ] read the banana-current scan summary
- [ ] generate presentation-ready PNG or PDF plots
- [ ] support one cleaned field-error versus coil-length plot
- [ ] support one current-configuration Poincare export step

Plot set:

- [ ] `iota_target` versus coil length
- [ ] `iota_target` versus max curvature
- [ ] `iota_target` versus QS error or field error
- [ ] banana current scale versus QS error
- [ ] banana current scale versus `iota`
- [ ] banana current scale versus Boozer success status
- [ ] cleaned field-error versus coil-length figure

Expected file touches:

- [ ] `examples/single_stage_optimization/plot_ishw_tradeoffs.py`
- [ ] `examples/single_stage_optimization/README.md`

#### Phase A4. Validate And Package Talk Outputs

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

This is the most important phase. Do not start plumbing until these are written down.

- [ ] Define the user-visible Stage 2 contract in one sentence.
- [ ] Decide whether Stage 2 will certify:
  - hardware only
  - hardware plus `iota` bootability
  - or hardware plus `iota` bootability only when the feature flag is enabled
- [ ] Define the status vocabulary. The existing schema in `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py` already emits `HARDWARE_CONSTRAINTS_OK` and `HARDWARE_CONSTRAINT_VIOLATIONS` — new fields must be layered into `build_hardware_constraint_artifact_payload_fields(...)` (or a companion payload helper) rather than introduced as parallel lower-case keys:
  - reuse: `HARDWARE_CONSTRAINTS_OK` (already means: hardware-feasible under current Stage 2 contract)
  - add: `BOOZER_BOOTABLE` (Boozer surface initialization succeeded on the reference surface)
  - add: `IOTA_FEASIBLE` (measured reference-surface `iota` sits within `±tol` of the target and outside any rational-neighborhood blocklist)
  - add: `STAGE2_ROOT_FIX_ENABLED` (feature-flag marker so stale consumers can detect the extended schema)
- [ ] Reuse the same bootability-status payload helper for:
  - unified runner probe and repair artifacts
  - Stage 2 reporting-only mode
  - any later Stage 2-native soft or hard `iota` path
- [ ] Decide whether incumbent selection will require both exact hardware pass and `iota` feasibility.
- [ ] Decide whether the reported `iota` target applies on:
  - the plasma boundary
  - an outer Boozer surface
  - or a dedicated reference surface ratio
- [ ] Decide whether the Stage 2 feature should target:
  - raw `iota`
  - `Jiota`
  - or a thresholded `iota_penalty`
- [ ] Decide on initial acceptance tolerances.

Open questions that must be resolved here:

- [ ] Which reference surface is physically meaningful and numerically stable for Stage 2?
- [ ] Should rational-surface neighborhoods be explicitly avoided in the default target list?
- [ ] Is a donor that is hardware-exact but `iota`-bad still worth preserving as a secondary artifact?

#### Phase B1. Factor A Shared Boozer And Iota Probe Seam

Goal:

- reuse the single-stage Boozer and `Iotas` logic without copy-pasting large blocks into Stage 2

Recommended implementation:

- [ ] extract the minimum shared helper seam from single-stage Boozer initialization into a shared `banana_opt/` helper module
- [ ] define one small helper that can:
  - build a reference Boozer surface
  - evaluate `Iotas`
  - return both scalar value and gradient-ready object handles
- [ ] keep the helper solver-owned, not runner-owned
- [ ] keep Stage 2-specific policy outside the helper

Expected file touches:

- [ ] `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- [ ] one new or refactored `examples/single_stage_optimization/banana_opt/` helper module
- [ ] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`

Acceptance criteria:

- [ ] single-stage behavior is unchanged after the helper extraction
- [ ] the helper can be invoked from a Stage 2-only smoke path
- [ ] the helper can also be invoked from the unified runner's probe-only mode

#### Phase B2. Implement The Unified Runner And Reporting-Only Probe

This phase is the concrete execution plan captured in
`docs/stage2_single_stage_unified_runner_plan_2026-04-16.md`.

Goal:

- make the producer-consumer mismatch observable before changing the Stage 2 hot loop
- avoid building a second, separate donor-repair stack later

Implementation:

- [ ] add the unified runner wrapper
- [ ] implement probe-only mode first
- [ ] reuse the shared Boozer and `iota` helper from Phase B1
- [ ] report bootability and `iota` statuses through the shared payload helper
- [ ] preserve existing standalone Stage 2 and standalone single-stage entrypoints

Acceptance criteria:

- [ ] one command can load or generate a Stage 2 donor and classify its bootability
- [ ] results use the same SSOT status vocabulary planned for later Stage 2-native work
- [ ] no second probe or repair schema is introduced outside the shared helper path

#### Phase B3. Add Reporting Before Optimization Gating

Goal:

- make Stage 2 able to compute and report `iota` before making it part of the optimization objective

Implementation:

- [ ] add optional Stage 2 CLI flags for the `iota` probe
- [ ] compute reference-surface `iota` for the initial donor
- [ ] report the value in `results.json`
- [ ] report Boozer success or failure explicitly
- [ ] record elapsed time for the probe

Expected file touches:

- [ ] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- [ ] Stage 2 results or artifact helper modules under `banana_opt/`
- [ ] `examples/single_stage_optimization/README.md`

Acceptance criteria:

- [ ] Stage 2 can run in reporting-only mode with `iota` probe enabled
- [ ] results clearly show whether the donor is hardware-valid but `iota`-bad

#### Phase B4. Implement The Soft Stage 2 Iota Prototype

Goal:

- test whether Stage 2 can be nudged toward single-stage-usable donors without full ALM constraint plumbing

Implementation:

- [ ] add Stage 2 config for:
  - `--stage2-iota-mode {off,report,soft,alm}`
  - `--stage2-iota-target`
  - `--stage2-iota-weight`
  - `--stage2-iota-tol`
- [ ] create an optional `Jiota` term for Stage 2
- [ ] wire the scalar term into the Stage 2 penalty objective path
- [ ] preserve current behavior when `--stage2-iota-mode=off`
- [ ] record final and initial `iota`
- [ ] record whether the best exact-hardware artifact also meets the `iota` tolerance

Expected file touches:

- [ ] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- [ ] `examples/single_stage_optimization/banana_opt/stage2_objectives.py` if the shared objective path needs extension
- [ ] Stage 2 artifact summary helpers

Runtime and robustness checks:

- [ ] benchmark one canonical Stage 2 run with and without the soft `iota` term
- [ ] measure objective evaluation slowdown
- [ ] log Boozer failures separately from ordinary constraint failures
- [ ] check for unstable sign flips or noisy gradients near rational targets

Acceptance criteria:

- [ ] at least one canonical case improves donor `iota` materially without losing exact hardware feasibility
- [ ] slowdown is measured and documented
- [ ] feature-off behavior matches current Stage 2 semantics

#### Phase B5. Decide Whether Hard Stage 2 ALM Iota Is Worth It

This phase is a decision gate, not an automatic implementation step.

- [ ] If the soft prototype clearly improves donor quality and runtime is acceptable, proceed to the hard ALM design.
- [ ] If the soft prototype is unstable or too expensive, stop here and keep the unified-runner / donor-repair path as the practical root fix.

Decision inputs:

- [ ] runtime multiplier
- [ ] donor bootability improvement
- [ ] implementation complexity still remaining
- [ ] whether the unified-runner probe and repair path already solves the practical workflow problem

#### Phase B6. Implement Hard Stage 2 ALM Iota Constraint

Implementation:

- [ ] add `iota` or `iota_penalty` to the Stage 2 ALM constraint-name set
- [ ] extend `evaluate_stage2_alm_problem(...)` to compute:
  - hard signed value
  - surrogate value if needed
  - gradient
  - feasibility value
- [ ] keep the Stage 2 constraint vocabulary explicit and separate from single-stage names where semantics differ
- [ ] update Stage 2 incumbent selection logic so the promoted artifact satisfies:
  - exact hardware pass
  - `iota` tolerance pass
- [ ] preserve a secondary artifact when exact hardware passes but `iota` fails, if that remains scientifically useful

Important schema rule:

- [ ] Do not claim Stage 2 needs `alm_state.partial.json` unless Stage 2 is explicitly given its own partial-state artifact path.
- [ ] If Stage 2 partial ALM persistence is desired, introduce a Stage 2-specific schema intentionally.

Expected file touches:

- [ ] `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
- [ ] `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
- [ ] Stage 2 artifact-contract helper modules
- [ ] `examples/single_stage_optimization/README.md`

Acceptance criteria:

- [ ] Stage 2 ALM reports `iota` feasibility explicitly
- [ ] best preserved rescue artifact is both hardware-exact and `iota`-feasible when such a point exists
- [ ] feature-off behavior remains unchanged

#### Phase B7. Optional Standalone Donor-Repair Entrypoint

If the unified runner proves useful, a standalone donor-repair entrypoint can still be added
later for batch workflows. It should reuse the same bridge helper and status schema rather than
introducing a separate repair stack.

Recommended artifact:

- `examples/single_stage_optimization/run_single_stage_donor_repair.py`

Responsibilities:

- [ ] load a Stage 2 donor
- [ ] run a cheap init-only or short-budget Boozer and `iota` acquisition pass
- [ ] optionally scan a small list of `iota` targets
- [ ] emit one repaired donor or a ranked list of bootable donors

Success criteria:

- [ ] the repaired donor can start the real single-stage workflow reliably
- [ ] the standalone entrypoint reuses the unified-runner bridge helpers and payload schema

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

- [ ] Add at least one small Stage 2 `iota` smoke test or fixture-driven regression test.
- [ ] Add at least one runner test for the new sweep wrappers that validates command construction and summary schema.
- [ ] Validate that feature-disabled Stage 2 output remains contract-compatible with existing readers.
- [ ] Validate that Poincare runners can be invoked non-interactively from the new scan scripts.
- [ ] Validate that sweep scripts continue past per-case failures and preserve partial results.

### Canonical Scientific Validation

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

- [ ] Stage 2 can report `iota` and Boozer bootability when enabled.
- [ ] Stage 2 can influence donor selection using `iota` when enabled.
- [ ] Exact hardware-pass salvage behavior is preserved.
- [ ] The repo explicitly distinguishes hardware feasibility from `iota` feasibility.
- [ ] The runtime overhead is measured and documented.
- [ ] The team has a decision record choosing between:
  - soft Stage 2 `iota`
  - hard Stage 2 ALM `iota`
  - donor repair fallback

## Risks And Open Questions

- [ ] Boozer solve cost inside Stage 2 may make fresh scans impractical.
- [ ] `iota` can be numerically unstable near rational surfaces.
- [ ] The best reference surface for Stage 2 `iota` targeting is not yet frozen.
- [ ] Low-current banana-current scan points may fail Boozer initialization even when Poincare still provides useful information.
- [ ] Existing JD or Rithik scan data may use different assumptions or metric definitions than the desired updated plots.
- [ ] If the root problem is mostly donor repair rather than Stage 2 optimization, a deep Stage 2 change may be unnecessary.

## Recommended Execution Order

- [ ] First, implement Track A Phase A0 through A4 for the talk deliverables.
- [ ] Second, implement Track B Phase B0 through B2 so Stage 2 can report `iota` before optimizing it.
- [ ] Third, prototype Track B Phase B3 soft `iota`.
- [ ] Fourth, decide whether to continue to Phase B5 or stop at the donor-repair fallback in Phase B6.

## Definition Of Done

This work is done only when both conditions are satisfied:

1. The talk deliverables are reproducible from scripts and artifact summaries.
2. The team has a measured, evidence-based answer to the root-fix question:
   - Stage 2 should own `iota`
   - or single-stage donor repair should own it

Until then, the repository may have useful new plots, but it does not yet have a resolved architecture decision.
