# Single-Stage Modularization TDD Plan

Date: 2026-04-27

Scope: simplify the single-stage banana coil optimization code without changing
objective physics, Stage 2 handoff semantics, solver behavior, artifact
contracts, or CLI defaults.

## Context

- Current HEAD includes `b525b8de2 fix: split Stage 2 plasma geometry`.
- That commit already moved Stage 2 plasma geometry behavior into
  `examples/single_stage_optimization/banana_opt/stage2_geometry.py`.
- The single-stage refactor plan should not re-extract that Stage 2 geometry
  work.
- The commit is still important because it changes the Stage 2 geometry
  contract that single-stage seeds depend on:
  - LCFS is scaled to `target_lcfs_major_radius_m`.
  - Working surface is scaled by the same LCFS-derived scale factor.
  - Geometry preflight may select a smaller LCFS target or working surface
    candidate before Stage 2 runs.
  - Stage 2 `SquaredFlux` remains on the working surface.
  - Stage 2 coil-to-plasma clearance remains against LCFS.

Current code shape:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  is still the oversized single-stage driver.
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py`
  contains the SSOT function for pure single-stage objective formula assembly:
  `build_total_objective`. The file also contains ALM/frontier evaluation
  helpers, so do not treat the whole file as pure.
- `examples/single_stage_optimization/banana_opt/stage2_geometry.py`
  is the SSOT for Stage 2 plasma geometry after `b525b8de2`.
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  remains the Stage 2 runtime entrypoint.

Current single-stage hot spots:

- `single_stage_banana_example.py:938` - `parse_args`.
- `single_stage_banana_example.py:3674` - `build_single_stage_objective_bundle`.
- `single_stage_banana_example.py:6654` - compatibility wrapper for
  `build_total_objective`.
- `single_stage_banana_example.py:7026` - `evaluate_search_step`.
- `single_stage_banana_example.py:7951` - giant `if __name__ == "__main__"` body.
- `single_stage_objectives.py:55` - pure objective formula function.

Cross-module consumers:

- `examples/single_stage_optimization/banana_opt/frontier_evaluator.py` imports
  `SINGLE_STAGE.single_stage_banana_example as single_stage`.
- That production module uses entrypoint constants and wrappers including:
  `apply_default_stage2_seed_args`, `validate_stage2_seed_contract`,
  `resolve_surface_mode_contract`, `build_surface_configs`,
  `initialize_boozer_surface`, `build_single_stage_objective_bundle`,
  `build_run_identity_config`, and `safe_evaluate_topology_gate`.
- It also uses private constraint helper re-exports such as
  `_smooth_min_curve_curve_signed_constraint`,
  `_smooth_min_curve_surface_signed_constraint`,
  `_smooth_max_curvature_signed_constraint`, and hard-signal variants.
- Rule: any phase that moves a symbol consumed by `frontier_evaluator.py` must
  update `frontier_evaluator.py` in the same slice. Do not remove entrypoint
  re-exports until this production consumer imports the real `banana_opt`
  modules directly.

Test consumers:

- `tests/geo/test_single_stage_example.py` calls
  `load_single_stage_example_module()` 122 times.
- `tests/geo/test_surface_mode_contracts.py` calls
  `load_single_stage_example_module()` 7 times.
- Phase 5 wrapper deletion is therefore a large mechanical migration, not a
  routine cleanup.

## Non-Negotiable Invariants

- [ ] Do not change the pure objective formula while modularizing.
- [ ] Keep `build_total_objective` in `single_stage_objectives.py` as the SSOT
      for pure objective math.
- [ ] Keep Stage 2 geometry behavior from `b525b8de2` locked by tests.
- [ ] Keep single-stage CLI defaults unchanged.
- [ ] Keep Stage 2 artifact and single-stage result metadata keys unchanged.
- [ ] Keep L-BFGS-B, ALM, basin-hopping, Boozer solve, and checkpoint behavior
      unchanged.
- [ ] Keep imports static. Do not add dynamic imports.
- [ ] Do not add defensive fallback behavior.
- [ ] Move first, simplify only after parity is proven.
- [ ] Do not change public signatures covered by
      `tests/geo/test_banana_modularization_parity.py::SnapshotParityTests`
      unless the snapshot commit is intentionally bumped in the same change.

## Target Final Shape

This shape is additive. Existing `banana_opt` modules not listed here keep
their current roles unless a phase explicitly names them.

```text
examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py
    Thin CLI entrypoint only.

examples/single_stage_optimization/banana_opt/single_stage_cli.py
    CLI arguments, parser defaults, argument validation glue.

examples/single_stage_optimization/banana_opt/single_stage_runner.py
    Top-level single-stage workflow orchestration.

examples/single_stage_optimization/banana_opt/single_stage_objective_bundle.py
    Builds objective terms and bundles from surfaces/coils/config.

examples/single_stage_optimization/banana_opt/single_stage_objectives.py
    Existing module. build_total_objective remains the pure objective formula
    SSOT; ALM/frontier evaluation helpers may remain here until Phase 6.

examples/single_stage_optimization/banana_opt/single_stage_search_loop.py
    evaluate_search_step, fun, accept/reject bookkeeping, search metrics.

examples/single_stage_optimization/banana_opt/single_stage_geometry.py
    Existing module. Surface stack, hardware geometry, topology gate geometry.

examples/single_stage_optimization/banana_opt/single_stage_phase1.py
    Existing module. L-BFGS-B phase-1/phase-2 optimization control.

examples/single_stage_optimization/banana_opt/stage2_geometry.py
    Existing module. Stage 2 plasma geometry SSOT. Already split by b525b8de2.
```

Existing modules such as `single_stage_constraints.py`,
`single_stage_search_policy.py`, `single_stage_banana_current_mode.py`,
`incumbents.py`, and the `frontier_*` modules remain part of the final layout.

## Phase 0 - Regression Lock Before Moving Code

Goal: characterize current behavior before any extraction.

### Objective Formula Locks

- [ ] Add or confirm tests that `build_total_objective(...)` assembles:
  - [ ] `JnonQSRatio`
  - [ ] `RES_WEIGHT * JBoozerResidual`
  - [ ] `IOTAS_WEIGHT * Jiota`
  - [ ] `LENGTH_WEIGHT * JCurveLength`
  - [ ] `CC_WEIGHT * JCurveCurve`
  - [ ] `CS_WEIGHT * JCurveSurface`
  - [ ] `CURVATURE_WEIGHT * JCurvature`
  - [ ] optional `VOLUME_WEIGHT * JVolume`
  - [ ] optional `SURF_DIST_WEIGHT * JSurfSurf`
  - [ ] optional `POLOIDAL_EXTENT_WEIGHT * JPoloidalExtent`
- [ ] Test target mode with `JVolume is None`.
- [ ] Test frontier mode with `JVolume is not None`.
- [ ] Test objective formula imports directly from
      `banana_opt.single_stage_objectives`.

### Objective Bundle Locks

- [ ] Add tests for `build_single_stage_objective_bundle(...)` current behavior.
- [ ] Lock returned bundle keys:
  - [ ] `JF`
  - [ ] `JVolume`
  - [ ] `JnonQSRatio`
  - [ ] `JnonQSRatioObjective`
  - [ ] `JBoozerResidual`
  - [ ] `JBoozerResidualObjective`
  - [ ] `Jiota`
  - [ ] `JCurveLength`
  - [ ] `JCurveCurve`
  - [ ] `JCurveSurface`
  - [ ] `JSurfSurf`
  - [ ] `JCurvature`
  - [ ] `JPoloidalExtent`
- [ ] Lock target-mode bundle behavior.
- [ ] Lock frontier-mode bundle behavior.
- [ ] Lock Boozer residual class selection by stage.

### Search Step Locks

- [ ] Add tests for `evaluate_search_step(...)` behavior using fakes.
- [ ] Inventory every entrypoint global read or mutated by `evaluate_search_step`
      and helpers it calls before moving it.
- [ ] Lock fallback behavior for `globals().get(...)` sites used in the search
      path, especially:
  - [ ] `surface_mode_contract`
  - [ ] `TOPOLOGY_GATE_PENALTY_SCALE` defaulting to `4.0`
  - [ ] `HARDWARE_SEARCH_PENALTY_SCALE` defaulting to `4.0`
  - [ ] `CONSTRAINT_METHOD` defaulting to `"penalty"` where currently used
  - [ ] `banana_current_state`
- [ ] Successful surface solve:
  - [ ] calls surface-stack solve once.
  - [ ] evaluates objective once.
  - [ ] records accepted metrics.
  - [ ] returns same `total` and `grad` as current code.
- [ ] Boozer/surface solve failure:
  - [ ] increments surface-solve rejection counters.
  - [ ] restores accepted `JF.x`.
  - [ ] restores surface state.
  - [ ] returns elevated objective.
  - [ ] returns previous accepted gradient.
- [ ] Hardware rejection:
  - [ ] increments hardware rejection counters.
  - [ ] preserves repair-first exception behavior.
  - [ ] preserves frontier penalty behavior.
- [ ] Topology rejection:
  - [ ] preserves broken topology invalidation.
  - [ ] preserves non-frontier hard rejection.
  - [ ] preserves frontier penalty behavior.
- [ ] ALM path:
  - [ ] returns `constraint_values`.
  - [ ] returns `constraint_grads`.
  - [ ] returns `max_violation`.
  - [ ] returns `stationarity_norm`.
  - [ ] returns activity tolerances and feasibility values.

### CLI and Runner Locks

- [ ] Snapshot parser defaults for hardware and objective flags:
  - [ ] `--constraint-weight`
  - [ ] `--constraint-method`
  - [ ] `--vol-target`
  - [ ] `--length-target`
  - [ ] `--cc-dist`
  - [ ] `--cs-dist`
  - [ ] `--curvature-threshold`
  - [ ] `--single-stage-goal-mode`
  - [ ] ALM flags
  - [ ] basin-hopping flags
  - [ ] banana-current flags
  - [ ] surface-mode flags
- [ ] Lock run identity fields affected by CLI defaults.
- [ ] Lock output artifact names and required metadata keys.
- [ ] Lock path defaults derived from `EXAMPLE_ROOT`, `SCRIPT_DIR`,
      `SIMSOPT_ROOT`, and `REPO_ROOT`:
  - [ ] `DEFAULT_EQUILIBRIA_DIR`
  - [ ] `DEFAULT_LOCAL_STAGE2_ROOT`
  - [ ] `DEFAULT_DATABASE_STAGE2_ROOT`
  - [ ] `DEFAULT_SINGLE_STAGE_OUTPUT_ROOT`
  These are currently derived from
  `configure_local_simsopt_imports(__file__)` in the entrypoint. CLI extraction
  must preserve those exact roots.

### Stage 2 Geometry Contract Locks From b525b8de2

- [ ] Keep tests proving `load_plasma_geometry(...)` scales LCFS to
      `target_lcfs_major_radius_m`.
- [ ] Keep tests proving working surface uses the same LCFS-derived scale
      factor.
- [ ] Keep tests for `select_plasma_geometry_preflight_candidate(...)`.
- [ ] Keep tests proving preflight prefers requested candidate when it fits.
- [ ] Keep tests proving preflight can select a smaller LCFS target when the
      max target violates vessel clearance.
- [ ] Keep tests proving preflight rejects when no candidate fits.
- [ ] Keep Stage 2 smoke test proving `CurveSurfaceDistance` uses LCFS.
- [ ] Keep Stage 2 smoke test proving proxy plasma-current coil receives
      `plasma_geometry.scale_factor`.
- [ ] Keep Stage 2 objective behavior:
  - [ ] `SquaredFlux(new_surf, new_bs)` on working surface.
  - [ ] `CurveSurfaceDistance(objective_curves, lcfs_surf, CS_THRESHOLD)` on
        LCFS.

### Cross-Module Consumer Locks

- [ ] Add a focused test or static check that `frontier_evaluator.py` no longer
      relies on entrypoint wrappers for any symbol moved in a phase.
- [ ] Before deleting private re-exports from `single_stage_banana_example.py`,
      update `frontier_evaluator.py` to import these helpers directly from
      `banana_opt.single_stage_constraints`:
  - [ ] `_smooth_min_curve_curve_signed_constraint`
  - [ ] `_smooth_min_curve_surface_signed_constraint`
  - [ ] `_smooth_max_curvature_signed_constraint`
  - [ ] `_smooth_min_curve_curve_signed_constraint_with_hard_signal`
  - [ ] `_smooth_min_curve_surface_signed_constraint_with_hard_signal`
  - [ ] `_smooth_min_surface_surface_signed_constraint`
  - [ ] `_smooth_min_surface_surface_signed_constraint_with_hard_signal`
  - [ ] `_smooth_min_surface_stack_signed_constraint`
  - [ ] `_smooth_min_surface_stack_signed_constraint_with_hard_signal`
- [ ] Keep `frontier_evaluator.py` import migration in the same commit as the
      moved symbol it consumes.

### Snapshot Parity Locks

- [ ] Preserve the signatures and semantics of functions loaded by
      `SnapshotParityTests`:
  - [ ] smoothing helpers compared to `SINGLE_STAGE_SNAPSHOT`
  - [ ] `evaluate_single_stage_hardware_constraints`
  - [ ] `build_surface_configs`
  - [ ] `compute_single_stage_surface_vessel_min_dist`
  - [ ] `topology_gate_deficit`
- [ ] If a public signature must change, update the snapshot commit constants
      in `test_banana_modularization_parity.py` in the same reviewed slice.

## Phase 1 - Extract Single-Stage Objective Bundle

Goal: move objective term construction out of the giant entrypoint.

Create:

```text
examples/single_stage_optimization/banana_opt/single_stage_objective_bundle.py
```

Move candidates:

- [ ] `build_single_stage_iota_objective`
- [ ] `build_single_stage_volume_objective`
- [ ] `build_boozer_derived_objective_terms`
- [ ] `boozer_residual_class_for_stage`
- [ ] `build_single_stage_objective_bundle`
- [ ] Do not move `apply_single_stage_objective_bundle` in Phase 1.

TDD steps:

- [ ] Write tests against the new module first.
- [ ] Keep the old wrapper in `single_stage_banana_example.py` temporarily.
- [ ] Prove old wrapper and new module return equivalent bundle keys and term
      values.
- [ ] Prove the final `JF` still comes from `single_stage_objectives.py`.
- [ ] Update `frontier_evaluator.py` to call the new objective-bundle module
      directly or keep a temporary entrypoint wrapper explicitly for that
      production consumer.
- [ ] Run parity tests before deleting any wrapper.

Acceptance:

- [ ] `single_stage_banana_example.py` delegates bundle construction to the new
      module.
- [ ] `apply_single_stage_objective_bundle` remains in
      `single_stage_banana_example.py` as the entrypoint-globals binder because
      it assigns the globals consumed by `evaluate_search_step`,
      `refresh_accepted_search_state`, callback/reporting code, and current
      tests.
- [ ] No objective formula is duplicated in the entrypoint.
- [ ] `frontier_evaluator.py` still works after the bundle move.
- [ ] Existing tests pass.

## Phase 2 - Extract Search Loop

Goal: isolate the line-search trial evaluation from the entrypoint.

This is an atomic refactor. The move cannot be behavior-preserving by only
leaving an old wrapper behind, because tests currently mutate globals on
`single_stage_banana_example.py`. Context introduction, entrypoint wiring, and
test fixture migration must ship together.

Create:

```text
examples/single_stage_optimization/banana_opt/single_stage_search_loop.py
```

Move candidates:

- [ ] `evaluate_search_step`
- [ ] `fun`
- [ ] `new_search_step_metrics`
- [ ] `search_step_metrics_for_run`
- [ ] `record_search_step_objective_eval`
- [ ] `record_search_step_acceptance`
- [ ] `record_search_step_rejection`
- [ ] `search_step_rejection_reason`
- [ ] `search_step_metrics_payload`
- [ ] curvature traversal precheck helpers if they are only used by search-loop
      evaluation.

Introduce explicit context:

```python
@dataclass
class SingleStageSearchContext:
    objective: object
    surface_data: list
    run_dict: dict
    vessel_surface: object
    biot_savart: object
    surface_iota_terms: list
    surface_mode_contract: object | None
    banana_curve: object
    banana_current_state: object | None
    thresholds: SingleStageSearchThresholds
    surface_ramp: SingleStageSurfaceRamp
    hardware_policy: HardwareSearchPolicy
    topology_policy: SingleStageTopologyPolicy
    frontier_policy: SingleStageFrontierSearchPolicy
    diagnostics: SingleStageSearchDiagnostics
    callbacks: SingleStageSearchCallbacks
```

The exact type names may change, but the context must cover the current
`evaluate_search_step` reads and writes. Required fields include:

- [ ] `run_dict`
- [ ] `surface_data`
- [ ] `JF` as a live object reference, not a copied snapshot
- [ ] `VV`
- [ ] `bs`
- [ ] `surface_iota_terms`
- [ ] `surface_mode_contract`
- [ ] `banana_curve`
- [ ] `banana_current_state`
- [ ] `SURFACE_GAP_THRESHOLD`
- [ ] `SS_DIST`
- [ ] `CC_DIST`
- [ ] `CS_DIST`
- [ ] `CURVATURE_THRESHOLD`
- [ ] `MULTISURFACE_RAMP_ITERATIONS`
- [ ] `INNER_SURFACE_INITIAL_WEIGHT`
- [ ] `CONSTRAINT_METHOD`
- [ ] `HARDWARE_SEARCH_MODE`
- [ ] `HARDWARE_SEARCH_SOFT_ITERATIONS`
- [ ] `TOPOLOGY_GATE_TMAX`
- [ ] `TOPOLOGY_GATE_TOL`
- [ ] `TOPOLOGY_GATE_SURVIVAL_THRESHOLD`
- [ ] `TOPOLOGY_GATE_PENALTY_SCALE`
- [ ] `HARDWARE_SEARCH_PENALTY_SCALE`

Do not over-abstract before the move is stable, but do not leave hidden reads
to the old entrypoint globals.

TDD steps:

- [ ] Add tests for successful candidate behavior.
- [ ] Add tests for rejected candidate behavior.
- [ ] Add tests for state restoration.
- [ ] Add tests for ALM return payload.
- [ ] Add tests for frontier penalty application.
- [ ] Port every `module.<global> = ...` search-step fixture to construct a
      `SingleStageSearchContext`.
- [ ] Update the entrypoint runtime to construct and pass the same context.
- [ ] Keep a temporary entrypoint wrapper only if it delegates to the new module
      using an explicitly constructed context from entrypoint globals.
- [ ] Prove the wrapper does not read stale defaults after the move.

Acceptance:

- [ ] `evaluate_search_step` no longer lives in
      `single_stage_banana_example.py`.
- [ ] Search-loop state dependencies are explicit in a context object.
- [ ] `context.objective.x = run_dict["accepted_x"]` preserves current `JF.x`
      restore semantics.
- [ ] `run_dict` counter semantics are unchanged.
- [ ] Tests no longer mutate entrypoint globals to configure search-loop
      behavior.

## Phase 3 - Extract Runner

Goal: move the runtime workflow out of the script body.

Create:

```text
examples/single_stage_optimization/banana_opt/single_stage_runner.py
```

Move:

- [ ] The body under `if __name__ == "__main__"`.
- [ ] Runtime setup after argument parsing.
- [ ] Stage 2 seed loading and upgrade orchestration.
- [ ] Surface and coil initialization orchestration.
- [ ] Optimizer mode selection.
- [ ] Final artifact writing and diagnostics orchestration.

Add:

```python
def run_single_stage(args) -> int:
    ...
```

TDD steps:

- [ ] Add tests that call `run_single_stage(args)` directly with fakes.
- [ ] Keep CLI smoke tests passing.
- [ ] Verify metadata emitted by runner matches current entrypoint output.
- [ ] Verify init-only, penalty, ALM, and basin-hopping paths route to the same
      solver calls as before.

Acceptance:

- [ ] `single_stage_banana_example.py` calls `run_single_stage(args)`.
- [ ] No optimizer behavior changes.
- [ ] No artifact schema changes.

## Phase 4 - Extract CLI

Goal: isolate parser and CLI default wiring.

Create:

```text
examples/single_stage_optimization/banana_opt/single_stage_cli.py
```

Move candidates:

- [ ] `parse_args`
- [ ] argument-group builders.
- [ ] CLI-only validation helpers.
- [ ] default application helpers that do not need runtime state.

Path-default ownership:

- [ ] Do not recompute repository/example roots from the new CLI module's
      `__file__`.
- [ ] Keep path roots derived from the original entrypoint location, or pass an
      explicit root config into the CLI builder:

```python
@dataclass(frozen=True)
class SingleStageCliPaths:
    example_root: str
    script_dir: str
    repo_root: str
    database_equilibria_dir: str
    default_equilibria_dir: str
    default_local_stage2_root: str
    default_database_stage2_root: str
    default_single_stage_output_root: str
```

- [ ] `DEFAULT_SINGLE_STAGE_OUTPUT_ROOT` must remain relative to the
      `SINGLE_STAGE` entrypoint directory unless intentionally changed in a
      separate behavior patch.

TDD steps:

- [ ] Snapshot current defaults.
- [ ] Snapshot selected parsed args from representative CLI command strings.
- [ ] Verify environment-variable defaults still work where currently
      supported.
- [ ] Verify run identity fields remain unchanged.
- [ ] Verify all path defaults match pre-extraction values.

Acceptance:

- [ ] `single_stage_banana_example.py` imports parser/main from CLI/runner
      modules.
- [ ] CLI behavior is unchanged.
- [ ] Path defaults are unchanged.

## Phase 5 - Remove Compatibility Wrappers

Goal: remove the old entrypoint-as-API pattern.

Targets:

- [ ] Replace tests importing helper functions from
      `single_stage_banana_example.py` with imports from `banana_opt.*`.
      Current migration size: 122 `load_single_stage_example_module()` call
      sites in `test_single_stage_example.py` and 7 in
      `test_surface_mode_contracts.py`.
- [ ] Decide whether to migrate all 129 test load sites in one mechanical
      slice or keep a thin compatibility entrypoint for low-value legacy tests.
- [ ] Update `frontier_evaluator.py` before deleting any wrapper or private
      helper re-export it uses.
- [ ] Delete wrappers like:

```python
def build_total_objective(...):
    return _build_total_objective_impl(...)
```

- [ ] Remove `# noqa: F401 - re-exported for importlib-loaded tests` imports
      once tests import real modules directly.
- [ ] Run `rg "_impl\\(" single_stage_banana_example.py` and remove remaining
      wrapper-only patterns.

Acceptance:

- [ ] `single_stage_banana_example.py` is a thin executable entrypoint.
- [ ] Tests no longer depend on entrypoint internals.
- [ ] `frontier_evaluator.py` no longer depends on entrypoint internals.
- [ ] Direct module imports are the only supported test surface for helper
      logic.

## Phase 6 - Final Simplification Pass

Goal: simplify only after behavior parity is locked.

Tasks:

- [ ] Remove duplicated helper logic.
- [ ] Remove dead imports.
- [ ] Reduce module-global reads in extracted modules.
- [ ] Convert hidden global state to explicit context where local and practical.
- [ ] Inventory every `globals().get(...)` site before changing it.
- [ ] For each `globals().get(...)` site, either:
  - [ ] move the value into an explicit context field, or
  - [ ] preserve it as a documented module-level default.
- [ ] Do not silently drop fallback values such as:
  - [ ] `SINGLE_STAGE_GOAL_MODE` defaulting to `"target"`
  - [ ] `CONSTRAINT_METHOD` defaulting to `"penalty"`
  - [ ] `TOPOLOGY_GATE_PENALTY_SCALE` defaulting to `4.0`
  - [ ] `HARDWARE_SEARCH_PENALTY_SCALE` defaulting to `4.0`
  - [ ] objective weight fallbacks defaulting to `0.0`
  - [ ] replay/config fallbacks used by preserved-timeout artifacts
- [ ] Keep objective and geometry SSOT boundaries clear:
  - [ ] single-stage objective formula in
        `single_stage_objectives.py::build_total_objective`.
  - [ ] single-stage bundle construction in `single_stage_objective_bundle.py`.
  - [ ] Stage 2 plasma geometry in `stage2_geometry.py`.
- [ ] Keep public runtime behavior unchanged.

## Required Validation Commands

Run after each phase:

```bash
python -m py_compile \
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  examples/single_stage_optimization/banana_opt/*.py

python -m pytest tests/geo/test_single_stage_example.py
python -m pytest tests/geo/test_banana_objective_modules.py tests/geo/test_banana_modularization_parity.py
python -m pytest tests/geo/test_banana_helper_modules.py
python -m pytest tests/geo/test_surface_mode_contracts.py
```

Run after Phase 2 and later:

```bash
python -m pytest tests/geo -k "single_stage and (alm or hardware or topology or frontier or search)"
```

Run after any Stage 2 geometry-sensitive change:

```bash
python -m pytest tests/geo/test_banana_helper_modules.py -k "Stage2GeometryHelperTests"
python -m pytest tests/geo/test_single_stage_example.py -k "Stage2RuntimeSmokeTests"
```

Run after moving any entrypoint wrapper used by frontier evaluation:

```bash
python -m pytest tests/geo -k "frontier"
```

Run before commit:

```bash
git diff --check
git status --short
```

## Risk Ranking

1. `evaluate_search_step` extraction.
   - Highest risk because it mutates `run_dict`, `JF.x`, surface state,
     topology status, hardware status, ALM payloads, and diagnostics.
   - It is an atomic context-and-test-fixture refactor, not a simple move.
2. Runner extraction.
   - High risk because the current script body relies on many globals.
3. CLI extraction.
   - Medium-high risk because defaults affect run identity and artifact paths.
4. Objective bundle extraction.
   - Medium risk if kept term-by-term and tested; also update
     `frontier_evaluator.py` if it consumes moved symbols.
5. Wrapper deletion.
   - Medium risk because it includes `frontier_evaluator.py` and 129 current
     test load sites.

## Recommended Execution Order

- [ ] Phase 0: add characterization tests.
- [ ] Phase 1: extract objective bundle.
- [ ] Phase 2: extract search loop.
- [ ] Phase 3: extract runner.
- [ ] Phase 4: extract CLI.
- [ ] Phase 5: remove wrappers.
- [ ] Phase 6: final simplification.

Best first implementation slice:

- [ ] Start with Phase 1 only.
- [ ] Do not touch `evaluate_search_step` in the first slice.
- [ ] Do not touch the runner body in the first slice.
- [ ] Do not change Stage 2 geometry in the first slice.

## Definition of Done

- [ ] `single_stage_banana_example.py` is a thin CLI wrapper.
- [ ] Pure objective math is only in
      `single_stage_objectives.py::build_total_objective`.
- [ ] Objective bundle construction is isolated and tested.
- [ ] Search-loop trial evaluation is isolated and tested.
- [ ] Runner orchestration is callable directly from tests.
- [ ] CLI parser/defaults are isolated and snapshot-tested.
- [ ] Stage 2 geometry contract from `b525b8de2` is locked by regression tests.
- [ ] `frontier_evaluator.py` imports real helper modules directly instead of
      depending on the entrypoint as an API.
- [ ] Snapshot parity contracts are either preserved or intentionally bumped.
- [ ] Existing optimization behavior and artifacts are unchanged.
