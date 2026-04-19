# Single-Stage Surface Mode Split Implementation Plan

Date: 2026-04-19
Status: proposal only, not yet implemented
Scope: `examples/single_stage_optimization/` single-stage CLI semantics, surface-mode contracts, objective assembly, metadata, wrappers, tests, and docs

## Goal

Replace the current implicit `num_surfaces` behavior split with three explicit user-facing modes:

- `single_surface`
- `published_multisurface`
- `experimental_multisurface`

The implementation must preserve the current single-surface baseline, quarantine the current custom two-surface search as experimental, and add a new published-aligned multisurface contract without rewriting the whole single-stage solver.

## Executive Summary

The current repo mixes two different ideas under one knob:

- a legitimate implementation detail: how many Boozer surfaces are solved
- a physics contract: what optimization problem the user thinks they are solving

That coupling is the root problem.

Today:

- `num_surfaces=1` means one outer surface with edge-focused `iota` and volume targeting
- `num_surfaces=2` means a custom hybrid contract:
  - QS and Boozer residual are averaged across both surfaces
  - `iota` and volume still come from the outer surface only
  - continuation ramps relax the inner-surface weight and gap / vessel thresholds early in the search
  - topology gating is enabled only in this path
  - ALM is disabled in this path

This is not obviously invalid physics, but it is not a clean published contract either.

The implementation target is:

1. preserve current `num_surfaces=1` behavior as `single_surface`
2. preserve current `num_surfaces=2` custom behavior as `experimental_multisurface`
3. add a new `published_multisurface` mode that uses:
   - a fixed surface grid
   - explicit surface weights
   - the same surface-local physics terms on every configured surface
   - fixed stack-validity semantics with no custom continuation ramp

The key design rule is:

- mode names describe the physics contract
- surface count becomes an internal detail of the selected mode

## Why This Change Is Needed

### 1. `num_surfaces` currently overstates what the user is selecting

`num_surfaces=2` does not mean "standard multisurface optimization". It means "the current repo-specific two-surface contract".

### 2. The current two-surface path is asymmetric in a non-obvious way

Current objective assembly in `SINGLE_STAGE/single_stage_banana_example.py`:

- builds `Iotas(...)` for each surface
- builds `NonQuasiSymmetricRatio(...)` for each surface
- builds `BoozerResidual(...)` / `BoozerResidualExact(...)` for each surface
- then averages QS and Boozer residual across surfaces
- but builds `Jiota` and `JVolume` from the outer surface only

That asymmetry is currently an implementation fact, not an explicit contract.

### 3. The current multisurface search also bakes in heuristic search control

Current multisurface flow in `banana_opt/single_stage_geometry.py`:

- ramps inner-surface weight from an initial value to `1.0`
- ramps surface-gap and vessel-gap thresholds with the same scale
- enforces nesting only after the gate is fully tightened

That continuation behavior may be useful, but it is optimizer policy, not the scientific definition of the problem.

### 4. Current feature compatibility already shows the path is not treated as the mainline

- topology gate is only enabled when `num_surfaces > 1`
- ALM currently requires `num_surfaces == 1`
- Boozer-stage refinement also requires `num_surfaces == 1`

This is a sign that the repo already treats the multisurface path as special-case behavior rather than a stable contract.

### 5. Published precedent supports multisurface optimization in general, but not this exact custom recipe

External validation on 2026-04-19 found:

- peer-reviewed support for direct coil optimization using nested surfaces and Boozer-surface-based objectives
- official SIMSOPT and DESC docs documenting multisurface QS / objective patterns in related equilibrium and QS workflows
- no canonical official example matching the repo's exact "outer-only `iota` / volume plus continuation-gated two-surface search" contract

Therefore the correct response is not to delete the custom lane blindly. It is to name it accurately and add a clean published-aligned lane beside it.

## Current Repo Behavior

### User-facing knobs

Current main single-stage entrypoint:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`

Relevant CLI / metadata seams:

- `--num-surfaces`
- `--inner-surface-ratio`
- `NUM_SURFACES`
- `INNER_SURFACE_RATIO`
- `SURFACE_NAMES`
- `SURFACE_SEED_LABELS`
- `SURFACE_TARGET_VOLUMES`
- `BOOZER_SURFACE_TARGET_VOLUMES`
- `PLASMA_CURRENT_SURROGATE_SCOPE`

Current wrappers also pass through this behavior, including:

- `run_single_stage_goal_mode_comparison.py`
- `run_stage2_to_single_stage.py`
- `run_single_stage_donor_repair.py`

Some wrappers inherit the surface flags transitively through parent parsers rather than redeclaring them locally.

### Single-surface path

Current `num_surfaces=1` behavior should be treated as the baseline contract:

- one outer Boozer surface
- `iota` target on the outer surface
- volume objective on the outer surface
- vessel spacing objective directly on the outer surface
- no topology gate during search
- ALM supported

This is the incumbent production baseline and should remain the default until the new published multisurface lane proves better in controlled comparisons.

### Current multisurface path

Current `num_surfaces=2` behavior:

- builds inner + outer surfaces from `inner_surface_ratio`
- uses shared coil DOFs for both
- averages surface-local QS and Boozer residual terms
- keeps outer-only `iota` and volume terms
- uses fixed stack checks plus a continuation gate
- enables topology gate only in this mode
- does not support ALM

This behavior should be preserved exactly first, then renamed to `experimental_multisurface`.

## Target User-Facing Contract

### New primary CLI

Add:

- `--surface-mode {single_surface,published_multisurface,experimental_multisurface}`

Make `single_surface` the default.

### Legacy compatibility

Keep these flags in the first rollout:

- `--num-surfaces`
- `--inner-surface-ratio`

Compatibility rules:

- if `--surface-mode` is set, it wins
- if `--surface-mode` is unset and `--num-surfaces=1`, map to `single_surface`
- if `--surface-mode` is unset and `--num-surfaces=2`, map to `experimental_multisurface`
- reject any unsupported `--num-surfaces` value
- emit a deprecation warning whenever legacy-only mapping is used

Do not silently reinterpret existing `--num-surfaces=2` runs as `published_multisurface`.

Migration note:

- legacy `--num-surfaces=3` remains rejected even though `published_multisurface` internally uses a 3-surface default stack in v1

### Results metadata

Add new fields:

- `SURFACE_MODE`
- `SURFACE_MODE_VERSION`
- `SURFACE_MODE_SOURCE`
- `SURFACE_LABEL_FRACTIONS`
- `SURFACE_WEIGHTS`
- `SURFACE_STACK_POLICY`
- `SURFACE_PHYSICS_CONTRACT`
- `LEGACY_NUM_SURFACES`
- `LEGACY_INNER_SURFACE_RATIO`

Definitions:

- `SURFACE_MODE`: one of the three user-facing names
- `SURFACE_MODE_VERSION`: starts at `1`
- `SURFACE_MODE_SOURCE`: `explicit_cli`, `legacy_num_surfaces_mapping`, or wrapper-defined
- `SURFACE_LABEL_FRACTIONS`: normalized radial fractions used to construct the surface stack
- `SURFACE_WEIGHTS`: explicit per-surface objective weights
- `SURFACE_STACK_POLICY`: `single_surface_direct`, `published_fixed_stack`, or `experimental_continuation_stack`
- `SURFACE_PHYSICS_CONTRACT`: stable text summary of what is optimized on which surfaces

Keep the old keys for backward compatibility.

`SURFACE_MODE_VERSION` bump policy:

- bump the version whenever default fractions, default weights, stack-validity policy, or objective mathematics change for any named mode
- do not bump the version for additive metadata, docs-only changes, or wrapper plumbing that leaves the runtime contract unchanged

## Mode Contracts

## `single_surface`

Definition:

- exactly the current `num_surfaces=1` contract

Surface set:

- one outer surface only

Physics terms:

- outer-surface `iota`
- outer-surface volume
- outer-surface vessel spacing term
- outer-surface QS / Boozer residual because there is only one surface
- global engineering terms as they exist today

Search policy:

- current single-surface policy unchanged

Feature compatibility:

- ALM remains supported
- current refinement behavior remains supported
- current metadata remains readable

Acceptance bar:

- short-run and init-only outputs remain numerically consistent with the current baseline
- legacy `--num-surfaces=1` workflows continue to behave the same after mapping

## `experimental_multisurface`

Definition:

- exactly the current `num_surfaces=2` custom contract, renamed only

Surface set:

- current inner + outer construction derived from `inner_surface_ratio`

Physics terms:

- averaged QS across surfaces
- averaged Boozer residual across surfaces
- outer-only `iota`
- outer-only volume
- current engineering terms

Search policy:

- current continuation ramp
- current stack-gate ramp
- current topology-gate behavior

Feature compatibility:

- ALM remains unsupported in v1
- current refinement restrictions remain unchanged in v1

Acceptance bar:

- legacy `--num-surfaces=2` mapped runs reproduce current behavior within numerical tolerance
- all current multisurface tests continue to pass after the rename

## `published_multisurface`

Definition:

- a published-aligned multisurface contract using a fixed surface grid, explicit weights, uniform surface-local physics terms, and fixed stack-validity semantics

Important wording rule:

- this mode is "published-aligned", not a claim that the repo exactly reproduces one specific paper line-for-line

### Initial v1 design

Surface set:

- default three-surface stack with normalized fractions `[0.6, 0.8, 1.0]`
- fractions are applied relative to the current outer-seed label construction path
- support explicit override by CLI in a later sub-step, but land the fixed default first

Weights:

- default `[1.0, 1.0, 1.0]`
- no continuation ramp in v1

Surface-local physics terms:

- QS metric on every configured surface
- Boozer residual on every configured surface
- aggregate by explicit weighted mean

Outer/global physics terms:

- outer-surface volume stays as the global size control
- edge / outer-surface `iota` stays as the transform control in v1

Rationale for v1 asymmetry:

- volume is already an outer-boundary quantity in the current solver
- the repo does not currently have a physically justified target `iota` profile interface
- a documented edge-transform target is cleaner than silently reusing the current custom asymmetry
- published precedent such as Wechsung et al. (2023) uses a broader multisurface transform / volume contract, including mean or volume-wide transform treatment and per-surface target volumes
- therefore `published_multisurface` v1 is only aligned at the multisurface QS + Boozer-residual level, not yet at the full published `iota` / volume contract level

Stack-validity policy:

- solve all configured surfaces every evaluation
- use fixed, always-on stack validity checks
- adjacent-surface gap is a hard feasibility check in v1, not an objective term
- outer-surface vessel gap is a hard feasibility check in v1, not an objective term
- no continuation ramp
- no topology gate in the optimization objective in v1
- topology remains a validation / reporting metric, not part of the published mode contract in v1

Feature compatibility:

- weighted-sum mode only in v1
- ALM remains disabled in v1 unless a later dedicated follow-up adds a clean contract
- current refinement should remain disabled in v1 until the multisurface state contract is made explicit

Acceptance bar:

- explicit metadata clearly distinguishes this mode from the experimental lane
- objective assembly tests show uniform surface-local terms across all configured surfaces
- stack-validity tests run with fixed thresholds and no continuation behavior

### Non-goals for `published_multisurface` v1

- do not reproduce the full bilevel nested-surface optimization papers exactly
- do not add profile-`iota` targets yet
- do not add topology gating to the objective
- do not reuse the experimental continuation ramp under a published name

## Recommended Implementation Order

Implement in phases. Do not rewrite everything at once.

## Phase 0: Freeze Current Behavior

Deliverables:

- document current `single_surface` and current custom two-surface behavior as the baseline
- add test snapshots or assertions for:
  - current single-surface objective composition
  - current multisurface objective composition
  - current continuation-gate semantics
  - current metadata payloads

Rationale:

- the current custom lane must be preserved before it is renamed, otherwise the rename and physics changes will be entangled

Snapshot contract to freeze in this phase:

- objective composition for current `num_surfaces=1`
- objective composition for current `num_surfaces=2`
- continuation-weight and stack-gate semantics
- topology-gate enablement semantics
- metadata payload keys:
  - `NUM_SURFACES`
  - `INNER_SURFACE_RATIO`
  - `SURFACE_NAMES`
  - `SURFACE_SEED_LABELS`
  - `SURFACE_TARGET_VOLUMES`
  - `BOOZER_SURFACE_TARGET_VOLUMES`
  - `PLASMA_CURRENT_SURROGATE_SCOPE`
- short-run regression tolerances:
  - exact equality for mode names, metadata keys, and capability flags
  - numerical checks use a stated tolerance in each test, with `atol` / `rtol` recorded next to the assertion instead of ad hoc defaults

Likely files:

- `tests/geo/test_single_stage_example.py`
- `tests/geo/test_banana_objective_modules.py`

Acceptance:

- existing behavior is pinned by tests before any contract refactor lands

## Phase 1: Add Mode Taxonomy And Metadata

Deliverables:

- add `SurfaceMode` enum or string constants
- add `--surface-mode`
- add new result metadata fields
- add legacy mapping from `--num-surfaces`

Recommended new helper module:

- `examples/single_stage_optimization/banana_opt/surface_mode_contracts.py`

Helper responsibilities:

- resolve user-facing surface mode from CLI
- validate supported combinations
- emit metadata fields
- expose mode capability flags

Do not yet change the objective mathematics in this phase.

Likely touched files:

- `SINGLE_STAGE/single_stage_banana_example.py`
- `banana_opt/surface_mode_contracts.py`
- `run_single_stage_goal_mode_comparison.py`
- `run_stage2_to_single_stage.py`
- `run_single_stage_donor_repair.py`

Acceptance:

- `single_surface` and legacy `--num-surfaces=1` produce the same runtime behavior
- `experimental_multisurface` and legacy `--num-surfaces=2` produce the same runtime behavior
- results metadata includes both legacy and new keys

## Phase 2: Extract Surface Set Construction From Physics Contract

Deliverables:

- separate "which surfaces exist" from "how the objective treats them"
- refactor current `build_surface_configs(...)` usage so the caller passes a mode contract or explicit surface fractions

Recommended helper split:

- `resolve_surface_label_fractions(surface_mode, args) -> list[float]`
- `build_surface_configs_from_fractions(...)`
- `resolve_surface_weights(surface_mode, args) -> np.ndarray`
- `resolve_surface_stack_policy(surface_mode, args) -> dataclass`

Behavior:

- `single_surface` returns `[1.0]`
- `experimental_multisurface` returns current two-surface fractions derived from outer fraction + `inner_surface_ratio`
- `published_multisurface` returns `[0.6, 0.8, 1.0]` in v1

Likely touched files:

- `banana_opt/single_stage_geometry.py`
- `banana_opt/surface_mode_contracts.py`
- `SINGLE_STAGE/single_stage_banana_example.py`

Acceptance:

- surface construction tests pass for all three modes
- current single-surface and experimental fractions remain unchanged

## Phase 3: Preserve `single_surface` Exactly

Deliverables:

- route all single-surface behavior through the new mode seam
- make `single_surface` the documented default

Requirements:

- no objective drift
- no metadata drift beyond additive fields
- no wrapper drift

Tests:

- parser tests
- init-only smoke tests
- short-run regression tests for objective composition

Acceptance:

- the default path remains stable enough to serve as the baseline for future A/B comparisons

## Phase 4: Quarantine The Existing Custom Two-Surface Path As `experimental_multisurface`

Deliverables:

- rename current `num_surfaces=2` behavior to `experimental_multisurface`
- preserve current continuation ramp and topology-gate logic behind explicit mode gating
- remove any implication in docs that this is the canonical multisurface contract

Implementation rule:

- keep the current code path as intact as possible
- do not "clean it up" in the same commit as the rename

Recommended code changes:

- replace raw `num_surfaces > 1` checks in policy code with explicit capability checks:
  - `surface_mode_supports_topology_gate(mode)`
  - `surface_mode_uses_continuation_gate(mode)`
  - `surface_mode_supports_alm(mode)`

Likely touched files:

- `SINGLE_STAGE/single_stage_banana_example.py`
- `banana_opt/single_stage_geometry.py`
- `banana_opt/surface_mode_contracts.py`

Acceptance:

- old two-surface behavior still exists, but only under the experimental name

## Phase 5: Implement `published_multisurface`

Deliverables:

- add a new mode-specific objective assembly path
- add a fixed three-surface stack
- add explicit equal weights
- add fixed stack-validity policy with no ramp

Recommended objective builder shape:

- one primary dispatcher keyed by a `SurfaceModeContract` dataclass or equivalent SSOT contract object
- mode-specific behavior selected through that dispatcher, not three parallel top-level builder functions that can drift

Shared lower-level helpers:

- `build_surface_local_terms(surface_data, coils, stage)`
- `aggregate_surface_terms(terms, weights)`
- `build_outer_global_terms(surface_data, ...)`
- `build_engineering_terms(...)`

Published-mode v1 behavior:

- `NonQuasiSymmetricRatio` for each configured surface
- `BoozerResidual` / `BoozerResidualExact` for each configured surface
- weighted average of those terms
- outer `iota`
- outer volume
- global engineering terms
- stack validity checked every evaluation using fixed thresholds
- no continuation ramp
- no topology gate during search

Implementation rule:

- do not share the experimental continuation helper with published mode

Likely touched files:

- `SINGLE_STAGE/single_stage_banana_example.py`
- `banana_opt/single_stage_objectives.py`
- `banana_opt/single_stage_geometry.py`
- `banana_opt/surface_mode_contracts.py`

Acceptance:

- new tests prove that published mode does not reuse experimental continuation behavior
- metadata identifies the published stack and weights explicitly

## Phase 6: Update Wrappers, Runner Scripts, And Docs

Deliverables:

- plumb `--surface-mode` through single-stage wrapper scripts
- update `examples/single_stage_optimization/README.md`
- update user-facing docs to explain the three mode names and default behavior

Required doc changes:

- mode summary table
- legacy mapping note
- examples for each mode
- warning that `experimental_multisurface` is not the literature-aligned default

Likely touched files:

- `examples/single_stage_optimization/README.md`
- wrapper scripts under `examples/single_stage_optimization/`
- one or more top-level docs under `docs/`

Acceptance:

- `--help` text is explicit and unambiguous
- wrapper-generated `results.json` includes the new mode metadata

## Phase 7: Validation, Benchmarking, And Recommendation

Deliverables:

- controlled comparison between:
  - `single_surface`
  - `published_multisurface`
  - `experimental_multisurface`

Compare on the same donor seeds and optimizer budgets.

Pre-register before running the comparison:

- donor seed set
- optimizer budget and stopping rules
- exact metrics to compare
- the decision rule for "beat or match"

Metrics:

- Boozer solve success rate
- accepted-iteration count
- runtime
- final QS metric
- final Boozer residual
- final `iota`
- final volume
- engineering metrics
- final topology / confinement diagnostics

Success criterion:

- `published_multisurface` must beat or match `single_surface` under the pre-registered rule before it should replace the default

Non-success criterion:

- if `published_multisurface` is cleaner but materially worse or much less stable, keep it as an opt-in lane and leave `single_surface` as default

## Capability Matrix

Initial target capability matrix:

| capability | single_surface | published_multisurface | experimental_multisurface |
| --- | --- | --- | --- |
| weighted-sum objective | yes | yes | yes |
| ALM | yes | no in v1 | no |
| topology gate in search | no | no in v1 | yes |
| continuation ramp | no | no | yes |
| fixed surface weights | trivial | yes | no |
| legacy `num_surfaces` mapping | yes | no | yes |

Any future change to this matrix must update:

- CLI help
- results metadata
- tests
- docs table

## Proposed New Modules And Responsibilities

### New module: `banana_opt/surface_mode_contracts.py`

Responsibilities:

- mode enum / constants
- legacy mapping
- capability checks
- metadata builders
- surface fraction resolution
- weight resolution

### Possible follow-up helper splits

If `single_stage_banana_example.py` remains too large after phase 2:

- `banana_opt/surface_mode_objectives.py`
- `banana_opt/surface_mode_runtime.py`

Do not create these extra modules until phase 2 proves the main script remains too coupled.

## Validation Plan

### Unit tests

Add or extend tests for:

- CLI mapping and compatibility rules
- mode metadata emission
- surface fraction resolution
- surface weight resolution
- capability matrix checks

Likely files:

- `tests/geo/test_single_stage_example.py`
- new `tests/geo/test_surface_mode_contracts.py`

### Objective-construction tests

Add mode-specific tests for:

- `single_surface` objective terms
- `published_multisurface` surface-local aggregation
- `experimental_multisurface` preservation of current asymmetry

### Search-policy tests

Add or extend tests for:

- published mode has no continuation ramp
- experimental mode keeps continuation ramp
- topology gate only runs where intended
- ALM gating errors are explicit and mode-aware

### Integration smoke tests

Run:

- `--init-only` per mode
- short bounded optimizer smoke runs on representative donors
- wrapper-level smoke runs where surface-mode metadata reaches final `results.json`

### Regression acceptance

Required before recommending default changes:

- current single-surface baseline remains stable
- legacy `num_surfaces=2` path preserved under `experimental_multisurface`
- published mode produces reproducible metadata and does not inherit experimental continuation behavior

## Risks And Mitigations

### Risk 1: behavior drift during the rename

Mitigation:

- freeze current single-surface and experimental behavior with tests before refactoring

### Risk 2: over-claiming the published lane

Mitigation:

- document it as "published-aligned v1"
- do not claim exact reproduction of a specific paper
- keep surface-local terms uniform across surfaces and keep policy simple

### Risk 3: CLI confusion from both old and new flags

Mitigation:

- add explicit precedence rules
- emit deprecation warnings
- record both old and new metadata during the transition

### Risk 4: runtime blow-up

Mitigation:

- keep published mode simple in v1
- no topology gate in search
- no ALM in v1
- benchmark against fixed short-run budgets before broader rollout

### Risk 5: mode-specific code branches become unmaintainable

Mitigation:

- centralize capability flags and metadata in one SSOT module
- keep objective differences explicit rather than encoded through scattered `if num_surfaces > 1` checks

## Non-Goals

- do not modify Stage 2 in this workstream
- do not remove legacy metadata in the first rollout
- do not add a new physics target-profile system for `iota` in v1
- do not make `published_multisurface` the default until benchmarking justifies it
- do not delete the current custom multisurface lane before the new mode split lands

## Immediate Next Task After This Plan

Start with a narrow first implementation slice:

1. add `--surface-mode`
2. add the mode taxonomy helper module
3. map legacy `--num-surfaces` to `single_surface` or `experimental_multisurface`
4. add new result metadata
5. do not change objective math yet

That first slice creates the contract seam needed for all later physics work without destabilizing the current solver.
