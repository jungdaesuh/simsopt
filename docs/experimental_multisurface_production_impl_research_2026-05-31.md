# experimental_multisurface Production Implementation Research

Date: 2026-05-31
Repo: `/Users/suhjungdae/code/columbia/simsopt-surrogate`
Observed HEAD: `d1eea50ec feat(single-stage): --free-tf-geometry -- unfreeze TF coil geometry + TF HW penalties`
Branch/status at doc-review time: `surrogate-confinement-v2`, ahead of
`fork/surrogate-confinement-v2`; untracked `.claude/` plus this research note.

## Research Method

This note applies the GPD research/map workflow to the live checkout rather than
starting a numbered GPD phase. The useful checks for this task are:

- source-grounded implementation map, not recollection;
- direct-vs-proxy separation for topology/KAM-style gates;
- numerical convergence and benchmark/replay acceptance before production claims;
- explicit missing-evidence markers where a run artifact was not inspected.

The GPD `numerical-computation` protocol is generic, but its production-grade
requirements map cleanly here: convergence tests, independent benchmarks,
error/uncertainty budgets, and direct-observable validation.

## Current Implementation Map

The current mode is real code, not a stub.

- Contract SSOT:
  `examples/single_stage_optimization/banana_opt/surface_mode_contracts.py`
  defines `single_surface`, `published_multisurface`, and
  `experimental_multisurface` (`:7-14`), the frozen `SurfaceModeContract`
  dataclass (`:32-43`), legacy `--num-surfaces=2` mapping to
  `experimental_multisurface` (`:72-104`), inner-ratio validation (`:58-69`),
  label fractions/weights/stack policy/physics contract (`:107-154`), and
  capability predicates (`:256-288`).
- Surface stack construction:
  `banana_opt/single_stage_geometry.py` builds legacy one/two-surface configs
  (`:71-143`) and contract-driven stacks (`:146-214`), including strict
  inner-to-outer target-volume ordering.
- Runtime stack validation:
  `evaluate_surface_stack` checks Boozer solve success, self-intersection,
  volume ordering, adjacent gaps, optional nesting, and iota-collapse
  rejection (`single_stage_geometry.py:301-413`).
- Continuation policy:
  `continuation_inner_surface_weight`, `build_surface_search_weights*`, and
  `build_surface_search_gate*` make the experimental ramp a diagnostics/gate
  policy, not a moving descended objective (`single_stage_geometry.py:1006-1180`).
- Objective assembly:
  production paths call `resolve_current_surface_objective_terms` and pass
  global QS/Boozer objectives through `evaluate_total_objective`, so
  non-uniform diagnostic weights do not retarget the optimized objective
  (`SINGLE_STAGE/single_stage_banana_example.py:6848-7135`,
  `banana_opt/single_stage_objectives.py:437-560`).
- Shear support:
  `IotaShearShortfall` implements an opt-in axis-to-edge iota-spread shortfall
  term for multisurface runs (`banana_opt/single_stage_objectives.py:71-141`).
  CLI knobs are `--shear-target` and `--shear-weight`
  (`single_stage_banana_example.py:2117-2134`), wired into the bundle at
  `:6375-6460` and runtime config at `:12194-12285`.
- Runtime entry:
  the main script resolves and validates the surface-mode contract before
  current, objective, and surface initialization (`single_stage_banana_example.py:11695-11810`).
- Multisurface init:
  `initialize_surface_data_for_contract` uses the published initializer only
  for `published_multisurface`; otherwise it initializes configured surfaces
  in order, which is the experimental path (`single_stage_banana_example.py:3242-3293`).
- Current handling:
  `published_multisurface` is vacuum-locked, but `experimental_multisurface`
  falls through to ordinary finite-current settings
  (`banana_opt/current_contracts.py:781-848`).
- Topology gate:
  `surface_mode_supports_topology_gate` enables the gate for published and
  experimental multisurface; `evaluate_search_topology_gate` and
  `final_topology_gate_for_results` thread the contract (`single_stage_banana_example.py:5488-5558`).
- Artifacts:
  surface artifacts are written per surface (`single_stage_geometry.py:1444-1499`);
  run metadata records surface names, seed labels, target volumes, initial/final
  volumes, initial/final iotas, stack status, and interior low-order-rational
  diagnostics (`single_stage_geometry.py:1501-1548`). Final `results.json` is
  written once at completion (`single_stage_banana_example.py:14347-14348`).
  Preserved partial artifacts have their own result payload path
  (`single_stage_banana_example.py:9370-9585`, `:9674-9834`).

## Current Test Evidence

Existing focused tests cover the mode contract and key runtime wiring:

- `tests/geo/test_surface_mode_contracts.py:73-159` verifies legacy mapping,
  explicit experimental contracts, ALM support, topology-gate support, and
  published runtime support.
- `tests/geo/test_surface_mode_contracts.py:338-376` asserts Boozer-stage
  refinement rejects explicit multisurface contracts and ALM allows
  `experimental_multisurface`.
- `tests/geo/test_single_stage_example.py:2988-3035` checks two-surface config
  construction and inner target-volume derivation.
- `tests/geo/test_single_stage_example.py:4113-4228` pins the continuation
  ramp, gate scaling, nesting relaxation, and published fixed-gate behavior.
- `tests/geo/test_single_stage_example.py:12605-12752` covers relaxed nesting
  during continuation and state restoration after multisurface fallback.
- `tests/geo/test_banana_objective_modules.py:6504-6663` covers the shear
  shortfall objective and default-off objective assembly.

Prior docs also record a targeted validation pass:
`docs/multisurface_validation_followups_impl_plan_2026-05-27.md:141-151`
reports a focused pytest command with 162 passing selected tests after the
May 27 multisurface follow-up patch.

Current live validation run for this note:

```bash
.conda-env/bin/python -m pytest tests/geo/test_surface_mode_contracts.py tests/geo/test_single_stage_example.py tests/geo/test_banana_objective_modules.py -k "surface_mode or multisurface or continuation or surface_stack or shear or topology_gate or checkpoint" -q
```

Observed result: `68 passed, 1 failed, 528 deselected, 2 warnings`. The failure
is `SingleStageSurfaceModeIntegrationTests.test_make_run_identity_config_uses_effective_surface_contract`:
the test fixture's `SimpleNamespace` lacks `topology_scorer_min_returns`, while
`make_run_identity_config` now reads that field. This is fixture drift in the
current checkout, not caused by the research note and not evidence that the
experimental surface stack itself is stubbed.

## Production Verdict

The implementation is functionally wired and already suitable as an opt-in
research lane. It is not production-equivalent to `single_surface` because the
contract still deliberately exposes weaker runtime guarantees:

1. `surface_mode_runtime_supported()` is currently a no-op that validates the
   mode name and returns `True` for every mode (`surface_mode_contracts.py:279-288`).
2. Boozer-stage refinement is still single-surface only; the validator rejects
   any multisurface contract when `--boozer-stage-refinement` is enabled
   (`single_stage_banana_example.py:5319-5340`).
3. The mode is exactly a two-surface continuation stack (`inner`, `outer`);
   it is not a general N-surface solver (`surface_mode_contracts.py:26-29`,
   `:107-154`).
4. Iota and volume remain outer-surface primary controls by default, while QS
   and Boozer residual aggregate across surfaces. Shear exists, but only as an
   opt-in objective term.
5. The search policy is continuation-gated. This is intentional, but production
   needs a reproducible acceptance contract around `accepted_iterations`,
   `gate_scale`, gap thresholds, topology-gate failures, and collapse rejections.
6. Final `results.json` has surface-mode and per-surface metadata, but
   checkpoint telemetry is still incomplete for production triage: checkpoint
   directories get `biot_savart.json` and surface sidecars
   (`single_stage_banana_example.py:11480-11514`), not a complete
   per-checkpoint `results.json`.

## Production Definition

Call `experimental_multisurface` production-grade only after all of the following
are true in the current checkout:

- The contract layer is the only source of mode truth: mode capabilities,
  refinement policy, telemetry schema, current policy, topology policy, and
  required knobs are represented in `SurfaceModeContract` or helpers owned by
  `surface_mode_contracts.py`.
- Runtime support is meaningful and fail-closed. Unsupported combinations
  raise before expensive initialization.
- Two-surface shear work has a first-class contract: per-surface iotas,
  iota spread, shear target, shear weight, and objective contribution are
  serialized and tested.
- The final-refinement story is explicit. Either multisurface Boozer-stage
  refinement is implemented, or production mode has a named replacement
  refinement/certification pass with equivalent field-error/nesting evidence.
- Every accepted checkpoint has enough structured telemetry to resume and audit
  without scraping iteration text logs.
- A reduced smoke run, a replay from a known seed, and a long-horizon direct
  topology/Poincare validation have passed under pinned commands.

## Gap Analysis

### 1. Runtime Support Predicate Is Not a Gate

Current:
`surface_mode_runtime_supported()` returns `True` for all modes once the mode
name resolves. The real compatibility checks live elsewhere.

Production change:
turn runtime support into a capability audit that checks the selected mode,
constraint method, current mode, Boozer-refinement policy, topology-gate policy,
surface count, ratio requirements, and artifact/resume compatibility in one
place. Keep the detailed CLI validators, but make the contract predicate no
longer ceremonial.

Acceptance:
unit tests must prove each unsupported combination fails before surface
initialization, including multisurface + Boozer-stage refinement until that
feature lands.

### 2. Final Boozer-Stage Refinement Is Missing For Multisurface

Current:
`--boozer-stage-refinement` is rejected unless the mode is `single_surface`.

Production options:

- Preferred: implement two-surface final-stage refinement. Rebuild the objective
  bundle for the final Boozer residual stage, preserve both surface states, run
  bounded refinement, and emit per-surface final-stage state sidecars.
- Acceptable interim: keep refinement unsupported but rename the production
  certification requirement to a separate multisurface final validation pass
  that runs strict stack status, field error, and direct topology evidence on
  the final incumbent. Do not imply parity with single-surface refinement.

The preferred path is cleaner if the mode is to become a real production lane.

### 3. Shear Is Implemented But Not Yet Production Telemetry

Current:
`IotaShearShortfall` exists and is default-off. Run metadata already records
`INITIAL_SURFACE_IOTAS`, `FINAL_SURFACE_IOTAS`, and interior low-order-rational
diagnostics (`single_stage_geometry.py:1501-1548`;
`tests/geo/test_single_stage_example.py:13466-13560`).

Production change:
do not duplicate the existing iota arrays. Add these derived shear-specific
fields whenever `len(surface_data) >= 2`:

- `FINAL_IOTA_SPREAD`
- `FINAL_IOTA_SPREAD_ABS`
- `SHEAR_TARGET`
- `SHEAR_WEIGHT`
- `SHEAR_OBJECTIVE`
- `SHEAR_SHORTFALL`
- `SHEAR_OBJECTIVE_ENABLED`

Acceptance:
tests must prove the existing iota arrays remain present for experimental
multisurface, the new shear fields are present or explicitly disabled according
to `SHEAR_WEIGHT`, and topology-only failures do not corrupt the serialized
surface/iota telemetry.

### 4. Checkpoint Artifacts Are Not Self-Describing Enough

Current:
periodic checkpoints write `biot_savart.json` and per-surface surface/Boozer
artifacts, while the final `results.json` is written only at completion.
Preserved partial artifacts have richer payloads, but that is a timeout/incumbent
path, not a regular checkpoint schema.

Production change:
for each checkpoint, write a compact `checkpoint_results.json` containing:

- surface-mode metadata;
- accepted iteration and gate scale;
- per-surface volumes/iotas/gaps/nesting status;
- topology-gate status;
- objective breakdown, including shear when enabled;
- artifact paths for each `surf_*_boozer_surface.json` and state sidecar;
- resume checkpoint path and source stage.

Acceptance:
add tests that create a checkpoint payload from a fake two-surface run and assert
the checkpoint has enough information to triage without reading `log.txt`.

### 5. Low-Iota Collapse Handling Needs A Named Production Policy

Current:
`solve_surface_stack_at_dofs` forwards iota-collapse guard kwargs where supported
and `evaluate_surface_stack` rejects iota collapse as defense in depth. This is
stronger than the original experimental path, but the launch policy is still an
operator recipe rather than a contract.

Production change:
define an explicit low-iota policy:

- default low-iota launch guard thresholds;
- required warm-start conditions for shear-push runs from low-iota seeds;
- repair-first and trust-radius defaults if this mode is launched from a fragile
  seed;
- serialized collapse-rejection counters and reasons.

Acceptance:
tests should cover a collapse rejection, state restoration, and telemetry fields.
A real production signoff still requires a reduced run from a low-iota seed and
a known higher-iota seed.

### 6. Direct Topology Evidence Must Stay Separate From Proxy Progress

Current:
the search-time topology gate is an acceptance signal. It is not equivalent to
long-horizon direct confinement validation.

Production change:
make production reports separate:

- search gate: cheap fieldline survival, penalty/rejection behavior;
- final direct validation: long-horizon topology/Poincare/KAM evidence;
- proxy objectives: QS, Boozer residual, shear shortfall, residue objective.

Acceptance:
results metadata must not mark a candidate production-feasible based only on
short search-time topology survival.

## Design-It-Twice

### Option A: Harden The Existing `experimental_multisurface` Mode

Keep the public mode name and make its contract production-grade. This is the
recommended path because existing runs, wrappers, tests, and operator knowledge
already target the name. It also avoids expanding the mode beyond its actual
physics: a custom two-surface continuation stack.

Implementation center:
`surface_mode_contracts.py` remains SSOT. `single_stage_geometry.py` owns stack
mechanics. `single_stage_banana_example.py` should lose ad hoc surface-mode
conditionals over time, but only after tests pin the contract behavior.

### Option B: Add A New `two_surface_production` Mode

This would preserve `experimental_multisurface` as-is and add a production alias
with stricter behavior. It is not recommended now. It creates a migration
surface, duplicates mode semantics, and fights the existing preference for the
three mode names already present in the code.

Use this only if production behavior must intentionally diverge from the legacy
two-surface continuation lane.

## Implementation Plan

### Phase 1: Contract Hardening

- Add fields or contract-owned predicates for:
  `requires_inner_surface_ratio`, `surface_count_policy`,
  `final_refinement_policy`, `current_policy`, `topology_policy`,
  `telemetry_schema_version`, and `production_support_level`.
- Replace the no-op runtime predicate with a real compatibility audit.
- Keep legacy `--num-surfaces=2` mapping to `experimental_multisurface`.
- Add tests in `test_surface_mode_contracts.py` for each supported/unsupported
  combination.

### Phase 2: Production Telemetry Schema

- Build a surface-mode telemetry helper that emits final and checkpoint payloads.
- Include per-surface state, shear metrics, gate state, topology status, and
  collapse counters.
- Write `checkpoint_results.json` in checkpoint directories.
- Add tests with fake surface data; do not require expensive Boozer solves.

### Phase 3: Multisurface Refinement Or Certification

- Preferred: support final-stage multisurface Boozer refinement for two-surface
  stacks.
- If deferred, add a named production final-certification pass and keep
  `--boozer-stage-refinement` rejected with explicit wording.
- Ensure final metadata distinguishes "refined" from "validated only".

### Phase 4: Shear Production Contract

- Promote shear telemetry to first-class output whenever two surfaces exist.
- Keep `--shear-weight=0` default-off.
- Keep the existing reduced objective tests that prove `JShear` changes the
  objective by exactly `SHEAR_WEIGHT * JShear`; add serialization tests for the
  active/inactive shear telemetry fields.

### Phase 5: Low-Iota Launch Policy

- Encode the anti-collapse launch recipe as CLI/env defaults or a named mode
  profile, not tribal knowledge.
- Serialize repair-first/trust-radius/collapse settings into results.
- Add a reduced low-iota regression and one real smoke run.

### Phase 6: Empirical Production Signoff

Required commands/artifacts before a production claim:

```bash
.conda-env/bin/python -m pytest tests/geo/test_surface_mode_contracts.py tests/geo/test_single_stage_example.py tests/geo/test_banana_objective_modules.py -k "surface_mode or multisurface or continuation or surface_stack or shear or topology_gate or checkpoint" -q
```

This command is intentionally the signoff gate even though the current checkout
fails it because one test fixture is missing `topology_scorer_min_returns`.
Fixing that fixture drift is a prerequisite before claiming production readiness.

Then run at least:

- init-only experimental two-surface smoke;
- short bounded optimizer smoke with `--constraint-method=alm`;
- shear-enabled smoke with `--shear-weight > 0`;
- replay from a known low-iota seed with anti-collapse policy active;
- replay from a known higher-iota seed;
- final long-horizon direct topology/Poincare validation on the best result.

## Non-Goals

- Do not turn this into a general N-surface solver unless a separate physics
  contract demands it.
- Do not rename the existing mode unless production semantics intentionally
  diverge.
- Do not claim direct confinement validation from QS/Boozer/shear proxies alone.
- Do not make `published_multisurface` inherit the experimental continuation
  ramp.

## Missing Evidence

The pasted "6-lane lever matrix" result was not found as tracked source text in
this checkout during source search. Treat that as external run evidence until
the actual artifact directory is inspected. It can be used for motivation, not
as a current-checkout production proof.

## Recommended Next Step

Implement Phase 1 first. It is low-risk, keeps all behavior centralized, and
turns the biggest production smell (`surface_mode_runtime_supported` always
passing) into a real fail-closed contract without touching numerical behavior.
