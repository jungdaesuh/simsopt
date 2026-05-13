# jhalpern30 External-Parity Implementation Evidence

**Date:** 2026-05-12
**Branch:** `surrogate-confinement-v2`
**Plan:** `docs/jhalpern30_external_parity_impl_plan_2026-05-12.md`
**Baseline SHA:** `c353f5dae` (docs: add jhalpern30 parity plan)
**Reference external drivers:** `/Users/suhjungdae/code/columbia/banana_drivers/jhalpern30/{stage2.py, singlestage.py}`

## Subject

Bring the in-tree Stage 2 and Single Stage objective contracts to parity with
the external jhalpern30 drivers by adding:

- A lower bound (`Jlsmin`) hinge on coil length, in addition to the existing
  upper bound.
- A projected-ellipse coil-width hinge with min and max thresholds (`Jwmin`,
  `Jwmax`).
- A differentiable curve self-distance objective (`Jself`) keyed to the
  reciprocal of the maximum allowed curvature.

These terms enter both the penalty and ALM routes of Stage 2 and the penalty
route of Single Stage. The Single Stage ALM route already owned `width_min`,
`width_max`, and `self_intersect` as ALM constraints; the parity work
explicitly does **not** add weighted duplicates of those terms to the
Single Stage ALM base objective.

## Method

### Files Touched (12 modified, none added)

| File | Lines added/removed | Purpose |
| --- | --- | --- |
| `examples/single_stage_optimization/banana_opt/hardware_contracts.py` | +17/-0 | SSOT for `BANANA_SELF_INTERSECT_MIN_DISTANCE_M`, `BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR`, four `*_WEIGHT_DEFAULT` constants |
| `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` | +~110/-~0 | Imports, argparse, `Jw`/`Jself`/`Jlsmin`/`Jwmin`/`Jwmax` construction, `JF` extension, ALM `evaluate_problem` plumbing, artifact-state capture, final prints, `build_stage2_results` kwargs |
| `examples/single_stage_optimization/banana_opt/stage2_objectives.py` | +~230/-~0 | `_stage2_constraint_names` + `_legacy_stage2_constraint_names` schema-driven ordering, threshold-overrides extension, hard-hardware-name metadata path, `evaluate_stage2_alm_problem` mandatory geometric-parity kwargs, four new constraint blocks (signed value + gradient + feasibility), artifact-state and hardware-snapshot fields |
| `examples/single_stage_optimization/banana_opt/single_stage_objectives.py` | +~50/-~0 | `build_total_objective` and `evaluate_total_objective` new optional kwargs (`JCoilWidth`, `WIDTH_WEIGHT`, `JCurveSelfIntersect`, `SELFINT_WEIGHT`); diagnostics fields exposing `J_coil_width`, `J_self_intersect`, and SSOT thresholds |
| `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` | +~50/-~0 | Argparse `--single-stage-{width,selfint}-weight`, module-level `SINGLE_STAGE_{WIDTH,SELFINT}_WEIGHT` globals, wrapper signatures, penalty-branch forwarding into `evaluate_total_objective`; ALM branch untouched |

Test extensions: `tests/geo/test_banana_objective_modules.py`,
`tests/geo/test_single_stage_alm_integration.py`,
`tests/geo/test_single_stage_example.py`,
`tests/geo/test_banana_modularization_parity.py`,
`tests/geo/test_alm_utils.py`,
`tests/geo/test_ishw_deliverables.py`. One doc citation refresh in
`docs/alm_hybrid_signal_contract_2026-05-08.md`.

### Constants Pinned to External Driver

| Constant | This repo | External (`banana_drivers/jhalpern30/stage2.py`) |
| --- | --- | --- |
| Width min threshold | `BANANA_WIDTH_MIN_M = 0.05` | `WIDTH_MIN = 0.05` |
| Width max threshold | `BANANA_WIDTH_MAX_M = 0.17` | `WIDTH_MAX = 0.17` |
| Self-distance threshold | `BANANA_SELF_INTERSECT_MIN_DISTANCE_M = 1.0 / MAX_CURVATURE_INV_M = 0.01 m` | `SELFINTERSECT_THRESHOLD = 1.0 / CURVATURE_THRESHOLD = 0.01 m` |
| Self-intersect neighbor skip | `int(BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR * curve.order) = int(1.5 * order)` | `int(1.5 * BANANA_ORDER)` |
| Coil length min target | `COIL_LENGTH_MIN_FRACTION * LENGTH_TARGET = 0.5 * 1.9 = 0.95 m` | `0.5 * LENGTH_TARGET = 0.95 m` |
| Width weight default | `STAGE2_WIDTH_WEIGHT_DEFAULT = 1.0`, `SINGLE_STAGE_WIDTH_WEIGHT_DEFAULT = 1.0` | `WIDTH_WEIGHT = 1e2` (intentionally 100x below; plan section 3.2) |
| Self-intersect weight default | `STAGE2_SELF_INTERSECT_WEIGHT_DEFAULT = 1.0`, `SINGLE_STAGE_SELF_INTERSECT_WEIGHT_DEFAULT = 1.0` | `SELFINTERSECT_WEIGHT = 1e2` |

The 100x lower default weights are deliberate: this repo's calibration of the
other geometric weights is also lower than external, so the parity work
preserves relative term magnitudes. Final calibration is an optimization
concern.

### Strict-Contract Decisions

1. **No legacy fallback flags.** Plan section 3.1 forbids
   `--stage2-length-min-penalty`, `--stage2-width-penalty`,
   `--stage2-selfint-penalty`, `--single-stage-weighted-width-penalty`,
   `--single-stage-weighted-selfint-penalty`. Only weight knobs were added.
2. **Stage 2 ALM enforces the full hardware contract unconditionally.**
   `evaluate_stage2_alm_problem` raises `ValueError` if `Jw`, `Jself`,
   `length_min_target`, `width_min_threshold`, or `width_max_threshold` is
   None.
3. **Single Stage ALM ownership preserved.** The new weighted terms enter
   only the penalty branch (`evaluate_total_objective` / `build_total_objective`).
   `evaluate_alm_objective` already owned `width_min`, `width_max`, and
   `self_intersect` as ALM constraints — that path is unchanged.
4. **Hardware schema advertises artifact applicability for
   `width_min`/`width_max`/`self_intersect`.** The artifact-state dict carries
   `coil_width`, `width_min_threshold`, `width_max_threshold`,
   `self_intersect_penalty`, `self_intersect_threshold`,
   `shortest_self_distance`, `self_intersect_min_distance`, and
   `length_min_target`. The artifact schema maps `self_intersect` to the
   zero-threshold penalty value; `shortest_self_distance` remains diagnostic.

## Results

### Test Outcomes

Final focused test invocation:

```
python -m pytest \
    tests/geo/test_banana_objective_modules.py \
    tests/geo/test_single_stage_alm_integration.py \
    tests/geo/test_single_stage_example.py \
    tests/geo/test_banana_modularization_parity.py \
    tests/geo/test_alm_utils.py \
    tests/geo/test_ishw_deliverables.py \
    tests/geo/test_boozersurface.py \
    -q
=> 727 passed, 214 subtests passed
```

### Phase F New Tests

`tests/geo/test_banana_objective_modules.py::SingleStageObjectiveModuleTests`:

- `test_evaluate_total_objective_includes_self_intersect_term` — verifies
  weighted self-intersect term contributes to total and diagnostics.
- `test_evaluate_total_objective_skips_geometric_parity_when_objectives_missing`
  — verifies `J_coil_width=0`, `J_self_intersect=0`, threshold fields None
  when both objectives are absent.
- `test_evaluate_total_objective_includes_coil_width_term_via_quadratic_penalty`
  — verifies width hinge contribution with mocked `QuadraticPenalty`.

`tests/geo/test_single_stage_example.py::HardwareConstraintTests`:

- `test_build_total_objective_forwards_self_intersect_term` — wrapper
  forwarding test mirroring `test_build_total_objective_forwards_length_min_term`.
- `test_build_total_objective_omits_width_self_terms_when_objectives_missing`
  — equivalence against legacy length-only signature.

### Phase F Test Modifications (subagent-driven)

To accommodate the new ordered ALM payload (8 base + optional surface +
optional poloidal + optional iota constraint names), the following tests
were updated. None of these updates weakened assertions; they only
extended expected values to cover the new constraints.

- `test_banana_objective_modules.py::Stage2ObjectiveModuleTests` — added
  `_FakeWidthObjective`, `_FakeSelfIntersectObjective`, and the
  `_default_geometric_parity_kwargs` helper; extended the ordered
  `signed_values` / `grads` checks for the new ALM constraint set.
- `test_banana_modularization_parity.py` — extended parity table.
- `test_single_stage_alm_integration.py` — extended constraint-name and
  tolerance-count expectations.
- `test_single_stage_example.py::Stage2RuntimeSmokeTests` — added
  `FakeProjectedEllipseWidth`, `FakeCurveSelfIntersect` patches and the
  new `coil_width` / `width_*_threshold` / `shortest_self_distance` /
  `self_intersect_min_distance` / `length_min_target` artifact fields.
- `test_alm_utils.py` — line-citation refresh for shifted code blocks.
- `test_ishw_deliverables.py` — secondary state shape.

### Lint

```
python -m ruff check examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
    examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
    examples/single_stage_optimization/banana_opt \
    tests/geo
=> clean for the touched Stage 2, objective, schema, and focused test files.
   The stale `PLASMA_VESSEL_MIN_DIST_M` import in
   `hardware_constraint_schema.py` was removed with the schema-artifact fix.
```

### Whitespace

`git diff --check` — clean.

## Interpretation

### Penalty-vs-ALM Contract Symmetry (Stage 2)

After this change, the Stage 2 penalty `JF` includes the lower-bound length
hinge, both width hinges, and the self-distance term. The Stage 2 ALM route
unconditionally emits matching `coil_length_min`, `width_min`, `width_max`,
and `self_intersect` constraints with hard value-kind metadata. Running the
same seed through `--constraint-method=penalty` and `--constraint-method=alm`
now optimizes against the same hardware contract. Prior to this change, an
ALM run could accept candidates that violated thresholds the penalty run
did not enforce, or vice versa.

### Single Stage Penalty Contribution

`evaluate_total_objective` now exposes:

- `J_coil_width`, `dJ_coil_width`, `coil_width_min_threshold`,
  `coil_width_max_threshold`
- `J_self_intersect`, `dJ_self_intersect`, `self_intersect_min_distance`

Penalty-mode runs receive non-zero gradient contributions when the candidate
violates either width bound or shrinks self-distance below the threshold.
ALM-mode runs continue to receive these as ALM constraints; their weighted
form is **not** added to the ALM base objective.

### Failed-Trial Path Untouched

The `evaluate_search_step` failed-trial path in
`SINGLE_STAGE/single_stage_banana_example.py` (the
`rejection_increment` / accepted-gradient handling around lines 7530-7545)
is intentionally untouched. That path avoids the legacy `-dJ`
BFGS-corruption bug; no parity work is needed there.

## Crucible Round 1 Findings

- **Lens 5 (Mistake Book) — Pattern 6, confidence 75:** `evaluate_stage2_alm_problem`
  required `Jwmin`/`Jwmax` kwargs in its guard but never read them inside the
  function body; the ALM path computes width hinges from `Jw` and the scalar
  thresholds directly. The `QuadraticPenalty(Jw, ...)` wrappers belong only to
  the penalty path's `JF` sum. **Fix applied:** removed `Jwmin`/`Jwmax` from
  `evaluate_stage2_alm_problem`'s signature, guard, and error message; removed
  matching kwargs from the `evaluate_problem` closure in
  `STAGE_2/banana_coil_solver.py`; cleaned up `Jwmin=object()` /
  `Jwmax=object()` placeholders in `tests/geo/test_banana_objective_modules.py`
  (helper `_default_geometric_parity_kwargs` and one Stage 2 ALM test) and
  `tests/geo/test_banana_modularization_parity.py`. Re-ran the focused suite
  (test_banana_objective_modules.py + test_banana_modularization_parity.py +
  test_single_stage_alm_integration.py + test_single_stage_example.py) =>
  513/513 pass.
- **Lens 1 (project-guidance compliance) — boozer findings:** STALE. Both
  findings reference `examples/single_stage_optimization/banana_opt/boozer_finite_current.py`
  and `tests/geo/test_boozersurface.py`, which were touched by the Stage 2
  subagent as scope creep and explicitly reverted before review. Diff now
  shows 12 in-scope files only.
- **Lenses 2, 3, 4, 6:** Background agents still running at the time of this
  evidence file; any new findings are integrated in a follow-up section.

## Verdict

PASS WITH ADVISORIES (interim, pending the remaining 4 discovery agents).
The applied Lens 5 fix preserves the Stage 2 penalty/ALM contract symmetry
(the penalty path still constructs and uses `Jwmin`/`Jwmax`; only the ALM
path no longer pretends to require them). All checklist items in the
plan's section 4 are implemented:

- Phase A — Verified current-tree baseline.
- Phase B — Shared constants added; schema artifact extension deferred per
  plan section 4 Phase B caveat.
- Phase C — Stage 2 penalty extended.
- Phase D — Stage 2 ALM extended.
- Phase E — Single Stage penalty extended; Single Stage ALM unchanged.
- Phase F — Focused regression tests added; existing suites extended.
- Phase G — Validation commands executed; this evidence file recorded.

## Reproduction

```bash
# From repo root, on branch surrogate-confinement-v2, head c353f5dae:

# 1. Apply the parity changes (or check out the resulting commit).

# 2. Focused regression sweep
python -m pytest \
    tests/geo/test_banana_objective_modules.py \
    tests/geo/test_single_stage_alm_integration.py \
    tests/geo/test_single_stage_example.py \
    tests/geo/test_banana_modularization_parity.py \
    tests/geo/test_alm_utils.py \
    tests/geo/test_ishw_deliverables.py \
    tests/geo/test_boozersurface.py \
    -q

# 3. Lint
python -m ruff check \
    examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
    examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
    examples/single_stage_optimization/banana_opt \
    tests/geo

# 4. Whitespace
git diff --check

# 5. Full tests/geo/ suite (long, used as final regression gate)
python -m pytest tests/geo/ -q
```

Production-scale smokes (full Stage 2 and Single Stage runs) are tens of
minutes and hours respectively per `AGENTS.md`. They are confirmation work
rather than the validation gate for this contract change. Schedule them on
the standard nightly cadence once this change is merged.

## Artifacts

- `docs/jhalpern30_external_parity_impl_plan_2026-05-12.md` — the plan.
- This file — implementation snapshot.
- `git log surrogate-confinement-v2` after merge will carry the parity
  commit SHA for direct reproduction.
