# jhalpern30 External-Parity Implementation Plan

**Date:** 2026-05-12
**Branch:** `surrogate-confinement-v2`
**Status:** Implemented and validated against the current tree, external drivers, and official docs
**Reference external drivers:** `/Users/suhjungdae/code/columbia/banana_drivers/jhalpern30/stage2.py`, `singlestage.py`

---

## 0. Review Corrections

This file replaces the earlier draft. The prior version had these executable-plan bugs:

- It said Stage 2 has no ALM machinery. Current `STAGE_2/banana_coil_solver.py` has both `--constraint-method=penalty` and `--constraint-method=alm`, so Stage 2 parity terms must be wired into both paths.
- It used the wrong `CurveSelfIntersect` call shape. The repo API is `CurveSelfIntersect(curve, minimum_distance, neighbor_skip=..., normalize=False)`, with diagnostic `shortest_self_distance()`.
- It assumed a repo-level `BANANA_ORDER`. Current Stage 2 resolves the actual order at runtime from `new_banana_curve.order`.
- It proposed default-on compatibility opt-out flags. This violates the strict no-fallback contract. Do not add legacy behavior switches.
- It proposed a generic `alm_owned_constraints` set for Single Stage. Current code already has the needed split: penalty mode uses `evaluate_total_objective`, ALM mode uses `evaluate_alm_objective`.
- It missed the `JCurveLengthMin` wrapper path in `SINGLE_STAGE/single_stage_banana_example.py::build_total_objective`. The current tree already contains the wrapper fix and focused regression test; keep them as Phase A's verified baseline.

---

## 1. Verified Current State

| Area | Current repo state | External reference | Required action |
|---|---|---|---|
| Stage 2 length floor | `hardware_contracts.py` and artifact hardware evaluation know `coil_length_min`, but Stage 2 penalty JF only uses max-length `QuadraticPenalty(Jls, LENGTH_TARGET, "max")`; Stage 2 ALM also constrains only the upper length bound. | `stage2.py:518-520`, `stage2.py:578-579` | Add lower-bound length penalty to Stage 2 penalty and ALM paths. |
| Stage 2 width min/max | `ProjectedEllipseWidth` exists and Single Stage constructs it; Stage 2 does not construct width objectives or measure final width. | `stage2.py:530-532`, `stage2.py:583` | Add width min/max objectives, ALM constraints, final metadata, and artifact hardware fields. |
| Stage 2 self-distance | `CurveSelfIntersect` exists and Stage 2 only has post-hoc `is_self_intersecting(new_banana_curve)`. | `stage2.py:534`, `stage2.py:584` | Add differentiable self-distance objective, ALM constraint, final metadata, and consistency test against post-hoc topology. |
| Single Stage width weighted path | `JCoilWidth` is constructed and ALM owns `width_min`/`width_max`; penalty-mode `evaluate_total_objective` does not include weighted width terms. | `singlestage.py:858` | Add weighted width terms only to Single Stage penalty mode. |
| Single Stage self-intersect weighted path | `JCurveSelfIntersect` is constructed and ALM owns `self_intersect`; penalty-mode `evaluate_total_objective` does not include weighted self-distance. | `singlestage.py:859` | Add weighted self-distance only to Single Stage penalty mode. |

Repo-superior behavior that must remain unchanged:

| Preserve | Current location | Reason |
|---|---|---|
| Failed-trial gradient handling | `SINGLE_STAGE/single_stage_banana_example.py::evaluate_search_step` | Rejected trials return elevated objective with copied accepted gradient, avoiding the old `-dJ` BFGS corruption path. |
| Finite-current Boozer plumbing | `BoozerSurfaceFiniteI`, `RefinedBoozerResidual`, `BoozerResidualExact` call path | More general than the external vanilla Boozer path. |
| Stage 2 coil-surface spacing | `STAGE_2/banana_coil_solver.py` constructs `Jcsdist = CurveSurfaceDistance(...)` | Keeps repo-specific coil-surface clearance in Stage 2. |
| Stage 2 Boozer/iota hot loop | `banana_opt/stage2_objectives.py`, `--stage2-iota-mode={soft,alm}` | Repo-only capability; do not rewrite while adding geometric terms. |

---

## 2. API and Math Grounding

Official docs and local APIs support the following contract:

- `QuadraticPenalty(obj, cons, f)` computes `0.5 * f(obj.J() - cons)^2`; `f="min"` is the lower-bound hinge and `f="max"` is the upper-bound hinge.
- SIMSOPT optimizables with `J()` and `dJ()` compose by scalar multiplication and addition, so these terms belong in the existing objective sum.
- `SquaredFlux(..., definition="normalized")` remains the Stage 2 flux objective; do not rewrite the Boozer/iota loop while adding geometric parity terms.
- SciPy L-BFGS-B `maxcor` belongs in `minimize(..., options={...})`; do not add deprecated `disp` plumbing.
- `ProjectedEllipseWidth(curve, R_winding, a_winding, Z_winding=0.0, scale=..., epsilon=...)` returns a width in meters and wraps cleanly in `QuadraticPenalty`.
- `CurveSelfIntersect(curve, minimum_distance, neighbor_skip=3, normalize=False)` exposes `shortest_self_distance()`.
- The self-distance threshold that matches external is `1.0 / MAX_CURVATURE_INV_M == 0.01 m`.
- The neighbor skip must be computed from the actual runtime curve order: `int(BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR * banana_curve.order)`.

---

## 3. Design Decisions

### 3.1 No Legacy Fallback Flags

Do not add `--stage2-length-min-penalty`, `--stage2-width-penalty`, `--stage2-selfint-penalty`, `--single-stage-weighted-width-penalty`, or `--single-stage-weighted-selfint-penalty`.

The new parity terms are part of the strict objective contract. Old byte-identical replays are out of scope; run historical commits for historical behavior.

### 3.2 Weight Knobs Are Allowed

Keep weight knobs because the repo already exposes scalarization weights (`--length-weight`, `--cc-weight`, `--curvature-weight`) and because calibration is an optimization concern, not a fallback.

Add only:

- `--stage2-width-weight`, env `STAGE2_WIDTH_WEIGHT`, default `STAGE2_WIDTH_WEIGHT_DEFAULT`.
- `--stage2-selfint-weight`, env `STAGE2_SELF_INTERSECT_WEIGHT`, default `STAGE2_SELF_INTERSECT_WEIGHT_DEFAULT`.
- `--single-stage-width-weight`, env `SINGLE_STAGE_WIDTH_WEIGHT`, default `SINGLE_STAGE_WIDTH_WEIGHT_DEFAULT`.
- `--single-stage-selfint-weight`, env `SINGLE_STAGE_SELF_INTERSECT_WEIGHT`, default `SINGLE_STAGE_SELF_INTERSECT_WEIGHT_DEFAULT`.

Initial defaults: `1.0` for each new weight. This is 100x below the external `1e2` width/self weights and matches this repo's lower Stage 2 geometric calibration. Final calibration is validation work, not a structural prerequisite.

### 3.3 Stage 2 Must Cover Penalty and ALM

Stage 2 has two objective routes:

- Penalty route: `JF = weighted sum`; add length-min, width-min/max, and self-distance directly.
- ALM route: `BASE_OBJECTIVE = SQUARED_FLUX_WEIGHT * Jf`; add length-min, width-min/max, and self-distance as inequality constraints in `banana_opt/stage2_objectives.py`.

Do not add these terms to only one route. That would create a downstream regression where `run_stage2_alm.py` and direct Stage 2 penalty runs enforce different hardware contracts.

### 3.4 Single Stage Must Preserve ALM Ownership

Single Stage already passes width and self-intersection to `evaluate_alm_objective`, and the ALM constraint-name registry includes `width_min`, `width_max`, and `self_intersect`. Only penalty mode needs weighted terms:

- Extend `banana_opt/single_stage_objectives.py::build_total_objective` and `evaluate_total_objective` with optional `JCoilWidth`, `width_weight`, `JCurveSelfIntersect`, and `selfint_weight`.
- Extend the `single_stage_banana_example.py` wrapper and penalty-mode `evaluate_search_objective` call to pass those terms.
- Do not route these weighted terms through `evaluate_alm_objective`.

---

## 4. Implementation Checklist

### Phase A - Verified Current-Tree Baseline

- [x] Verify `SINGLE_STAGE/single_stage_banana_example.py::build_total_objective` accepts and forwards `JCurveLengthMin`.
- [x] Verify `tests/geo/test_single_stage_example.py::test_build_total_objective_forwards_length_min_term` covers that wrapper path.

### Phase B - Shared Constants and Schema

- [x] In `banana_opt/hardware_contracts.py`, add:
  - `BANANA_SELF_INTERSECT_MIN_DISTANCE_M = 1.0 / MAX_CURVATURE_INV_M`
  - `BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR = 1.5`
  - `STAGE2_WIDTH_WEIGHT_DEFAULT = 1.0`
  - `STAGE2_SELF_INTERSECT_WEIGHT_DEFAULT = 1.0`
  - `SINGLE_STAGE_WIDTH_WEIGHT_DEFAULT = 1.0`
  - `SINGLE_STAGE_SELF_INTERSECT_WEIGHT_DEFAULT = 1.0`
- [x] In `banana_opt/hardware_constraint_schema.py`, update `width_min`, `width_max`, and `self_intersect` `applies_to` only after the implementation writes measured values in the penalty/artifact path. Do not advertise artifact support without artifact fields.

### Phase C - Stage 2 Penalty Objective

- [x] Import `ProjectedEllipseWidth`, `CurveSelfIntersect`, and the new constants in `STAGE_2/banana_coil_solver.py`.
- [x] Add `--stage2-width-weight` and `--stage2-selfint-weight`; do not add boolean penalty toggles.
- [x] Construct:
  - `Jlsmax = QuadraticPenalty(Jls, LENGTH_TARGET, "max")`
  - `Jlsmin = QuadraticPenalty(Jls, COIL_LENGTH_MIN_FRACTION * LENGTH_TARGET, "min")`
  - `Jw = ProjectedEllipseWidth(new_banana_curve, BANANA_WINDING_SURFACE_MAJOR_RADIUS_M, BANANA_WINDING_MINOR_RADIUS_M)`
  - `Jwmin = QuadraticPenalty(Jw, BANANA_WIDTH_MIN_M, "min")`
  - `Jwmax = QuadraticPenalty(Jw, BANANA_WIDTH_MAX_M, "max")`
  - `Jself = CurveSelfIntersect(new_banana_curve, BANANA_SELF_INTERSECT_MIN_DISTANCE_M, neighbor_skip=int(BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR * new_banana_curve.order))`
- [x] Add to penalty `JF`:
  - `LENGTH_WEIGHT * (Jlsmax + Jlsmin)`
  - `args.stage2_width_weight * (Jwmin + Jwmax)`
  - `args.stage2_selfint_weight * Jself`
- [x] Add print/log metadata for raw width, width penalties, self-distance penalty, `shortest_self_distance()`, and thresholds.
- [x] Add final result fields and artifact-state fields for width and self-distance so `hw-audit` can verify the same contract that optimization used.

### Phase D - Stage 2 ALM Constraints

- [x] Extend `STAGE_2/banana_coil_solver.py::stage2_alm_constraint_names` and `banana_opt/stage2_objectives.py::_stage2_constraint_names` to include:
  - `coil_length_min`
  - `width_min`
  - `width_max`
  - `self_intersect`
- [x] Extend `banana_opt/stage2_objectives.py::evaluate_stage2_alm_problem` arguments to receive the raw length-min threshold, `Jw`, and `Jself`; keep `Jwmin`/`Jwmax` in the penalty path only.
- [x] Implement signed constraints:
  - `coil_length_min`: `length_min_target - coil_length`, gradient `-length_grad`.
  - `width_min`: `BANANA_WIDTH_MIN_M - Jw.J()`, gradient `-dJw`.
  - `width_max`: `Jw.J() - BANANA_WIDTH_MAX_M`, gradient `dJw`.
  - `self_intersect`: `Jself.J()`, gradient `dJself`, threshold `0.0`.
- [x] Thread metadata, threshold overrides, raw constraints, feasibility values, activity tolerances, and ALM result payloads through the same SSOT helpers used by existing Stage 2 constraints.
- [x] Keep `--stage2-iota-mode=soft|alm` behavior unchanged except for receiving the expanded geometric constraint set in ALM mode.

### Phase E - Single Stage Penalty Objective

- [x] Extend `banana_opt/single_stage_objectives.py::build_total_objective` with optional weighted width/self terms:
  - `JCoilWidth=None`
  - `width_weight=0.0`
  - `JCurveSelfIntersect=None`
  - `selfint_weight=0.0`
- [x] In `build_total_objective`, use SSOT constants and `QuadraticPenalty(JCoilWidth, BANANA_WIDTH_MIN_M, "min")`, `QuadraticPenalty(JCoilWidth, BANANA_WIDTH_MAX_M, "max")`; add `selfint_weight * JCurveSelfIntersect`.
- [x] Extend `evaluate_total_objective` diagnostics to expose `coil_width`, width thresholds, width penalty values, `self_intersect_penalty`, and `self_intersect_threshold`.
- [x] Extend `SINGLE_STAGE/single_stage_banana_example.py` parse args and wrapper plumbing for the two new weights.
- [x] Pass `JCoilWidth`, `JCurveSelfIntersect`, and weights only in the penalty-mode `evaluate_search_objective` branch.
- [x] Keep ALM mode routed through `evaluate_alm_objective`; do not double-count width/self in ALM base objective.

### Phase F - Tests

- [x] Add Stage 2 penalty tests for length-min, width-min, width-max, and self-distance objective inclusion and gradients.
- [x] Add Stage 2 ALM tests proving `coil_length_min`, `width_min`, `width_max`, and `self_intersect` appear in ordered ALM payloads with correct signs, scales, thresholds, and gradients.
- [x] Add artifact tests proving final Stage 2 metadata/hardware status includes width and self-distance fields when the terms are active.
- [x] Add Single Stage penalty-mode tests proving weighted width/self terms change total value and gradient.
- [x] Add Single Stage ALM-mode tests proving weighted width/self terms are not added to the ALM base objective.
- [x] Add a smoke test that constructs real `ProjectedEllipseWidth` and `CurveSelfIntersect` objectives with the repo's banana test curve.

### Phase G - Validation

Run baseline before implementation, then repeat after each phase:

```bash
python -m pytest tests/geo/test_banana_objective_modules.py -q
python -m pytest tests/geo/test_single_stage_alm_integration.py -q
python -m pytest tests/geo/test_single_stage_example.py -q
```

After implementation:

```bash
python -m pytest tests/geo/ -q
python -m ruff check examples/single_stage_optimization/STAGE_2/banana_coil_solver.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py examples/single_stage_optimization/banana_opt tests/geo
git diff --check
```

Production smokes to record in `docs/parity_evidence/`:

- Stage 2 penalty run with new terms active.
- Stage 2 ALM run with new terms active.
- Single Stage penalty run proving weighted width/self terms contribute.
- Single Stage ALM run proving no weighted double-count.

---

## 5. Done When

- [x] All implementation checklist items are complete.
- [x] Stage 2 penalty and ALM paths enforce the same length-min, width, and self-distance contracts.
- [x] Single Stage penalty path includes width/self weighted gradients; Single Stage ALM path keeps width/self as ALM constraints only.
- [x] `hw-audit` sees the same measured width/self-distance fields that optimization used.
- [x] Focused tests and `tests/geo/` pass with no new failures relative to the recorded baseline.
- [x] `docs/parity_evidence/` contains dated smoke records with git SHA, command, key metrics, and pass/fail status.
