# ALM Normalization Audit v2 — 2026-05-08

## Summary

Recent commits `a169f296a` (scale-floor provenance + threshold rejection) and
`2e9acced2` (iota-threshold operator-units doc clarification) closed two of the
three findings raised in `.alm_audit/normalization_review.md` (F1 zero/negative
threshold acceptance, F2 missing floor source label) and addressed the v1
FIX_PLAN H1/M7 items at the metadata-constructor and schema layers. F3
(`_normalize_alm_run_inputs` length check) was also closed by routing
`initial_multipliers` through `validate_initial_multipliers`
(`alm_utils.py:2272`, called at `:2320`).

This audit identifies **eight new findings**. Two are HIGH severity and reflect
incomplete delivery of the v1 H1/M10 fixes:

- **F1 (HIGH)** — `run_single_stage_thresholded_physics_alm.py` defaults
  `--alm-length-penalty-threshold=0.0`, which the new
  `require_positive_alm_threshold` gate now rejects. Default invocation of the
  thresholded-physics runner is broken at the boundary.
- **F2 (HIGH)** — Help text and `README.md` describe
  `--alm-iota-penalty-threshold` as `(iota - target)**2 * iotas_weight`, but
  the live constraint is `0.5 * (iota - target)**2 ≤ T` with **no** `iotas_weight`
  factor (because `Jiota = QuadraticPenalty(iota, target)` and the constraint
  is on `Jiota.J()`, not `iotas_weight * Jiota`). Operators following the
  documented formula will set thresholds **2× too large** for a target
  deviation, and rescaling by `iotas_weight` is meaningless for the constraint.

Three are MEDIUM:

- **F3 (MEDIUM)** — `validate_alm_cli_args` still does not run the four physics
  threshold checks (the `weighted_sum + alm` blind spot called out in v1
  FIX_PLAN H1 implementation checklist line 143 was not applied). A user
  passing a stale `--alm-length-penalty-threshold=0.0` from a prior config
  while running `weighted_sum + alm` (or any non-thresholded formulation that
  still wires the flag through) bypasses the gate.
- **F4 (MEDIUM)** — `scale_floor_applied: bool` exists on
  `ALMConstraintMetadata` (`hardware_constraint_schema.py:59`) but is never
  serialized in `alm_constraint_metadata_payload`
  (`hardware_constraint_schema.py:413-438`). Provenance survives only as a
  `:floored` source-string suffix; downstream consumers that want a typed
  boolean (filtering, dashboards) cannot get one without parsing strings.
- **F5 (MEDIUM)** — Stage 2 hard signal is normalized via raw `np.asarray /
  constraint_scales` at `stage2_objectives.py:1918-1921`, bypassing
  `normalize_alm_constraint_signals`'s shape and positivity guards. It is
  arithmetically equivalent in the happy path but breaks the SSOT for
  normalization invariants.

Three are LOW:

- **F6 (LOW)** — `_resolved_threshold` (`hardware_constraint_schema.py:672-683`)
  passes user override values through without positivity validation. Defense
  is one layer downstream in `_resolved_alm_scale_with_provenance`. Belt-and-
  suspenders gap, not a live bug because `require_positive_alm_threshold`
  catches it.
- **F7 (LOW)** — `run_stage2_alm.py` does not call `validate_alm_cli_args`. The
  ALMSettings `__post_init__` covers most numeric flags, but
  `--alm-distance-smoothing` and `--alm-curvature-smoothing` are validated
  only by `validate_alm_cli_args` (`alm_utils.py:429-432`). Stage 2 silently
  accepts negative or zero smoothing.
- **F8 (LOW)** — Stage 2's iota tolerance becomes the activity tolerance for
  the iota-penalty constraint via `max(stage2_iota_penalty_threshold(tol),
  _SMOOTHING_EPS)` (`stage2_objectives.py:1305`). The activity tolerance is
  thus the same as the scale (after `0.5 * tol²` conversion). After
  normalization, `activity_tolerance / scale = 1.0` — the constraint is
  considered active until violation drops to zero. This is unusually large
  compared with hardware constraints (`ALM_ACTIVITY_TOLERANCE_FRACTION =
  1e-3`) and may keep iota constraints active longer than intended. Possibly
  by design but undocumented.

Confirmed correct: all three F-findings from `.alm_audit/normalization_review.md`,
block-penalty removal completeness, surrogate vs hard normalized with the same
scale (sign-mismatch invariant under scaling), gradient/value normalization
consistency, sign preservation under positive-scale division, and per-constraint
ρ removal.

## Methodology

1. Read prior audit `.alm_audit/normalization_review.md` and v1 FIX_PLAN to
   establish what was reportedly closed.
2. Inspected `git show a169f296a` and `git show 2e9acced2` to see what landed.
3. Walked the normalization call graph end-to-end:
   `alm_utils.normalize_alm_constraint_{signals,grads}` →
   `_physics_alm_metadata` / `_stage2_alm_constraint_metadata` /
   `_resolved_alm_scale_with_provenance` →
   `require_positive_alm_threshold` → `resolve_alm_scale_with_provenance` →
   metadata payload → artifact emission.
4. Verified F1, F2, F3 of the v1 audit are closed by reading the relevant
   commit hunks against current HEAD `e7b836464`.
5. Cross-checked CLI defaults in `run_single_stage_thresholded_physics_alm.py`
   against the new `require_positive_alm_threshold` gate.
6. Traced `Jiota` semantics through `build_single_stage_iota_objective` →
   `QuadraticPenalty.__init__` (`src/simsopt/objectives/utilities.py:166-194`)
   to confirm the `0.5 *` factor.
7. Audited test coverage in `tests/geo/test_alm_utils.py`,
   `tests/geo/test_banana_objective_modules.py`, and
   `tests/geo/test_alm_fixture_benchmarking.py` for normalization invariance,
   sign preservation, and provenance pinning.

## Findings

### F1: Default `--alm-length-penalty-threshold=0.0` runtime-rejected after H1 [HIGH]

- File: `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py:43`
- Code:
  ```python
  DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0
  ```
  argparse hookup at `:201-205`:
  ```python
  parser.add_argument(
      "--alm-length-penalty-threshold",
      type=float,
      default=DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD,
  )
  ```
  forwarded to inner script at `:313-314`:
  ```python
  "--alm-length-penalty-threshold",
  str(args.alm_length_penalty_threshold),
  ```
- Bug: `validate_single_stage_alm_formulation_args` now calls
  `require_positive_alm_threshold(flag_name, value)` for every required
  threshold (`single_stage_banana_example.py:2994-2995`), and
  `require_positive_alm_threshold` rejects `0.0`
  (`alm_utils.py:381-385`):
  ```python
  if not np.isfinite(value_f) or value_f <= 0.0:
      raise ValueError(
          f"ALM threshold {name!r} must be a finite positive value; got {value!r}"
      )
  ```
  The thresholded-physics runner therefore aborts on its own default.
- Why: The `0.0` default predates commit `a169f296a`, which tightened the
  acceptance from `< 0.0` to `<= 0.0`. The runner was not updated. Search
  confirms the literal is also propagated into
  `run_stage2_to_single_stage.py:539`:
  ```python
  alm_length_penalty_threshold=(
      recovery_runner.DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD
      if args.alm_length_penalty_threshold is None
      else args.alm_length_penalty_threshold
  ),
  ```
  so the recovery path is also broken.
- Impact: HIGH. Default-args invocation of the canonical thresholded-physics
  runner fails immediately at validation. CI / smoke-test scripts that don't
  override this flag stop dead. There is no scaling-math corruption (the
  validator catches it before normalization), but the production-grade
  guarantee is broken: a default-args run of a documented entry point cannot
  start.
- Suggested fix: Change the default to a small positive value that matches
  the operator-meant overshoot tolerance after the
  `0.5 * max(L − L_target, 0)²` semantics. Per the constraint formula, an
  overshoot tolerance of 1 cm (0.01 m) corresponds to a threshold of
  `0.5 * 0.01² = 5e-5` m². Recommend `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD
  = 5.0e-5` (or tighter — 1 mm overshoot would be `5e-7`). Add a CLI test
  that runs `parse_args` with no arguments and checks the threshold is
  positive.

### F2: `--alm-iota-penalty-threshold` operator-unit formula is wrong by factor of 2 and incorrectly includes `iotas_weight` [HIGH]

- File: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1405-1416`
- Code (help text):
  ```python
  help=(
      "thresholded_physics-mode upper bound for the Jiota penalty objective. "
      "Units: squared-penalty units, NOT iota deviation. The constraint is "
      "Jiota_penalty = (iota - iota_target)**2 * iotas_weight <= threshold, "
      "so this threshold scales with iotas_weight. To target a desired iota "
      "deviation d, set --alm-iota-penalty-threshold = (d**2) * iotas_weight "
      "(e.g. iotas_weight=1.0 with target deviation 0.01 -> threshold 1e-4). "
      ...
  ),
  ```
  Mirrored in `examples/single_stage_optimization/README.md:574-593`:
  ```
  --alm-iota-penalty-threshold (single-stage --alm-formulation thresholded_physics)
    The constraint is Jiota_penalty <= threshold where
    Jiota_penalty = (iota - iota_target)^2 * iotas_weight.
  --alm-iota-penalty-threshold = d**2 * iotas_weight
  Example: --alm-iota-penalty-threshold = 0.01**2 * 1.0 = 1e-4
  ```
  Live constraint construction:
  - `single_stage_banana_example.py:3709-3714`:
    ```python
    Jiota = build_single_stage_iota_objective(
        surface_iota_terms[-1],
        iota_target,
        ...
    )
    ```
  - `single_stage_banana_example.py:3495-3496` (target mode used by ALM):
    ```python
    if goal_mode == "target":
        return QuadraticPenalty(surface_iota_term, iota_target)
    ```
  - `src/simsopt/objectives/utilities.py:185-194`:
    ```python
    def J(self):
        ...
        elif self.f == 'identity':
            return 0.5*diff**2
    ```
  - `single_stage_objectives.py:933-937`:
    ```python
    "iota_penalty": _objective_upper_bound_constraint(
        Jiota,
        iota_penalty_threshold,
        objective_optimizable,
    ),
    ```
  - `single_stage_objectives.py:117-124` (signed value computation):
    ```python
    def _objective_upper_bound_constraint(objective, threshold, objective_optimizable):
        ...
        signed_value = float(objective.J()) - float(threshold)
    ```
- Bug: The constraint actually evaluated by ALM is
  `Jiota.J() - threshold ≤ 0` where `Jiota.J() = 0.5 * (iota - iota_target)²`.
  No `iotas_weight` enters; that weight is part of `JF` (the soft objective),
  not the ALM constraint. The constraint is therefore equivalent to
  `|iota - iota_target| ≤ sqrt(2 * threshold)`. For target deviation `d`
  the correct threshold is `0.5 * d²`, **not** `d² * iotas_weight`.
- Why: The help text and README appear to confuse the soft-objective
  contribution (`iotas_weight * Jiota`) with the ALM constraint (which is on
  the un-scaled `Jiota`). Compare to Stage 2's correct formulation:
  `stage2_objectives.py:230-234`:
  ```python
  def stage2_iota_penalty_threshold(iota_tolerance: float) -> float:
      tolerance = float(iota_tolerance)
      if tolerance <= 0.0:
          raise ValueError("Stage 2 iota tolerance must be positive.")
      return 0.5 * tolerance * tolerance
  ```
  Stage 2 includes the `0.5` factor; single-stage docs omit it.
- Impact: HIGH. An operator following the documented formula and intending
  iota deviation `d = 0.01` (a typical stellarator iota tolerance) sets
  `--alm-iota-penalty-threshold = 1e-4` per the README example. The
  effective deviation tolerance is then
  `sqrt(2 * 1e-4) = 0.0141` — about 1.4× the intended limit. If the user
  also tunes `--iotas-weight`, they will rescale by a factor that has no
  effect on the constraint, mis-attributing iota tightness changes to the
  weight. Combined with the prior assumption that the README is correct,
  this can produce pareto-frontier comparisons where iota tightness varies
  systematically with `iotas_weight` for reasons unrelated to optimization.
- Suggested fix: Update both the help string at
  `single_stage_banana_example.py:1405-1416` and the README block at
  `examples/single_stage_optimization/README.md:574-593` to:
  ```
  Jiota_penalty = 0.5 * (iota - iota_target)**2 <= threshold
  Operator conversion: target deviation d => threshold = 0.5 * d**2
  Example: target deviation 0.01 => threshold = 5e-5
  ```
  Confirm `iotas_weight` is **not** part of the constraint and remove all
  references to it in the threshold formula. Add a regression test that
  verifies for a fixed `Jiota.J() = 0.5 * d²` the constraint is exactly at
  the boundary when `--alm-iota-penalty-threshold = 0.5 * d²`.

### F3: `validate_alm_cli_args` does not enforce physics-threshold positivity for non-`thresholded_physics` modes [MEDIUM]

- File: `examples/single_stage_optimization/alm_utils.py:389-432`
- Code:
  ```python
  def validate_alm_cli_args(args) -> None:
      if args.alm_max_outer_iters <= 0:
          raise ValueError("--alm-max-outer-iters must be positive")
      ...
      if distance_smoothing is not None and distance_smoothing <= 0.0:
          raise ValueError("--alm-distance-smoothing must be positive")
  ```
  No reference to `alm_qs_threshold`, `alm_boozer_threshold`,
  `alm_iota_penalty_threshold`, or `alm_length_penalty_threshold`.
  The check exists only inside
  `single_stage_banana_example.py:2948-2995`
  (`validate_single_stage_alm_formulation_args`), which gates on
  `args.alm_formulation == "thresholded_physics"` and returns early
  otherwise.
- Bug: v1 FIX_PLAN H1 implementation checklist line 143 explicitly required
  "Add a top-level CLI validator hook in `validate_alm_cli_args`
  (`alm_utils.py:318`) that runs the helper for the four ALM threshold flags
  **regardless of formulation** (so `weighted_sum + alm` is also covered)."
  This step was not implemented in commit `a169f296a`.
- Why: H1 was partially landed: the helper exists, the metadata constructors
  call it, and `thresholded_physics` mode validates the four flags. But the
  belt-and-suspenders guard at the top-level CLI boundary was skipped.
- Impact: MEDIUM. Today, only `thresholded_physics` consumes the four
  thresholds, so a stale `0.0` value passing through `weighted_sum + alm` is
  silently dropped (single-stage doesn't add physics constraints in
  `weighted_sum`). However, the contract is "any explicit ALM threshold the
  CLI accepts must be a positive float," not "any threshold reaching the
  metadata constructor must be positive." If a future formulation routes any
  of these four flags into a different metadata path, the gate will be
  missing.
- Suggested fix: Add a helper-call loop in `validate_alm_cli_args` for the
  four physics-threshold flags, gated on `getattr(args, name, None)`:
  ```python
  for flag_name, attr in (
      ("--alm-qs-threshold", "alm_qs_threshold"),
      ("--alm-boozer-threshold", "alm_boozer_threshold"),
      ("--alm-iota-penalty-threshold", "alm_iota_penalty_threshold"),
      ("--alm-length-penalty-threshold", "alm_length_penalty_threshold"),
  ):
      value = getattr(args, attr, None)
      if value is not None:
          require_positive_alm_threshold(flag_name, value)
  ```
  Add a unit test in `tests/geo/test_alm_utils.py` named
  `test_validate_alm_cli_args_rejects_zero_thresholds_in_weighted_sum_mode`
  (the FIX_PLAN H1 test case that was never written; line 165 of v1
  FIX_PLAN).

### F4: `scale_floor_applied` not exposed in `alm_constraint_metadata_payload` [MEDIUM]

- File: `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py:413-438`
- Code:
  ```python
  def alm_constraint_metadata_payload(
      constraint_names: Iterable[str],
      metadata_by_name: Mapping[str, ALMConstraintMetadata],
  ) -> dict[str, object]:
      ordered_metadata = [metadata_by_name[name] for name in constraint_names]
      return {
          "constraint_scales": [metadata.scale for metadata in ordered_metadata],
          "constraint_blocks": [metadata.block for metadata in ordered_metadata],
          "constraint_scale_sources": [metadata.source for metadata in ordered_metadata],
          ...
          "raw_thresholds": [metadata.raw_threshold for metadata in ordered_metadata],
      }
  ```
  No `scale_floor_applied` entry.
- Bug: The `scale_floor_applied: bool` field added in commit `a169f296a`
  (`hardware_constraint_schema.py:59`) is set inside metadata constructors
  but never serialized to the artifact payload. Provenance reaches artifacts
  only as a `:floored` suffix on the source string
  (`hardware_constraint_schema.py:74-76`). A consumer that wants a typed
  boolean column for filtering / dashboards / regression checks must parse
  the source string.
- Why: The commit message for `a169f296a` documents the schema field
  addition and the `:floored` suffix, but the payload exporter wasn't
  updated. v1 FIX_PLAN M7 implementation checklist line 631 said
  "[ ] Update artifact payload emission to include the `scale_floor_applied`
  field where consumers read it."
- Impact: MEDIUM. The information is recoverable from the source string, so
  no operator-facing data is lost. But the typed-payload contract that other
  metadata fields enjoy (e.g. `raw_thresholds`, `constraint_scales`) is
  inconsistent. Tooling that emits Parquet/JSONL columns from this payload
  silently drops the boolean flag.
- Suggested fix: Add `"scale_floor_applied": [metadata.scale_floor_applied
  for metadata in ordered_metadata]` to the payload dict at line 437. Update
  any downstream readers — `alm_utils.py:1346` enumerates the
  pass-through string list and would need a parallel bool list.
  Add a test in `tests/geo/test_banana_objective_modules.py` that asserts
  the payload contains `scale_floor_applied` and that it correctly reflects
  the per-constraint floor decision.

### F5: Stage 2 hard-signal normalization bypasses `normalize_alm_constraint_signals` shape and positivity guards [MEDIUM]

- File: `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1918-1921`
- Code:
  ```python
  normalized_hard_signed_constraint_values = (
      np.asarray(sanitized_hard_signed_constraint_values, dtype=float)
      / constraint_scales
  )
  ```
- Bug: This divides a hard-signal vector by `constraint_scales` directly,
  without going through `normalize_alm_constraint_signals` (`alm_utils.py:510-532`).
  The latter validates `scales` finite-and-positive
  (`alm_utils.py:517-518`) and that all input shapes match the scale shape
  (`alm_utils.py:522-527`). The direct division skips both checks, instead
  relying on the in-broadcast division to raise on a length mismatch only
  when one happens to be a multiple of the other.
- Why: The surrogate signal goes through `normalize_alm_constraints` (which
  internally calls `normalize_alm_constraint_signals`) at `:1901-1907`, but
  hard signal normalization was inlined. The other Stage 2 normalization at
  `:1934-1939` (dual_update + feasibility) goes through the helper.
- Impact: MEDIUM. In the happy path the math is identical. But:
  1. SSOT for normalization is lost — three normalization sites (helper,
     direct division, raw assignment) violate DRY.
  2. A scale-array containing a 0 / inf / nan that escaped the metadata
     constructor would bypass the check here. Today the metadata
     constructor cannot emit such a scale (validators present), but the
     defense-in-depth is asymmetric.
  3. A test that mocks the metadata to inject an invalid scale would
     surface a different error (broadcast vs ValueError), making test
     output less useful for diagnosing which guard fired.
- Suggested fix: Replace the direct division with a call to
  `normalize_alm_constraint_signals(sanitized_hard_signed_constraint_values,
  np.maximum(sanitized_hard_signed_constraint_values, 0.0),
  raw_constraint_activity_tolerances, constraint_scales)["normalized_signed_values"]`,
  or extract a new helper `normalize_alm_constraint_signed_array(values,
  scales)` that owns the positivity + shape checks for a single-array
  normalization. Update `tests/geo/test_alm_utils.py` to pin the helper.

### F6: `_resolved_threshold` does not validate user override positivity [LOW]

- File: `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py:672-683`
- Code:
  ```python
  def _resolved_threshold(
      spec: HardwareConstraintSpec,
      threshold_overrides: Mapping[str, float] | None,
  ) -> float:
      if threshold_overrides is not None and spec.name in threshold_overrides:
          return float(threshold_overrides[spec.name])
      if spec.threshold <= 0.0:
          raise ValueError(
              f"Constraint {spec.name!r} has no canonical hardware threshold; "
              "supply an explicit positive threshold override."
          )
      return float(spec.threshold)
  ```
- Bug: When an override is present, the function returns the user value
  unchecked. A negative or zero override is then forwarded to
  `_resolved_alm_scale_with_provenance` (line 290-302), which **does** call
  `require_positive_alm_threshold` (line 294) and raises. So the live path
  is correct, but the failure mode is "raises in scale resolution" instead
  of "raises in threshold resolution" — making the stack trace misleading.
- Why: The scale-resolution path was retrofitted to use
  `require_positive_alm_threshold` in commit `a169f296a` but the upstream
  threshold-resolution path was not. Schema thresholds (the non-override
  branch) are well-controlled (`HARDWARE_CONSTRAINT_SCHEMA` defines them).
  Override values come from user CLI / programmatic callers and are the
  unbounded input.
- Impact: LOW. The validation does happen, just one layer further out. A
  test asserting "an `_resolved_threshold` call with a negative override
  raises" would currently fail.
- Suggested fix: Either (a) call `require_positive_alm_threshold` on the
  override branch in `_resolved_threshold`, matching the schema-zero check
  semantics, or (b) document that `_resolved_threshold` is "raw passthrough"
  and rely on `_resolved_alm_scale_with_provenance` for positivity. (a) is
  the safer choice and matches the SSOT pattern at the metadata
  constructors.

### F7: `run_stage2_alm.py` skips `validate_alm_cli_args` [LOW]

- File: `examples/single_stage_optimization/run_stage2_alm.py` (entire file)
- Code: No reference to `validate_alm_cli_args`. ALMSettings construction
  goes through `build_stage2_alm_settings` at `stage2_objectives.py:534-550`,
  which calls `ALMSettings(...)` and relies on `__post_init__`
  (`alm_utils.py:37-77`) for validation.
- Bug: `ALMSettings.__post_init__` covers ~90% of CLI args (penalty,
  tolerance, trust radius, multiplier_max, history_max_entries) but does
  **not** cover `alm_distance_smoothing` and `alm_curvature_smoothing`,
  which are validated only in `validate_alm_cli_args` at
  `alm_utils.py:429-432`:
  ```python
  if curvature_smoothing is not None and curvature_smoothing <= 0.0:
      raise ValueError("--alm-curvature-smoothing must be positive")
  if distance_smoothing is not None and distance_smoothing <= 0.0:
      raise ValueError("--alm-distance-smoothing must be positive")
  ```
- Why: Stage 2's smoothing values reach
  `single_stage_constraint_activity_tolerances` and feed into the
  `_SMOOTHING_EPS`-clipped activity-tolerance vectors. A non-positive value
  passed to e.g. `softmin_selection_window(distance_smoothing)` is clipped
  by `max(value, _SMOOTHING_EPS)` downstream, but operators get no error
  warning that their smoothing setting was overridden.
- Impact: LOW. Negative smoothing is silently floored. Math doesn't break,
  but operator intent is silently overridden.
- Suggested fix: Either (a) move the smoothing checks into
  `ALMSettings.__post_init__` (smoothing isn't actually a member of
  `ALMSettings`, so this would require expanding the dataclass — out of
  scope), or (b) call `validate_alm_cli_args(args)` in
  `run_stage2_alm.py` before `build_stage2_alm_settings`. (b) is the
  cheaper and more consistent fix.

### F8: Stage 2 iota-penalty activity tolerance equals scale, producing always-active behavior [LOW]

- File: `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1303-1305`
- Code:
  ```python
  if include_iota_penalty:
      tolerances.append(
          max(stage2_iota_penalty_threshold(iota_tolerance), _SMOOTHING_EPS)
      )
  ```
- Bug: For the iota_penalty constraint, the activity tolerance equals
  `0.5 * iota_tolerance²` (the same value used for the scale). After
  normalization, `activity_tolerance_norm = activity_tolerance / scale =
  1.0`. Compare hardware constraints
  (`hardware_constraint_schema.py:312-316`):
  ```python
  def _alm_activity_tolerance_from_scale(spec, scale):
      return float(scale) * float(spec.alm_activity_tolerance_fraction)
  ```
  with `ALM_ACTIVITY_TOLERANCE_FRACTION = 1e-3`. After normalization,
  hardware activity tolerance is `1e-3` (i.e. constraint counted active
  when `c_norm > -1e-3`). The iota constraint, by contrast, is counted
  active when `c_norm > -1.0`, which is essentially always (`c_norm` is
  bounded by definition to be ≤ 1 only if the constraint is satisfied at
  100% of threshold).
- Why: Activity tolerance for hardware uses a fraction of the scale; iota
  uses the whole scale. This yields a 1000× wider activation band after
  normalization for the iota constraint. The constraint is therefore
  always in the active set (per `_kkt_stationarity_norm` at
  `alm_utils.py:2063-2064`), even when iota is far from the target.
- Impact: LOW. Stationarity / KKT diagnostics include the iota constraint
  even when it is well below the threshold. This inflates the active set,
  potentially making `_kkt_stationarity_norm` larger than it should be at
  the converged iterate.
- Suggested fix: Use the same `ALM_ACTIVITY_TOLERANCE_FRACTION` factor
  for the iota constraint:
  ```python
  if include_iota_penalty:
      iota_scale = stage2_iota_penalty_threshold(iota_tolerance)
      tolerances.append(
          max(iota_scale * ALM_ACTIVITY_TOLERANCE_FRACTION, _SMOOTHING_EPS)
      )
  ```
  Verify against test expectations — this changes activity-set inclusion
  for runs where iota is between `(1 − 1e-3) * scale` and `scale`.

## Confirmed-Correct Items

The following normalization invariants and contracts hold under audit:

- **Sign preservation under normalization.** `normalize_alm_constraint_signals`
  (`alm_utils.py:510-532`) and `normalize_alm_constraint_grads`
  (`alm_utils.py:535-544`) divide by `scale`, which is enforced finite and
  strictly positive (`:517-518`, `:537-538`). Sign is preserved by
  construction.
- **Surrogate vs hard share scale.** Single-stage
  (`single_stage_objectives.py:1055-1072`) and Stage 2
  (`stage2_objectives.py:1900-1921`) both normalize surrogate and hard signals
  with the **same** `constraint_scales` array. The
  `surrogate_minus_hard_normalized_gap` diagnostic (`alm_utils.py:948-957`)
  and `surrogate_hard_sign_mismatch_by_constraint`
  (`alm_utils.py:958-961`) detect a true math signal, not a scaling artifact.
- **Gradient and value normalization consistent.** Both helpers divide by
  the same scale array; no double-scaling on the gradient.
- **No double-scaling of penalty.** `_penalty_values` (`alm_utils.py:1677-1687`)
  validates penalty finite-and-positive but does **not** apply scale; the
  `augmented_inequality_objective` formula
  `positive_shift = λ_norm + ρ * c_norm` (`alm_utils.py:459`) uses normalized
  values throughout. The convention "ρ is a normalized-space tunable" is
  applied uniformly.
- **Block-penalty removal is complete.** No live `block_penalt*` control
  code paths remain. Legacy nullable fields persist
  (`alm_utils.py:1233-1241, 2426, 2612, 2878-2880`) — all set to `None` and
  documented as legacy schema in
  `docs/alm_scalar_hardening_block_penalty_removal_plan_2026-05-06.md:85, 147`.
- **Per-constraint vs global penalty.** Production path uses scalar penalty
  (`_handle_alm_dual_update_transition` at `alm_utils.py:3090-3110` against a
  scalar `penalty_argument`). `_penalty_values` accepts a vector for legacy
  test calls only.
- **Provenance partially recorded.** The `:floored` suffix on
  `metadata.source` (`hardware_constraint_schema.py:74-76`) flows through
  `alm_constraint_metadata_payload` at line 421 and out to the artifact
  field `ALM_CONSTRAINT_SCALE_SOURCES`
  (`single_stage_banana_example.py:6088-6090`,
  `stage2_objectives.py:1018-1020`). String-based but recoverable.
- **Zero-floor / nonpositive threshold rejection works for the registered
  constraint paths.** `require_positive_alm_threshold` is called in
  `validate_single_stage_alm_formulation_args`
  (`single_stage_banana_example.py:2994-2995`), `_physics_alm_metadata`
  (`single_stage_objectives.py:503`),
  `_require_explicit_stage2_alm_threshold`
  (`stage2_objectives.py:701`), and
  `_resolved_alm_scale_with_provenance`
  (`hardware_constraint_schema.py:294`). Tests in
  `tests/geo/test_alm_utils.py:1243-1281` and
  `tests/geo/test_banana_objective_modules.py:686-700` lock zero, negative,
  NaN, +Inf rejection.
- **Scale fixed across outer iterations.** Scales depend only on
  user-supplied thresholds and schema constants (no x-dependence).
  Multiplier transformation `λ_raw = λ_norm / scale` is invariant when
  scales are stable.
- **`_normalize_alm_run_inputs` validates initial multipliers.** Routes
  through `validate_initial_multipliers` (`alm_utils.py:2272-2295`,
  `:2320`), which checks shape, finiteness, and non-negativity. Closed F3
  from prior audit.

## Verdict

The April–May 2026 normalization rework is sound at the math layer. Sign,
shape, scaling, and surrogate/hard alignment are all correctly engineered.
The recent (`a169f296a`) provenance + threshold-rejection commit closes most
of the v1 FIX_PLAN H1 / M7 work. Remaining gaps are entry-boundary defaults
and operator-facing documentation.

The two HIGH findings (F1, F2) require operator action: F1 will block
default invocations of the thresholded-physics runner, and F2 sends operators
to set thresholds 2× larger than they intend. Both are surface-level fixes
(default value, documentation) but ship-blocking for production usage.

F3-F8 are operability / SSOT hygiene; none break correctness today.

Recommend land F1 and F2 immediately; schedule F3-F5 in the next normalization
cleanup pass.
