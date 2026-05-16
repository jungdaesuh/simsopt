# ALM Physics Audit v2 — 2026-05-08

Branch: `surrogate-confinement-v2`, HEAD `e7b836464`
Audit scope: physics correctness of the ALM constraint formulation.
Out of scope: ALM iteration math, control flow, JAX/GPU, tests.

## Summary

**Verdict: physics formulation is mostly sound, but two operator-facing bugs introduced or surviving since the prior audit must be fixed before any new run is configured.**

The physical formulas (`c ≤ 0 ⟺ feasible`), gradient signs, frame conventions, and dimensional normalization are all correct in the actual ALM code path — the underlying constraints `0.5·(ι−ι_target)² − thresh ≤ 0`, `|I|−I_max ≤ 0`, `min_dist − actual ≤ 0`, etc. are all correctly written and consistently signed. Stage 2 ↔ single-stage handoff still re-validates via the schema-driven status builder. The recent commit `d61648f50` removed the ACCEPT_OFFSPEC env-flag escape hatches completely (verified with grep — no references remain).

However, the *operator-facing units* of the iota constraint in `--alm-formulation thresholded_physics` mode are documented incorrectly in two places, and one runner default cannot run. These are the live findings:

- **F1 (CRITICAL)**: README and `--alm-iota-penalty-threshold` help text claim the constraint is `(ι − ι_target)² · iotas_weight ≤ threshold` and tell the operator to set `threshold = d² · iotas_weight`. The actual code computes `0.5·(ι − ι_target)² − threshold ≤ 0` and does NOT multiply by `iotas_weight`. With the default `--iotas-weight 100`, an operator following the README to target deviation 0.01 ends up with an effective deviation tolerance of 0.141 — **14× looser** than intended.
- **F2 (HIGH)**: `run_single_stage_thresholded_physics_alm.py` ships with `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0`, but `validate_single_stage_alm_formulation_args` calls `require_positive_alm_threshold(...)` which rejects 0. The runner cannot start with its own defaults.
- **F3 (MEDIUM)**: In `--alm-formulation thresholded_physics` mode, the constraint set contains BOTH `coil_length_upper_bound` (`L − L_target ≤ 0`, m) AND `length_penalty` (`0.5·max(L − L_target, 0)² − thresh ≤ 0`, m²). They enforce overlapping inequalities through two multipliers. Same physics as the prior audit's Finding 1, just relocated — the prior fix only hardened weighted_sum mode.

The prior audit's other minor findings (sub-differential at I=0, `Σ|I_TF|` vs `Σ I_TF` for `G_0`, no rational-surface guard) remain technically present but are not new and remain non-blocking.

## Methodology

1. Read `.alm_audit/physics_review.md` (prior audit verdict).
2. Read `examples/single_stage_optimization/alm_utils.py` (4847 LOC) focused on:
   - `upper_bound_residual` / `lower_bound_residual` / `augmented_inequality_objective` / `normalize_alm_constraint_*` / `require_positive_alm_threshold` (L373–545)
   - `_explicit_raw_signed_constraint_values` and `_surrogate_hard_sign_mismatch` (L678–832)
   - `_extract_stage2_constraint_signal_state` and `_constraint_routing_state` (L1979–2137)
3. Read `banana_opt/single_stage_constraints.py` end-to-end (435 LOC).
4. Read `banana_opt/hardware_constraint_schema.py` end-to-end (841 LOC).
5. Read `banana_opt/single_stage_objectives.py` end-to-end (1171 LOC).
6. Read `banana_opt/stage2_objectives.py` (focused L88–308 iota state, L560–701 constraint metadata, L1740–2067 ALM problem evaluation).
7. Read `banana_opt/surface_mode_contracts.py` end-to-end (270 LOC) — confirmed it is a stack-policy contract, not a Fourier-mode amplitude constraint.
8. Read `run_single_stage_thresholded_physics_alm.py` end-to-end and `run_stage2_alm.py` (focused on iota arg semantics).
9. Read `SINGLE_STAGE/single_stage_banana_example.py` focused on:
   - `--alm-iota-penalty-threshold` / `--alm-length-penalty-threshold` parsing (L1397–1430)
   - `validate_single_stage_alm_formulation_args` (L2948–2996)
   - `evaluate_alm_objective` call site (L4262–4307)
   - `build_single_stage_iota_objective` (L3478–3504) — confirmed `Jiota = QuadraticPenalty(...).J() = 0.5·(ι−ι_target)²`.
10. Cross-checked `src/simsopt/objectives/utilities.py:166-211` (`QuadraticPenalty.J() = 0.5·diff²` with `f="identity"`).
11. Verified ACCEPT_OFFSPEC removal: `grep -rn "OFFSPEC\|off_spec\|off-spec\|escape.hatch" examples/single_stage_optimization/ --include="*.py"` returns no hits.
12. Verified iota constraint test pinning at `tests/geo/test_banana_objective_modules.py:1738-1798` (Stage 2 path; expected `signed_values[2] == 0.1` for `iota=0.18, target=0.20, penalty_threshold=0.5` — consistent with `0.5·(0.18-0.2)² - 0.5 = 0.0002 - 0.5 = -0.4998` ... wait, that test uses a mocked `Stage2IotaState(penalty=0.6)` directly, so it's `0.6 - 0.5 = 0.1` — consistent with the formula `signed = penalty - penalty_threshold`).
13. Confirmed the Stage 2 path conversion `penalty_threshold = 0.5·tol²` in `stage2_objectives.py:230-234`, validating my reading of the constraint formula `0.5·(ι−target)² ≤ 0.5·tol²` ⇔ `|ι−target| ≤ tol`.
14. Verified the single-stage path bypasses this conversion: `single_stage_objectives.py:933-937` passes the user-supplied threshold directly to `_objective_upper_bound_constraint(Jiota, threshold, ...)`.

## Findings

### F1: Iota-penalty threshold conversion documented wrong by factor of `√(2·iotas_weight)` [CRITICAL]

- **Files**:
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1397-1416`
  - `examples/single_stage_optimization/README.md:576-593`
  - Code that defines the actual constraint: `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:933-937` and `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:117-124`
  - `Jiota` definition: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3478-3504` (target mode → `QuadraticPenalty(surface_iota_term, iota_target)`)
  - `QuadraticPenalty.J()` = `0.5·diff²`: `src/simsopt/objectives/utilities.py:185-196`

- **Code (help text in single_stage_banana_example.py:1397-1417)**:
  ```python
  parser.add_argument(
      "--alm-iota-penalty-threshold",
      type=float,
      default=(
          float(os.environ["ALM_IOTA_PENALTY_THRESHOLD"])
          if "ALM_IOTA_PENALTY_THRESHOLD" in os.environ
          else None
      ),
      help=(
          "thresholded_physics-mode upper bound for the Jiota penalty objective. "
          "Units: squared-penalty units, NOT iota deviation. The constraint is "
          "Jiota_penalty = (iota - iota_target)**2 * iotas_weight <= threshold, "
          "so this threshold scales with iotas_weight. To target a desired iota "
          "deviation d, set --alm-iota-penalty-threshold = (d**2) * iotas_weight "
          "(e.g. iotas_weight=1.0 with target deviation 0.01 -> threshold 1e-4). "
          ...
      ),
  )
  ```

- **Code (README.md:576-593)**:
  ```markdown
  - `--alm-iota-penalty-threshold` (single-stage `--alm-formulation thresholded_physics`)
    Units: squared-penalty units, scaled by `--iotas-weight`. The constraint is
    `Jiota_penalty <= threshold` where `Jiota_penalty = (iota - iota_target)^2 * iotas_weight`.
    ...
  Operator-facing conversion. To target a desired iota deviation `d` in single-stage ALM:

  ```
  --alm-iota-penalty-threshold = d**2 * iotas_weight
  ```

  Example mapping for `--iotas-weight 1.0` and target deviation `0.01`:

  ```
  --alm-iota-penalty-threshold = 0.01**2 * 1.0 = 1e-4
  ```
  ```

- **Code (actual constraint, single_stage_objectives.py:933-937)**:
  ```python
  "iota_penalty": _objective_upper_bound_constraint(
      Jiota,
      iota_penalty_threshold,
      objective_optimizable,
  ),
  ```
  with `_objective_upper_bound_constraint` (L117-124):
  ```python
  signed_value = float(objective.J()) - float(threshold)
  ```
  and `Jiota.J() = 0.5·(iota - iota_target)²` (target mode).

- **Bug**: The README and help text both claim:
  1. The constraint is `(ι − ι_target)² · iotas_weight ≤ threshold` — i.e., `iotas_weight` enters the constraint and there is no `0.5` factor.
  2. The operator should set `threshold = d² · iotas_weight` to bound deviation by `d`.

  Neither claim is true:
  1. The constraint actually computed is `0.5·(ι − ι_target)² − threshold ≤ 0`. There is no `iotas_weight` multiplier in the constraint — `IOTAS_WEIGHT` only appears in the `weighted_sum` base objective at `single_stage_objectives.py:78` and `:422`, not in the `iota_penalty` constraint definition. In `thresholded_physics` mode the base objective is zeroed out (`single_stage_objectives.py:430-431`), so `IOTAS_WEIGHT` plays no role at all.
  2. The correct conversion is `threshold = 0.5·d²` (independent of `iotas_weight`), giving deviation `d = √(2·threshold)`.

- **Why** (expected formula): `Jiota = QuadraticPenalty(iota_term, iota_target, f="identity")`, so `Jiota.J() = 0.5·(ι − ι_target)²` (per `simsopt/objectives/utilities.py:194`, the identity branch returns `0.5*diff**2`). The constraint is `Jiota.J() − threshold ≤ 0`, so feasibility is `0.5·(ι − ι_target)² ≤ threshold` ⇔ `|ι − ι_target| ≤ √(2·threshold)`.

- **Impact** (operator-facing):
  - For `iotas_weight = 1.0` (claimed in README example), an operator targeting `d = 0.01` follows the README and sets `threshold = 0.01² · 1.0 = 1e-4`. Actual band: `|ι − target| ≤ √(2·1e-4) ≈ 0.0141`. **41% looser than intended.**
  - For default `iotas_weight = 100` (`single_stage_banana_example.py:1612`), an operator targeting `d = 0.01` follows the README and sets `threshold = 0.01² · 100 = 0.01`. Actual band: `|ι − target| ≤ √(2·0.01) ≈ 0.1414`. **~14× looser than intended.** This is the dominant operational case.
  - Stage 2 path is unaffected — Stage 2 takes the deviation `tolerance` directly and converts internally via `0.5·tol²` (`stage2_objectives.py:230-234`), which is correct.
  - Default `--alm-iota-penalty-threshold = 1e-4` in `run_single_stage_thresholded_physics_alm.py:42` corresponds to actual deviation tolerance `0.0141`, not the `0.01` implied by the README's example.

- **Suggested fix**: Pick one of:
  1. Change README and help text to say `threshold = 0.5·d²` (and drop the `iotas_weight` rescaling guidance entirely). Recommended — the constraint physics and the help text disagree about which weight participates, so the right fix is to correct the docs.
  2. Or, for symmetry with Stage 2, expose a new `--alm-iota-deviation-tolerance` flag in the runner and convert internally to `0.5·d²`. Stage 2 uses this pattern and it is operator-friendly (`stage2_objectives.py:230-234`).

  In either case, also retire the `iotas_weight`-scaling claim in the README (L577 + L584): `IOTAS_WEIGHT` does NOT enter the constraint formula and therefore should not enter the operator-facing conversion.

---

### F2: `run_single_stage_thresholded_physics_alm.py` ships an unrunnable default `--alm-length-penalty-threshold = 0.0` [HIGH]

- **Files**:
  - `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py:43`
  - `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py:201-205`
  - Validation: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:2979-2995`
  - Validator implementation: `examples/single_stage_optimization/alm_utils.py:373-386`

- **Code (runner default, L40-43)**:
  ```python
  DEFAULT_ALM_QS_THRESHOLD = 3.0e-3
  DEFAULT_ALM_BOOZER_THRESHOLD = 1.0e-2
  DEFAULT_ALM_IOTA_PENALTY_THRESHOLD = 1.0e-4
  DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0
  ```

  ```python
  parser.add_argument(
      "--alm-length-penalty-threshold",
      type=float,
      default=DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD,
  )
  ```

- **Code (downstream validator, single_stage_banana_example.py:2979-2995)**:
  ```python
  required_thresholds = {
      "--alm-qs-threshold": args.alm_qs_threshold,
      "--alm-boozer-threshold": args.alm_boozer_threshold,
      "--alm-iota-penalty-threshold": args.alm_iota_penalty_threshold,
      "--alm-length-penalty-threshold": args.alm_length_penalty_threshold,
  }
  missing_thresholds = [
      flag_name for flag_name, value in required_thresholds.items() if value is None
  ]
  if missing_thresholds:
      raise ValueError(...)

  for flag_name, value in required_thresholds.items():
      require_positive_alm_threshold(flag_name, value)
  ```

- **Code (validator, alm_utils.py:373-386)**:
  ```python
  def require_positive_alm_threshold(name: str, value) -> float:
      value_f = float(value)
      if not np.isfinite(value_f) or value_f <= 0.0:
          raise ValueError(
              f"ALM threshold {name!r} must be a finite positive value; got {value!r}"
          )
      return value_f
  ```

- **Bug**: `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0` flows from `run_single_stage_thresholded_physics_alm.py` → forwarded as `--alm-length-penalty-threshold 0.0` (L313-315) → caught by `require_positive_alm_threshold` (commit `a169f296a` made nonpositive thresholds reject). Any user invoking this runner without an explicit `--alm-length-penalty-threshold` override gets a `ValueError` at parse time inside the single-stage child process.

- **Why** (intended behavior): The thresholded_physics formulation requires every constraint to have a positive scale (the threshold doubles as the normalization scale per `_physics_alm_metadata` at `single_stage_objectives.py:497-519`). A zero threshold would make the constraint redundant with the `coil_length_upper_bound` linear constraint AND would create a zero-scale normalization. Both are correctly forbidden.

- **Impact**: The single-stage thresholded-physics ALM rerun runner is unrunnable from defaults. Any operator running `python run_single_stage_thresholded_physics_alm.py --plasma-surf-filename ... --stage2-bs-path ...` without explicitly overriding `--alm-length-penalty-threshold` to a positive value gets an immediate failure. This is a regression introduced by commit `a169f296a` (`fix(alm-hardware): record scale-floor provenance and reject nonpositive thresholds`) which tightened the validator without updating this runner default.

- **Suggested fix**:
  ```python
  # Replace
  DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0
  # with a small positive default that is dominated by the linear coil_length_upper_bound
  # constraint, i.e. effectively neutralized but passes the positivity gate:
  DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 1.0e-6  # m^2; ~1.4 mm slack vs L_target
  ```
  or, preferred, drop `length_penalty` from `SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES` entirely (see F3) and remove this runner CLI flag.

---

### F3: `coil_length_upper_bound` and `length_penalty` both active in `thresholded_physics` mode [MEDIUM]

- **Files**:
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:326-331` (constraint name list)
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3026-3030` (constraint name resolution)
  - `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:787-792` (linear constraint)
  - `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:938-942` (quadratic constraint)
  - `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3730` (`JCurveLength` definition)

- **Code (constraint name list)**:
  ```python
  SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES = (
      "qs_error",
      "boozer_residual",
      "iota_penalty",
      "length_penalty",
  )
  ```

  ```python
  if alm_formulation == "thresholded_physics":
      names.extend(SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES)
  return names
  ```

- **Code (linear length constraint via hardware schema)**:
  ```python
  if coil_length_objective is not None and coil_length_threshold is not None:
      hardware_constraints["coil_length_upper_bound"] = _objective_upper_bound_constraint(
          coil_length_objective,
          coil_length_threshold,
          objective_optimizable,
      )
  ```
  with `coil_length_objective = curvelength` (single_stage_banana_example.py:4290), so `signed = L − L_target` in meters.

- **Code (quadratic length constraint via thresholded physics)**:
  ```python
  "length_penalty": _objective_upper_bound_constraint(
      JCurveLength,
      length_penalty_threshold,
      objective_optimizable,
  ),
  ```
  with `JCurveLength = QuadraticPenalty(curvelength, length_target, "max")` so `JCurveLength.J() = 0.5·max(L − L_target, 0)²` in m². Signed: `0.5·max(L − L_target, 0)² − thresh`.

- **Bug**: In `--alm-formulation thresholded_physics`, BOTH constraints are active simultaneously:
  1. `coil_length_upper_bound`: `L − L_target ≤ 0` (linear, m, scale = `L_target` ≈ 1.9 m).
  2. `length_penalty`: `0.5·max(L − L_target, 0)² − thresh ≤ 0` (quadratic, m², scale = `thresh`).

  Both are zero on the same set `{L ≤ L_target}` and both push the iterate down when `L > L_target`. The linear constraint is strictly tighter for any `thresh ≥ 0` (it activates at the boundary, while the quadratic only activates when the squared overshoot exceeds `2·thresh`). They share no functional purpose; the second one is redundant.

  This is the same redundancy pattern as the prior audit's Finding 1 (which was about base-objective `LENGTH_WEIGHT * JCurveLength` + `coil_length_upper_bound` in `weighted_sum` mode). The prior fix (`single_stage_banana_example.py:2957-2966`) only blocked the `weighted_sum + LENGTH_WEIGHT > 0` case. The same fix did not look at `thresholded_physics`, where `LENGTH_WEIGHT` is intentionally zeroed out but the redundancy reappears as a duplicated constraint pair.

- **Why** (expected behavior): Each physical inequality should have exactly one ALM multiplier. Two multipliers on the same physical constraint corrupt the KKT interpretation and roughly double the gradient magnitude on the boundary (each constraint contributes `(λ + ρc)·dc/dx` to the augmented gradient).

- **Impact**:
  1. KKT analysis: at convergence with the constraint active, the standard ALM dual interpretation is `λ ≈ ∂J_base/∂c`. Here two distinct `λ` are reported — one for `coil_length_upper_bound` (with units 1/m) and one for `length_penalty` (with units 1/m²). The pair adds to a meaningful total but the split is meaningless.
  2. Gradient magnitude: when `L > L_target`, both constraints contribute. The `coil_length_upper_bound` gradient is `(λ_1 + ρ·(L − L_target))·dL/dx`. The `length_penalty` gradient is `(λ_2 + ρ·(0.5·(L − L_target)² − thresh))·(L − L_target)·dL/dx`. The directions agree but magnitudes both grow with `(L − L_target)`, so the trust-region calibration is shifted relative to a single-constraint formulation.
  3. Diagnostics: the saved `ALM_FINAL_MULTIPLIERS` and `ALM_FINAL_CONSTRAINT_VALUES` arrays contain two entries for what is physically one inequality, complicating archive comparison across runs that use different formulations.

- **Suggested fix**: Drop `"length_penalty"` from `SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES`. The `coil_length_upper_bound` constraint already enforces the same physical inequality, with the correct linear residual and clean operator-interpretable scale. Also remove the now-orphan `--alm-length-penalty-threshold` CLI flag and corresponding runner default (which fixes F2 simultaneously). 5–10 LOC change.

---

## Confirmed-Correct Items

The following physics-correctness items either remained correct since the prior audit or were independently re-verified:

- **MAJOR_RADIUS frame and coil-frame convention**. `VACUUM_VESSEL_MAJOR_RADIUS_M = 0.976` (`hardware_contracts.py:20`) is treated as the coil-frame value everywhere. `validate_major_radius` rejects deviations >1e-12 (`hardware_contracts.py:97-105`). Banana winding-surface center sits at the same `R₀ = 0.976 m` (`hardware_contracts.py:22`). Confirmed against project memory `project_hbt_sidecar_conventions.md`.
- **ACCEPT_OFFSPEC env-flag escape hatches removed.** Verified by `grep -rn "OFFSPEC\|off_spec\|off-spec\|escape.hatch" examples/single_stage_optimization/ --include="*.py"` returning no hits. The `env_flag` helper at `hardware_contracts.py:138-139` is still defined but only used for diagnostic toggles (`BANANA_CURRENT_DIAGNOSTICS`, `BANANA_CURRENT_FD_DIAGNOSTICS`), never for spec relaxation. Commit `d61648f50` is complete.
- **Sign convention uniform: `c ≤ 0 ⟺ feasible`.** Every `signed_value` in the schema is constructed so that the augmented Lagrangian `(max(0, λ + ρc))²/(2ρ)` activates only when feasibility is violated. Verified for:
  - lower-bound (distance) constraints: `c = thresh − dist` (`single_stage_constraints.py:130`, `:221`, `:294`, `:397`)
  - upper-bound (curvature, length, poloidal-extent, qs/boozer/iota/length penalty): `c = value − thresh` (`single_stage_objectives.py:122`, `:236-238`)
  - box-bound (banana current): `c = |I| − I_max` (`single_stage_objectives.py:127-129`, `stage2_objectives.py:2049-2054`)
- **Distance gradient sign-flip preserved.** `smooth_min_*_signed_constraint` returns `-grad` to convert `d(smooth_min)/dx` into `d(min_dist − smooth_min)/dx` (`single_stage_constraints.py:130-134`, `:221-225`, `:294-298`, `:397-399` — explicit comments).
- **Banana-current sub-differential at `I = 0`.** `_scalar_abs_upper_bound_constraint` at `single_stage_objectives.py:127-136` picks `sign = +1` at `I = 0`. Same as prior audit Finding 3 — non-blocking.
- **TF-current convention.** `validate_tf_current_limit` requires `-TF_CURRENT_HARD_LIMIT_A ≤ I_TF < 0` (`hardware_contracts.py:49-56`). The artifact-only `tf_current` hardware constraint uses the box-bound `|I_TF| − I_max ≤ 0` form, which correctly handles the negative-CW convention.
- **Iota constraint dimensional consistency** (the *formula* itself). `0.5·(ι − ι_target)²` is dimensionless, threshold is dimensionless, scale (= threshold or `ALM_OBJECTIVE_SCALE_FLOOR = 1e-12`) is dimensionless, normalized residual is dimensionless. No unit mismatch in the math. The bug in F1 is purely the operator-facing documentation.
- **Stage 2 iota-tolerance conversion** is correctly implemented and correctly documented after commit `2e9acced2`. `stage2_iota_penalty_threshold(tol) = 0.5 * tol * tol` (`stage2_objectives.py:230-234`) gives the correct `|ι − target| ≤ tol` band. The Stage 2 help text was updated by `2e9acced2`; only the single-stage path was missed.
- **Solver-failure handling on iota.** `Stage2GuardedBoozerEvaluator` at `stage2_objectives.py:106-204` correctly snapshots the last successful Boozer state, restores it on solver failure or self-intersection, and reports `iota_violation = max(threshold, 1.0)` (`stage2_objectives.py:1768-1771`). Failed iterates drive multipliers AWAY from the failure region. Same as prior audit, still correct.
- **Hybrid surrogate-vs-hard signal contract.** Geometry constraints use surrogate signed values for the inner objective/gradient and hard signed values for dual updates and feasibility certification (`hardware_constraint_schema.py:80-114`, `single_stage_objectives.py:560-577`, `stage2_objectives.py:704-759`). Sign mismatch detection is wired through `_surrogate_hard_sign_mismatch` (`alm_utils.py:826-832`) and `_constraint_routing_state` (`alm_utils.py:2079-2137`). Engineering trade-off documented in commit `3671c479c`.
- **Constraint-scale normalization.** Each `c̃ = c / scale` is dimensionless (`alm_utils.py:510-545`); each multiplier is dimensionless; the augmented term is dimensionless. The shared penalty `ρ` and multipliers are correctly comparable across constraints with different physical dimensions (m, 1/m, A, dimensionless).
- **No surface mode amplitudes constraint exists.** The "surface mode" referenced in the prompt is satisfied by the surface stack policy in `surface_mode_contracts.py`, not Fourier mode amplitude constraints. There are no Fourier mode amplitude inequality constraints in the ALM schema; `mpol/ntor` are only Boozer surface quadrature parameters (`stage2_single_stage_handoff.py:646-692`). No bug.
- **`_explicit_raw_signed_constraint_values` reads the right field.** Returns `evaluation["raw_constraint_values"]` first, falling back to `evaluation["raw_surrogate_signed_constraint_values"]` (`alm_utils.py:678-686`). Both are populated from the same `constraint_values` and `surrogate_signed_values` lists in `evaluate_alm_objective`. Field naming is consistent across single-stage (`single_stage_objectives.py:1112-1131`) and Stage 2 (`stage2_objectives.py:1973-1982`).

## Constraint-by-Constraint Table (delta from prior audit)

| Constraint | Sign | Units | Frame | Verdict |
|---|---|---|---|---|
| `coil_coil_spacing` | `c = thresh − dist` | m | coil | OK |
| `coil_surface_spacing` | `c = thresh − dist` | m | coil↔plasma | OK |
| `surface_vessel_spacing` | `c = thresh − dist` | m | plasma↔vessel | OK |
| `surface_surface_spacing` | `c = thresh − dist` | m | adjacent boozer | OK; gated by `SURFACE_GAP_THRESHOLD > 0` |
| `max_curvature` | `c = κ_smooth − thresh` | 1/m | coil | OK |
| `coil_length_upper_bound` | `c = L − L_target` | m | coil | OK; **F3 redundant with `length_penalty` in thresholded_physics mode** |
| `poloidal_extent` | `c = max\|θ\| − π/4` | rad | banana-winding-surface | OK |
| `banana_current_upper_bound` | `c = \|I\| − I_max` | A | hardware | OK; `sign(0) := +1` (prior F3, non-blocking) |
| `iota_penalty` (Stage 2) | `c = 0.5·(ι−target)² − 0.5·tol²` | dimensionless | flux-surface | OK; help text fixed in `2e9acced2` |
| `iota_penalty` (single-stage) | `c = 0.5·(ι−target)² − thresh` | dimensionless | flux-surface | **F1: README/help-text formula wrong** |
| `qs_error` (single-stage) | `c = J_QS − thresh` | dimensionless | flux-surface | OK |
| `boozer_residual` (single-stage) | `c = J_Boozer − thresh` | dimensionless | flux-surface | OK |
| `length_penalty` (single-stage) | `c = 0.5·max(L−L_target,0)² − thresh` | m² | coil | **F3 redundant; F2 default `0.0` rejected by validator** |
| `tf_current` (artifact) | `c = \|I\| − I_max` | A | hardware | OK; not in solver loop |
| `lcfs_*_radius` (artifact) | `c = R − R_max` | m | plasma | OK; not in solver loop |

## Verdict

**Three actionable findings.** The constraint formulas and ALM machinery are physically correct in the actual code path; this audit's findings are operator-facing surface area:

1. **F1 (CRITICAL)** — fix the README and help text for `--alm-iota-penalty-threshold` to remove the spurious `iotas_weight` factor and add the missing `0.5` factor. Operators currently get an iota deviation tolerance that is `√(2·iotas_weight)` larger than they intended (14× looser at default `iotas_weight=100`). This is the highest-priority fix because it silently degrades physics-correctness of the deployed runs.
2. **F2 (HIGH)** — fix the runner default `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0`; the current default is rejected by `require_positive_alm_threshold` and prevents the runner from launching with its own defaults. This regressed in commit `a169f296a` and was not paired with a runner-default update.
3. **F3 (MEDIUM)** — drop `length_penalty` from the thresholded_physics constraint set since `coil_length_upper_bound` already enforces the same physical inequality with cleaner units. Mostly cleanup; not a correctness bug, but the parallel of the prior audit's Finding 1 in a different mode and was not closed by the prior fix.

Beyond these three, no new sign flips, no new frame mismatches, no new silent unit bypasses, no off-spec escape hatches surviving, and no new double-feeds into the same multiplier compared to the prior audit. The Stage 2 iota path, the Boozer guarded evaluator, the surrogate-vs-hard signal contract, the schema-driven status builder, and the dimensional normalization remain physics-correct.
