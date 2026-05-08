# ALM Physics Audit

Branch: `surrogate-confinement-v2`
Audit scope: physics correctness of ALM constraint formulation only.
Out of scope: ALM iteration math, control flow, JAX/GPU, tests.

## Executive Summary

**Verdict: physically sound, with five sub-major / minor findings.** The ALM
constraint formulation is correctly written. Sign conventions are consistent
(`c ≤ 0 ⟺ feasible`), units check out after normalization, frame conventions
match the project memory (coil-frame `R₀ = 0.976 m`, banana-winding-surface
center at the same R₀ but smaller minor radius), and the recent normalization
work (`7cb96e21e`) preserves dimensional consistency. The Stage-2 → single-stage
handoff actively re-validates every artifact constraint via the same
schema-driven status builder, so values cannot silently rescale or strip.
Solver-failure handling on the iota path correctly biases iterates **away** from
infeasibility instead of admitting failed states.

The five sub-major / minor findings are:

1. **(SUB-MAJOR)** In `weighted_sum` ALM mode, the coil-length quadratic
   penalty enters the base objective with weight `LENGTH_WEIGHT` AND an
   inequality constraint `length ≤ length_target` enters as an ALM augmented
   term, both centered at the same threshold. This is a redundant penalty that
   biases the iterate strictly below `length_target` and corrupts the KKT
   interpretation of the multiplier.
2. **(MINOR)** The single-stage `iota_penalty` constraint takes a
   user-facing threshold in *squared-penalty units* (`--alm-iota-penalty-threshold`),
   while Stage 2 takes it in *iota-deviation units* (`--stage2-iota-tolerance`)
   and squares internally. Same physics, different operator-facing semantics.
3. **(MINOR)** The banana-current ALM constraint uses a sub-differential at
   `value = 0` (`sign(0) := +1`). Harmless in practice but worth noting.
4. **(INFO)** The `compute_tf_G0` helper sums **|I_i|** rather than `Σ I_i`.
   Inherits simsopt's convention; both helpers (`stage2_single_stage_handoff`
   and `surfaceobjectives.py:55`) agree, so no internal inconsistency. Just be
   aware that the seed magnitude is sign-blind to the CW/CCW TF convention.
5. **(INFO)** `iota_target` has no rational-surface guard — relying entirely on
   `Stage2GuardedBoozerEvaluator` to handle Newton failure. This is correct
   defensive behaviour, but means the user can pick `iota_target = 1/2` etc.
   and the solver may simply spin up failed iterates.

No sign flips. No frame mismatches on the current strict-contract path. No
silent unit-bypass. No double-feeding into the same multiplier.

---

## Findings

### Finding 1 (SUB-MAJOR): Coil-length penalty + ALM constraint double-feed in weighted_sum mode

**File**: `banana_opt/single_stage_objectives.py:417-421` (base objective build),
`banana_opt/single_stage_objectives.py:781-786` (ALM constraint build),
`SINGLE_STAGE/single_stage_banana_example.py:3707` (`JCurveLength` construction).

**Physics statement.** In `weighted_sum` ALM mode the base objective contains
`LENGTH_WEIGHT * QuadraticPenalty(curvelength, length_target, "max") =
LENGTH_WEIGHT * 0.5 * max(0, length - length_target)^2`. Simultaneously, the
ALM constraint set contains `coil_length_upper_bound` with signed value
`c = curvelength.J() - length_target`, scale `length_target`, and augmented
term `(λ + ρ * c̃)² / (2ρ)` activated for `c > 0`. Both terms drive the
iterate when `length > length_target`; both are exactly zero when
`length ≤ length_target`. They are not mutually cancelling, but they ARE the
same physical penalty applied through two different mechanisms.

**Consequence.**
1. KKT analysis: at convergence with the constraint active, the standard ALM
   dual interpretation is `λ ≈ ∂J_base/∂c|_{c=0}`. Here `J_base` already
   contains a term that is differentiable in `c` with slope
   `LENGTH_WEIGHT * c|_{c=0+} → 0`, so the multiplier is well-defined, but the
   *reported* `λ` carries an offset that depends on `LENGTH_WEIGHT` and the
   iterate's distance to the boundary. Operator interpretation of "saved
   multiplier" across runs with different `LENGTH_WEIGHT` is incoherent.
2. Optimizer behaviour: when the constraint is approached from inside, the
   base penalty does nothing (both pieces are zero on `c ≤ 0`). When it's
   approached from above, both push down, which is OK directionally but
   roughly doubles the gradient magnitude near the boundary. The trust-region
   logic and step-rejection thresholds in ALM were calibrated against the
   constraint-only gradient, not the doubled-gradient case.
3. Default behaviour: `--length-weight` defaults to `1.0`, so this redundancy
   IS active in the default single-stage configuration.

**Why it isn't catastrophic.** Both the penalty and the augmented term are
zero on the same set, both drive in the same direction off the set, both
gradients point along the same `dlength/dx`. So the *direction* of the
augmented gradient is correct; only its magnitude is contaminated. The
strictness bias ("solver settles slightly below `length_target`") is
*physically conservative* — coils that are slightly shorter than the limit
remain feasible.

**Fix (in order of preference).**
- **Preferred**: when `constraint_method == "alm"` and `weighted_sum` mode is
  active for the length variable, set `LENGTH_WEIGHT = 0.0` and rely solely on
  the ALM constraint, so the multiplier carries the entire penalty. The user
  can still steer length elsewhere via the constraint threshold.
- **Acceptable**: pass a different `target` to `JCurveLength` so the base
  penalty centers on a softer engineering preference (e.g. `1.85 m`) while
  the ALM constraint enforces the hard target (`1.9 m`). This makes the two
  channels physically distinct (target vs. ceiling).
- **Document**: at minimum, add a comment in `evaluate_alm_objective` and
  `HARDWARE_CONSTRAINTS.md` flagging this redundancy and pointing the operator
  at `--length-weight=0`.

This finding is sub-major because it does not invalidate physics-correctness
of any reported result — it just makes the dual estimate and gradient
magnitude harder to interpret.

---

### Finding 2 (MINOR): iota constraint threshold semantics differ between Stage 2 and single-stage

**File**: `banana_opt/stage2_objectives.py:228-237` (`stage2_iota_penalty_threshold`),
`SINGLE_STAGE/single_stage_banana_example.py:1395-1403` (`--alm-iota-penalty-threshold`).

**Physics statement.** Stage 2 takes `--stage2-iota-tolerance` (an iota
deviation in dimensionless units, like `0.05`) and internally computes
`threshold = 0.5 * tol²`. Single-stage in `thresholded_physics` mode takes
`--alm-iota-penalty-threshold` directly (a squared-penalty value, like
`1.25e-3`). Both produce the same ALM constraint
`c = 0.5*(iota - target)² - threshold ≤ 0`, but the operator must know which
flag wants which kind of number.

**Consequence.** A user who is comfortable with Stage 2's `iota_tolerance =
0.05` and translates that to single-stage as
`--alm-iota-penalty-threshold=0.05` is silently asking for a tolerance
equivalent to `√(2 * 0.05) ≈ 0.316` in iota — six times looser than intended.

**Fix.** Either:
- expose `--alm-iota-deviation-tolerance` in single-stage and square it
  internally, or
- accept both flags and translate consistently.

Sign and dimension are both correct; this is purely a UX coherence issue.

---

### Finding 3 (MINOR): Sub-differential choice at value=0 for box-bound constraints

**File**: `banana_opt/single_stage_objectives.py:125-134` (`_scalar_abs_upper_bound_constraint`).

**Physics statement.** For `c = |I| - I_max`, the gradient at `I = 0` is the
sub-differential `[-1, 1]` of `|I|`. The code picks `sign = +1` at `I = 0`
unconditionally.

**Consequence.** When banana current is initialized at zero, the constraint
is strictly inactive (`c = -I_max < 0`), so the multiplier never activates
and the gradient direction is irrelevant in practice. If an iterate ever
lands exactly at `I = 0` while the multiplier is positive (would require a
very particular sequence of penalty updates), the gradient could oscillate
between `+1` and `-1` flavour as the iterate crosses zero. Standard ALM
behaviour, not a bug.

**Fix.** Optional. Use a smoothed `|·|`, e.g. `√(I² + δ²) - I_max` with a
small `δ`, if convergence near zero ever becomes a problem. Today it isn't.

---

### Finding 4 (INFO): `compute_tf_G0` uses `Σ |I_i|`, not `Σ I_i`

**File**: `banana_opt/stage2_single_stage_handoff.py:399-404`.

**Physics statement.** `G0 = 2π * Σ|I_i| * (μ₀/(2π)) = μ₀ * Σ|I_i|`. This
matches `simsopt/geo/surfaceobjectives.py:55`, so the two callers agree.

**Why it is fine.** Boozer Newton uses the seed `G0` only as a starting guess
and re-solves the actual `G` to satisfy the residual. The sign of the seed
does not bind the sign of the converged `G`. With HBT-EP's CW TF convention
(all 20 TF coils carry the same negative current), `Σ I_i = -Σ|I_i|`, so the
absolute-value sum supplies a magnitude that is correct up to sign. The
solver finds the correct sign.

**Why it is worth flagging anyway.** A future hardware configuration with
mixed-sign TF coils (e.g. transient counter-current bias coil) would yield
an incorrect magnitude here, because `Σ|I_i| ≠ |Σ I_i|`. Today HBT-EP has
exactly one TF current direction, so this is fine.

---

### Finding 5 (INFO): No iota_target rational-surface guard

**File**: `SINGLE_STAGE/single_stage_banana_example.py:1415-1425`,
`STAGE_2/banana_coil_solver.py:560-570`.

**Physics statement.** The user can pass any `--iota-target`, including
values arbitrarily close to `1/2`, `2/5`, etc. The constraint formula
`c = 0.5*(iota - target)² - threshold` is continuous and well-defined for
any real `target`, so the constraint itself is not pathological. However,
the **iota Boozer solve** can fail near rational surfaces because of magnetic
island formation and surface destruction.

**Why it is handled.** `Stage2GuardedBoozerEvaluator` (`stage2_objectives.py:104-202`)
correctly handles solve failure by:
- snapshotting the last successful Boozer solve state,
- restoring it on failure or self-intersection,
- setting `iota_violation = max(threshold, 1.0)` instead of silently
  fabricating a feasible value.

The raised violation drives the multiplier away from the failed region. This
is the right design.

**What is missing.** No upstream validation. A user running with
`--iota-target=0.5` may waste many iterations bouncing off Boozer failures
before the ALM gives up. This is operator-time waste, not a correctness
problem.

**Fix.** Optional. Print a warning when `iota_target` is within ~0.01 of
`{1/2, 1/3, 2/3, 1/4, 1/5, 2/5, 1/6, ...}`. No physics change required.

---

## Constraint-by-Constraint Table

| Constraint | Sign | Units (raw signed value) | Frame | Normalization scale | Source file:line | Verdict |
|---|---|---|---|---|---|---|
| `coil_coil_spacing` | `c = thresh − dist` (≤0 feasible) | m | coil frame | `COIL_COIL_MIN_DIST_M = 0.05 m` | `single_stage_constraints.py:130`, `single_stage_objectives.py:692-697` | OK |
| `coil_surface_spacing` | `c = thresh − dist` | m | coil↔plasma frame | `COIL_PLASMA_MIN_DIST_M = 0.015 m` | `single_stage_constraints.py:221`, `single_stage_objectives.py:704-713` | OK |
| `surface_vessel_spacing` | `c = thresh − dist` | m | plasma↔vessel frame | `PLASMA_VESSEL_MIN_DIST_M = 0.04 m` | `single_stage_constraints.py:294`, `single_stage_objectives.py:739-754` | OK |
| `surface_surface_spacing` | `c = thresh − dist` | m | adjacent boozer frames | `SURFACE_GAP_THRESHOLD` (only active when `>0`) | `single_stage_constraints.py:397`, `single_stage_objectives.py:760-775` | OK; gated by `single_stage_surface_stack_alm_enabled` |
| `max_curvature` | `c = κ_max − thresh` | 1/m | coil frame | `MAX_CURVATURE_INV_M = 100 m⁻¹` | `single_stage_constraints.py:52`, `single_stage_objectives.py:714-719` | OK |
| `coil_length_upper_bound` | `c = length − target` | m | coil frame | runtime `length_target` ∈ (0, 2.0] m | `single_stage_objectives.py:115-122, 781-786` | OK; **redundant with base in weighted_sum** (Finding 1) |
| `poloidal_extent` | `c = max\|θ\| − π/4` | rad | banana-winding-surface frame (R=0.976, Z=0) | `POLOIDAL_EXTENT_HALF_WIDTH_RAD = π/4` | `poloidal_extent.py:30-117`, `single_stage_objectives.py:814-835` | OK |
| `banana_current_upper_bound` | `c = \|I\| − I_max` | A | hardware | `BANANA_CURRENT_HARD_LIMIT_A = 16 kA` | `single_stage_objectives.py:125-134, 787-803`, `stage2_objectives.py:2035-2058` | OK; subdiff choice at I=0 is `+1` (Finding 3) |
| `tf_current` (artifact only) | `c = \|I\| − I_max` | A | hardware | `TF_CURRENT_HARD_LIMIT_A = 80 kA` | `hardware_contracts.py:6-9, 49-56` | OK; not in solver loop |
| `iota_penalty` (Stage 2) | `c = 0.5·(ι − target)² − 0.5·tol²` | dimensionless | flux-surface | `0.5 * tol²` | `stage2_objectives.py:228-237, 1745-1773` | OK; uses guarded Boozer solver |
| `iota_penalty` (single-stage) | `c = 0.5·(ι − target)² − thresh` | dimensionless | flux-surface | user-explicit `--alm-iota-penalty-threshold` | `single_stage_objectives.py:927-931, 547-552` | OK; threshold semantic differs from Stage 2 (Finding 2) |
| `qs_error` (single-stage) | `c = J_QS − thresh` | dimensionless (mean QS residual²) | flux-surface | user-explicit `--alm-qs-threshold` | `single_stage_objectives.py:917-921, 547-552` | OK |
| `boozer_residual` (single-stage) | `c = J_Boozer − thresh` | dimensionless | flux-surface | user-explicit `--alm-boozer-threshold` | `single_stage_objectives.py:922-926, 547-552` | OK |
| `length_penalty` (single-stage) | `c = J_len − thresh` | m² (squared length) | coil frame | user-explicit `--alm-length-penalty-threshold` | `single_stage_objectives.py:932-936, 547-552` | OK |
| `lcfs_major_radius` (artifact only) | `c = R_LCFS − R_max` | m | plasma frame | `TARGET_LCFS_MAX_MAJOR_RADIUS_M = 0.92 m` | `hardware_contracts.py:31, 69-76` | OK; not in solver loop |
| `lcfs_minor_radius` (artifact only) | `c = a_LCFS − a_max` | m | plasma frame | `TARGET_LCFS_MAX_MINOR_RADIUS_M = 0.15 m` | `hardware_contracts.py:32, 79-86` | OK; not in solver loop |

Sign convention is uniform: every signed value is constructed so that
**`c ≤ 0` ⟺ the iterate is feasible for that constraint**. The augmented
Lagrangian `(max(0, λ + ρc))² / (2ρ)` (`alm_utils.py:374-414`) consequently
activates only for `c > 0`, and the projection `max(0, ·)` defines the
positive-shift KKT estimate. Multiplier sign is non-negative throughout, as
required for inequality ALM.

After normalization (`alm_utils.py:437-459`), every `c̃ = c / scale` is
dimensionless, so multipliers `λ_i` are also dimensionless and the augmented
term is dimensionless throughout. The penalty `ρ` and multipliers are
therefore directly comparable across constraints.

---

## Things Verified Correct

- **MAJOR_RADIUS frame.** `VACUUM_VESSEL_MAJOR_RADIUS_M = 0.976` is treated as
  the COIL FRAME everywhere on the current strict-contract path, matching the
  project memory at
  `~/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/project_hbt_sidecar_conventions.md`.
  `validate_major_radius` (`hardware_contracts.py:97-105`) raises on any
  off-spec value within 1e-12. The Stage-2 → single-stage handoff at
  `stage2_single_stage_handoff.py:344-396` re-validates the seed
  `MAJOR_RADIUS` field, the banana winding-surface radius, the curvature
  ceiling, the poloidal-extent threshold, and the LCFS-vessel clearance
  before launching the Boozer solver. **No silent rescaling.**
- **Banana winding surface.** Sits at the same R₀ as the vacuum vessel
  (0.976 m) but smaller minor radius (0.21 m vs 0.222 m), and the
  `PoloidalExtent` constraint pivots at this winding-surface center
  (`single_stage_banana_example.py:3716-3720`,
  `banana_coil_solver.py:1641-1645`). Geometrically correct frame for
  inboard-arc measurement.
- **Hard ceiling vs soft target separation (commit `c378395f1`).** The hard
  ceiling `COIL_LENGTH_HARD_LIMIT_M = 2.0 m` is enforced **at parse time** —
  `--length-target > 2.0` is rejected before solver launch
  (`single_stage_banana_example.py:8364-8368`,
  `banana_coil_solver.py:1610-1612`,
  `constraint_contract.py:281-285`). The actual ALM constraint then uses the
  user's `length_target` (≤ 2.0) as the inequality. There is **no scenario
  where both 1.9 and 2.0 feed the same multiplier.** Finding 1 above is
  about a different overlap (base-objective penalty + ALM constraint at the
  same threshold) and does not contradict this design.
- **Sign convention preservation (commit `bfd4b5195`).** Every distance
  constraint returns `signed_value = threshold − distance` (lower-bound) and
  every upper-bound returns `signed_value = value − threshold`. The
  smooth-min/smooth-max gradients are correctly negated so that
  `d(signed_value)/dx` matches the gradient of the *signed* expression, not
  the underlying min/max. See
  `single_stage_constraints.py:130-134, 221-225, 294-298, 397-399` for the
  documented sign-flip comments.
- **Constraint normalization (commit `7cb96e21e`).** `_resolved_alm_scale`
  uses `max(raw_threshold, sys.float_info.epsilon)` so a degenerate scale
  (e.g. zero threshold) is bounded away from zero by machine precision. The
  validator `_validate_alm_metadata` then re-checks finiteness and
  positivity. The `surface_surface_spacing` ALM constraint (the only schema
  entry with `threshold = 0`) is gated off entirely when `SURFACE_GAP_THRESHOLD
  = 0`, so the floor-to-epsilon is never actually exercised.
- **Dimensional consistency.** Each `c_i / scale_i` is dimensionless; each
  `λ_i` is dimensionless; the augmented term and the base objective are
  dimensionally homogeneous. The multipliers can be saved and reloaded across
  runs without unit conversion.
- **Banana-current contract enforcement (commit `daa6d5484`).** The
  banana-current upper bound enters as `c = |I| − I_max` (box bound,
  `hardware_constraint_schema.py:153-160`). The banana-current schema entry
  also feeds penalty-search box bounds via
  `apply_penalty_traversal_forbidden_box_bounds`
  (`current_contracts.py:268-299`), so penalty-mode and ALM-mode both honour
  the hardware limit. Independent-current banana coils are handled with one
  `banana_current_<i>_upper_bound` constraint per controllable current
  (`single_stage_objectives.py:137-155`), preserving the per-coil sign and
  gradient.
- **Stage-2 → single-stage handoff (commit `cee1db38d`).** The handoff
  `validate_stage2_seed_contract` re-runs **the same**
  `build_hardware_constraint_status` over the artifact's measured values that
  the live solver runs over its computed values. Because the schema is the
  SSOT, no rescaling happens — the same units, same thresholds, same
  signed-value formulas apply on both sides. The handoff additionally rejects
  any seed whose `MAJOR_RADIUS`, `banana_surf_radius`, `CURVATURE_THRESHOLD`,
  `POLOIDAL_EXTENT_THRESHOLD_RAD`, `SURFACE_VESSEL_MIN_DIST`, or
  `TF_CURRENT_A` falls outside the live contract.
- **Solver-failure handling on iota.** `Stage2GuardedBoozerEvaluator`
  (`stage2_objectives.py:104-202`) snapshots the last successful Boozer
  solution and restores it on solver failure or self-intersection. Failed
  iterates are reported as `iota_violation = max(threshold, 1.0)` so the
  multiplier and feasibility metrics drive the iterate AWAY from the
  failure, not into it. **No silent acceptance of solver failure as
  feasibility.**
- **Iota constraint at degenerate iota.** The constraint
  `0.5·(ι − target)² − threshold ≤ 0` is well-defined for any real `ι`,
  including `0` (no rotational transform) and `1` (unit transform). No
  geometric singularity in the constraint formula itself. Behaviour at
  rational surfaces is governed by the Boozer solver, which is properly
  guarded as above.
- **Poloidal-extent geometry at degenerate iota.** The poloidal-extent
  constraint operates on the banana **coil** geometry, not the plasma flux
  surfaces, so it is invariant to the rotational-transform value and remains
  geometrically meaningful at any iota, including `iota = 0` and rational
  iota. The arc is measured against the fixed winding-surface center
  (`R = 0.976 m, Z = 0`), so the geometry is independent of the magnetic
  configuration.
- **ALM dual-update value-kind contract.** The schema enforces that
  `geometry` block constraints can use surrogate signals for the inner solve
  but must use hard signals for the dual update if `use_hard_geometry_signals`
  is true; `current` block constraints use hard signals exclusively;
  `physics` block constraints use raw-physics signals
  (`hardware_constraint_schema.py:62-97`). This prevents the multiplier from
  drifting away from the engineering meaning of the constraint while still
  giving the inner solver smooth surrogate gradients.
- **Hard surrogate diagnostics.** When `hard_surrogate_diagnostics=True`,
  `_resolve_hard_signal` emits the *hard* (non-smoothed) signed values and
  violations alongside the smoothed surrogates
  (`single_stage_objectives.py:843-913`). The dual-update array can then be
  driven by the hard signal while the inner solver still sees the smooth
  one. This is the right separation of concerns and matches the contract
  documented in `HARDWARE_CONSTRAINTS.md:60-79`.
- **Coil-length objective vs constraint.** The ALM constraint receives the
  RAW `CurveLength` (units: m), not the squared `QuadraticPenalty` form.
  The signed value is `length − length_target` in meters, which is the
  correct linearizable inequality form. The base objective separately takes
  `JCurveLength = QuadraticPenalty(curvelength, length_target, "max")` if
  `LENGTH_WEIGHT > 0`. The two enter at different orders (linear residual vs.
  quadratic-in-residual penalty), so they ARE physically distinguishable —
  see Finding 1 for the operator-interpretation problem this creates.

---

## Inputs Audited

- `examples/single_stage_optimization/alm_utils.py` (4637 LOC; focused on
  L370-475 for augmented_inequality_objective and normalization).
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py`
  (1166 LOC, full read).
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
  (2058 LOC; constraint construction and iota guard).
- `examples/single_stage_optimization/banana_opt/single_stage_constraints.py`
  (full read for sign conventions and gradient signs).
- `examples/single_stage_optimization/banana_opt/poloidal_extent.py`
  (full read for frame and gradient).
- `examples/single_stage_optimization/banana_opt/hardware_contracts.py`
  (full read for contract values and validators).
- `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py`
  (full read for schema, ALM metadata, signed-value semantics).
- `examples/single_stage_optimization/banana_opt/current_contracts.py`
  (full read for current-related contracts and Boozer-I conversion).
- `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py`
  (focused on validate_stage2_seed_contract, compute_tf_G0, classify_bootability_result).
- `examples/single_stage_optimization/HARDWARE_CONSTRAINTS.md` (SSOT doc).
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  (focused on objective construction L3675-3740, ALM call site
  L4192-4284, length-target validation L8364-8368, and arg parsing).
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  (focused on parse-time contract enforcement and Boozer setup).
- Project memory `project_hbt_sidecar_conventions.md`.

Recent commits inspected: `bfd4b5195`, `daa6d5484`, `c378395f1`,
`6c1d359e2`, `cee1db38d`, `7cb96e21e`.
