# Banana audit — remaining three items implementation plan (2026-06-12)

## Purpose

Close the three first-principles audit items that are still untouched after the
2026-06-11/12 remediation rounds (island-gate wiring, resonant q-cap, and the
`banana_curves` NameError are already DONE and are NOT in scope here). Each item
was scoped read-only against the live tree by a dedicated agent; this plan records
the exact integration points so an engineer (or executor agent) can implement
without rediscovery.

The three items:
1. **Audit-2** — smooth rational-ι repulsion objective on the OUTER Boozer iota
   (keep iota off low-order rationals during optimization; near-shearless field,
   target ι≈0.08–0.12, live threats 1/9, 1/10, 1/12).
2. **Survival-domain ownership** — add an optional FIXED external reference domain
   to the field-line survival metric so "survival" measures a physical boundary,
   not the candidate's own (movable) surface.
3. **REGCOIL floor diagnostic** — one-off offline current-potential lower bound on
   the winding surface to decompose the flux residual into winding-surface-physics
   vs coil-order-parameterization deficit (decides order-3→4 vs move-the-surface).

## Goals

- A default-OFF, opt-in `IotaRationalRepulsion` objective term that repels the
  outer Boozer iota from low-order rationals (q ≤ 13), with an exact gradient
  through the existing `Iotas` adjoint and a finite-difference gradient test.
- An additive, default-inert fixed-reference-domain survival verdict recorded
  alongside the existing candidate-relative one, in both the in-run topology
  scorer and the strict Poincaré path.
- A standalone `run_regcoil_floor.py` diagnostic that, given a run artifact dir,
  reports the REGCOIL floor on the winding surface in the SAME units as the
  stored `field_objective`, plus the achieved/floor decomposition.
- Zero behavior change to existing runs until each feature is explicitly enabled
  (byte-identical objective graph / inert stopping criteria / no objective edits).

## Non-Goals

- No edits to `src/simsopt/**` — all three items are `examples/`-only (the `Iotas`
  adjoint, `SurfaceClassifier`, and `SurfaceRZFourier` already expose everything
  needed; verified).
- Not changing the existing interior-surface rational DIAGNOSTIC
  (`interior_iota_low_order_rational_diagnostics`) — it deliberately excludes the
  outer surface; the new objective owns the outer surface, so they are complementary.
- Not putting REGCOIL in the optimizer loop — it is offline decision-support only.
- Not vendoring upstream simsopt's `CurrentPotential*`/REGCOIL modules (drag in
  C++/winding-surface deps this fork does not build) — write a self-contained numpy
  LSQ kernel instead.
- Not enabling any of the three by default in the live spec-143 lanes; enablement
  and weight tuning are a separate campaign decision.

## Current Context

- Tree: `/Users/suhjungdae/code/columbia/simsopt-surrogate`, branch
  `surrogate-confinement-v2` (dirty tree expected; a shared multi-coil refactor
  by another session is in flight — coordinate before editing
  `single_stage_banana_example.py`, per the spec-143 plan's quarantine note).
- The opt-in, default-OFF objective-term pattern already exists and is the template
  for item 1: `IotaShearShortfall` + `build_single_stage_shear_objective`
  (`banana_opt/single_stage_objectives.py:79-149`), wired into `build_total_objective`
  via a `JTerm=None, WEIGHT=0.0` param + `if JTerm is not None: objective = objective
  + WEIGHT * JTerm` guard (`single_stage_objectives.py`: def `:369`, `JMinLGradB`
  guard `:460`, `JShear` guard `:466`).
- **There are TWO `build_total_objective` functions**, both taking `JShear`/`SHEAR_WEIGHT`:
  the term-library one above, AND a local copy in the example file at
  `single_stage_banana_example.py:10948` (`JShear=None, SHEAR_WEIGHT=0.0` at `:10986-10987`).
  `JShear`/`SHEAR_WEIGHT` appears at ~27 sites across `single_stage_banana_example.py`
  (the bundle path, the in-run `resolve_current_surface_objective_terms` resolver, and
  `globals().get("JShear")` forwarding at `:7665`). The new term must mirror EVERY one of
  those sites or it will be inert in the in-run optimization loop even though it appears
  in the bundle.
- The outer-surface iota leaf the new term must wrap is `surface_iota_terms[-1]`,
  already wrapped by `QuadraticPenalty(...)` for iota targeting
  (`single_stage_banana_example.py:6844-6849`; `Iotas` value+adjoint at
  `src/simsopt/geo/surfaceobjectives.py:1001-1051`).
- A band-calibrated rational enumerator already exists and should be reused:
  `enumerate_resonant_rationals(iota_target, delta, q_max)` with
  `MAX_RESONANT_DENOMINATOR = 13` (`banana_opt/stage2_resonant_flux.py:97-138`).
- Survival is purely candidate-surface-relative: `build_stopping_criteria(surface,...)`
  builds the only `SurfaceClassifier` from the candidate's own surface
  (`topology_scorer.py:834-889`, classifier at `:879-881`); `survival_fraction =
  survived/nfieldlines` (`:1040`). The strict path does the same
  (`POINCARE_PLOTTING/poincare_surfaces.py:533-547`). Commit `f81eb1aac` fixed the
  RESOLUTION of the candidate surface, not its OWNERSHIP.
- A ready-made fixed external domain already exists with no candidate dependency:
  `build_banana_reference_surfaces(...)` returns `vessel` (0.976/0.222),
  `lcfs_clearance_reference`, and `coil_winding_surface` (0.903/0.142) as plain
  `SurfaceRZFourier` tori (`banana_opt/reference_surfaces.py:21-54`).
- REGCOIL: this fork has NO `CurrentPotential*`/`CurrentPotentialSolve`/`WindingSurface`/
  `regcoil` solver (only `SurfaceRZFourier.from_nescoil_input` reader at
  `surfacerzfourier.py:397`). The achieved flux to compare against is
  `results.json["field_objective"]` =
  `0.5·mean((B·n̂)²·|n|)` over the working surface (key written at
  `banana_coil_solver.py:2907`; evaluator `_evaluate_stage2_flux_objective_on_own_grid`
  at `:2541`, `return float(Jf.J())` at `:2544`; `Jf = SquaredFlux(new_surf, new_bs)`
  at `:3815`; `new_surf = plasma_geometry.working_surface` at `:3579`; exact reduction
  in `src/simsoptpp/integral_BdotN.cpp:102`). Plasma target + circulations load via
  `load_plasma_geometry(...)` (`stage2_geometry.py:258-285`); G from 20 TF coils
  (`create_equally_spaced_curves` at `banana_coil_solver.py:3484`, signed-G policy
  `current_contracts.py:37`), I via `physical_current_to_boozer_I`
  (`current_contracts.py:210`); flux label `s` from `--toroidal-flux` →
  `validate_normalized_toroidal_flux` at `banana_coil_solver.py:3517`. Offline-script
  template: `run_residue_probe.py`.
  **NOTE: `banana_coil_solver.py` is under active edit by another session — its line
  numbers drift; navigate by the symbol names above, not the line numbers.**
- **Winding-surface number discrepancy to resolve:** the on-spec contract is
  (R0, a) = (0.903, 0.142) (`hardware_contracts.py:71-72`), but the campaign's ruled
  target is (0.920, 0.143). The REGCOIL diagnostic must take R0/a as explicit args.

## Rationale

- Each item maps to a distinct payoff and a distinct subsystem, so they are
  independent and parallelizable; none shares a file with another's primary edit.
- Item 1 attacks the dominant physics failure mode (resonance-cliff island growth
  in a shearless field) DURING optimization rather than discovering it post-hoc in
  Poincaré — highest leverage for the stated goal of rebuilding ι≥0.08 at 0.143.
  Reusing `IotaShearShortfall`'s three-layer wiring and the existing `Iotas` adjoint
  keeps it to ~30 lines of genuinely new code (the multi-well J/dJ) with no new
  linear algebra.
- Item 2 is metric hygiene: it makes "survival N/50" comparable across candidates
  and tie to a real boundary, and closes a Goodhart corridor (a puffed-out
  candidate surface inflating its own survival). Additive + default-inert mirrors
  the proven `f81eb1aac` keyword-only style, so risk to existing verdicts is nil.
- Item 3 converts the order-3→4 vs move-the-surface decision from intuition into a
  measurement. It is offline and self-contained; the only real cost is the net-new
  REGCOIL LSQ kernel, justified because the alternative (vendoring upstream) drags
  in unbuildable C++ deps.

## Assumptions

- The outer-surface Boozer solve iota is stable (not branch-hopping) in the regime
  where item 1 is enabled; the campaign observed exact-Newton branch hops
  (label 0.0954 → exact 0.1186), and the repulsion's restoring direction is keyed to
  `sign(ι − p/q)`, so it must not run while iota is hopping. Verify with a short
  iota-stability probe before any production enablement. [ASSUMPTION — verify]
- For item 2, the design LCFS-clearance reference (inboard ≈0.761 / outboard ≈1.045)
  is the intended "fixed physical domain"; the vessel torus (0.976/0.222) is the
  harder hardware backstop. Default the new verdict to the LCFS-clearance surface,
  but expose the choice. [ASSUMPTION — confirm with user]
- For item 3, a from-scratch numpy REGCOIL kernel resolved to convergence in
  (mpol, ntor) reproduces a meaningful unrestricted lower bound on this winding
  surface; the single-shared-current banana family cannot reach it, so the result
  is a strict lower bound (interpret the gap as an upper bound on the deficit).
- The `field_objective` units (`0.5·mean((B·n)²·|n|)`) and the same scaled
  `working_surface` grid are used on both sides of the comparison.

## Implementation Plan

1. **Phase 1 — Audit-2: `IotaRationalRepulsion` objective term**
   - [ ] Add `IotaRationalRepulsion(Optimizable, depends_on=[iota_term])` to
         `banana_opt/single_stage_objectives.py`, mirroring `IotaShearShortfall`
         (`:79-131`): `J = 0.5·Σ_q w_q·max(0, δ_q − |ι − p/q|)²`; `dJ = g'(ι) ·
         iota_term.dJ(partials=True)` so the gradient flows through the `Iotas`
         adjoint. Handle the `|ι − p/q|` cusp with an explicit per-rational
         `sign(ι − p/q)` prefactor (mirror the `sign`/`shortfall==0` handling at
         `single_stage_objectives.py:123-125`).
   - [ ] Build the rational set by importing `enumerate_resonant_rationals` from
         `banana_opt/stage2_resonant_flux.py` (q≤13, band-aware, δ≈0.02 default).
         Do NOT reuse `nearest_low_order_rational` (single-nearest, denom=8,
         diagnostic-only).
   - [ ] Add `build_single_stage_rational_repulsion_objective(...)` returning the
         term or `None` (mirror `build_single_stage_shear_objective` `:134-149`).
   - [ ] Wire into `build_total_objective`: add params
         `JRationalRepulsion=None, RATIONAL_REPULSION_WEIGHT=0.0` (mirror `:407-412`)
         and a guarded add-block `if JRationalRepulsion is not None: objective =
         objective + RATIONAL_REPULSION_WEIGHT * JRationalRepulsion` (mirror `:466-467`).
   - [ ] CLI in `SINGLE_STAGE/single_stage_banana_example.py`: add
         `--rational-repulsion-weight` (+ `--rational-repulsion-delta` default ~0.02;
         `--rational-repulsion-qmax` default 13) following the `--shear-weight` /
         `--shear-target` pattern (`:2293-2312`), each with `os.environ.get` default
         "0.0"/off.
   - [ ] Bind to UPPERCASE module vars near `:13136` (next to `SHEAR_WEIGHT`).
   - [ ] Mirror `JShear`/`SHEAR_WEIGHT` at EVERY site (verify with
         `grep -n 'JShear\|SHEAR_WEIGHT' single_stage_banana_example.py` — ~27 sites
         as of 2026-06-12), NOT only the two below. Following only the bundle path
         leaves the term inert in the in-run loop. The known sites: construct in the
         bundle near `:7002` on `surface_iota_terms[-1]` (the OUTER surface leaf);
         forward into the imported `build_total_objective` call near `:7099`; add the
         param + guarded add-block to the LOCAL `build_total_objective` def at
         `:10948` (params near `:10986`); the in-run resolver
         `resolve_current_surface_objective_terms` (`:6457`); and the
         `globals().get("JShear")`/`globals().get("SHEAR_WEIGHT", 0.0)` forwarding at
         `:7664-7665`. Also thread it through the
         `build_single_stage_objective_bundle` call site (`:13197`) where `SHEAR_*`
         args are passed.
   - [ ] Confirm the term reads the POST-solve `res['iota']` (same value the adjoint
         linearizes, `surfaceobjectives.py:1032`) so value/gradient are
         self-consistent within an evaluation.

2. **Phase 2 — Survival-domain ownership: fixed reference domain**
   - [ ] `topology_scorer.py` `build_stopping_criteria` (`:834`): add keyword-only
         `reference_surface=None`; when provided, build a second
         `SurfaceClassifier(_full_torus_surface(reference_surface), h=0.03, p=2)` and
         append a second `LevelsetStoppingCriterion`; extend the returned
         `stop_labels` with a new `"reference_domain_exit"` (new
         `STOP_LABELS_VALIDATION_WITH_REFERENCE`), preserving index ordering used by
         `stop_reason_label`/`stop_reasons_indicate_broken` (`:772-794`). Do NOT add
         `reference_domain_exit` to `BROKEN_STOP_REASONS` (it is a real physical exit,
         like `surface_exit`).
   - [ ] `trace_metrics` (`:967`): derive a second `reference_survival_fraction`
         (lines hitting neither iteration-limit nor `reference_domain_exit`) and a
         `reference_validation_status`; new optional kwargs only.
   - [ ] `score_topology` (`:1388`) + `safe_score_topology` (`:1580`): add
         keyword-only `reference_surface=None`, thread into the
         `build_stopping_criteria` call (`:1439-1443`), add new results keys
         (`reference_survival_fraction`, `reference_survived_lines`,
         `reference_validation_status`, `reference_domain` metadata block) and
         matching stub keys in `empty_topology_score_result` (`:1324-1370`).
   - [ ] `POINCARE_PLOTTING/poincare_surfaces.py` (`:533-547`): construct the
         reference surface once via `build_banana_reference_surfaces(nfp=surf.nfp,
         banana_surf_radius=..., banana_surf_major_radius=...)` (import from
         `banana_opt/reference_surfaces.py`), pass into both `build_stopping_criteria`
         calls, thread through `build_poincare_render_modes` to the `trace_metrics`
         call (`:615-626`) and the `build_poincare_mode_artifact` keys (`:138-196`).
   - [ ] Default the strict-Poincaré reference to the LCFS-clearance surface; expose
         vessel vs LCFS as an explicit choice (env/flag). Keep existing
         `survival_fraction`/`validation_status` candidate-relative and unchanged.

3. **Phase 3 — REGCOIL floor diagnostic (offline, standalone)**
   - [ ] New `examples/single_stage_optimization/run_regcoil_floor.py` mirroring
         `run_residue_probe.py` structure: `configure_local_simsopt_imports` boilerplate
         (`:13-19`), `parse_args()` with positional run output dirs + `--report-path`
         (`:51-197`), `evaluate_output_dir(...)` (`:236-322`), `main()` writing a JSON
         sidecar (`:343-437`). Optionally factor the kernel into
         `banana_opt/regcoil_floor.py` for unit-testability (SRP).
   - [ ] Implement the REGCOIL LSQ kernel (net-new, ~150-300 LOC): single-valued
         current potential Φ on the winding surface in a Fourier basis
         {sin/cos(mθ−nφ)} + secular G·φ/2π and I·θ/2π terms; surface current
         K = n × ∇Φ on the winding-surface grid; Biot–Savart K→B·n inductance matrix A
         on the plasma grid; solve `min_x ‖Ax − b‖²` (λ→0 for the true floor, optional
         Tikhonov). Use `SurfaceRZFourier.gamma/gammadash1/gammadash2/normal`.
   - [ ] Reuse geometry/IO: load the plasma target via `load_plasma_geometry(...)`
         (`stage2_geometry.py:258`) with the SAME `s_working`, `target_lcfs_major_radius`,
         `nphi`, `ntheta`, wout path read from the run's `results.json`; build the
         winding torus via `build_banana_reference_surfaces(...).coil_winding_surface`
         OR a direct `SurfaceRZFourier` at explicit `--winding-major-radius`/
         `--winding-minor-radius` (default to contract 0.903/0.142, allow the campaign
         0.920/0.143 and sweeps).
   - [ ] Net currents: `G = μ₀·(20·tf_current_A)/(2π)` with the signed-TF convention
         (`current_contracts.py:37`); `I` via `physical_current_to_boozer_I(...)`
         (`current_contracts.py:210`) — 0 for the vacuum floor, matched proxy I for a
         finite-current run.
   - [ ] **Units lock:** compute the floor as `0.5·mean((Ax−b)²·|n|)` on the SAME
         scaled `working_surface` grid so it is directly comparable to the stored
         `results.json["field_objective"]` (`integral_BdotN.cpp:102`).
   - [ ] Emit JSON: `winding_surface`, `plasma_surface` (s, scale, grid),
         `net_currents` (G, I, convention), `regcoil` (mpol, ntor, n_basis, lambda,
         `floor_field_objective`), `achieved.field_objective`, `decomposition`
         (`achieved_over_floor_ratio`, `physics_floor`,
         `parameterization_deficit_upper_bound`), and the lower-bound `caveats` string.

## Validation Plan

- [ ] Phase 1 unit test: add `IotaRationalRepulsionTests` to
      `tests/geo/test_banana_objective_modules.py`, mirroring `IotaShearShortfallTests`
      (`:7303-7373`) and reusing the `_LinearIotaLeaf` stub (`:7279`) so no solved
      BoozerSurface is needed: value test (J>0 inside a δ_q band, J=0 outside all
      bands), and `test_rational_repulsion_gradient_matches_finite_difference`
      (central FD vs analytic dJ, `atol=1e-7`).
- [ ] Phase 1 byte-identity: a default-run objective graph is unchanged with weight 0
      (assert the `if JRationalRepulsion is not None` block is skipped); run the
      existing single-stage objective tests.
- [ ] Phase 1 in-loop reachability (catches the multi-site wiring gap — byte-identity
      will NOT): with `--rational-repulsion-weight > 0`, assert the term's contribution
      actually appears in the objective the IN-RUN loop minimizes (e.g. the in-run
      `resolve_current_surface_objective_terms` path and the local
      `build_total_objective` at `:10948`), not only in the bundle. A grep audit that
      every `JShear`/`SHEAR_WEIGHT` site has a matching `JRationalRepulsion`/
      `RATIONAL_REPULSION_WEIGHT` site is the cheapest form of this check.
- [ ] Phase 1 δ_q-overlap check: with q≤13 over ι∈[0.08,0.12], confirm adjacent
      wells (1/12≈0.083, 1/10=0.100, 1/9≈0.111) do not merge into a gradient-dead
      plateau at the chosen δ; reduce δ if they do.
- [ ] Phase 2 byte-identity: with `reference_surface=None`, assert
      `survival_fraction`/`validation_status` and all stop-label indices are
      bit-unchanged (mirror the `f81eb1aac` inert-default guarantee); run the topology
      scorer tests and `tests/geo/test_single_stage_example.py::HardwareConstraintTests`.
- [ ] Phase 2 functional: on one known candidate, assert the new
      `reference_survival_fraction` is produced and that a deliberately puffed-out
      candidate surface no longer inflates the reference number (Goodhart check).
- [ ] Phase 3 units: on a finished run, assert the diagnostic's plasma grid/scaling
      equals the run's and that an all-zero current potential reproduces the
      vacuum-only B·n baseline; verify `floor_field_objective` ≤
      `achieved.field_objective` (floor is a lower bound).
- [ ] Phase 3 convergence: sweep `--regcoil-mpol/--regcoil-ntor` and confirm the
      floor is resolution-converged before reporting.
- [ ] Gate regression (all phases): `cd /Users/suhjungdae/code/columbia/simsopt-surrogate
      && .conda-env/bin/python3.11 -m pytest tests/geo/test_banana_objective_modules.py
      tests/geo/test_single_stage_example.py::HardwareConstraintTests -q` stays green
      (currently 164 passed).

## Risks and Mitigations

- Risk: Phase 1 repulsion pushes the wrong way if the outer Boozer iota branch-hops
  between evaluations (sign keyed to `sign(ι − p/q)`).
  Mitigation: read post-solve `res['iota']` (value/gradient self-consistent per eval);
  keep δ_q narrow so wells don't merge; gate behind default-0 weight; run an
  iota-stability probe and do NOT enable while iota is hopping.
- Risk: Phase 1 multi-well term creates a gradient-dead plateau or fights the
  `--iotas-weight` target (default 100), stalling L-BFGS.
  Mitigation: FD gradient test + δ-overlap check above; treat repulsion as a small
  perturbation on the anchored iota target; one-rung A/B before campaign use.
- Risk: Phase 2 extended stop-label indexing drifts from
  `stop_reason_label`/`stop_reasons_indicate_broken`, corrupting verdicts.
  Mitigation: add a test asserting label→index mapping for both the with- and
  without-reference label lists; default param `None` keeps the old path bit-identical.
- Risk: Phase 3 floor is computed with mismatched units or grid → meaningless ratio.
  Mitigation: units-lock test (identical `0.5·mean((B·n)²·|n|)` reduction on the same
  scaled `working_surface`); zero-current baseline test; convergence sweep.
- Risk: Phase 3 over-read — REGCOIL floor is an UNRESTRICTED optimum the
  single-shared-current banana family cannot reach, so `achieved − floor`
  over-attributes to "coil-order deficit".
  Mitigation: report it explicitly as a lower bound; label the deficit
  `parameterization_deficit_upper_bound`; carry the caveat string in the JSON.
- Risk: editing `single_stage_banana_example.py` (Phase 1) collides with the
  in-flight shared multi-coil refactor.
  Mitigation: coordinate with the owning session first; keep the validation gate
  green before/after; never patch their in-flight code unilaterally.

## Completion Criteria

- [ ] `IotaRationalRepulsion` implemented, wired default-OFF, FD-gradient test green,
      default objective graph byte-identical with weight 0.
- [ ] Fixed-reference-domain survival verdict produced (in-run scorer + strict
      Poincaré) with new `reference_*` keys; existing candidate-relative verdict
      bit-unchanged when no reference surface is passed.
- [ ] `run_regcoil_floor.py` runs on a real artifact dir and emits the floor +
      decomposition JSON in `field_objective` units, floor ≤ achieved, convergence
      checked.
- [ ] Full gate (`test_banana_objective_modules.py` +
      `test_single_stage_example.py::HardwareConstraintTests`) green.
- [ ] No `src/simsopt/**` changes; changes committed in scoped sets with tests, on
      user instruction.

## Open Questions

- Item 1: q-cap — reuse `MAX_RESONANT_DENOMINATOR=13` directly, or expose
  `--rational-repulsion-qmax` (default 13)? And single δ vs per-q δ_q?
- Item 1: should the term ever be enabled in the live spec-143 lanes, or stay a
  research/insurance lane until the iota-stability probe passes? — campaign decision.
- Item 2: default fixed reference = LCFS-clearance (≈0.761/1.045) vs vessel
  (0.976/0.222)? Recommend LCFS-clearance for the verdict, vessel as a second
  optional report. — confirm with user.
- Item 2: should `reference_domain_exit` ever count toward promotion failure, or
  remain report-only initially? Recommend report-only first.
- Item 3: evaluate the floor at the contract winding (0.903/0.142), the campaign
  ruled form (0.920/0.143), or sweep both for the standoff lever? Recommend a small
  R0/a sweep since that is the decision the diagnostic informs.
- Item 3: factor the LSQ kernel into `banana_opt/regcoil_floor.py` (unit-testable)
  vs inline in the script? Recommend the module for SRP/testability.
