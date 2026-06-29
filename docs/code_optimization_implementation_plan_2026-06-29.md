# Single-Stage Code Optimization Plan (2026-06-29)

## Purpose

Consolidate the performance defects found in two independent audits (an empirical
micro-event diagnosis of a 30+min A100 pre-optimization stall, plus a 5-lens
upstream "slow-code" sweep and a 70-agent AD-redundancy audit) into one
execution-ready plan. It supports fixing the production single-stage stall and
the structural costs behind it, while explicitly fencing off the audit findings
that are phantom, latent, or unsafe in **this** repo
(`simopt-jax-clean-local`, the canonical single-stage prod checkout).

## Goals

- Eliminate the per-successful-step minutes-long stall in the startup/per-step
  hardware-validity audit on the default JAX single-stage lane.
- Remove the O(n³) coil-spec rebuild amplifier so host geometry views
  (hardware audit, `kappa`, startup snapshots) are O(1)-amortized per DOF state.
- Cut startup snapshot cost (~55s currents / ~106s geometry, observed).
- Keep every fix on the optimizer objective/gradient **bit-identical**; keep
  reporting-only fixes value-preserving (no loss of the clearance margin).
- Pay down the systemic per-arg/per-output reverse-mode AD redundancy in the
  lanes that actually use it (force / framed-curve / ALM), without touching the
  single-stage default objective.

## Non-Goals

- Swapping the upstream CPU `CurveCurveDistance`/`CurveSurfaceDistance` for the
  `*JAX` adapters. **Verified red herring**: `CurveCurveDistanceJAX.shortest_distance`
  (`src/simsopt_jax_adapters/geo/curve_objectives.py:411`) and the `*JAX` twin
  (`:513`) are *also* unconditional host `cdist`, their candidate culler is dead,
  and they are not on any production caller.
- Skipping the `boozersurface.py:719` residual recompute. **Verified unsafe**:
  it is the failure/rollback branch (`persist_solved_state=False`, lines 715-724),
  not per-eval, and its residual feeds `resdict['gradient']` which IS consumed
  (rolled-back candidate gradient, cf. commit `6a2d2208b`). Not a free win.
- Fixing `banana_opt/self_intersect.py` or `banana_opt/fold_buildability.py`
  (`RotationAwareCurvatureExcessPenalty`). **These files do not exist in this
  repo** (`find`/`grep` = 0 hits); that audit scanned a different checkout.
- Changing physics/selection/smooth-min math anywhere (perf/memory restructure only).

## Current Context

- Default single-stage lane: `bs = SingleStageRuntimeSpecBiotSavartJAX`
  (`examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:14658`;
  the "example/driver" below) ⇒ coils are
  `SpecBackedCoil` with `SpecBackedCurve`/`SpecBackedCurrent` views.
- Hardware audit SSOT: `_evaluate_single_stage_hardware_status`
  (`single_stage_banana_example.py:1686-1700`) calls `cc/cs/surf.shortest_distance()`
  + `banana_curve.kappa()`. Invoked at startup (`:15538`), per **successful**
  step (`:13439`, gated by Boozer-success + non-self-intersecting), and at
  finalization (`:12883`). Reporting-only: feeds `run_dict["hardware_constraint_status"]`;
  the optimizer's `J()/dJ()` never depend on it.
- `objectives["cc"]` / `["cs"]` are the **upstream CPU** classes imported at
  `single_stage_banana_example.py:128-131` and built at `:9068-9069` with
  `downsample=1`, `num_basecurves=len(curves)` (~21–30 curves: 20 TF + banana).
- `objectives["surf"]` is the JAX adapter `SurfaceSurfaceDistance`
  (`src/simsopt_jax_adapters/geo/surface_objectives.py`).
- ALM lane is **non-default** (`CONSTRAINT_METHOD` default `penalty`,
  example `:4617`; ALM gated `:16001`); its constraint helpers live in
  `examples/single_stage_optimization/banana_opt/single_stage_constraints.py`.

## Rationale

Two audits converged on one shared, real, hot finding (the cc/cs empty-candidate
brute force) but each missed the other's key point. The micro-event run proved
the stall is in `cc.shortest_distance()`; the upstream sweep proved *why* it is
minutes not milliseconds (uncached per-pair `.gamma()` ⇒ O(n³) coil-spec
rebuilds); the AD audit found systemic `grad(argnums=k)`-per-arg redundancy that
is real but, on validation, mostly **latent** for the single-stage default lane.
Prioritizing by *reachability on the healthy default path* (not raw site count)
puts the candidate-gated brute force + the uncached-gamma amplifier first, the
ALM memory restructure second (only if ALM is used), and the AD sweep last.

## Assumptions

- `_coil_dof_state_token` (`biotsavart_backend.py:787` init, `:872` refresh) is
  refreshed on **every** mutation of `owner.x` / coil DOFs. **Must be verified
  before P1.3 lands** — a stale memo here would corrupt the optimizer.
- In the healthy/converged production case, coils are pairwise farther than
  `cc_dist`/`cs_dist`, so `compute_candidates()` returns empty and the brute-force
  fallback is the path taken (consistent with the observed stall).
- Cost models below are **static estimates** (per-call cost × frequency), not
  GPU-profiled on this machine. Wall-time claims are to be confirmed in Validation.
- Surf sizes: boozer plasma surface `nphi*ntheta = 255*64 = 16320`; vessel
  `SurfaceRZFourier` default `61*62 = 3782`.

## Implementation Plan

### Phase 1 — Single-stage default hot path (production stall) [PRIORITY]

1. **cc/cs empty-candidate brute force — value-preserving gamma-hoist** (reporting-only; upstream `src/simsopt/geo/curveobjectives.py`)
   - [ ] `CurveCurveDistance.shortest_distance` (`:259-265`): materialize
     `gammas = [c.gamma()[::self.downsample] for c in self.curves]` **once**, then
     index `gammas[i]/gammas[j]` in the `cdist` double loop (870→~30 `.gamma()`
     calls at n=30). Do **not** change the returned value (keep exact min).
   - [ ] `CurveCurveDistance.shortest_distance_among_candidates` (`:253-257`):
     same one-shot `gammas` hoist (also re-fetches `.gamma()` per candidate pair).
   - [ ] `CurveSurfaceDistance.shortest_distance` (`:374-380`) +
     `shortest_distance_among_candidates` (`:368-372`): hoist the per-curve
     `.gamma()` and reuse the single `xyz_surf` already computed; add a
     `downsample` stride (class `__init__ :348` currently has none — add the param,
     default 1, applied to both curve and surface samples in the fallback).
   - [ ] Do **not** apply the "return `self.minimum_distance` when candidates empty"
     shortcut — it discards the exact clearance margin used in
     `threshold_margins`. Keep the exact computed min.

2. **surf shortest_distance — host-numpy blocked cdist** — DONE (uncommitted)
   - [x] `SurfaceSurfaceDistance.shortest_distance` rewritten to host blocked
     `cdist` (`src/simsopt_jax_adapters/geo/surface_objectives.py`); dead
     `_shortest_distance` jit + orphaned `pairwise_min_distance_pure` import removed;
     test strengthened from `>= 0.0` to brute-force match. Secondary (the stall is cc),
     but keep it: it removes a real reporting-only jit compile.

3. **O(n³) amplifier — memoize the coil-spec rebuild** (highest structural leverage; `src/simsopt_jax_adapters/field/biotsavart_backend.py` + `src/simsopt_jax/core/field.py`) [HIGHER RISK — gates on the assumption above]
   - [ ] FIRST: enumerate every site that refreshes/sets `_coil_dof_state_token`
     and confirm it covers all `owner.x` mutations (grep `_coil_dof_state_token`,
     `_new_coil_dof_state_token`; sites incl. `:787`, `:872`, `:1484`, `:1510`).
     Block the rest of P1.3 until this holds.
   - [ ] Add an owner-level memo: cache the result of
     `coil_specs_from_dof_extraction_spec(owner.coil_dof_extraction_spec(), owner.x)`
     on the owner keyed by `_coil_dof_state_token` (recompute only when the token
     changes). Route `SpecBackedCurve._current_curve_spec` (`:453-457`),
     `SpecBackedCurrent.get_value` (`:403-408`), and `SpecBackedCoil` (`:771`)
     through it. Turns O(n)-per-access into O(1) amortized.
   - [ ] Add a combined `gamma_and_gammadash()` accessor: `gamma()` (`:480-482`)
     and `gammadash()` (`:484-486`) each call `curve_gamma_and_dash_from_spec`
     and discard half — collapse the 2× compute when both are needed
     (e.g. in `snapshot_static_tf_geometry`).
   - [ ] Keep the traceable JAX value/grad path untouched (this is host-view only).

4. **Audit caller trim** (reporting-only; `single_stage_banana_example.py:1686-1700`)
   - [ ] Subsumed by P1.1+P1.3 for cost; additionally, evaluate computing the three
     minima from the JAX gammas already resident from the solve (device
     min-distance) instead of host `cdist`, OR run the full audit **once at
     finalization** rather than every successful step (`:13439`). Pick the smaller
     diff that preserves the per-step `hardware_constraint_status` semantics the
     promotion gate relies on.

### Phase 2 — ALM lane (off default; optimizer value+grad — bit-identical restructure only)

5. Only relevant under `--constraint-method alm`. Fix worst-first; each must return
   numerically identical value AND gradient (keep the selected set a superset of
   entries within `hard_min + 4*temperature`).
   - [ ] `single_stage_constraints.py:124-140` (curve-surface): two-pass streaming
     (pass 1 global `hard_min` from per-curve mins, discard blocks; pass 2 recompute
     per curve, apply selection, accumulate grad via `np.add.at`, discard). ~2GB→~67MB.
   - [ ] `single_stage_constraints.py:184-214` (surface-surface): chunk the
     `[16320,3782,3]` pairwise over rows (mirror `SurfaceSurfaceDistance` block=1024)
     so the 1.48GiB array is never materialized.
   - [ ] `single_stage_constraints.py:67-83` (curve-curve): `gamma_i` is already hoisted
     (`:76`) but `gamma_j` is re-fetched inside the inner loop (`:79`, ~n(n-1)/2≈435
     uncached re-fetches, banana-dominated) and every `(i,j,diffs,dists)` is retained in
     `pair_blocks` (`:83`). Precompute all `gammas` once + stream per-pair blocks; skip
     constant fixed-TF/TF pairs or add a sopp prefilter.

### Phase 3 — Systemic AD redundancy (codebase hygiene; latent for single-stage default)

6. Per-arg/per-output reverse-mode passes that re-run forward+backward N× where one
   multi-primal `grad`/`vjp` suffices (numerically identical). Scope to the lanes
   that use each; **do not** touch the single-stage default objective.
   - [ ] `src/simsopt/geo/framedcurve.py:57-82` (and `:237+`): collapse
     `binormgrad_vjp0..5` / `torsiongrad_vjp0..5` (12 per-input `vjp` jits) into one
     `vjp` returning all 6 input cotangents. (Mechanism is per-input `vjp`, not
     `grad(argnums)`.) Gate: only when a finite-build/framed lane is exercised.
   - [ ] `src/simsopt/field/force.py`: collapse the separate `grad(self.J_jax, argnums=k)`
     jits into one `grad(argnums=(...))` in `B2Energy.dJ` (class `:456`, dJ `:537`, jits
     `:502/508/514`), `NetFluxes.dJ` (`:666`/`:752`, jits `:720/726`), `SquaredMeanForce.dJ`
     (`:897`/`:1030`, jits `:984-986`), `LpCurveForce.dJ` (`:1205`/`:1349`), `LpCurveTorque`
     (`:1521`/`:1665`), `SquaredMeanTorque` (`:1818`/`:1954`). Gate: force-aware lanes only.
     (NOTE: the external audit's `MeanSquaredForce`/`LpCurveForce.dJ:97-159`/`:209-271` do
     not exist in this repo — that audit scanned a different checkout; the analogue here is
     `SquaredMeanForce`. Lines 97-159/209-271 are `_pure` helper bodies, not `dJ`.)
   - [ ] `src/simsopt/geo/curveobjectives.py:236-239` (cc `dJ_dgamma1/dl1/dgamma2/dl2`;
     `:235` is the `J_jax` value jit) + cs equivalents: one `grad(argnums=(0,1,2,3))`
     instead of 4 jits. Low single-stage payoff (candidate-gated: `dJ` loops empty
     candidates → returns 0 in the healthy case), but tidy and helps the close-coil regime.

## Validation Plan

- [ ] Ruff clean on every edited file: `ruff check <files>`.
- [ ] P1.1 correctness: extend `tests/geo/test_curve_objectives.py` to assert
  `CurveCurveDistance.shortest_distance()` and `CurveSurfaceDistance.shortest_distance()`
  equal an independent brute-force min (both empty- and non-empty-candidate cases),
  bit-identical pre/post hoist.
- [ ] P1.1/P1.3 no-regression on the objective: assert `J()`/`dJ()` byte-identical
  before/after (the shared `JF` must not move) — reuse the cross-backend parity test
  (`tests/integration/test_single_stage_objective_parity.py`).
- [ ] P1.3 memo safety: a test that mutates `owner.x`, reads `gamma()`/`get_value()`,
  and asserts the memo returns the **new** geometry (no staleness) across every
  token-refresh site; plus a Boozer/optimizer-step parity check vs pre-memo.
- [ ] P1 timing gate (GPU): time `_evaluate_single_stage_hardware_status` and the
  `snapshot_static_tf_*` startup steps before/after; expect the per-step audit to drop
  from minutes to <1s and reach the first value/grad promptly (`--benchmark-mode` as
  the A/B reference for "audit skipped").
- [ ] P2 (if ALM touched): ALM inner value+grad bit-identical vs current
  (`tests/integration/test_stage2_jax.py` + a fresh constraint-level parity test);
  peak-RSS check confirming the memory drop.
- [ ] P3: per-site, assert the collapsed `grad`/`vjp` equals the sum of the old
  per-arg results to machine precision before deleting the old jits.
- [ ] Full suite green under the repo interpreter (`PYTHONPATH=src python3 -m pytest`
  on the touched modules; note `...chunking_matches_dense` is a pre-existing,
  unrelated module-binding red).

## Risks and Mitigations

- Risk: P1.3 memo serves stale coil specs if `_coil_dof_state_token` misses a
  mutation path → silent optimizer corruption.
  Mitigation: gate P1.3 on the token-coverage audit task; add the mutate-then-read
  staleness test; land P1.1 first (fixes the stall without touching the optimizer adapter).
- Risk: editing vendored upstream `curveobjectives.py` diverges from upstream simsopt.
  Mitigation: keep P1.1 reporting-only and value-preserving (J/dJ untouched), minimal
  diff, documented; consider upstreaming.
- Risk: P2 ALM restructure changes the gradient (selection-set boundary).
  Mitigation: keep selected set a superset within `hard_min + 4*temperature`; bit-identical
  value+grad test as a merge gate.
- Risk: "optimization" that targets latent code (force/framed) yields zero
  single-stage benefit and adds churn.
  Mitigation: Phase 3 is explicitly gated behind the lanes that exercise it and ranked last.

## Completion Criteria

- [ ] Default single-stage lane reaches the first optimizer value/grad without a
  multi-minute pre-optimization audit; per-successful-step audit is <1s.
- [ ] Startup `snapshot_static_tf_*` no longer dominated by repeated full coil-spec rebuilds.
- [ ] `J()`/`dJ()` and (if touched) ALM value+grad proven bit-identical to pre-change.
- [ ] New correctness + staleness tests green; ruff clean; touched-module pytest green.
- [ ] Plan's Non-Goals respected (no CPU→JAX swap, no boozersurface residual skip,
  no phantom-file work).

## Open Questions

- Does `_coil_dof_state_token` provably cover every `owner.x` mutation? (Blocks P1.3.)
- P1.4: is per-step `hardware_constraint_status` actually consumed by the promotion
  gate every step, or is finalization-only sufficient? (Determines whether we can
  drop the per-step audit entirely vs. only make it cheap.)
- Are the static cost estimates borne out on GPU (esp. cc minutes, ALM ~2GB peak)?
  Needed to confirm prioritization and the timing gate thresholds.
- Should P1.1 be upstreamed to simsopt, or kept as a local vendored patch?
