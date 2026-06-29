# Stage-2 order-64 conditioning plan (Sobolev mode-scale + optional trust-region)

> Scoped 2026-06-28. Status: **SUPERSEDED for the active lane** by
> `docs/stage2_sobolev_metric_preconditioner_implementation_plan_2026-06-29.md`.
> This document remains the historical diagonal mode-scale + trust-region plan:
> the diagonal Lever A sweep was executed and found insufficient, and Lever B
> trust-region remains the fallback only if the successor non-diagonal metric
> diagnostic cannot produce `first_step_dJ < 0`.

## 1. Problem (measured, not assumed)

The edge-iota Stage-2 run (`banana_coil_solver.py`, penalty/L-BFGS-B path) cannot
take a first step from the order-64 slid_clean chomp seed. The
`--diagnose-seed-gradient` probe (committed `e61c2c656`) measured, at the seed:

- `n_dofs=271`, `J0=2.69e4`, **`‖grad‖₂=3.79e7`** (`‖grad‖∞=2.41e7`).
- `scale = identity` (winding-only scale map is empty here).
- raw `-grad` descent: `dJ=-38` @ε1e-6, `-375` @ε1e-5, **`+7.7e6` @ε1e-4** — the
  descent window is microscopic.
- untruncated first step `J(x0 - grad·scale²) - J0 = +1.09e31` — catastrophic
  overshoot.
- VERDICT: `descent_exists_along_minus_grad`.

**This is NOT a converged vertex** (‖grad‖ is enormous), **not weak edge coupling**
(the hardware gradient is what's huge), **not a scaling artifact** (scale is
identity). It is the classic **n²-Fourier ill-conditioning of high-order curve
modes** — the same root cause documented for order-64 slid_clean
(`memory/project_adam_baseline_unstable_order64_slid_clean_2026-06-23.md`: adam
diverges at step 0, κ 36.7→10308; prescription "trf + sobolev_h2").

### Where the gradient actually comes from (corrected)

The 3.79e7 gradient is **not** SquaredFlux/field-matching (bare SquaredFlux at this
seed is ~1e-3). It is dominated by the **high-weight clearance/keep-out hinges** —
`CurveSurfaceDistance` (weight 5000), and the on-by-default swept Type-KK
**hardware + vessel keep-out** (weight 1000 each, `hardware_contracts.py:156,165`) —
propagated through the **258 banana-curve `CurveCWSFourier` Fourier DOFs**
(`phic/phis/thetac/thetas(0..64)`, `curvecwsfourier.py:351-375`). The curvature
(`LpCurveCurvature`, w200) and fold (`CurveSurfaceGeodesicCurvature`) penalties are
**inactive** at the seed (κ 35.88 < threshold) but **re-activate the instant a step
excites high-k modes** — that is the `+7.7e6` overshoot at ε1e-4 (their n⁴ Hessian
stiffness). Either way **the conditioning target is the high-order curve modes**, so
a mode-order-dependent preconditioner is correctly aimed.

The chomp seed is a curvature-cleanup output evaluated against a *different* plasma
target (`wout_s01_1f082f`) with the swept finite-build pack + on-by-default
keep-outs (`FIELD_ERROR=0.0799` here vs ΔB 0.068% in its native context), so it sits
where the weight-5000/1000 hinges are active and stiff — far from *this* objective's
minimum.

## 2. Approach

Two independent levers; do them in order, the first is the primary fix.

### Lever A (PRIMARY, surgical) — Sobolev mode-scale on the EXISTING `u=x/scale` hook

The penalty path already runs L-BFGS-B under a `u = x/scale` transform:
`run_scaled_winding_minimize(minimize, fun, dofs, scale=build_winding_dof_scale_vector(JF.dof_names, winding_dof_scale_map), ...)` (`banana_coil_solver.py:5744`,
transform at `single_stage_geometry.py:1423-1472`). The first step in x-space is
`dx = -grad·scale²` (already reported by the diagnostic). Today `scale` is identity
on curve DOFs because `WINDING_DOF_CORRIDOR_SCALE_MAP` only matches winding-surface
suffixes (`rc(0,0)`, …; `stage2_geometry.py:85-89`).

**Fix:** build an order-dependent scale `scale_k = 1/(1+α·k^q)` for the curve Fourier
DOFs (k = mode order parsed from the `phic(k)/phis(k)/thetac(k)/thetas(k)` suffix),
composed multiplicatively with the existing winding map; leave the ~13
current/winding DOFs at 1.0. With `grad_k ~ k²` (stiff modes) and
`scale_k = 1/(1+α·k²)`, the step `dx_k ~ k²·scale_k² → O(1/k²)` — the blow-up is
damped exactly on the stiff modes. L-BFGS-B is **not** scale-invariant (unlike Adam),
so this diagonal preconditioner should bite. **Caveat — this combination is a
hypothesis, not yet proven:** the prior art validated the Sobolev scale only as
scipy_trf's `x_scale` (a trust-region least-squares solver), where it unlocked trf
(`FINDINGS:122,134`). Applying the same diagonal scale to the L-BFGS-B `u=x/scale`
path here is a NEW pairing whose first-step efficacy is exactly what Phase 2 measures
(`first_step_dJ < 0` under the scale) BEFORE any full run. If the scale alone does not
flip the first step to descent, escalate to Lever B (trust-region).

- Reuse the validated formula `_sobolev_x_scale(dof_modes, alpha)=1/(1+α·k²)`
  (`geometry_cleanup.py:429-443`, the H¹ form; α=1→κ33.05, α=4→κ32.60 on the
  geometry_cleanup curvature objective) and the mode-parse idiom
  (`geometry_cleanup.py:244`).
- `build_winding_dof_scale_vector` currently does exact-suffix match only
  (`single_stage_geometry.py:1410-1419`); extend it (or add a sibling builder) to
  also parse `phic/phis/thetac/thetas(k)` and apply `1/(1+α·k^q)`.
- **Default-off / byte-identical when off** (α=0 or flag absent → identity, exactly
  as today).

### Lever B (COMPLEMENTARY, more robust) — trust-region branch

scipy `minimize(method="trust-constr", jac=True)` drops straight into the existing
scalar `fun→(J,grad)` interface (`minimize` already imported,
`banana_coil_solver.py:20`), with `Bounds(lower,upper)` from `lbfgsb_bounds` and a
tunable **`initial_tr_radius`** that caps the first step a priori (the principled cure
for the overshoot). ~15 lines: one argparse pair (`--stage2-optimizer trust-region`,
`--stage2-initial-tr-radius`) + one `elif` in the dispatch ladder before the penalty
`else` (`banana_coil_solver.py:5743`); `selected_result_x` falls back to `res.x`
(`:5763`) so no downstream change. Pairs well with Lever A (precondition AND bound the
step). A cheaper variant is the in-repo ALM box-trust-region proxy
(`alm_utils.py:1827`, `_build_box_bounds`) wrapped around the L-BFGS-B call.

### Rejected options

- **`least_squares(method="trf")` for Stage-2** — no residual vector exists; the
  Stage-2 `JF` is a scalar `(J,grad)`. Would require re-expressing SquaredFlux + all
  penalties as a JAX residual + Jacobian (the geometry_cleanup trf works only because
  it is a tiny JAX curvature objective). High-risk rewrite. NO.
- **Reuse `geometry_cleanup` as a pre-conditioning Stage A** — it freezes the field
  and optimizes a curvature-only JAX objective on curve dofs (no currents, no flux);
  it cannot reach this objective's field minimum, and its own adam diverges at step 0
  on order-64. The conditioning must wrap the *Stage-2 field objective*. NO.
- **Add a Sobolev/MeanSquaredCurvature regularization TERM to `JF`** — changes the
  minimizer, needs weight tuning, and adds *more* n²-carrying gradient rather than
  conditioning it. NO.

## 3. Two-stage edge steering (why conditioning is a prerequisite)

`grad_edge=334 ≪ grad_hw=3.79e7` (at edge weight 1e2): the edge hinge is dynamically
irrelevant until the hardware gradient is near zero. So edge steering only bites
after the hardware objective is conditioned and near its minimum. Plan:

- **Stage A:** conditioned solve, **edge weight 0** (`--stage2-edge-iota-mode report`),
  → hardware/field minimum at order 64.
- **Stage B:** conditioned solve from A's output, **modest `--stage2-edge-iota-weight`
  + `--stage2-edge-iota-hinge linear`** (the linear hinge, committed `e61c2c656`, is
  designed exactly to drive a coil off a converged hardware minimum).

Both stages use the same conditioned solver (Lever A ± B) in `banana_coil_solver.py`.

## 4. Phases & gates

- **Phase 0 — confirm the target (cheap, local).**
  - Dump `JF.dof_names` for the order-64 seed; confirm 258 curve modes with
    `phic/phis/thetac/thetas(k)` suffixes and the ~13 non-curve DOFs. Gate: the
    mode-parse must cover exactly the curve block.
  - Record the baseline diagnostic numbers (already have: ‖grad‖=3.79e7, first_step
    +1.09e31).

- **Phase 1 — Sobolev curve-mode scale (Lever A).**
  - New/extended scale builder: parse k, `scale_k = 1/(1+α·k^q)`. **q=2 is the
    default and the validated form** — `_sobolev_x_scale = 1/(1+α·k²)`, which the
    source labels H¹ (`1+k²`; the legacy `sobolev_h2` name is a misnomer,
    `geometry_cleanup.py:435`). Expose the exponent so q∈{2 (H¹, validated), 4
    (H²-like, more aggressive `1/(1+α·k⁴)`)} is selectable (q=1 is **not** a Sobolev
    metric — do not offer it). Compose with the winding map. SSOT: reuse the
    `_sobolev_x_scale` formula.
  - CLI/env: `--stage2-curve-mode-precondition` (off by default) +
    `--stage2-sobolev-alpha` (default 0 ≡ off) + optional `--stage2-sobolev-power`
    (q, default 2; named `-power` not `-order` to avoid colliding with the curve
    `--order 64`). Wire into the penalty-path `scale=` at `banana_coil_solver.py:5748`
    AND the diagnostic at `:5605` (so the probe measures the conditioned gradient).
  - Tests: builder unit tests (mode parse, `1/(1+αk²)`, identity when α=0,
    composition with winding map, byte-identical-off); a diagnostic-level test that
    `scaled_grad_norm_2` drops and `first_step_dJ` flips negative under the scale on a
    synthetic order-N objective.
  - Gate (Tier-1b/2, crucible): default-off byte-identical; no behavior change to ALM
    / basin-hops (scale is penalty-path only — document it).

- **Phase 2 — VALIDATE WITH THE DIAGNOSTIC BEFORE ANY FULL RUN (the cheap filter).**
  - Run `--diagnose-seed-gradient --stage2-curve-mode-precondition
    --stage2-sobolev-alpha {1,4,16}` on Perlmutter `shared` (each ~12 min, the probe
    skips the optimizer). PASS = `scaled_grad_norm_2` drops by orders of magnitude AND
    `first_step_dJ < 0` (the untruncated first step now descends). This tells us the
    α/q that makes L-BFGS-B's first step viable WITHOUT a multi-hour run. If no α
    makes `first_step_dJ < 0`, escalate to Lever B (trust-region) before a full run.

- **Phase 3 — (conditional) trust-region branch (Lever B).** Only if Phase 2 shows
  the diagonal scale alone is insufficient (first_step still overshoots). Add
  `--stage2-optimizer trust-region` (trust-constr + `initial_tr_radius`). Validate the
  same way (diagnostic, then a short capped run).

- **Phase 4 — two-stage conditioned run (Perlmutter `shared`).**
  - Stage A: `report` mode, conditioned, maxiter ~100 → confirm it MOVES
    (MAX_CURVATURE / FIELD_ERROR change from seed; OPTIMIZER_SUCCESS true) and reaches
    a buildable hardware min (κ<43.31, keepout clear, vacuum).
  - Stage B: `soft`+`linear` from A's output, modest weight → run the post-run edge
    p10 oracle (`post_run_edge_iota_oracle_check.py`). WIN = p10≥0.10 ∧ survival>0;
    else documented negative (now a *real* steering result, not a step-0 abort).

## 5. Risks / decision points

- **q and α are regime-dependent** (`stage2_geometry.py:76-84` warns a scale <1 can
  under-converge soft DOFs in cold/full-convergence regimes — but here shrinking
  high-mode steps is the goal). Phase 2's diagnostic sweep tunes them cheaply.
- **Penalty-path only.** The scale is not applied under ALM/basin-hops; the two-stage
  plan uses the penalty path (CM=penalty, as the failing run does). Document loudly.
- **Conditioning ≠ confinement.** Even a successful conditioned steer is judged by the
  p10 trace oracle; the chaotic edge may still give a negative (the steering is a
  proxy, not a guarantee). That's the existing, honest framing.
- **Upstream/parity:** Lever A touches `single_stage_geometry.py` + `banana_coil_solver.py`
  (examples tree, not `src/simsopt`), so no upstream-mirror concern.

## 6. Effort

- Phase 1: small (extend one builder + 1 CLI group + thread into 2 call sites +
  ~4 unit tests). ~½ day + crucible.
- Phase 2: cheap (3 diagnostic jobs, no code). Same day.
- Phase 3: small (~15 lines) and CONDITIONAL.
- Phase 4: 2 shared-QOS runs + oracle judgment.

## 7. Pointers

- Diagnostic + L1 hinge: commit `e61c2c656`; `banana_opt/stage2_objectives.py`
  (`diagnose_seed_gradient`, `_add_stage2_edge_iota_objective`).
- The hook: `single_stage_geometry.py:1398-1472` (`build_winding_dof_scale_vector`,
  `run_scaled_winding_minimize`); map at `stage2_geometry.py:85-89`.
- Sobolev formula + mode parse: `geometry_cleanup.py:429-443`, `:244`.
- Curve DOF names: `src/simsopt/geo/curvecwsfourier.py:351-375`.
- Prior order-64 art: `autoresearch` `memory/project_adam_baseline_unstable_order64_slid_clean_2026-06-23.md`;
  `campaigns/.../slid_clean_cws_conversion_FINDINGS_2026-06-23.md`;
  `docs/geometry_cleanup_additive_optimization_plan_2026-06-23.md`.
- Edge-iota lane: `docs/edge_delivered_iota_lane_implementation_plan.md`;
  launch pkg `autoresearch/campaigns/balance_pareto_singlestage_2026-06-17/perlmutter/edge_iota_soft/`.
