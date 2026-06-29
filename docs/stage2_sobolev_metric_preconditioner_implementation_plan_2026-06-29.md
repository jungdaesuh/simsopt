# Stage-2 Order-64 Sobolev *Metric* Preconditioner — Implementation Plan

> Created 2026-06-29 · Tier-3 (root-cause conditioning cure) ·
> Successor to `docs/stage2_order64_sobolev_conditioning_plan_2026-06-28.md`
> (which covered Lever A = diagonal mode-scale, Lever B = trust-region).

## Status / Precedence

This plan records the metric lane for the order-64 slid-clean Stage-2
conditioning fix and its current fallback state. Local source is implemented in
`5dff32284` with follow-up diagnostic/test hardening in `45d2b3ae1`, review
hardening in `735400361`, and ALM Sobolev-alpha fail-closed hardening in
`cc91579dd`, with fallback status wording refreshed in `0d4647a46`; the
Perlmutter launch package and handoff are active in
`autoresearch` commits `39758405`, `5cbd31cd`, `d499d62f`, and wrapper
soft-control forwarding commit `df39e677`, with fallback handoff status refreshed
through `d51795ae`. The required remote gate is still
decisive: the H1 and H2 beta=1 Perlmutter sweeps on `shared`
assembled the metric but failed the descent gate for alpha 1, 4, 16, and 64.
Every `first_step_dJ` stayed positive (H1: `+1.933950e10`, `+1.941215e10`,
`+2.122981e10`, `+1.812255e10`; H2 beta=1: `+1.983622e10`,
`+1.903104e10`, `+2.202494e10`, `+1.702834e10`). The metric lane therefore
did not produce a passing real-seed diagnostic. The documented trust-region
fallback smoke completed as Perlmutter job `55271978` (`COMPLETED|0:0`) and
proved route engagement (`CONSTRAINT_METHOD='alm'`, `EDGE_IOTA_MODE='soft'`,
`HARDWARE_CONSTRAINTS_OK=True`), but it is not physics closure: the smoke
reported `EDGE_IOTA_STATUS='insufficient_samples'`. The full ALM fallback job
`55273370` is running on `regular_1` (started `2026-06-29T16:10:10`, 6h
limit) and has not yet produced a physics result.

## Purpose

The Stage-2 banana-coil penalty solve aborts at step 0
(`ABNORMAL_TERMINATION_IN_LNSRCH`) on the order-64 slid_clean chomp seed: the
n²-stiff Fourier parametrization gives `‖grad‖₂ ≈ 3.79e7` against `J0 ≈ 2.7e4`,
so L-BFGS-B's first (unbounded) step overshoots to `J ≈ 1e31` and the line
search gives up. This plan implements the **root-cause conditioning fix**: a
genuine **non-diagonal H¹/H² Sobolev *metric* preconditioner** on the curve
DOFs, plus **objective normalization**. Unlike globalization-only workarounds
(ALM trust box, scipy `trust-constr`), this targets the problem conditioning
itself. This plan initially wires only the penalty L-BFGS-B path and its
seed-gradient diagnostic; ALM/trust-region use remains gated on explicit
composition tests for their bound/trust-region mappings.

## Goals

- A symmetric-positive-definite (SPD) Sobolev metric `M` assembled from the
  curve's **coefficient Jacobian** (`dgammadash_by_dcoeff`, optionally
  `dgammadashdash_by_dcoeff`), applied as a preconditioner via a Cholesky
  variable transform — observably reducing the conditioning so the seed
  `first_step_dJ` (from the existing seed-gradient diagnostic) is **negative**
  (a real descent step) where the diagonal scale left it at `+3e23`.
- Objective normalization so `J` and `‖grad‖` are `O(1)` at the seed regardless
  of `--order`, making L-BFGS-B's absolute `ftol/gtol` heuristics meaningful and
  making acceptance diagnostics compare runs on a common objective scale. Metric
  strength (`alpha`) is still selected by diagnostic sweep and must not be
  assumed order-independent unless the implementation normalizes the metric and
  records that convention.
- **Byte-identical default-off**: with the metric disabled the penalty path is
  bit-for-bit the current behavior.
- The preconditioner **preserves the existing box bounds** on the winding-size
  and banana-current DOFs (no bound is silently dropped or rotated away).

## Non-Goals

- Replacing or re-tuning the ALM trust-region route (separate, complementary
  lever; ALM inner L-BFGS-B can use this metric only after its bound/trust-box
  composition tests pass).
- Wiring scipy `trust-constr` (already recorded OOM-prone at this scale, job
  `55167099`).
- A full quasi-Newton / Gauss-Newton Hessian preconditioner (a possible Tier-3b;
  noted under Open Questions, not built here).
- Changing the physics objective, the seed, the hardware contract, or any
  promotion gate.

## Current Context

- **Diagonal Lever A was implemented and proven insufficient.** The current
  metric implementation keeps the diagonal/off path as a compatibility baseline
  while adding the non-diagonal operator. Relevant files (all
  `examples/single_stage_optimization/`):
  - `banana_opt/single_stage_geometry.py`:
    - `build_sobolev_curve_mode_scale_vector(dof_names, alpha, power=2)` →
      `1/(1+alpha*k**power)` on curve Fourier DOFs, 1.0 else.
    - `build_winding_dof_scale_vector(dof_names, scale_map)`.
    - `Stage2PenaltyPreconditioner` and `run_scaled_winding_minimize(...)`
      now support either the legacy diagonal `scale` vector or the non-diagonal
      curve-block operator. The optimizer sees `u`; physical DOFs are mapped
      through `preconditioner.to_x(...)`, gradients through
      `preconditioner.grad_to_u(...)`, and final/callback DOFs are mapped back to
      physical `x`.
    - `_CURVE_FOURIER_DOF_PREFIXES = ("phic(","phis(","thetac(","thetas(")`.
  - `STAGE_2/banana_coil_solver.py`:
    - `resolve_stage2_penalty_preconditioner(...)` composes the winding
      diagonal scale with the non-diagonal curve metric when
      `--stage2-sobolev-metric h1/h2` is enabled, or returns the diagonal/off
      compatibility path otherwise.
    - `--stage2-sobolev-metric`, `--stage2-sobolev-alpha`,
      `--stage2-sobolev-h2-beta`, `--stage2-sobolev-power`, and
      `--stage2-objective-normalize` are penalty-path flags. `power` only
      applies to the legacy diagonal path when metric mode is `off`.
    - Penalty optimizer and seed-gradient diagnostic both receive the same
      `stage2_penalty_preconditioner`, so `first_step_dJ` is measured under the
      same operator as the L-BFGS-B path.
  - `tests/geo/test_winding_dof_scale.py`: 11 tests incl. 5 Sobolev-diagonal.
- **α-sweep verdict (why diagonal failed):** `‖grad·scale‖∞` stayed pinned at
  `2.41e7` for every `alpha ∈ {1,4,16,64}` and `first_step_dJ` saturated at
  `~+3e23` without flipping negative. The dominant bad direction is **not** a
  single high-k curve mode that diagonal mode-scaling can reach.
- **Operational handoff now records the failed metric gate and live fallback.**
  `autoresearch:.handoffs/order64-conditioning.md` records the diagonal and
  non-diagonal metric sweeps as insufficient, points to this plan for the metric
  implementation evidence, and keeps the ALM trust-region fallback as the live
  operational lane while its full run is pending.
- **Confirmed building blocks.** The Stage-2 path instantiates
  **`CurveCWSFourierCPP`** (`src/simsopt/geo/curvecwsfourier.py:145`, subclass of
  `Curve, sopp.Curve`) at `banana_opt/stage2_geometry.py:715` — *not* the
  separate Python `CurveCWSFourier` (`src/simsopt/geo/curve.py:1232`). The
  **public dense** accessors `dgammadash_by_dcoeff()` and
  `dgammadashdash_by_dcoeff()` are defined directly on `CurveCWSFourierCPP`
  (`curvecwsfourier.py:446,490`) and return shape `(npts, 3, ndofs)` — confirmed
  by live callers `src/simsopt/geo/accessibility.py:712,858-859` ("size npts x 3
  x ndofs"). The `_impl` / `_vjp` variants also exist but the dense accessors are
  what assembly uses.
- **Curve quadrature.** The banana curve is built
  `CurveCWSFourierCPP(np.linspace(0, 1, num_quadpoints, endpoint=False), order,
  surf)` (`stage2_geometry.py:715`) — uniform on `[0,1)`, so the H¹/H² integral
  is `(1/num_quadpoints)·Σ_q`. `num_quadpoints` is the `--num-quadpoints` CLI
  value (default **128** via `NUM_QUADPOINTS`; `banana_coil_solver.py:2160`); the
  `edge_iota_soft` slurm does not override it, so the order-64 run uses 128.
- **Confirmed bound structure.** The synthetic `_BOUNDS` fixture in
  `tests/geo/test_winding_dof_scale.py` proves curve Fourier DOFs (`thetas(k)`,
  etc.) are handled as `(-inf, inf)` and winding size bounds are mapped
  element-wise, but that fixture does **not** prove the production banana-current
  bound. In the live Stage-2 penalty path, banana-current bounds are applied by
  `apply_penalty_traversal_forbidden_box_bounds(...)`, optional VF bounds by
  `apply_vf_current_upper_bound(...)`, then `build_lbfgsb_bounds(JF)` snapshots
  `JF.lower_bounds` / `JF.upper_bounds`. The preconditioner must therefore prove
  every finite-bound non-curve DOF stays in the diagonal/identity block, and that
  only unbounded curve-Fourier DOFs enter the non-diagonal block.
- Interpreter: `PYTHONNOUSERSITE=1
  /Users/suhjungdae/code/columbia/simsopt-surrogate/.conda-env/bin/python`;
  pytest is run from `examples/single_stage_optimization` with absolute paths.
- Local box cannot run order-64 solves (OOM policy); correctness is validated by
  unit tests on small synthetic CurveCWSFourierCPP objects. Seed diagnostics and
  smokes run on Perlmutter `shared` QOS; the full ALM fallback is on `regular`.

## Rationale

Two load-bearing facts shape the design.

**1. A *naive* `1+αk²` "non-diagonal" metric would be a no-op — it must be the
true pullback metric.** For a plain Fourier curve on a uniform grid the standard
H^s inner products (`L²`, `H¹`, `H²`) are **diagonal in the Fourier basis** (the
modes are orthogonal over a period, `∫₀¹ ∝ δ_{kj}`; differentiation only
multiplies mode `k` by `k`). So `diag(1+αk²)` *is already* the naive H¹ metric —
i.e. exactly the diagonal scale that failed. The cross-mode coupling that diagonal scaling cannot
touch comes from the **`CurveCWSFourierCPP` surface pullback**: the physical
tangent `γ'(θ)` is the curve coefficients pushed through the winding-surface
embedding, whose metric varies along `θ`, so the H¹ inner product *in
coefficient space*,
`M_ij = ∫ (∂γ'/∂c_i)·(∂γ'/∂c_j) dθ`,
is **genuinely non-diagonal**. That is the object this plan assembles (from
`dgammadash_by_dcoeff`), and it is precisely what the diagonal approximation
threw away. The H² variant adds `β·∫ (∂γ''/∂c_i)·(∂γ''/∂c_j) dθ` via
`dgammadashdash_by_dcoeff`.

**2. A non-diagonal transform breaks box bounds — so precondition only the
unbounded block.** The variable transform for an SPD metric `M = LLᵀ` is
`u = Lᵀx` (so `grad_u = L⁻¹grad`, `x = L⁻ᵀu`). A full non-diagonal `L` rotates
the axis-aligned x-box into a polytope SciPy L-BFGS-B cannot represent, because
that method accepts box bounds only. Therefore the non-diagonal block is allowed
only on DOFs whose live `JF.lower_bounds` / `JF.upper_bounds` entries are
infinite. Every finite-bound non-curve DOF (winding size/current/optional VF
current) stays in the identity/diagonal block. Bounds are preserved exactly, and
the existing winding-corridor diagonal scale continues to handle the bounded
block. The curve↔surface cross-metric terms are intentionally dropped (the
surface DOFs stay in the diagonal bounded block); this is the modeling decision
that buys bound-safety at negligible cost (surface DOFs are few and already
corridor-scaled).

This generalizes `run_scaled_winding_minimize` from a diagonal `scale` vector to
a **linear operator** (with the diagonal case as the all-ones/identity special
case), keeping SSOT: one transform, one penalty call site, one composer.

## Assumptions

- The held-uncommitted diagonal Lever A code is the baseline this builds on
  (the new builder sits beside `build_sobolev_curve_mode_scale_vector`; the
  composer extends `resolve_stage2_penalty_dof_scale`). If Lever A is reverted,
  Phase 1 must also re-introduce the curve-prefix/`k`-parse helpers.
- *Confirmed*: `CurveCWSFourierCPP.dgammadash_by_dcoeff()` (and the
  `dgammadashdash_*` analogue) is a public dense accessor returning shape
  `(num_quadpoints, 3, ndofs)` (callers `accessibility.py:712,858-859`). The one
  thing **Phase 0 must still verify** is that the curve's local DOF order
  (the `ndofs` axis) aligns one-to-one with the global `JF.dof_names` suffixes
  (`phic(k)` etc.) so the per-curve block embeds into the right global indices.
- *Confirmed*: curve quadrature weights are uniform (`1/num_quadpoints`) on
  `[0,1)` (`np.linspace(0, 1, num_quadpoints, endpoint=False)`,
  `stage2_geometry.py:715`), so `M = (1/num_quadpoints)·Σ_q J_qᵀ J_q` is the
  exact periodic-trapezoidal H¹ integral.
- The metric is assembled **at the seed and frozen** for the solve (a
  frozen-frame preconditioner, standard Sobolev-gradient practice), recomputed
  at most per order-ladder rung — not per inner step.
- Each banana coil's curve block is assembled independently; with multiple
  identical banana coils the per-curve metric is reused.
- Global names come from `Optimizable.full_dof_names` prefixing local names; the
  block mapper must match on the stable local suffixes emitted by
  `CurveCWSFourierCPP._make_names` (`phic(0..order)`, `phis(1..order)`,
  `thetac(0..order)`, `thetas(1..order)`) and must reject any finite-bound index
  before assigning it to the non-diagonal curve block.

## Implementation Plan

### Phase 0 — Verify the DOF-ordering contract (read-only, blocking)
- [x] On a tiny `CurveCWSFourierCPP` (`order=2`), confirm the `ndofs` axis of
      `dgammadash_by_dcoeff()` aligns one-to-one (and in order) with the curve's
      `local_dof_names` (`phic/phis/thetac/thetas(k)`), and that those map onto
      the global `JF.dof_names` suffixes — so the per-curve block embeds into the
      correct global indices. (Shape `(num_quadpoints, 3, ndofs)` and uniform
      `[0,1)` quadrature are already confirmed — see Current Context.) Evidence:
      committed accessor/order tests in `tests/geo/test_sobolev_metric_precond.py`.
- [x] Confirm the curve-Fourier DOFs are `(-inf, inf)`-bounded in the live
      Stage-2 bound vector (so the non-diagonal block carries no finite bound).
      Evidence: `test_metric_operator_preserves_jf_snapshot_finite_bounds_and_metadata`
      builds bounds from a JF-like snapshot via `build_lbfgsb_bounds(...)` and
      asserts every curve-block index is unbounded while finite current/R0 bounds
      remain diagonal.

### Phase 1 — Assemble the Sobolev metric (`single_stage_geometry.py`)
- [x] Add `build_curve_sobolev_metric(curve, alpha, *, h2_beta=0.0)` returning an
      SPD `(n, n)` metric over the curve's `n` local Fourier DOFs:
  - `Jd = curve.dgammadash_by_dcoeff().reshape(num_quadpoints*3, n)`;
    `M_sob = (1/num_quadpoints) * Jd.T @ Jd` (H¹, PSD).
  - If `h2_beta > 0`: `M_sob += (h2_beta/num_quadpoints) * Jdd.T @ Jdd` from
    `dgammadashdash_by_dcoeff()` (H²).
  - `M = np.eye(n) + alpha * M_sob`. **SPD by construction** (identity is SPD,
    `M_sob` is PSD), so the constant/null directions (`Jd` is rank-deficient on
    the `k=0` mode) stay positive without any ridge/jitter, and `alpha=0` ⇒
    `M = I` *exactly* (the off/identity case). Assert symmetry to `rtol=1e-12`.
  - **Note (not a no-op vs. the held diagonal scale):** this metric is *not* the
    held diagonal `1/(1+αk²)` scale for `α>0`. The diagonal path implies a metric
    `M_diag = diag(1/scale²)` whose curve step is `dx = -scale²·grad`; here the
    curve step is `dx = -M⁻¹·grad` with a *full* (non-diagonal) `M`. They
    coincide only at `α=0` (both `= I`). The whole point is the off-diagonal
    pullback structure the diagonal form cannot represent.
  - Record whether CLI `alpha` is dimensional (`1/[M_sob]`) or whether `M_sob`
    is internally normalized before adding identity. Do not leave the metric-unit
    convention implicit; persist the chosen convention in `results.json` so later
    order sweeps compare the same object.
- [x] Add `build_curve_block_cholesky(M)` → lower-triangular `L`,
      `M = L Lᵀ` (`scipy.linalg.cholesky(M, lower=True)`); raise on non-SPD
      (fail-closed — with the identity baseline this can only fire on extreme-`α`
      numerics, never silently jittered).
- [x] Add `build_stage2_penalty_preconditioner(dof_names, curves_by_block,
      alpha, *, h2_beta, winding_dof_scale_map)` returning a **linear operator**
      object (SSOT) holding: the diagonal winding `scale` for the bounded block,
      and the per-curve `L`/`L⁻¹` factors mapped to the curve-Fourier index
      blocks of `dof_names`. `alpha=0` ⇒ the curve block is identity, so the
      operator reduces to the existing diagonal winding-scale (current behavior).

### Phase 2 — Generalize the transform (`single_stage_geometry.py`)
- [x] Generalize `run_scaled_winding_minimize` to accept **either** a diagonal
      `scale` vector **or** the preconditioner operator from Phase 1:
  - Forward: `x = P.to_x(u)` (block: `x_bounded = u_bounded*scale`,
    `x_curve = L⁻ᵀ u_curve`); objective returns `(J, P.grad_to_u(grad))`
    (`grad_u_curve = L⁻¹ grad_curve`, `grad_u_bounded = grad*scale`).
  - Bounds: only the bounded block maps (element-wise, as today); the curve
    block is `(-inf, inf)` ⇒ unchanged. **Bound-safety is structural.**
  - Invert: `res.x = P.to_x(res.x)` so all downstream consumers see physical
    DOFs. `res.nit/.success/.message/.status` untouched. All-ones/identity ⇒
    byte-identical to the current diagonal path.
- [x] Keep the diagonal `scale` signature working unchanged (the diagonal path
      is the `alpha=0`/identity special case) — no caller outside the penalty
      path changes.

### Phase 3 — Objective normalization (`single_stage_geometry.py` / solver)
- [x] Add `normalize_objective(fun, j_ref)` wrapper (mirrors
      `build_scaled_outer_problem`): returns `(J/j_ref, grad/j_ref)`; `j_ref =
      max(J(x0), eps)` computed once at the seed, or a caller-supplied constant.
- [x] Apply it around the penalty objective **before** the transform so the
      transformed gradient and L-BFGS-B `ftol/gtol` operate on `O(1)`
      quantities. Record `j_ref` in `results.json` for reproducibility.
- [x] Document honestly: normalization is a **uniform scalar** ⇒ it does not by
  itself change the condition number (`‖grad‖/J` is invariant); its job is
  to fix the absolute-scale mismatch that makes default tolerances and the
  first-step magnitude easier to interpret under SciPy's absolute tolerances.
  It does **not** select or normalize metric `alpha` by itself. The conditioning
  win comes from the **metric** (Phases 1–2), and `alpha` is accepted only when
  the operator-aware seed diagnostic proves a descent first step.

### Phase 4 — Solver wiring (`STAGE_2/banana_coil_solver.py`)
- [x] Extend `resolve_stage2_penalty_dof_scale` (or add
      `resolve_stage2_penalty_preconditioner`) to build the Phase-1 operator when
      `--stage2-sobolev-metric` is on, else return the existing diagonal scale
      (SSOT composer; default off).
- [x] Add flags: `--stage2-sobolev-metric {off,h1,h2}` (default `off`),
      `--stage2-sobolev-alpha` (reuse existing), `--stage2-sobolev-h2-beta`
      (default 0), `--stage2-objective-normalize` (default off). Env-mirrored
      like the existing `STAGE2_*` flags for the slurm.
- [x] Pass the operator to `run_scaled_winding_minimize` at the penalty call
      site and to the seed-gradient diagnostic so the
      diagnostic reports `first_step_dJ` **under the metric** (this is the
      cheap, read-only acceptance probe).
- [x] Generalize `banana_opt/stage2_objectives.py::diagnose_seed_gradient` to
      consume the same transform object as the optimizer. It currently accepts a
      diagonal `scale` vector and computes `x0 - grad * scale * scale`; with a
      metric operator it must compute the operator's identity-Hessian physical
      step (`dx = -P.step_from_gradient(grad)` or equivalent) and report the
      operator-aware `scaled_grad` / `first_step_dJ`.
- [x] Leave ALM/trust-region wiring off until separate composition tests prove
      the operator composes with their trust-box and bound transforms.

### Phase 5 — Campaign wiring (in the **autoresearch** repo, not this one)
- [x] Add an `edge_iota_soft_metric` slurm variant (copy of `autoresearch:`
      `campaigns/balance_pareto_singlestage_2026-06-17/perlmutter/edge_iota_soft/`)
      passing `--stage2-sobolev-metric h1 --stage2-sobolev-alpha <swept>
      --stage2-objective-normalize`. For the diagnostic gate, enable
      `STAGE2_DIAGNOSE_SEED_GRADIENT=1` (or pass `--diagnose-seed-gradient`) with
      the metric flags and assert `first_step_dJ < 0`; `SMOKE=1` / `--maxiter 1`
      by itself proves setup plus one capped optimizer step, not the seed
      diagnostic.

## Validation Plan

- [x] **CWS accessor / ordering contract test**
      (`tests/geo/test_sobolev_metric_precond.py`, new): instantiate a small
      `CurveCWSFourierCPP`, assert `local_dof_names` ordering matches
      `CurveCWSFourierCPP._make_names`, and assert
      `dgammadash_by_dcoeff().shape == (num_quadpoints, 3, num_dofs)` plus the
      same shape contract for `dgammadashdash_by_dcoeff()`.
- [x] **SPD / round-trip unit test** (`tests/geo/test_sobolev_metric_precond.py`,
      new): on a small `CurveCWSFourierCPP` (`order≈3`), assert `M` symmetric +
      SPD, `L Lᵀ == M` to `1e-10`, and `alpha=0` ⇒ `M == I` (exact identity).
- [x] **Transformed-gradient FD test**: capture the operator's `scaled_fun`
      closure (as `test_winding_dof_scale.py` does for the diagonal path) and
      central-FD-check the transformed gradient to `rtol≤1e-4`; assert
      `grad_u_curve == L⁻¹ grad_curve` to `1e-12` (mutation guard: dropping
      `L⁻¹` fails this).
- [x] **Conditioning-improvement test**: on a synthetic ill-conditioned curve
      block, assert `cond(L⁻¹ H L⁻ᵀ) < cond(H)` for a representative SPD `H`
      (or, on a constructed quadratic, that the metric step flips
      `first_step_dJ` negative where the diagonal scale leaves it positive) — the
      executable proof this is non-diagonal *and* helps.
- [x] **Byte-identical-off test**: with `--stage2-sobolev-metric off` and
      `--stage2-objective-normalize` off, `run_scaled_winding_minimize` result
      (`res.x`, `res.fun`, `res.nit`) equals a bare `minimize` to the bit
      (extend the existing `test_default_off_is_byte_identical_to_plain_minimize`).
- [x] **Live bound-inventory / preservation test**: from the same `JF` object used
      by the Stage-2 penalty solve, assert every index assigned to a non-diagonal
      curve block has `(-inf, inf)` in `JF.lower_bounds` / `JF.upper_bounds`, and
      assert every finite-bound index remains in the diagonal/identity block with
      transformed bounds equal to the current diagonal path.
- [x] **Diagnostic-operator test** (`tests/geo/test_seed_gradient_diagnostic.py`):
      prove `diagnose_seed_gradient` uses the same operator as the optimizer by
      comparing its reported first step against `P.step_from_gradient(grad)`;
      keep the existing vector-scale case unchanged.
- [x] **Objective-normalization test**: prove the wrapper divides both `J` and
      `grad` by the same frozen `j_ref`, records `j_ref`, and does not claim a
      condition-number change on a constructed quadratic.
- [x] **Local commands** (from `examples/single_stage_optimization`):
      `PYTHONNOUSERSITE=1 .../.conda-env/bin/python -m pytest
      <abs>/tests/geo/test_sobolev_metric_precond.py
      <abs>/tests/geo/test_seed_gradient_diagnostic.py
      <abs>/tests/geo/test_winding_dof_scale.py -q`. Evidence: 25 passed after
      `cc91579dd`.
- [ ] **Perlmutter seed probe**: `SMOKE=1` run on the slid_clean chomp seed
      with `STAGE2_DIAGNOSE_SEED_GRADIENT=1` asserts metric assembles and the
      seed diagnostic prints `first_step_dJ < 0` (the go/no-go for a full run).
      Evidence so far: H1 jobs `55264061`, `55265700`, `55265701`, and
      `55265702` plus H2 beta=1 jobs `55267436`, `55267438`, `55267443`,
      and `55267445` all assembled the metric but failed the descent gate.
      Positive `first_step_dJ` ranges were `+1.812255e10..+2.122981e10`
      for H1 and `+1.702834e10..+2.202494e10` for H2 beta=1.
- [x] **Trust-region fallback smoke**: Perlmutter job `55271978` completed
      successfully (`COMPLETED|0:0`, elapsed `00:14:42`) and wrote
      `results.json` proving `CONSTRAINT_METHOD='alm'`,
      `ALM_TRUST_RADIUS_INIT=0.02`, `ALM_MAX_OUTER_ITERS=1`,
      `EDGE_IOTA_MODE='soft'`, and `HARDWARE_CONSTRAINTS_OK=True`. The edge
      report was still `EDGE_IOTA_STATUS='insufficient_samples'`, so this is
      route-engagement proof only, not a full edge-iota success.
- [x] **Wrapper soft-control forwarding**: `autoresearch df39e677` forwards and
      records `--stage2-edge-iota-weight` and `--stage2-edge-iota-hinge` through
      `scripts/run_one.py`, with regression coverage on the `soft` wrapper path.
      The direct Perlmutter launcher already carried these flags; this closes
      the managed wrapper route.
- [ ] **Full ALM fallback result**: Perlmutter job `55273370` is running on
      `regular_1` with 6h limit. It is the next operational fallback evidence,
      but it is not complete and must not be reported as physics closure.
- [x] **Review/Crucible closure for delivered slices**: source, tests, plan, and
      handoff deltas have strict PASS review coverage (no defensive fallbacks,
      SSOT composer, no fake/jittered metric, regression tests non-tautological).
      This review closure is not physics closure; the full ALM fallback result
      remains the open operational gate.

## Risks and Mitigations

- **Risk:** `M` is singular/ill-conditioned itself (constant/null mode has zero
  derivative ⇒ `M_sob` rank-deficient).
  **Mitigation:** `M = I + α·M_sob` is SPD by construction for any `α ≥ 0`,
  `β ≥ 0` (identity SPD + PSD), so no ridge/jitter is needed; Cholesky failure is
  fail-closed (raise) and can only arise from extreme-`α` floating-point loss.
- **Risk:** dropping curve↔surface cross-metric terms weakens the
  preconditioner.
  **Mitigation:** required for bound-safety; surface DOFs are few and corridor-
  scaled; quantify residual conditioning via the seed diagnostic and, if the
  step still doesn't flip negative, escalate to Tier-3b (Gauss-Newton metric).
- **Risk:** assembly cost at order 64 (`n = 4·order+2 ≈ 258` curve DOFs,
  `num_quadpoints = 128` default).
  **Mitigation:** `Jdᵀ Jd` is `O(num_quadpoints·n²) ≈ 128·258²` (≈8.5M flops)
  once at the seed (frozen-frame), Cholesky `O(n³) ≈ 258³` is sub-second; never
  per-inner-step. Reuse across identical coils.
- **Risk:** normalization mistaken for a conditioning fix.
  **Mitigation:** plan + code docstring state it is a uniform scalar; the
  conditioning win is asserted only for the metric (the cond-improvement test).
- **Risk:** the geometric metric still misses the actual dominant low-order or
  non-curve direction, just as the diagonal Sobolev sweep did.
  **Mitigation:** require the operator-aware seed diagnostic to flip
  `first_step_dJ < 0` before launching a long run; if it does not, the current
  handoff's trust-region path remains the next executable action.
- **Risk:** plan/runbook drift.
  **Mitigation:** keep `autoresearch:.handoffs/order64-conditioning.md` and this
  plan synchronized on the actual lane state: failed metric gate, ALM fallback
  smoke passed, full ALM result pending.
- **Risk:** silent regression of the proven diagonal/off path.
  **Mitigation:** byte-identical-off + bound-preservation tests gate it; the
  operator's identity case must equal the diagonal vector path bit-for-bit.

## Completion Criteria

- [x] `build_curve_sobolev_metric` + Cholesky + block operator implemented,
      SPD/round-trip/FD/cond-improvement/byte-identical-off/bound-preservation
      tests pass locally on the conda interpreter. Evidence: `45d2b3ae1` plus
      `735400361` and `cc91579dd`; focused local pytest reports `25 passed` for
      `test_sobolev_metric_precond.py`, `test_seed_gradient_diagnostic.py`, and
      `test_winding_dof_scale.py`.
- [x] Penalty path + seed diagnostic consume the operator; `--stage2-sobolev-
      metric off` is byte-identical to current behavior, and
      `banana_opt/stage2_objectives.py::diagnose_seed_gradient` covers the same
      operator step as the optimizer. Evidence: `5dff32284` source wiring and
      `45d2b3ae1` operator-aware diagnostic formatter/test coverage,
      `735400361` for diagonal-metadata and unwired-route hardening, and
      `cc91579dd` for rejecting ALM runs with unused Sobolev alpha telemetry.
- [ ] On the slid_clean chomp seed (Perlmutter `SMOKE=1` with
      `STAGE2_DIAGNOSE_SEED_GRADIENT=1`), the seed-gradient diagnostic reports
      `first_step_dJ < 0` under the metric (the diagonal scale could not
      achieve this). Evidence so far: H1 and H2 beta=1 alpha values 1, 4,
      16, and 64 all assembled the metric but failed the gate. The metric
      completion criterion is not met; fallback smoke `55271978` completed and
      proved ALM/edge-soft route engagement only. Full ALM fallback job
      `55273370` is running, but no result is available yet.
- [x] Crucible/review strict PASS for the delivered source, tests, plan, and
      handoff slices; plan cross-referenced from
      `docs/stage2_order64_sobolev_conditioning_plan_2026-06-28.md` and the
      `autoresearch:.handoffs/order64-conditioning.md` handoff updated to record
      the failed metric gate and active ALM fallback, or explicitly left
      unchanged with this plan marked as a non-active experiment.
      Cross-reference/handoff evidence is in
      `45d2b3ae1`, `39758405`, `5cbd31cd`, `d499d62f`, `0d4647a46`, and
      `d51795ae`. This does not close the physics gate: job `55273370` still
      needs a full ALM fallback result.

## Open Questions

- **H¹ vs H² as default:** does the curvature-penalty-dominated objective need
  the `k⁴` H² term, or is H¹ enough to flip `first_step_dJ`? Decide from the
  seed-diagnostic probe (cheap), not a priori.
- **Tier-3b (escalation):** if the surface-pullback H¹/H² metric still leaves
  the dominant direction unconditioned, the next step is a **Gauss-Newton metric**
  from the field-residual Jacobian (`J_res^T J_res`) — captures the *objective*
  cross-coupling the geometric metric cannot. Larger effort; gated on whether
  Phase 4 actually flips the step. Out of scope here, recorded for sequencing.
- **Interaction with ALM inner solve:** the same operator should precondition
  ALM's inner L-BFGS-B (via `_build_box_bounds`); confirm the block transform
  composes with the trust-box bound mapping before enabling it there.
- **Order-ladder coupling:** recompute `M` per rung (cheap) or carry/interpolate
  across rungs? Default to recompute-per-rung unless profiling says otherwise.
