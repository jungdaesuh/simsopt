# Near-Target Newton Solver: Regime-Adaptive Dense-IR Implementation Plan

**Status:** In progress (Phase 0 shipped `37b65c7af`; Phase A v1 shipped
`ad3cc28b7`+`00b912d7f`+`1d7284afe`, local CPU validation complete — GPU
B7 lane and Phase B open)
**Last updated:** 2026-07-05

## Purpose

Root fix for the jax-CPU K1 Boozer-solve cost blowup diagnosed 2026-07-04
(perf-gap plan close-out, commits `9376cd835`/`e250656c5`): near the
convergence target the traced Newton polish runner solves an
ill-conditioned operator (κ(H) ≈ κ(J)²) with unpreconditioned GMRES to a
~1e-14 tolerance, so the Krylov iteration runs to essentially full
dimension (~651–1302 HVP matvecs per Newton iteration; ~600 s/iteration
on CPU at 255×64, ~15 s on A100). This plan replaces that regime with
direct factorization + iterative refinement, phased so each step is
independently shippable and validated.

## Goals

- jax-CPU reference lane viable at production resolution (255×64,
  mpol/ntor 10): eval-1 K1 forward completes in minutes, not hours.
- Near-target Newton iterations cost O(1) HVPs (LU presolve + ≤3
  refinement matvecs) after one amortized dense build per K1 call.
- Direction quality certifiable: success = measured backward error ≤
  tolerance, not assumed GMRES convergence.
- Default solver policy self-deciding (regime + byte-budget gated); env
  var demoted to comparator/debug override.
- GPU accepted-eval cost reduced (B6 ev56 pattern `[192,1308,1308,1308]`
  matvecs → `[~build, ~3, ~3, ~3]`).

## Non-Goals

- Changing the far-from-target loose Eisenstat-Walker GMRES path
  (measured optimal at 5–23 matvecs/iteration — B6 ev52).
- The exact-Newton runner (`_build_traceable_exact_newton_runner`):
  audited 2026-07-05 as KEEP-AS-IS (off default path — reachable only via
  `--boozer-stage final`; retry bounded fail-closed; nonsymmetric system
  has no cheap dense seam).
- The K2 adjoint path and its shared `(lu,piv)` byte-parity contract
  (`_traceable_solve_plu_linearization`) — untouched until Phase C.
- J-based LSMR/QR large-n solver (documented escape hatch only).

## Current Context (verified facts, file:line at `e250656c5` + Phase-0 edits)

- Solver mode SSOT: `src/simsopt_jax/geo/optimizers/optimizer.py:413-450`
  — env `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER`, modes/codes
  `{operator_gmres:1, dense_lu:2, hybrid_final_dense_lu:3}`, resolved once
  at import (`:450`); runner cache key includes the mode (`:5883-5892`).
- Hybrid routing predicate (LS polish runner body, `:6144+`):
  `_eisenstat_walker_strict_cap_applies(norm, tol)` — Phase 0 extends it
  with `| state["retry_linear_solve_at_strict_cap"]`.
- Dense in-loop solve `_solve_dense_square_operator_lu_system_with_status`
  (`:5186`): re-materializes EVERY call via `_dense_square_operator_matrix`
  (`:5119-5141`, `lax.map` batch 8 — deliberate anti-constant-fold, chunk
  const `:3777`, env override `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE`),
  then `lu_factor` + `lu_solve` + 1 IR step + backward-error/Hager-Higham
  gates (gates reuse cached `lu_piv`: 10 × O(n²) triangular solves, zero
  extra factorizations).
- GMRES budgets (`:4396-4407`): restart=min(n,64), maxiter=10 →
  651 matvecs/solve; refined (near-target) budget 1302 (+1 = 1303
  reported). Dense budget = n = 663. One dense build ≈ one single-pass
  GMRES ≈ half of one refined solve.
- Loop-invariant closure pattern (factor-once feasibility): body_fun
  already closes over out-of-carry tracers (`tol_value` `:5946`→`:6033`,
  `hvp_fn` `:5942`→`:6053`, solver-code constants `:5969-5992`); jit
  wrappers use no `donate_argnums` (`:6461-6484`). No structural blocker.
- Harness wiring (audited 2026-07-05): matrix children inherit the env var
  (`benchmarks/validation_ladder_common.py:296-330` — `dict(os.environ)`,
  var in no pop list); BOTH legs get the same value — per-leg seam =
  reference call site `benchmarks/single_stage_init_parity.py:2272` vs
  target `:2330`, with the `cuda_memory_env`-style `env.update` precedent
  (`validation_ladder_common.py:307-308`). No parity gate compares solver
  mode across legs in the default config (solver-metadata exact-compare is
  diagnostic-only: `strict_solver_contract=False` at sole call site
  `single_stage_init_parity.py:4841`; whole path off unless
  `MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE=1`).

## Measured baselines (2026-07-04/05, iota011 seed, 255×64, mpol/ntor 10)

| Config | eval-1 K1 forward | Notes |
|---|---|---|
| `67bdde1a7` operator (pre-retry-fix), local CPU | 593 s (evals 2-5: 659/475/477/485 s) | returns fast but eval-1 primal REJECTED (measured 2026-07-05: stalled, linear residual 499, ‖grad‖ stuck 3.6e-08; ALL 5 evals rejected — zero optimizer progress) |
| `bbe1a7452` operator, Perlmutter CPU | >6400 s, killed | A4 job 55499050, CASE_TIMEOUT 7200 |
| `bbe1a7452` operator, local CPU, newton-cap-5 | >3170 s, unfinished | `sample` stacks: XLA-CPU executing, not compiling |
| `bbe1a7452` **hybrid**, local CPU, prod cap 50 | **1552.2 s, returned** | K2 121.4 s; bounded, converging |
| A100 GPU operator (B5/B6) | 31–66 s/eval | acceptance bar already met (3.7× vs cpp) |

Interpretation: hybrid unblocks the lane at a 2.6× premium over pre-fix;
the premium is per-iteration re-materialization (663 HVPs), which
factor-once (Phase A) amortizes.

## Rationale

Regime-adaptive direct+IR beats the alternatives at production n:
preconditioning GMRES with the same factors degenerates to IR anyway;
always-dense (`dense_lu` mode, lane B4) pays materialization far from
target where loose GMRES costs 23 matvecs; J-based LSMR (κ un-squared)
still runs O(100+) Krylov iterations at 2 AD sweeps each while a 3.5 MB
factorization costs milliseconds — it is the correct tool only above the
dense byte budget. Exact directions also remove the root of the recurring
CPU/GPU fragility: marginal inexact directions flipping backtracking
accept/reject at ULP level (init stall, replay death, reject/retry churn
all trace to this).

## Assumptions

- Near-target steps are small enough that factors frozen at the regime
  entry point keep IR contractive (ρ(I − A₀⁻¹A) < 1); violations are
  caught by the IR residual gate and handled by one refresh, then
  fail-loud stall. (Validated implicitly by `dense_lu` mode converging
  B4 init in 1 iteration to 2.47e-14.)
- The ~3.5 MB `(factors, piv)` loop carry at n=663 is negligible against
  the 25.8 GiB (26,469 MiB) GPU / CPU footprints (measured B6).
- Local Apple-silicon CPU walls are a faithful relative proxy for the
  Perlmutter CPU leg (validated: same code ratio pre/post fix on both).

## Implementation Plan

### Phase 0 — Hybrid retry routing on existing machinery (2026-07-05, in flight)

1. Code + tests (DONE, uncommitted at time of writing)
   - [x] Route strict-cap retry to dense-LU in hybrid mode:
         `use_dense_lu_iteration |= state["retry_linear_solve_at_strict_cap"]`
         + rationale comment (LS polish runner, `optimizer.py:~6144`).
   - [x] Regression test
         `test_newton_polish_traceable_hybrid_routes_strict_cap_retry_to_dense_lu`
         (proven red at pre-fix HEAD: retry served by always-rejecting
         operator fake → stall; green post-fix).
   - [x] Full private suite: 142 passed, 5 skipped. Ruff clean.
   - [x] e2e: hybrid mode completes eval-1 K1 in 1552.2 s (table above).
2. Close-out
   - [x] Commit scoped (`optimizer.py` + private test file): `37b65c7af`,
         measured numbers in the message. (e2e run exited 124 at its
         3500 s cap mid-eval-2; eval-1 gate numbers locked beforehand.)
   - [x] Push to fork (`e250656c5..37b65c7af`).

### Phase A — `hybrid_final_dense_ir`: factor-once + iterative refinement (~1–2 days)

1. Solver mode plumbing (`optimizer.py`)
   - [x] Add `_TRACEABLE_NEWTON_LINEAR_SOLVER_HYBRID_FINAL_DENSE_IR`
         (code 4) to constants/CODES/`_resolve_traceable_newton_linear_solver`
         (+ aliases `dense-ir`, `hybrid-final-dense-ir`); update the
         resolver error message. (Shipped `ad3cc28b7`.)
2. LS polish runner: factor-once + IR branch
   - [x] v1: materialize + `lu_factor` once pre-loop (663 HVPs via
         existing `_dense_square_operator_matrix`), close `(lu, piv)` over
         body_fun (same pattern as `tol_value`/`hvp_fn`). (Shipped
         `ad3cc28b7`: uncounted `entry_hessian_matvec` build outside the
         counter closure.)
   - [x] Per near-target/retry iteration: `lu_solve` presolve + ≤3 IR
         steps, each 1 HVP against current `state["x"]` through the
         counted `matvec` (keeps NDJSON matvec telemetry honest);
         success = relative residual of the LAST IR step ≤ `linear_tol`;
         report backend code 4 + IR-step count as iterations; budget
         metadata = 3.
   - [ ] v2 (lazy + refresh): move factors into the loop carry
         (`hessian_factors`, `factors_ready`); `lax.cond`-materialize on
         first near-target iteration; ONE re-materialization when IR
         fails its residual gate; second failure → unconverged status →
         existing rejection/stall semantics unchanged (fail-loud, no
         silent fallback).
3. Harness: per-leg solver override
   - [x] Reference-leg-only env injection (shipped `00b912d7f`: keyword
         param on `repo_pythonpath_env`, resolver
         `_reference_leg_newton_linear_solver_override`, sole reference
         call site `single_stage_init_parity.py:2301`; target leg never
         injected), driven by `MATRIX_REFERENCE_NEWTON_LINEAR_SOLVER` in
         the K1 slurm (`export`ed — Crucible 2026-07-05 caught the
         unexported in-file default as a silent no-op).
4. Tests (private file, existing fake/monkeypatch conventions; distinct
   runner-cache keys per test)
   - [x] Routing: dense-IR serves near-target AND retry iterations
         (backend code 4 in trace), loose iterations stay operator.
         (`test_newton_polish_traceable_dense_ir_routes_near_target_and_retry`)
   - [x] IR convergence: 2%-stale factors vs live operator — refinement
         re-anchors to the live matvec, success only when measured
         residual ≤ tol.
         (`test_solve_dense_ir_system_refines_against_current_operator`)
   - [ ] Stale-factor refresh (v2): drifted-operator fake — first IR gate
         failure triggers exactly one re-materialization, then converges.
   - [x] IR failure stalls loud — covered at BOTH levels: solve-level
         (`test_solve_dense_ir_system_stale_factors_fail_loud`) and
         runner-level
         (`test_newton_polish_traceable_dense_ir_failed_direction_stalls_loud`:
         failed direction → immediate stall, no GMRES fallback, 1
         attempted iteration). Production confirmation: e2e eval-2
         garbage trial (entry ‖grad‖ 749.7) → final IR residual 178 →
         fail-loud stall → honest trial rejection at 1256.7 s.
   - [x] Numerical equivalence: dense-IR ≡ `dense_lu` to atol 1e-12 on a
         κ=1e5 diagonal quadratic (production near-target κ(J)² class;
         `test_newton_polish_traceable_dense_ir_matches_dense_lu_ill_conditioned`)
         plus the perfectly-conditioned real-path fixture. Delivered as
         ill-conditioned synthetic, NOT the originally-worded "real small
         Boozer fixture" — real-system equivalence evidence is the e2e
         eval-1 IR residual 1.4e-15 / ‖grad‖ 2.4e-14 (vs dense-LU-mode
         hybrid 2.0e-15 on the same seed).
   - [x] Matvec-count assertion: ≤3 counted matvecs per post-build
         near-target iteration (real-path test pins actual == 3; e2e
         eval-1 trace `[189, 3]` — build uncounted, 189 < n=663 proves no
         build leak into the counter).

### Phase B — Self-deciding default + retire compensation machinery (~1 day + soak)

   - [ ] Default policy in `_resolve_traceable_newton_linear_solver`:
         near-target dense-IR when `n²·8 ≤ max_dense_jacobian_bytes`
         (existing gate), operator-GMRES otherwise; env var becomes
         comparator/debug override only.
   - [ ] Update default-mode assertions
         (`test_boozersurface_jax_private.py:3526+`, and
         `tests/geo/test_boozersurface_jax.py` consumers found by the
         2026-07-05 vocabulary audit).
   - [ ] Remove the near-target refined-GMRES pass
         (`_refine_traceable_newton_operator_gmres_solution` call sites in
         the polish runner) once unreachable under the default policy;
         keep the helper for the forced-operator comparator mode.
   - [ ] Telemetry test: strict-cap retry never fires on the production
         seed under the default policy (assert zero retry iterations in a
         representative polish run).

### Phase C — Cross-eval factor reuse (separate campaign, design-review first)

   - [ ] Design doc: thread eval N's K2 `(lu,piv)` into eval N+1's K1 as
         initial factors (kernel signature + byte-parity contract review;
         the plan-doc "factor-reused predictor" open question).
   - [ ] Only after A/B soak.

### Large-n escape hatch (documented only)

   - [ ] One paragraph in `docs/using_jax_backend.md` (or successor):
         above the byte budget the near-target branch falls back to
         factor-preconditioned Krylov or J-based LSMR/QR (κ≈625
         un-squared), selected by the same byte-budget predicate.

## Validation Plan

- [x] Phase 0: private suite green (142 pass); e2e hybrid K1 wall
      recorded (1552.2 s); scoped commit `37b65c7af` + pushed.
- [x] Phase A local (measured 2026-07-05, run `cpu_a4repro_denseir2`):
      full private file green (146+2 pass / 5 skip); eval-1 K1
      **799.3 s `success=True`**, matvecs `[189, 3]`, final ‖grad‖
      2.4e-14, IR residual 1.4e-15, K2 120.7 s unchanged.
      DEVIATION vs the "≤ ~700 s" letter: +14% (includes fresh-graph
      compile). The ~700 s target was calibrated against the pre-fix
      593 s wall, which 2026-07-05 forensics showed was a REJECTED eval
      (fail-fast, zero progress) — not a valid success-wall comparator.
      Against valid comparators: 1.94× faster than hybrid (1552.2 s),
      and the only mode that is both successful and cheap. Goal
      criterion ("minutes not hours" + certified direction) met —
      ACCEPTED. Physics: gradient/IR certificates above; per-component
      Vol/Iota extraction not repeated locally, deferred to the B7
      parity lane (same seed family as the hybrid run).
- [x] Phase A GPU (measured 2026-07-05, lane B7, job 55547957,
      A100-40GB @b33a105e0, config = B5 twin): ALL GATES PASSED.
      Accepted-eval matvec actuals **`[192, 3]`** (was B6
      `[192,1308,1308,1308]`), K1 39.8 s (was 66.3 s), final ‖grad‖
      1.81e-15, K2 27.5 s unchanged → accepted eval 93.8 → 67.3 s
      (1.39×, now K2-dominated as forecast). Total wall **613.0 s ≤
      B5's 644.7 s** including the fresh dense-IR graph compile.
      Final-sync `reuses_objective_value_and_grad=True` (12.3 s). No
      OOM at auto-chunk, exit 0, L-BFGS-B converged (Vol 0.049164,
      Iota 0.110175). Trial eval honestly rejected via `[5, 653, 3]`
      (budget-exhausted GMRES → IR retry → reject). Artifacts:
      `/pscratch/sd/j/jungdae/k1_matrix_runs/crucible-gates-b33a105e0-laneB7/`.
- [x] Phase A parity — MEASURED (lane A5b, job 55549665, 2026-07-05,
      exclusive node, A4-twin config maxiter 20). Three results:
      (1) **A4 infeasibility CURED**: jax-CPU(dense-IR) reference leg
      completed the full 20-iteration optimization in 3958.1 s
      optimizer wall (script 4425.6 s) — inside CASE_TIMEOUT 7200
      that killed A4, and faster than the easier trial-skip-era twin
      (5877.3 s). GPU target 503.7 s.
      (2) **Per-solve parity proven at the shared eval-1 state**
      (identical bfgs pre=701 on both legs): dense-IR nit=1 →
      ‖grad‖ 2.43e-14 vs operator nit=6 → 9.52e-12 (June GPU-gold
      pattern) — same solution, dense-IR two orders deeper; the
      reference-leg-only env knob provably steered ONLY the
      reference leg (solver signatures differ per design).
      (3) **End-state diffs are cross-solver path divergence, not
      solver error**: vol rel 9.6e-8, iota abs 7.2e-6, field error
      0.011% apart — same physics basin after 20 independent outer
      iterations. Formal `passed` gate red ONLY via the
      machine-precision same-solver tolerances (iota 1e-10) applied
      to a cross-solver comparison — same class as the parked
      laneC/native tolerance-band decision; NOT loosened here
      (USER CALL, tracked in the perf-gap plan close-out).
      Ops constraint recorded: the CPU leg's XLA compile of the
      dense-IR graph needs >56 GiB host RAM at 255×64 (shared-slice
      -c 32 OOM-killed lane A5 attempt 1, job 55548579; MaxRSS
      62.8 GiB on the exclusive node).
      Byte-contract test surface untouched (K2 path not modified).
- [ ] Phase B: full parity matrix CPU+GPU under default AND forced
      `operator_gmres`; Crucible pass on the default-flip diff; one
      production-config soak run per platform before merge.

## Risks and Mitigations

- Risk: frozen factors non-contractive on an unusually hard trial point →
  IR gate failure loop.
  Mitigation: exactly one refresh then fail-loud stall (existing
  semantics); telemetry counts refreshes; hybrid per-iteration
  re-materialization remains available as forced mode.
- Risk: `lax.cond` materialization branch inflates the compiled graph /
  compile time (echo of the pole-1 constant-fold hang).
  Mitigation: reuse `_dense_square_operator_matrix` verbatim (its
  `lax.map` shape is the proven anti-fold form); v1 pre-loop variant as
  fallback if v2 compile regresses; compile-diagnostics recording in the
  validation lane.
- Risk: bit-level output drift vs operator mode trips a hidden strict
  gate.
  Mitigation: 2026-07-05 audit — solver-metadata exact compare is
  diagnostic-only and off by default; physics gates are tolerance-based;
  parity matrix run in Phase A validation confirms empirically.
- Risk: default flip (Phase B) changes behavior for downstream users of
  `scipy-jax-decomposed` on GPU.
  Mitigation: default flip gated on the Phase-A GPU lane showing
  equal-or-better wall AND physics parity; soak before merge; env
  override preserved.
- Risk: Perlmutter sshproxy key expiry (2026-07-05 14:29 EDT) blocks the
  GPU lane.
  Mitigation: refresh key before Phase-A GPU validation
  (`sshproxy -u jungdae`).

## Completion Criteria

- [x] Phase 0 commit on `simopt-jax-clean-local`, pushed to fork.
- [x] Phase A COMPLETE: dense-IR mode merged; local CPU eval-1 K1
      799.3 s `success=True` (~700 s letter missed 14%, adjudicated
      ACCEPTED); all v1 tests green (148/5); GPU B7 PASSED all gates
      (accepted-eval `[192, 3]`, 39.8 s K1, wall 613.0 ≤ 644.7 s,
      reused=True, no OOM); parity lane A5b PASSED substance (A4
      infeasibility cured 3958 s < 7200; per-solve parity at shared
      state 2.43e-14; same-basin end states; formal gate red only
      via the cross-solver tolerance-band USER DECISION).
- [ ] Phase B: self-deciding default merged after soak; refined-GMRES
      pass removed from the default path; Crucible PASS.
- [ ] Plan doc `docs/scipy_jax_decomposed_gpu_perf_gap_implementation_plan_2026-07-01.md`
      cross-referenced (its "optional batch A5" decision superseded if
      dense-IR makes the CPU leg fit CASE_TIMEOUT 7200).

## Open Questions

- v2 lazy-carry vs v1 pre-loop factorization: decide from Phase-A compile
  diagnostics (owner: implementer; default = ship v1 if v2 compile cost
  is visible).
- Should the Phase-B default ALSO apply to Stage-2 `scipy-jax` flows, or
  single-stage decomposed only? (Owner: user; affects deprecation story
  documented 2026-07-04 in `6f156d3b9`.)
- Does dense-IR make the jax-CPU reference leg fit CASE_TIMEOUT 7200 at
  maxiter 20 (projected yes: ~600 s/eval × ~9 evals + setup), letting the
  paired A5 matrix run on `shared_interactive` after all? Re-evaluate
  after Phase-A local numbers.
