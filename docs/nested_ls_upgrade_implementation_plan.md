# Nested-LS Prior-Art Upgrade — Implementation Plan

**Status:** Draft
**Last updated:** 2026-08-24

## Purpose

Execution plan for upgrading the nested single-stage lane (outer scipy
L-BFGS-B over coil DOFs, inner nested Boozer-LS solve) with the mechanisms
identified by the 2026-08-24 prior-art research (DESC, TORAX, SUNDIALS/CVODE,
AUTO, Ceres, Optimistix) and validated against the B37 root-cause synthesis.
It supersedes the ad-hoc "successor charter" notes scattered across the
synthesis scratchpad and the reviewed research memo, and encodes the
external review's corrected sequencing (merge first; tolerance budget before
any loose inner result feeds the adjoint).

Companion records: charter closure (`docs/jax_nested_ls_outer_charter.md`,
Closure section, commit `55e87b294`), track verdict
(`docs/receipts/nested_ls_reduced_track_20260820.md`), prior-art dossier
(published artifact "Solved Elsewhere"), root-cause synthesis
(session scratchpad `b37_root_fix_synthesis.md`).

## Goals

- One SSOT transactional state model shared by both nested children and the
  fused lane's host boundary (incumbent/candidate, commit-on-accept).
- Inner-solve robustness at outer trial points: predictor warm start,
  refresh-before-abandon, damping, sub-stepping — closing the measured lane
  fork (JAX inner 10 undamped Newton @1e-13 vs native LBFGS≤1500+Newton≤40
  @1e-11) without abandoning the JAX lane's per-evaluation speed edge.
- Typed, evidence-grade trial semantics: three-valued inner exit,
  `value_is_valid` ledger bit, per-leg binary provenance.
- A defensible tolerance/error budget before any coarse inner result feeds
  the implicit-function-theorem adjoint.
- Re-certified B3 v2 (and then B37 v2) receipts minted from the merged
  current-source lineage.

## Non-Goals

- Replacing the outer scipy L-BFGS-B (no faithful on-device L-BFGS-B exists;
  the sealed parity contract requires stock scipy in both lanes).
- Changing the iota branch-guard threshold (adjudicated: no threshold
  separates poisoning from healthy steps; structural predicates come later).
- Promoting flat-675 to a second product lane ("certified flat" is Phase 6+,
  explicitly deferred by review).
- Any timing claim from upgrade-era runs before a new sealed policy and
  fresh OMP sweep exist.

## Current Context (confirmed facts, 2026-08-24)

- Main `pr/jax-port-squashed` @ `051323e71` still carries the v1
  rolling-anchor implementation. The transactional containment lives only on
  branch `fix-b37-restart` (`01fefbadd` transactional trials + `19d5e65fb`
  evidence source-binding), worktree `.wt-simsopt-fix-b37-restart`
  (Codex-authored; adversarially reviewed GO; execution-proven by the
  38→39→38 replay, transcript committed as
  `docs/receipts/evidence/nested_ls_outer_b37_20260824_transaction_replay.log`).
- The containment is at the **child layer**: children restore the committed
  candidate after every evaluation. `nested_ls_reduced_scale.py:4743` still
  mutates the anchor per-eval inside `_solve_nested_inner_at_coils`; the
  restore makes it harmless but the mutation itself is un-deleted (SSOT debt).
- Key seams (main-tree line numbers, scale module identical on both
  branches): warm-start install `:4701`; inner solve call `:4704-4711`
  (`stab=F3_B37_IFT_STAB=0.0`, `maxiter=NESTED_LS_NEWTON_MAXITER=10`,
  `tol=NESTED_LS_NEWTON_TOL=1e-13`); per-eval anchor commit `:4743`;
  adjoint dense LU assembly `:4910-4917` (factor discarded after use).
- Predictor machinery exists in the fused lane:
  `surface_objectives_traceable.py:3154`
  (`_traceable_predict_warmstart_result_from_anchor`: JVP forcing +
  factor-reuse + anchor fallback) and the transactional host boundary
  `AcceptedIncumbentHostValueAndGrad` at `:422`.
- Inner-walk internals: `nested_ls_reduced.py:1146-1329` (undamped Newton +
  Armijo, counter semantics), `dense_schur_lu_preconditioner` `:986-996`,
  `apply_reduced_mixed_schur_coil_tangent` `:1389-1426`. Predictor expression
  already FD-validated by `tests/geo/test_nested_ls_reduced.py:1360,1471`
  (`test_implicit_adjoint_matches_surface_response_to_coil_step`,
  `test_unregularized_ift_adjoint_matches_reconverged_surface_fd`).
- Evidence in flight: B3 v2 receipt (producer `01fefbadd`, stem
  `nested_ls_outer_b3_20260824`) and the B37 v2 JAX-only diagnostic. A
  simsoptpp binary boundary (sha `41b2ca79`→`95190afa`, ~07:0x) falls
  between B3 v2 pairs 1 and 2; the cross-pair bitwise canary adjudicates it.
- Recorded replay inputs for predictor testing: the recovered 5090 ledger
  (`docs/receipts/evidence/nested_ls_outer_b37_20260823_recovered_jax.json`)
  and the A100 probe JSON carry per-eval coil DOFs and the endpoint surface.

## Rationale

Every mechanism below is lifted from verified prior art rather than invented:
commit-on-accept is CVODE/TORAX/Ceres/Optimistix standard; the predictor is
DESC's published perturbation method (order 1 = our chartered δs =
−Ĥ_ss⁻¹Ĥ_sc δc, order 2 available on the same factorization); refresh-
before-abandon is CVODE's ladder; damping + three-valued exit is TORAX's
Newton; sub-stepping is AUTO/TORAX step control. The external review's two
corrections are binding: (1) merge the transactional lineage before building
on it, and (2) no coarse inner result feeds the IFT adjoint until a gradient
error budget exists (the IFT formula assumes zero inner residual; DESC
accepts that error silently — we budget it).

## Assumptions

- The B3 v2 receipt lands green (pair-0 parity 2.6e-13 and pair-0/1 bitwise
  native determinism already observed). If pair-2 forks bitwise across the
  binary boundary, the merge proceeds but the re-baseline (Phase 1) must pin
  a single simsoptpp sha.
- Codex remains the owner of `fix-b37-restart` content-wise; merge conflicts
  with concurrent main-tree docs commits are mechanical only.
- scipy stays pinned at 1.17.1 for the certification era; the StopIteration
  control-flow pin (Phase 0) is version-scoped.

## Implementation Plan

0. **Tier-0 hardening (can land on `fix-b37-restart` before the merge; ~1 day)**
   - [ ] Pin scipy 1.17.1 `StopIteration`-from-callback control flow with a
         test: raise from a new-style callback under L-BFGS-B, assert the
         observed status/message/success triple (expected: caught, returned
         `success=False`; exact status code to be pinned by the test, not
         assumed — sources disagree between 2/99).
   - [ ] Add `value_is_valid: bool` to `_OuterEval` rows + native twin
         records; barrier rows carry `False`.
   - [ ] Barrier contraction assertion: unit test that dcsrch interpolation
         on the quadratic barrier contracts at least geometrically over
         `maxls=8` trials (Ceres-gap check).
   - [ ] `test_overstepping`-class regression (DESC pattern): noise the
         gradient so every step rejects, cripple the inner solve, assert the
         returned endpoint is bitwise the start state.
   - [ ] Parameterized sweep test of `nested_ls_outer_restart_reason` and
         `nested_ls_outer_endpoint_success` alone (TORAX `test_cond_fun`
         pattern) — extend `tests/geo/test_nested_ls_outer_transaction.py`
         (file exists on `fix-b37-restart` only until Phase 1 merges it).
   - [ ] Add `simsoptpp_sha256` (and `.so` path) to
         `nested_ls_runtime_identity` (`nested_ls_reduced_scale.py:398`).
   - [ ] Soften the `accept()` KeyError crash: on a missing candidate, emit
         an honest-failure payload (child writes JSON with
         `success=False`, reason `accept_without_candidate`) before exiting
         nonzero — fail closed without destroying evidence (dcsrch
         XTOL-promotion corner).

1. **Merge + re-baseline (review step 1–2; ~1 day + machine time)**
   - [ ] Squash-merge `fix-b37-restart` (`01fefbadd`+`19d5e65fb`+Tier-0)
         onto current main; resolve the charter Amendment-4 ordinal (their
         amendment renumbers to 5; closure section already records this).
   - [ ] One child→parent→dual-rejudge B3 integration test (fast, stubbed
         inner) proving the driver consumes both children's v2 schemas
         end-to-end.
   - [ ] Fresh native OMP sweep on the merged source; mint the
         source-bound B3 v2 receipt at the merged SHA (the in-flight
         `01fefbadd` receipt remains bridge evidence, cited not superseded).
   - [ ] Post-merge: DIAG4 execution-source manifest regeneration
         (`benchmarks/regenerate_execution_source_manifest.py`; pinned
         counts at `single_stage_native_equivalent_quality_successor_authority.py:212,451`).
   - [ ] Merge the parked `_lbfgs` worktree branch
         (`worktree-agent-a294f051ffd23f1c9` @ `92ec95f5`, strict-PASS).

2. **Predictor on recorded displacements (review step 3; ~2 days)**
   - [ ] Add `anchor_coil_dofs` to `NestedLsOuterState`
         (`nested_ls_reduced_scale.py:4568-4611`), written at commit time.
   - [ ] Cache the adjoint's dense LU (via `dense_schur_lu_preconditioner`)
         on the state at commit; predictor per trial = one
         `apply_reduced_mixed_schur_coil_tangent` JVP + one cached LU solve.
   - [ ] Trust-region cap on the prediction, DESC `tr_ratio` semantics
         (DESC *scales* the step to the bound `‖δs_pred‖ ≤ 0.1·‖s_anchor‖`,
         it does not reject): clip to the cap, then the reject-to-bare-anchor
         condition is ours, not DESC's — fall back when the predicted start's
         envelope gradient exceeds the anchor's at the same coils (one cheap
         `_envelope_value_and_grad` evaluation).
   - [ ] Offline validation on recorded states BEFORE wiring into children:
         replay legs 3/4 from the synthesis validation ladder — predictor at
         the recorded x₃₈→x₃₉ displacement (does it prevent the wrong-branch
         capture?) and at the post-poisoning x₃₉→x₃₈ trial (expected: no
         rescue — falsifiable prediction). Inputs from the recovered ledger
         and probe JSON. Quiet GPU, ~5 solves.
   - [ ] Wire into `_solve_nested_inner_at_coils` behind an explicit policy
         flag; transactional rollback covers the predictor state.

3. **Inner-solve robustness ladder (review step 4; ~3 days)**
   - [ ] Refresh-before-abandon (CVODE rule): on `inner_solve_failed` with a
         stale factorization, re-assemble/re-factor once and retry before
         declaring failure.
   - [ ] Damped Newton: residual-norm backtracking with NaN rejection in the
         inner walk (TORAX `root_newton_raphson` shape), replacing the
         undamped step at `nested_ls_reduced.py:1146-1329`; keep the
         existing Armijo/quality bails as-is where they are stricter.
   - [ ] Three-valued exit on `NestedLsSchurNewtonResult`:
         `{converged, coarse_converged, failed}` with
         `NESTED_LS_NEWTON_COARSE_TOL` (proposed 1e-8, DESC's production
         setting) — recorded in ledgers; **treated as failed** until Phase 4
         licenses coarse use.
   - [ ] Inner Δc sub-stepping from the committed anchor (1→2→4→8 legs,
         halve on failure, floor, honest halt).
   - [ ] Delete the per-eval `state.set_anchor` at
         `nested_ls_reduced_scale.py:4743-4745` and return a frozen trial
         record instead; delete FD-0's now-redundant manual pins
         (`:5044`, `:5291`) and prove FD-0 no-op (scatter 0.0, 11/11).

4. **Tolerance/error budget for the adjoint (review step 5; ~2 days, gates Phase 5)**
   - [ ] Derive and test the bound: gradient error of the IFT adjoint as a
         function of inner residual norm (measure empirically on the
         recorded states: solve to 1e-8 vs 1e-13, compare adjoint gradients;
         κ(Ĥ_ss) enters — record it, currently unmeasured).
   - [ ] Decide and document the licensed coarse tier: which inner residual
         levels may feed (a) line-search trial values only, (b) gradients,
         (c) committed anchors (always tight, 1e-13 polish at commit).
   - [ ] Red test: a coarse-converged result feeding the adjoint outside the
         budget must fail closed.

5. **New sealed policy + B37 v2 certification (~1 day + ~4 h machine)**
   - [ ] Seal policy `anchor_frozen_predictor_v3` (names all Phase 2–4
         behaviors); bump child/claim schemas; charter amendment.
   - [ ] Fresh OMP sweep; B3 v3 physics gate; then paired single-process
         B37 v2 (single-pair rule) on a quiet box; walls informational.
   - [ ] Verdict section + receipts to the track doc.

6. **Deferred (explicitly out of this plan's scope, revisit after Phase 5)**
   - Lineax reroute (test bit-identity under the B3 gate first),
     Optimistix inner `root_find` + iota hypercube clip, LU-derived branch
     test functions, order-2 (Halley) predictor term, periodic restoration /
     "certified flat" productization.

## Validation Plan

- [ ] Phase 0: new tests green in `tests/geo/test_nested_ls_outer_transaction.py`
      + the StopIteration pin test; `ruff` + targeted `pyright` clean
      (CPU venv, one file per process per repo rule).
- [ ] Phase 1: B3 integration test green; merged-source B3 v2 receipt
      physics-green (J-parity ≤1e-9 band, per-pair); DIAG4 manifest counts
      match pinned expectations.
- [ ] Phase 2: recorded-displacement replay results written to the track doc
      (both falsifiable predictions adjudicated). Predictor ON changes the
      warm start of **every** evaluation at coils ≠ anchor, so a
      predictor-ON B3 is a *new trajectory* — the gates are the per-pair
      physics parity band and FD-0 (not bitwise identity vs predictor-OFF).
      The one bitwise invariant that must hold: an evaluation at coils
      bitwise-equal to the committed anchor has δc = 0 ⇒ δs_pred = 0 and
      must reproduce the predictor-OFF result exactly (extends the stp=0
      invariant test).
- [ ] Phase 3: B37-class stress replay — re-run the recorded stall
      neighborhood (evals 38–44) under the full ladder; expected: zero
      `inner_solve_failed` at bitwise-repeated coils; sub-step ledger rows
      present.
- [ ] Phase 4: measured adjoint-error curve committed as evidence; red test
      fails closed outside budget.
- [ ] Phase 5: B37 v2 paired receipt; verdict either green (claim-eligible
      under the new policy) or an honest typed failure with the ladder's
      telemetry localizing the remaining gap.

## Risks and Mitigations

- Risk: merge races with concurrent sessions on main (three active peers +
  Codex).
  Mitigation: coordinate the merge window via the session-relay pattern
  already in use; squash to one commit; docs-only conflicts expected.
- Risk: damping/sub-stepping erodes the JAX lane's speed edge (the cheap
  inner IS the 2.29×/3.86×).
  Mitigation: ladder engages only on failure paths; healthy-path cost is one
  predictor solve (~O(n²) on cached LU); measure per-eval walls in the B3 v3
  rehearsal before sealing.
- Risk: predictor at stab=0.0 mis-predicts under indefinite Ĥ_ss.
  Mitigation: DESC cascade cap + envelope-gradient fallback (Phase 2), both
  cheap; κ(Ĥ_ss) measured in Phase 4.
- Risk: coarse-tier gradients silently bias optimization (JAXopt #466
  class).
  Mitigation: Phase 4 is a hard gate before Phase 5; coarse results are
  failed-by-default until budgeted.
- Risk: scipy version drift invalidates the StopIteration pin.
  Mitigation: the pin test asserts the triple under the installed version;
  CI fails loudly on upgrade.

## Completion Criteria

- [ ] Merged main carries the transactional lane; v1 rolling-anchor code
      deleted (not merely bypassed) — `:4743` mutations gone.
- [ ] All Phase 0–4 tests green; FD-0 no-op reproduced post-deletion.
- [ ] Source-bound B3 v2 (merged SHA) and B37 v2 receipts committed with
      verdict sections; per-leg simsoptpp sha recorded in identity blocks.
- [ ] Track doc + charter amendments record every policy change; no timing
      claim without the new sealed policy + fresh sweep.

## Open Questions

- Does the B3 v2 pair-2 bitwise canary confirm binary-boundary indifference
  (receipt in flight)? If not, Phase 1's re-baseline pins one sha.
- B37 v2 diagnostic endpoint (in flight): does the transactional lane reach
  native-class J at budget 37 unaided? Outcome decides how much of Phase 3
  is sufficiency-critical vs robustness-hygiene.
- Owner split: which phases run in this session vs Codex vs peer sessions
  (fix-branch ownership is Codex's; coordinate before Phase 1).
- `NESTED_LS_NEWTON_COARSE_TOL=1e-8`: adopt DESC's value or derive from the
  Phase 4 budget? (Plan assumes derive; 1e-8 is the starting hypothesis.)
