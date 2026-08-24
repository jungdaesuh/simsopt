# Nested-LS Prior-Art Upgrade — Implementation Plan

**Status:** In execution
**Last updated:** 2026-08-24 (execution pass 1)

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
  retry-with-regularization instead of abandoning on the first rejected
  step, Δc sub-stepping — closing the measured lane fork (JAX inner Newton
  capped at 10 iterations @1e-13 vs native LBFGS≤1500 + Newton≤40 @1e-11)
  without abandoning the JAX lane's per-evaluation speed edge. **Corrected
  from the draft, which said "undamped" and "refresh-before-abandon":** the
  inner walk is already damped (`nested_ls_reduced.py:803-831`), and the
  production `dense_lu` path re-factors every iteration so no stale factor
  exists to refresh. See Phase 3.
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

- **LANDED `d835c43ee`.** `fix-b37-restart` is squash-merged onto
  `pr/jax-port-squashed` (all five commits: `290fc4238` restartable stops,
  `6572daef3` stop classification, `ce8e5a31a` child payload staging,
  `01fefbadd` transactional trials, `19d5e65fb` evidence source-binding).
  The charter Amendment-4 ordinal collision is resolved: three sections had
  claimed 4 — `e118fa813` (prose correction) keeps 4, `997bbacd5`
  (fault-rerun) renumbered to **Amendment 5** with its three ordinal
  citations updated, and the branch's transactional containment landed as
  **Amendment 6** immediately above the Closure section. Upstream provenance
  unchanged: Codex-authored, adversarially reviewed GO, execution-proven by
  the 38→39→38 replay
  (`docs/receipts/evidence/nested_ls_outer_b37_20260824_transaction_replay.log`).
- **LANDED `d3bd48ecd`.** The parked `_lbfgs` worktree branch
  (`worktree-agent-a294f051ffd23f1c9` @ `92ec95f52`) is merged; its three
  test files are green (68 + 4 + 19 passed).
- The containment is at the **child layer**: children restore the committed
  candidate after every evaluation. `nested_ls_reduced_scale.py:4766` still
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
- Inner-walk internals: `nested_ls_reduced.py:1146-1329` (the Newton loop and
  its counter semantics) with the damping in `_schur_armijo_step` at
  `:803-831` — 8 halvings, non-finite trials rejected twice, accepted on the
  objective Armijo condition **or** residual-norm monotonicity. The walk's
  abandon point is `:1325` (`if not step_accepted: break`) and the quality
  bail at `:1258-1284`; those, not the step rule, are what Phase 3 targets.
  `dense_schur_lu_preconditioner` `:986-996`,
  `apply_reduced_mixed_schur_coil_tangent` `:1389-1426`. Predictor expression
  already FD-validated by `tests/geo/test_nested_ls_reduced.py:1360,1471`
  (`test_implicit_adjoint_matches_surface_response_to_coil_step`,
  `test_unregularized_ift_adjoint_matches_reconverged_surface_fd`).
- **B37 v2 JAX-only diagnostic: LANDED `59ccbe8a0`**, and it answers the
  plan's second open question — see "Adjudicated" below.
- B3 v2 bridge run (producer `01fefbadd`, stem `nested_ls_outer_b3_20260824`,
  worktree `.wt-b3v2-run`, started 05:12): **died at ~08:19 inside pair 1's
  first rejudge child and wrote NO receipt.** Its structured log is 12 lines
  with no traceback and the rejudge child payload is 0 bytes. What survived
  on disk is strong and is the evidence to cite: all six timed child payloads
  intact; all three pairs **bitwise identical on every endpoint field**
  (native `J = 0.012982793095001662`, `iota = 0.144818275423838`; JAX
  `J = 0.012982793095005024`); pair 0 fully recorded `physics=True`,
  `j_rel_gap = 2.5894998648152005e-13`, `reason=None`, both rejudges
  `noop=True iter=0` at `‖∇J‖ ≈ 3e-15` against a 1e-13 gate.
  **The pair-2 canary came back clean anyway**: pair 2's native leg ran on
  simsoptpp `95190afa` and matched pairs 0/1 on `41b2ca79` bitwise; pair 2's
  JAX leg ran on `d4a6e028` and matched the earlier JAX legs bitwise. The
  endpoints are indifferent to the swaps on the lanes that ran them, so
  pinning one `.so` in Phase 1 is hygiene rather than a correction for a
  measured fork. (Weaker on the JAX side — that lane barely touches the C++
  binary.) **Even completed, this receipt could not gate a merged-tree B37** — the merged driver
  refuses it on five independent interlocks: claim schema `v1` not `v2`
  (`benchmarks/nested_ls_outer_claim.py:476`), `git_head` `01fefbadd`
  not the B37 run's HEAD (`:476`), a v1 sweep artifact at
  `git_head 484b3fc26`, that artifact's `omp_set [8,14,16,20]` not the
  frozen host set, and sweep rows lacking `child_schema`. Phase 1's fresh
  sweep is therefore not polish; it is the only path to a B37-eligible B3.
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

- The re-baseline (Phase 1) **pins a single simsoptpp sha** — this is now a
  ruling, not a contingency. See "Adjudicated" below.
- Codex owned `fix-b37-restart` content-wise; the merge conflict was one
  docs section and was mechanical, as predicted.
- Every claim/sweep run needs a **dedicated clean worktree at the frozen
  SHA**. `_require_clean_tree()` (`benchmarks/nested_ls_outer_claim.py:317`
  → `benchmarks/nested_ls_shamanskii_attribution.py:114-123`) aborts before
  any child launches if any tracked-or-untracked path outside
  `docs/receipts/evidence/` is dirty, and this repo is a shared worktree with
  several concurrent sessions writing to it.
- scipy stays pinned at 1.17.1 for the certification era; the StopIteration
  control-flow pin (Phase 0) is version-scoped.

## Implementation Plan

0. **Tier-0 hardening (~1 day; lands on the merged main, not on the branch)**
   - [x] Pin scipy 1.17.1 `StopIteration`-from-callback control flow.
         **Observed, not assumed — and the 2-vs-99 disagreement is not a
         disagreement: both are real, at different boundaries.** Through
         `scipy.optimize.minimize(method="L-BFGS-B")` the triple is
         `status=99`, message ``"`callback` raised `StopIteration`."``,
         `success=False`; through `_minimize_lbfgsb` directly, before
         `minimize`'s override, it is `status=2`,
         `"STOP: CALLBACK REQUESTED HALT"`. The children go through
         `minimize`, so **99 is what they see**. The catch is in
         `scipy/_lib/_util.py:1006-1011` (`_call_callback_maybe_halt`), the
         status is set at `scipy/optimize/_lbfgsb_py.py:475-477,492,506-510`,
         and `minimize` overrides it at `scipy/optimize/_minimize.py:823-826`.
         The catch is `StopIteration`-specific (a `RuntimeError` from the same
         callback propagates); a `StopIteration` **subclass** is also caught.
         **Plan correction: the children pass OLD-style single-positional
         callbacks** — `nested_ls_outer_jax_child.py:584` (`def _callback(xk)`)
         and the native twin's counting wrapper around
         `run.accept` (`nested_ls_outer_native_child.py:803`) — not new-style. The triple,
         `nit`, `nfev` and `x` are identical either way, but a new-style
         callback receives an `OptimizeResult` whose `.x` **is the live buffer
         L-BFGS-B mutates in place** — a migration must copy at the record
         site or every recorded candidate collapses onto the last one.
         **Defect found and fixed while pinning this:**
         `nested_ls_outer_endpoint_success` excluded stop codes by name
         (`status != 2`), so a status-99 halt published as a good endpoint.
         Now an allow-list, `NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES`.
   - [x] Add `value_is_valid: bool` to `_OuterEval` rows + native twin
         records; barrier rows carry `False`. **Item was under-scoped: the
         same defect exists one level up.** scipy does not restore
         `result.fun` after a wholly-rejected attempt — it leaves the last
         rejected trial's barrier value (measured `0.31250004043721513`
         against a true anchor objective of `0.3125`), and both children
         published that straight through as `restart_attempts[*]["fun"]`
         (JAX also as top-level `result_fun`). Both levels now carry the bit.
   - [x] Barrier contraction assertion (Ceres-gap check). Measured **≤ 1/3
         per trial**, worst ratio 0.3333333247815527 over 320 configurations
         driving real scipy against the real barrier, corroborated
         analytically from MINPACK `dcstep`. **Honest limitation:** five
         sentinel variants (coherent barrier, stale-gradient, zero-gradient,
         constant penalty, under-reporting) ALL contract at ≤ 1/3, so
         contraction cannot discriminate a coherent barrier from an
         incoherent one, and **a non-contracting line search is not a viable
         explanation for the B37 stall**. The test is a guard on the sealed
         barrier and the scipy pin, not evidence for the mechanism.
   - [x] `test_overstepping`-class regression (DESC pattern). Asserts the
         endpoint coils, surface, iota, G **and the live Boozer objects** are
         bitwise the start state, driving the real candidate store and real
         `scipy.minimize`; failure names the drifting block and its ULP gap.
         Falsified against a no-op restore. Note: an all-rejected run reports
         `nit=0` and fires no callback, so the path exercised is `record()`
         priming plus restore-on-rejection — the accept path is not reached.
   - [x] Parameterized sweep of `nested_ls_outer_restart_reason` and
         `nested_ls_outer_endpoint_success` alone (TORAX `test_cond_fun`
         pattern), extending `tests/geo/test_nested_ls_outer_transaction.py`
         (38 → 101 tests). **Plan corrections:** the status set is
         `{0,1,2,99}`, not `{0,1,2}`; and status 2 is not synonymous with
         restartable — `"STOP: CALLBACK REQUESTED HALT"` is status 2 without
         the `ABNORMAL` prefix, terminal *and* unpublishable. A producer test
         drives scipy six ways and requires set equality so the table cannot
         go stale.
   - [x] Add `simsoptpp_sha256` (and `.so` path) to
         `nested_ls_runtime_identity`. This was a **de-duplication**:
         `nested_ls_receipt_provenance` already derived both keys and already
         spread the identity dict, so receipt provenance is byte-unchanged
         while **five bare-identity consumers that had no binding now get
         one**. Missing-`simsoptpp` fails closed (raises) rather than
         recording `None` — a native-lane receipt with no binary bound to it
         is exactly the drift the field exists to catch.
   - [x] Soften the `accept()` crash into an honest-failure payload.
         **Plan correction: it is a `RuntimeError`, not a `KeyError`**
         (`nested_ls_contract.py:314`) — an implementer who writes
         `except KeyError` would leave the crash untouched while the test
         appeared to pass. **And the item was half a fix as written:** the
         parent's `_run_child` raised on `rc != 0` *before* reading the
         payload, so a schema-valid failure receipt was invisible to the
         parent no matter how well the child wrote it; the parent now names
         `child_fault_reason` and the payload path (renamed from
         `failure_reason`, which collided with three other live vocabularies
         in the same call stack and was narrower than the sealed charter's
         definition of child failure). The shared vocabulary lives
         in `nested_ls_contract.py` as
         `NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON`, beside
         `restart_reason`'s vocabulary, not in one lane.
   - [x] **Not in the original list, and required to meet this phase's own
         validation line.** Both children buried their optimizer loop,
         callback and payload builder inside a function whose heavy imports
         were function-local, so no CPU test could reach any of them. Each is
         now split into a composition root (`_prepare_*` → a frozen context)
         and the logic (`_drive_*`). Phases 2–4 need the same seam.

1. **Merge + re-baseline (review step 1–2; ~1 day + 11–14 h machine)**
   - [x] Squash-merge `fix-b37-restart` onto current main; resolve the
         charter Amendment-4 ordinal. **Done `d835c43ee`** — landed as
         Amendment 6, with `997bbacd5` renumbered 4→5 and its three ordinal
         citations updated. All five branch commits went in, not the two the
         draft named.
   - [ ] One child→parent→dual-rejudge B3 integration test (fast, stubbed
         inner) proving the driver consumes both children's v2 schemas
         end-to-end.
   - [ ] **Pin one simsoptpp sha for the whole re-baseline** and record it in
         `nested_ls_runtime_identity` before the first leg starts (Phase 0).
   - [ ] Fresh native OMP sweep on the merged source — 16 legs (the frozen
         host set `(4,8,12,14,16,20,24,32)` × 2 interleaved repeats,
         `src/simsopt_jax_adapters/geo/nested_ls_contract.py:133-143`), in a
         **dedicated clean worktree**; then mint the source-bound B3 v2
         receipt at the merged SHA. The `01fefbadd` receipt stays bridge
         evidence, cited not superseded — and cannot gate B37 (Current
         Context lists the five refusals).
   - [ ] Post-merge: DIAG4 execution-source manifest regeneration
         (`benchmarks/regenerate_execution_source_manifest.py --admit … --expect-count …`).
         **Corrected citation.** The live pin is
         `single_stage_native_equivalent_quality_successor_authority.py:383`
         (`DIAG5_EXECUTION_SOURCE_ENTRY_COUNT = 642`), with derived checks at
         `:1732` (`+3`) and `:3829` (`+2`). `:212` is only the default used
         when a caller passes no explicit count — every live caller passes
         one — and `:451` is a **sealed historical record of the
         predecessor's failed DIAG4 run**, compared against archived evidence
         at `:895-903` and `:2196`; it must not be touched. Regeneration moves
         `:383` 642 → 660 and needs **18 explicit `--admit` flags**. All 18
         entrants (the 8 `nested_ls_*` benchmarks, including the outer claim
         and both children, plus 10 from earlier campaigns) were **never**
         members, so this is a debt refreeze the merge forces into the open,
         not merge-caused drift. Note what admission actually means here:
         `_diag4_execution_source_membership` (`…successor_authority.py:4828-4843`)
         is a **broad sweep** — every regular `*.py` under `benchmarks/`,
         `examples/`, `src/`, unioned with the qualified and frozen sets — so
         `--admit` is an operator acknowledgement that files entered, not a
         curation decision about whether each belongs. The count must be
         **recomputed at run time**, not copied from here: this plan's own
         Phase-2 work adds `benchmarks/nested_ls_outer_predictor_replay.py`
         under a broad root, taking it to 661 with 19 admissions, and any
         other session's new benchmark moves it again. Consequence: the merge's own net effect on the count is
         zero, because `tests/` is not one of
         `DIAG4_EXECUTION_SOURCE_BROAD_ROOTS = ("benchmarks","examples","src")`
         and the new test file therefore does not enter membership. Run only
         on a quiescent, committed tree at the exact SHA the receipts cite —
         the regenerator hashes **working-tree** bytes, so a dirty peer edit
         gets certified.
   - [x] Merge the parked `_lbfgs` worktree branch
         (`worktree-agent-a294f051ffd23f1c9` @ `92ec95f52`, strict-PASS).
         **Done `d3bd48ecd`**; 68 + 4 + 19 tests green, pinned ruff clean.

2. **Predictor on recorded displacements (review step 3; ~2 days)**
   - [ ] Add `anchor_coil_dofs` to `NestedLsOuterState`
         (`nested_ls_reduced_scale.py:4592-4646`), written at commit time.
         Do this **together with** Phase 3's deletion of the per-eval
         `set_anchor` (see the ordering note under Phase 3): building the
         predictor on the still-mutable rolling anchor means plumbing it
         twice.
   - [ ] Cache the adjoint's dense LU (via `dense_schur_lu_preconditioner`)
         on the state at commit; predictor per trial = one
         `apply_reduced_mixed_schur_coil_tangent` JVP + one cached LU solve.
         The LU already exists and is thrown away: `nested_ls_outer_value_and_grad`
         assembles `materialize_stabilized_schur_dense` at
         `nested_ls_reduced_scale.py:4933-4940` and discards it after the
         adjoint solve.
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
         rescue — falsifiable prediction). Quiet GPU, ~5 solves; the committed
         replay log shows 43–547 s per solve, so budget ~20–45 min.
         **Input audit (done).** Leg 3 is fully reconstructable offline:
         `nested_ls_outer_b37_20260823_recovered_jax.json` has
         `endpoint_eval_index = 38`, and `outer_evals[38].coil_dofs` and
         `.inner_surface_sha256` both equal the endpoint's, so the eval-38
         anchor is complete on disk — coils (11), surface (661), ι, G — and
         x₃₉ = `outer_evals[39].coil_dofs` with `‖Δc‖₂ = 4.4728e-3`.
         Leg 4 is **replay-then-predict, not pure replay**: the trial is
         recorded (eval 43's `coil_dofs` are bitwise x₃₈, and it failed
         `inner_solve_failed`), but the poisoned eval-39 *anchor surface* is
         stored only as a hash (`inner_surface_sha256 = 052923e7b92e…`), so
         s₃₉ must be regenerated by one inner solve at x₃₉ from the recorded
         s₃₈ and verified against that hash. The regeneration is already
         execution-proven by the transaction replay log (J = 0.07471552895095307,
         9 inner iterations, from ledger surface `07c00c33e7bfddbd…`). Costs
         one of the ~5 solves. Neither JAX ledger carries `vessel_dofs` (the
         native twin does, at `$.endpoint.vessel_dofs`) and neither carries
         the start surface — that comes from the `$.lane.path` bundle.
   - [ ] Wire into `_solve_nested_inner_at_coils` behind an explicit policy
         flag; transactional rollback covers the predictor state.

3. **Inner-solve robustness ladder (review step 4; ~3 days)**

   **Re-scoped against the code and the B37 v2 evidence.** Two of the draft's
   rungs rested on premises that do not survive contact with the source, and
   the sufficiency question that motivated the ladder is now answered (see
   "Adjudicated"). What remains is real but is **hygiene and headroom, not
   sufficiency-critical** — with one rung the data actively motivates.

   - [ ] ~~Refresh-before-abandon (CVODE rule): on `inner_solve_failed` with
         a stale factorization, re-assemble/re-factor once and retry.~~
         **Premise false as written.** The production path is
         `linear_solver="dense_lu"` (`nested_ls_reduced_scale.py:4727-4734`),
         and that path re-factors `factor_reduced_nested_ls_schur` at the top
         of *every* Newton iteration (`nested_ls_reduced.py:1158-1163`). A
         stale factorization exists only in `shamanskii` mode
         (`stale_apply`, `:1144`, `:1217-1234`), which production does not
         use. **Replaced by the rung the measurement actually points at:**
         *don't abandon on the first rejected step.* In the B37 v2 diagnostic
         all three late rejections (evals 39, 43, 53) bail with **9–10 of 10
         Newton iterations unspent** at `‖g‖ ≈ 1e-3`, i.e. the walk quit on
         the Armijo/quality bail (`nested_ls_reduced.py:1258-1284`,
         `if not step_accepted: break` at `:1325`), not on budget. Retry the
         rejected iteration once with increased `stab` before declaring
         failure, and record the retry in the step ledger.
   - [ ] ~~Damped Newton: residual-norm backtracking with NaN rejection,
         replacing the undamped step at `nested_ls_reduced.py:1146-1329`.~~
         **Already implemented.** `_schur_armijo_step`
         (`nested_ls_reduced.py:803-831`) is a backtracking line search over
         `NESTED_LS_SCHUR_BACKTRACKING_MAX_STEPS = 8` halvings that rejects
         non-finite trials twice (`:815`, `:822`) and accepts on **either**
         the objective Armijo condition **or** residual-norm monotonicity
         (`trial_norm <= current_norm`, `:823`). The walk is damped; the
         draft's "undamped Newton" is wrong. Nothing to do beyond the retry
         rung above.
   - [ ] Three-valued exit on `NestedLsSchurNewtonResult`:
         `{converged, coarse_converged, failed}` with
         `NESTED_LS_NEWTON_COARSE_TOL` (proposed 1e-8, DESC's production
         setting) — recorded in ledgers; **treated as failed** until Phase 4
         licenses coarse use. Keep: this is the typed-evidence half of the
         plan and Phase 4 depends on it.
   - [ ] Inner Δc sub-stepping from the committed anchor (1→2→4→8 legs,
         halve on failure, floor, honest halt). Keep, and it is the natural
         partner to the retry rung: the three late failures are at
         `‖Δc‖ = 7.3e-3, 3.2e-3, 7.2e-3` from a committed anchor, which is
         exactly the regime sub-stepping addresses.
   - [ ] Delete the per-eval `state.set_anchor` at
         `nested_ls_reduced_scale.py:4766` and return a frozen trial
         record instead; delete FD-0's now-redundant manual pins
         (`:5067`, `:5314`) and prove FD-0 no-op (scatter 0.0, 11/11).
         **Ordering note: do this FIRST, before Phase 2.** It is the enabling
         SSOT refactor — once the anchor moves only at commit, Phase 2's
         `anchor_coil_dofs` and cached LU have exactly one write site. Doing
         Phase 2 first means plumbing the predictor through a mutable anchor
         and then re-plumbing it. Caller inventory is bounded: 3 read sites in
         `nested_ls_reduced_scale.py`, 9 in `benchmarks/nested_ls_outer_jax_child.py`,
         and a fake state plus 3 assertions in
         `tests/geo/test_nested_ls_outer_transaction.py:1095-1148`.

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

5. **New sealed policy + B37 v2 certification (~1 day + 15–18 h machine)**
   - [ ] Seal policy `anchor_frozen_predictor_v3` (names all Phase 2–4
         behaviors); bump child/claim schemas; charter amendment.
   - [ ] Fresh OMP sweep; B3 v3 physics gate; then paired single-process
         B37 v2 (single-pair rule) on a quiet box; walls informational.
   - [ ] Verdict section + receipts to the track doc.

   **Machine budget, corrected.** The draft's "~4 h machine" covers only the
   paired B37 leg. Grounded per-leg: the sweep is 16 legs, and the one
   executed sweep (`nested_ls_outer_native_omp_sweep_20260823.json`, 8 legs,
   Σ `process_wall_seconds` = 12 579 s) scales by ×2 for the doubled leg
   count and ×~1.43 for v2's `nfev` 7 → 10, giving **8–11 h**; B3 v3 must be
   3 pairs (the B37 interlock at `benchmarks/nested_ls_outer_claim.py:476`
   requires exactly 3) at **~3.3 h** measured live; paired B37 at
   `--pairs 1 --skip-prime` is **~4.1 h** (JAX leg 6866 s from the v2
   diagnostic's `wall_splits.process_elapsed_seconds`, native leg 7646 s from
   the recovered ledger's `walls.child_total_seconds`, plus 2 rejudges).
   **Total 15–18 h**, and the same arithmetic makes Phase 1 11–14 h.
   `--skip-prime` is legitimate only against a cache demonstrably warm in
   *that worktree's* `.artifacts/nested-ls-outer-xla` — the cache is
   per-worktree, so a fresh certification worktree starts cold.

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
      physics-green — the B3 gates are the per-lane C++ LS Newton reconstruct
      no-op and `fail_closed_reason is None`, and the endpoint-J gap is
      **measured and published, not gated** (`--j-parity-rtol` is forbidden at
      budget 3, `benchmarks/nested_ls_outer_claim.py:1367`); the 1e-9
      band is the B37 gate frozen from this measurement. DIAG4 manifest
      counts match pinned expectations (`…successor_authority.py:383`, not
      `:212,451`).
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

## Adjudicated (execution pass 1, 2026-08-24)

- **The pair-2 bitwise canary is unresolvable as designed; Phase 1 pins one
  simsoptpp sha.** The plan assumed a single binary boundary
  (`41b2ca79` → `95190afa`) between B3 v2 pairs 1 and 2. A **third** binary
  replaced it at 07:39:04, ten minutes into pair 2's native leg
  (`d4a6e028…`, hardlinked into both venvs, while a peer rebuilt
  `src/simsoptpp/permanent_magnet_optimization.cpp`). The three binaries are
  pinned by process maps: the driver (PID 1133505, 05:12) holds the deleted
  inode of `41b2ca79`, pair 2's native child (PID 2187449, 07:29) holds
  `95190afa`, and pair 2's JAX leg plus all six rejudge children will load
  `d4a6e028`. Pair 2 therefore straddles two binaries whatever its native leg
  reports, and the `95190afa` → `d4a6e028` step is unmeasured on the native
  lane entirely. **Outcome, measured after the fact: it did not matter.**
  Pair 2's native leg on `95190afa` reproduced pairs 0/1's endpoint on
  `41b2ca79` bitwise on every field, and pair 2's JAX leg on `d4a6e028`
  reproduced the earlier JAX legs bitwise. So the *designed* canary — one
  pair-2 comparison adjudicating one boundary — was made unresolvable by the
  third binary, while the *observed* answer is indifference across all three.
  The ruling is unchanged and is now hygiene rather than a correction:
  **freeze the `.so`, record its sha in `nested_ls_runtime_identity`
  (Phase 0, done), and run the fresh sweep + B3 v2 from first leg to last
  rejudge on that one binary.** The bridge run is evidence with a disclosed
  three-binary boundary, not a source-bound baseline — and it died before
  writing a receipt at all.
- **B37 v2 endpoint (Open Question 2): YES, the transactional lane reaches
  native-class J at budget 37 unaided.** Diagnostic `59ccbe8a0`
  (`nested_ls_outer_b37v2_20260824_jax_diagnostic.json`, schema
  `nested-ls-outer-jax-child.v4`): `endpoint_j = 0.006746235108545951` against
  the v1 native reference `0.006737776589829025` — **+0.1255 %**, where v1
  stalled **+7.65 %** away at `nit = 27`. Full budget consumed
  (`nit = 37`, `nfev = njev = 60`, `restart_count = 0`), honest exit
  (`status = 1`, ITERATIONS LIMIT, `ftol_zero_stop = False`), endpoint on the
  optimizer's own x, `endpoint_inner_grad_l2 = 1.9e-15`. The poisoning class
  is survived by construction: `outer_evals[25]` is a feasible probe at
  `j = 10.6957`, `grad_l2 = 19627`, `iota_branch_delta = 0.0071` (under the
  0.05 guard) that the line search rejected with the committed anchor intact —
  the exact event that killed v1 at eval 39. **Consequence: Phase 3 is
  robustness hygiene, not sufficiency-critical**, and Phase 3 is re-scoped
  accordingly. Qualifier: single-lane, single-trajectory, non-certifying — no
  native twin ran under v2 semantics and the GPU was shared, so walls are
  informational.
- **Where the remaining headroom is.** All 8 v2 rejections are
  `inner_solve_failed`, zero `iota_branch_guard`, at evals 1–5 (start
  transient) and 39, 43, 53. The three late ones bail with **9–10 of 10
  Newton iterations unspent** at `‖g‖ ≈ 1e-3` and `‖Δc‖` of 7.3e-3, 3.2e-3,
  7.2e-3 — an Armijo/quality bail on the *first* Newton step, not budget
  exhaustion. That is what Phase 3's retry-with-regularization and
  Δc sub-stepping rungs must target. The stall neighbourhood itself is gone:
  v2 walks 38 → 39(rej) → 40 → 41 → 42 → 43(rej) → 44 and keeps descending,
  where v1 was terminal after the eval-39 capture.
- **B3 does not gate on J parity.** `--j-parity-rtol` is *forbidden* at
  `--budget 3` (`benchmarks/nested_ls_outer_claim.py:1367`) and
  `endpoint_j_within_frozen_band` must be `None` on B3 pairs (`:605-609`).
  B3 measures; 1e-9 is the **B37** gate, frozen from B3. Measured on the
  in-flight pairs 0/1 (bitwise identical to each other on every endpoint
  field): native `0.012982793095001662`, JAX `0.012982793095005024`,
  relative gap **2.5895e-13** — inside 1e-9 by 3862×, but with the sign
  flipped versus v1 (v1 measured −1.588e-12 with worse-direction 0.0), so a
  B3 v2 receipt carries `measured_j_rel_gap_max = 2.59e-13` rather than 0.0.

## Open Questions

- Owner split: which phases run in this session vs Codex vs peer sessions.
  The B3 v2 bridge run is a peer's; the B37 v2 diagnostic was a peer's; this
  session owns the merge, Phase 0, and Phases 2–5 implementation.
- Scheduling the certification runs. Phase 1 (11–14 h) plus Phase 5
  (15–18 h) is 26–32 h of machine time that wants a quiet box, and this box
  currently carries a DESC continuation, a pytest suite, and the bridge B3
  run. Whether to serialize them here, defer, or move a leg to the A100 is a
  scheduling decision, not a technical one.
- `NESTED_LS_NEWTON_COARSE_TOL=1e-8`: adopt DESC's value or derive from the
  Phase 4 budget? (Plan assumes derive; 1e-8 is the starting hypothesis.)
