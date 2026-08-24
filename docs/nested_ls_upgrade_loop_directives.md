# Nested-LS Upgrade — Loop Directives

**Status:** Ready to execute
**Last updated:** 2026-08-24

Self-contained briefs for the sessions executing
`docs/nested_ls_upgrade_implementation_plan.md` as closed optimization loops
("iterate on X with a harness and dataset until it hits Y"). One owner
session runs Loops 1→2→4 sequentially; an optional second session may run
Loop 3 concurrently (disjoint files, its own worktree). Do not run Loops 1
and 2 in different sessions — they edit the same seam files.

## Setup (once per session)

```bash
# Owner session (Loops 1, 2, 4):
git worktree add ../.wt-upgrade-main -b upgrade/nested-robustness
# Side session (Loop 3 only):
git worktree add ../.wt-upgrade-tolbudget -b upgrade/adjoint-tol-budget
```

Known-good run environment (every python launch, GPU or CPU):

```bash
WT=<absolute worktree path>
env PYTHONPATH="$WT:$WT/src" \
    JAX_ENABLE_X64=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR="$WT/.artifacts/nested-ls-outer-xla" \
    /home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-gpu/bin/python ...
```

Traps that cost hours on 2026-08-24 — respect them:
- The venv installs meta-path finders that override `PYTHONPATH` in
  worktrees. If imports resolve to the wrong tree, neutralize via a
  `sitecustomize.py` on `PYTHONPATH` that strips finders whose module
  matches `__editable__`/`scikit_build` (pattern in the memory note
  `worktree-validation-metapath-trap`). Verify with
  `python -c "import simsopt_jax_adapters, sys; print(simsopt_jax_adapters.__file__)"`.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` on EVERY GPU process, no exceptions.
- Check `nvidia-smi` for other tenants before any GPU run; coordinate via
  cross-session message if the card is busy. One GPU, one owner at a time.
- JAX test files: one file per pytest process.
- Native legs: OMP pinned to the sweep artifact's value; never bare.
- Scripts written by the Write tool are 0644 — `chmod +x` or run via
  `bash script.sh`.

## Invariant gates — green on EVERY iteration, no exceptions

1. `tests/geo/test_nested_ls_outer_transaction.py` — all green.
2. FD-0 no-op when the seam changed: scatter 0.0, 11/11 directions.
3. The δc=0 ⇒ δs_pred=0 bitwise invariant (an eval at coils bitwise-equal
   to the committed anchor reproduces the no-predictor result exactly).
4. `ruff check` + `ruff format --check` + targeted `pyright` on touched
   modules (CPU venv).
5. No timing claims — walls are diagnostics; label everything
   diagnostic-not-certifying until Phase 5 mints sealed receipts.

## Iteration protocol (all loops)

edit → run the loop's harness → append one line to the loop's log
(`<worktree>/.artifacts/loop<N>_log.md`: iteration, change, metric, verdict)
→ commit if the metric improved with gates green, revert if not. Stop and
write a report (best result, blockers, ranked next hypotheses) after
**3 consecutive iterations without metric improvement** — do not thrash.

## Loop 1 — tangent predictor (plan Phase 2)

- **X (what you edit):** port `_traceable_predict_warmstart_result_from_anchor`
  (`surface_objectives_traceable.py:3154`) into the inner-solve seam
  (`nested_ls_reduced_scale.py:~4701`), + `anchor_coil_dofs` on the state,
  + cached-LU reuse, + tr_ratio clip (scale to `0.1·‖s_anchor‖`, DESC
  semantics) + envelope-gradient fallback.
- **Dataset:** consecutive (anchor, trial) coil pairs from
  `docs/receipts/evidence/nested_ls_outer_b37_20260823_recovered_jax.json`
  and `..._probe_restart.a100.json` (per-eval `coil_dofs`,
  `endpoint_surface_dofs` = the anchor surface).
- **Harness:** a replay script (pattern:
  `docs/receipts/evidence/nested_ls_outer_b37_20260824_transaction_replay.log`'s
  producer) that installs a recorded anchor, applies the predictor, runs the
  inner solve, and reports iterations + success per pair. 5090, minutes per
  sweep.
- **Metrics:** (a) mean inner Newton iterations over accepted-class pairs;
  (b) `inner_solve_failed` count over the 16 recorded A100 failure trials.
- **Y (stop-green):** (a) ≤ 2.0 (baseline ≈ 3–4) AND the two falsifiable
  predictions adjudicated and written to the track doc: does the predictor
  prevent the eval-39 wrong-branch capture (synthesis predicted "plausibly
  yes", low-med confidence)? does it fail to rescue the post-poisoning
  x₃₉→x₃₈ trial (predicted "yes, fails")?

## Loop 2 — robustness ladder (plan Phase 3; after Loop 1)

- **X:** in order, one mechanism per commit: (1) refresh-before-abandon
  (re-factor once on failure with a stale factorization, then retry),
  (2) damped inner Newton (residual-norm backtracking + NaN rejection,
  TORAX shape) in `nested_ls_reduced.py:1146-1329`, (3) three-valued exit
  {converged, coarse_converged @1e-8 (recorded, still FAILED until Loop 3
  licenses it), failed}, (4) Δc sub-stepping from the committed anchor
  (1→2→4→8 legs, floor, honest halt), (5) delete the per-eval
  `set_anchor` at `nested_ls_reduced_scale.py:4743-4745` + FD-0 manual
  pins, prove FD-0 no-op.
- **Dataset:** the stall neighborhood — recorded evals 38–44 + the probe's
  16 restart-era failures.
- **Metric:** fraction of dataset trials ending in {solved | honest
  sub-stepped solve | honest typed failure} vs budget-exhausted rejection.
- **Y:** 16/16 non-pathological AND a B3-budget rehearsal whose healthy-path
  per-eval wall regresses < 5%.

## Loop 3 — adjoint tolerance budget (plan Phase 4; own worktree, may run concurrently)

- **X:** a standalone measurement harness (new file under `benchmarks/`) +
  one red test. Does NOT edit the seam files.
- **Dataset:** 5 recorded anchor states from the recovered ledger.
- **Harness:** for inner tol in {1e-13, 1e-11, 1e-9, 1e-8, 1e-7, 1e-6}:
  solve, compute the IFT adjoint gradient, compare against the 1e-13
  reference. Also record κ(Ĥ_ss) from the dense factorization.
- **Metric/deliverable:** the error curve ‖Δg‖/‖g‖ vs inner residual,
  committed as evidence + the derived licensed tier (max tol with gradient
  error < 1e-2 relative) + a red test failing closed outside it.
- **Y:** curve + tier + red test committed. Gates any coarse-trial use in
  Loop 2's item 3.

## Loop 4 — certification (plan Phase 5; ONLY after Loops 1–3, quiet box)

- Fresh native OMP sweep at the merged SHA → B3 v2 receipt (v2 schemas,
  source-bound; note `865412667` already fixed the rejudge-binding defect
  that would have failed every v2 pair) → B37-budget diagnostics until
  37/37 + endpoint inside the J-parity band vs a same-semantics native leg
  AND jax wall < native wall → seal policy `anchor_frozen_predictor_v3`,
  mint the certified paired B37 v2 receipt. ~5 h machine time total; run
  when the box can be quiet end-to-end.

## Ownership & coordination

- Loops 1/2/4: one owner session in `.wt-upgrade-main`. Loop 3: optional
  second session in `.wt-upgrade-tolbudget`. Nobody edits the shared
  checkout.
- GPU etiquette: announce takeover/release via cross-session message; the
  Codex session is unreachable by message — route through the user.
- On completion of each loop: commit the loop log + update the checkboxes
  in `docs/nested_ls_upgrade_implementation_plan.md`.
