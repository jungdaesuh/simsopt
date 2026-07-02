# scipy-jax-decomposed GPU Perf Gap Closure Plan

**Status:** In progress
**Last updated:** 2026-07-02

## Purpose

Execution plan for the follow-up work identified by the 2026-07-01 root-cause
deep-dive into why `lbfgs-scipy-jax-decomposed` on GPU ran slower than the
native cpp/CPU reference. Companion to
`docs/scipy_jax_decomposed_newton_polish_and_reporting_reuse_report_2026-07-01.md`
(the diagnosis/fix report). That report closed the workflow-redundancy layer in
code; this plan covers (1) the still-missing runtime proof of those fixes,
(2) the trial-policy quality gate, and (3) the remaining structural per-solve
gap. Historical file:line citations in the diagnosis section are anchored at
commit `0cf4230cb`; the execution-status section reflects later implementation
commits and should be read as the current source-of-truth for completion state.

Abbreviations: `EX` = `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`,
`OPT` = `src/simsopt_jax/geo/optimizers/optimizer.py`,
`BZ` = `src/simsopt_jax_adapters/geo/boozer_surface.py`.

## Goals

- Prove at GPU runtime that `0cf4230cb` (jax.Array memo-key normalization,
  `EX:7491-7498`) eliminates both the trace-lane duplicate K1 solve and the
  final-sync K1 re-solve on the production non-trace path.
- Prove the trial-only Newton-polish `skip` default (`EX:265`) does not degrade
  optimizer trajectory quality vs `run` on an accepted-result run.
- Reduce the per-Newton-iteration HVP count in the traceable Newton polish
  (`newton_polish_traceable`, `OPT:5908`), measured before designed.
- Replace the hardcoded dense-operator chunk batch default
  (`_DENSE_OPERATOR_CHUNK_BATCH_SIZE`, `OPT:3607`) with byte-budget
  auto-sizing.
- Bound the trial pre-Newton L-BFGS budget so trial K1 stops paying for a
  1500-iteration budget that Newton rescues in ~6 iterations anyway.

## Non-Goals

- Making `lsmr_j` the default adjoint solver (stays an experimental comparator
  behind `SIMSOPT_ADJOINT_LINEAR_SOLVER=lsmr_j`, requires `newton_stab > 0`,
  `OPT:5153-5242`).
- Switching the outer optimizer to `lbfgs-ondevice` (monolithic-compile risk;
  decided against in the report).
- Any change to the native CPU/cpp reference path or to physics/objective math.
- Re-running the compile-cache operational work (persistent-cache dir handling
  is a known separate thread; see HANDOFF.md §9).

## Current Context

Confirmed facts from the deep-dive (A100 artifacts `a100_run_artifacts_2026-06-29/`,
Perlmutter jobs 55353209 / 55358522 / 55363238, and code reads at `0cf4230cb`):

- Pre-fix warm smoke (m3, maxiter cap 3, mpol10/ntor10/nphi255/ntheta64):
  1053s progress-clock total, three objective evaluations, and one accepted
  optimizer iteration. One diverging line-search trial (objective 34547 vs
  7.2e-4) cost 493s (246s solve-with-full-polish + 247s duplicate trace
  re-solve) = 47% of the run. Rejected-trial Newton polish alone measured
  175s / 24 stalled iterations on Perlmutter (job 55353209).
- Fix chain: `9ed630123` (trace memo) and `dd604859e` (final-sync cache +
  trial `skip` policy) were **dormant** for real target-lane inputs because the
  memo guard was `isinstance(coil_dofs, np.ndarray)` while the lane passes
  `jax.Array`; `0cf4230cb` host-normalizes the key and activates both. Local
  regressions pass; the follow-up GPU smoke (job 55364581) timed out before
  allocation, so **the linchpin commit has no GPU runtime proof yet**.
- Trial policy plumbing: default `skip` (`EX:265`), resolved at `EX:9963`,
  applied per K1 solve call via
  `wrap_target_lane_solved_pair_with_boozer_overrides` (`EX:7384`). Trial K1 at
  HEAD = pre-Newton L-BFGS only (`BZ:6301`, skip branch `BZ:6324-6354`).
  Initialization, accepted-step, final-sync, and reference paths keep `run`.
- Structural cost: traceable Newton polish solves each correction with
  operator-GMRES over HVPs (restart 64 × maxiter 10 + 1 refinement pass,
  `OPT:4207-4210`, `OPT:4735-4821`) — worst case ~1280 HVPs/iteration, each a
  full fwd+bwd BiotSavart pass over the 255×64 half-period grid. The
  factor-once path materializes the dense n×n Hessian (n=663 at
  mpol10/ntor10 stellsym, run metadata) via `lax.map` chunks
  (`OPT:3669-3684`): batch 8 → 83 sequential chunks, batch 4 (40GB A100) → 166.
  Native C++ Newton pays ONE analytic `sopp.boozer_residual_ds2` assembly +
  ONE LAPACK LU per iteration (`src/simsopt/geo/boozersurface.py:536-554`).
- Boozer init evidence: on-device L-BFGS ran 701 iterations, failed
  tol 1e-10, Newton converged in 6 iterations (m3 `boozer_init_progress.json`).
- A/B harness exists: `benchmarks/perlmutter/single_stage_k1_matrix_gpu.slurm`
  with cases `baseline_dense_run_chunk8`, `lbfgs_run_chunk8`,
  `dense_skip_chunk8`, `dense_run_lsmrj_chunk8`, `dense_run_chunk16`,
  selectable via `MATRIX_CASES`.

## Execution Status (2026-07-02)

Authoritative state as of the 2026-07-02 scoped implementation pass:

- Phase 1 is now assertable but not GPU-proven. The K1 matrix launcher has an
  opt-in progress gate (`MATRIX_ASSERT_K1_PROGRESS=1`) backed by
  `benchmarks/single_stage_k1_progress_assertions.py`. By default it checks K1
  return events, trial Newton skip semantics, and optional final-sync reuse; when
  `MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE=1`, it additionally checks
  exactly-one K1 return per `objective_evaluation` and optional trace-forward
  reuse. Remote Perlmutter
  validation passed for py-compile, the focused synthetic pytest slice, and a
  CLI synthetic progress smoke. Fresh GPU submission of this assertion-enabled
  job is currently blocked by Slurm policy: even minimal `sbatch --test-only`
  GPU jobs for `gpu_debug`, `gpu_interactive`, `gpu_shared`, and `gpu_regular`
  returned `Job request does not match any supported policy`. Existing earlier
  GPU jobs remain pending, but they target older source commits and therefore
  do not prove this assertion-enabled HEAD.
- Phase 3 measurement plumbing is partially implemented at `61d1cb99a`.
  `SIMSOPT_TRACEABLE_NEWTON_MATVEC_COUNTS=1` records actual operator matvec
  callback counts into `newton_trace_linear_solve_matvec_actual` and
  `newton_last_linear_solve_matvec_actual`; default behavior is unchanged.
  Remote Perlmutter validation passed the focused private optimizer tests.
  The decision gate still requires a real GPU run artifact because the local
  JAX public `gmres` `info` value is status-only on the pinned runtime, not an
  iteration count.
- Phase 4 code for byte-budget chunk sizing has landed, but the required
  no-explicit-override GPU validation has not run yet.
- Phase 5 trial-budget plumbing has landed in the child single-stage runner,
  the parent parity wrapper, and the K1 matrix launcher. The launcher now
  accepts `MATRIX_TRIAL_BOOZER_BFGS_TOL` and
  `MATRIX_TRIAL_BOOZER_BFGS_MAXITER`, records them in each case manifest, and
  forwards them as `--target-lane-trial-boozer-bfgs-*` to the child. No
  accepted-result A/B quality gate or measured 200/300/500 sweep has completed
  yet.
- Phase 6 same-resolution runtime seed-spec bypass landed at `651639189` and
  was validated remotely with a real CLI same-shape copy smoke
  (`payload_equal=true`, identical SHA-256 source/output).

## Rationale

The redundancy fixes are committed but the activating commit is unproven on
GPU; running validation first is cheap and de-risks everything downstream.
The trial-`skip` policy is a deliberate contract change (line search now sees
L-BFGS-only merit values), so it needs a trajectory-quality gate before it can
be trusted in production sign-off. Only then is structural work justified:
measure actual GMRES matvec counts first, because Eisenstat–Walker adaptive
tolerances (`OPT:4466-4495`) may make the worst-case bound irrelevant — if
measured HVPs/iteration is already below ~n (663), swapping GMRES for a
materialize-once-per-iteration + direct-solve scheme is not a win.

## Assumptions

- The prior Perlmutter diagnostic GPU was A100-SXM4-40GB
  (`gpu_shared_interactive`). Use the available interactive GPU for validation,
  but compare wall time only against the same GPU class and record memory,
  chunk batch, and preallocation. On H100-class hardware the lane already beat
  cpp (1.14×, 2026-06-23 cross-GPU benchmark), so "GPU slower than cpp/cpu" is
  an A100-tier + workflow-multiplier statement, not universal.
- Run metadata condition estimates hold (`ls_condition_estimate ≈ 6.1e6`;
  squared-system GMRES stagnation gate documented at `OPT:3641-3652`).
- The dirty working tree stays under concurrent edits; every phase commits
  scoped (`git commit --only -- <paths>`).

## Implementation Plan

1. **Phase 1 — GPU runtime proof of the committed fixes (no code changes)**
   - [ ] Submit the K1 matrix at `0cf4230cb` (or current HEAD) with
         `MATRIX_CASES=baseline_dense_run_chunk8,dense_skip_chunk8`,
         `MATRIX_MAXITER=1`:
         `sbatch --export=ALL,REPO_ROOT=<checkout>,RUN_ROOT=<scratch>,MATRIX_CASES=... benchmarks/perlmutter/single_stage_k1_matrix_gpu.slurm`
   - [ ] Assert in progress events: exactly one
         `target_lane_decomposed_k1_forward_returned` per eval (no duplicate
         K1), and `newton_polish_skipped=true` / `newton_iter=0` on trial
         evals. Do **not** require `pre_newton_iter=0`: the `skip` policy
         bypasses Newton polish only; the pre-Newton L-BFGS stage still runs and
         must report `pre_newton_iter` / `pre_newton_nfev` / `pre_newton_ngev`.
   - [ ] In trace-enabled runs, assert the wrapper emits
         `target_lane_optimizer_objective_eval_N_forward_result_reused`
         (not `..._started`) after K2 for the same candidate.
   - [ ] Run one case **without** `--record-objective-evaluation-trace` to
         prove the production non-trace path populates the sync cache
         (`build_single_stage_target_lane_objective_evaluation_sync_cache_wrapper`,
         `EX:13421`) and final sync shows
         `target_lane_reporting_forward_result_started reused=true`.
   - [ ] Record per-eval wall deltas against the 2026-06-29 A100 baseline
         (warm: 168s / 493s / 141s per eval). Treat the report's post-fix
         K1 33-63s figures as a partial signal only: that run proved final-sync
         reuse but still reported `newton_polish_skipped=false`, so the final
         `0cf4230cb` smoke is the first valid trial-skip + trace-reuse proof.

2. **Phase 2 — Trial-skip trajectory quality gate**
   - [ ] Same seed/targets (iota011_R0935: iota 0.11015671329581699,
         vol 0.04920000000000004), `maxiter` 20–50, A/B
         `--target-lane-trial-boozer-newton-polish-policy skip` vs `run`.
   - [ ] Compare: final J, FINAL_IOTA / FINAL_VOLUME / FINAL_BOOZER_RESIDUAL,
         accepted-iteration count, total line-search eval count,
         rejected-trial rate, wall time.
   - [ ] Gate: `skip` reaches equal-or-better J at equal accepted iterations
         (tolerance: the two trajectories may diverge; judge by objective
         quality, not step-for-step parity), and the rejected-trial rate does
         not increase enough to erase the per-trial savings.

3. **Phase 3 — Structural: cut HVPs per Newton iteration (measure first)**
   - [ ] Extend the `1a9deabac`/`945a010b2` diagnostics to record the actual
         GMRES matvec count per Newton iteration (Eisenstat–Walker-adjusted),
         surfaced through the existing K1 subtimer events.
   - [ ] Decision gate (write results into this file): if measured
         HVPs/iteration ≥ ~n (663), implement Option A; if well below,
         document and close this phase as not-a-win.
   - [ ] Option A: per-iteration dense materialization
         (`_materialize_dense_linear_operator`, `OPT:3669`) + on-device PLU
         solve inside the traceable Newton loop, reusing the factor-once
         dispatch machinery (`BZ:3527-3548`). Env-gated comparator first
         (mirror the `lsmr_j` precedent), default off.
   - [ ] Option B (fallback): keep GMRES, precondition with a stale PLU factor
         refreshed every k iterations. Only if Option A's per-iteration
         assembly cost regresses.
   - [ ] Equivalence gate: optimizer objective/gradient unchanged
         (bit-identical where the contract requires; otherwise the repo's
         ≤1e-12 equivalence harness) before any default flip.

4. **Phase 4 — Chunk-batch byte-budget auto-sizing**
   - [ ] Derive the dense-operator chunk batch from the GPU byte budget
         (`SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU`, env name at
         `runtime.py:126`, default at `runtime.py:234`) instead of the
         hardcoded env default 8 (`OPT:3607`); keep
         `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE` as explicit override.
         Constraint: `lax.map` needs a static batch at import (`OPT:3601-3606`).
   - [ ] Verify the auto value on the lowest-memory target GPU, including the
         prior 40GB A100 diagnostic class when available, with
         `XLA_PYTHON_CLIENT_PREALLOCATE=true` does not OOM. Historical handoff
         evidence records batch=32 OOM around a 49.35 GiB effective allocation
         at mpol10/nphi255 (HANDOFF.md §5); the auto-sizer must be validated on
         the actual allocated GPU rather than assuming 32 is safe on every
         nominal-memory class.

5. **Phase 5 — Bounded trial pre-Newton budget**
   - [x] Add a trial-context `bfgs_maxiter`/`bfgs_tol` override alongside the
         trial polish policy (same plumbing:
         `build_target_lane_trial_boozer_overrides`, `EX:8644`) and expose it
         through the parent parity wrapper plus the K1 matrix launcher.
         Evidence for sizing: init used 701/1500 iterations and still failed
         1e-10.
   - [ ] Run a measured sweep (e.g. 200/300/500) on the Phase 2 config.
   - [ ] Guard: accepted/final/reference solves keep the full budget;
         policy recorded in progress metadata like `newton_polish_policy`.

6. **Phase 6 (independent) — seed-spec projection bypass**
   - [ ] Bypass same-resolution runtime-seed-spec projection when the
         serialized surface already matches the requested resolution (the
         staging hang noted in the report's Runtime boundary section). Not a
         perf-gap item; unblocks official warm-start staging for Phases 1–2.

## Validation Plan

- [ ] Local regressions (repo tests require the meta-path workaround from
      HANDOFF.md §4 — drop `ScikitBuildRedirectingFinder`, force `src/`):
      `tests/integration/test_single_stage_jax_cpu_reference.py -k "trace_wrapper_uses_decomposed_k1_cache_after_value_grad or decomposed_trace_reuse_hits_through_real_optimizer_to_coil_transform"`,
      `tests/integration/test_single_stage_newton_polish_policy.py`,
      `tests/geo/test_boozersurface_jax_private.py`.
- [ ] Phase 1 progress-event assertions (listed inline above) on the actual
      Perlmutter artifacts, not just exit codes.
- [ ] `/usr/bin/time` per-case wall comparison across matrix cases; GPU memory
      high-water from `nvidia-smi_before/after` snapshots the harness writes.
- [ ] Any Phase 3–5 change that can touch the objective path passes the
      equivalence gate before default-on; env-gated until then.

## Risks and Mitigations

- Risk: trial-`skip` changes line-search merit values enough to alter
  trajectories and mask a quality regression on longer runs.
  Mitigation: Phase 2 A/B gate on final objective/physics, not wall time alone;
  keep `run` reachable via flag for reference runs.
- Risk: Phase 3 Option A pays n HVPs of assembly on iterations where
  Eisenstat–Walker GMRES would have converged in far fewer matvecs.
  Mitigation: the measure-first decision gate; comparator stays env-gated.
- Risk: auto-sized chunk batch OOMs under XLA preallocation on lower-memory
  cards.
  Mitigation: conservative byte model validated against the historical
  batch=32 / ~49.35 GiB OOM report; explicit env override preserved.
- Risk: concurrent Codex/agent edits in the dirty tree clobber plan work
  (has happened before in this repo's dirty worktree).
  Mitigation: scoped commits per phase (`git commit --only -- <paths>`),
  verify guards present before committing.
- Risk: single-entry K1 memo is insufficient if a future optimizer evaluates
  value and gradient at different points in one step.
  Mitigation: L-BFGS-B calls `value_and_grad` at one x per eval today; add a
  regression asserting memo-hit on the final-sync DOFs so any change breaks
  loudly.

## Completion Criteria

- [ ] GPU smoke at HEAD shows: 1 K1 solve per eval, trial `newton_iter=0`
      with pre-Newton metrics still recorded, trace-wrapper
      `forward_result_reused` for the post-K2 same-candidate check,
      final-sync `reused=true` on the non-trace path, and per-eval wall at or
      below ~50% of the 2026-06-29 warm baseline on the same A100 class.
- [ ] Phase 2 quality gate passed and recorded (or `skip` default reverted
      with the evidence written into the report doc).
- [ ] Phase 3 decision gate resolved either way with measured HVP counts in
      this file; if implemented, equivalence gate green.
- [ ] Phases 4–5 landed with regressions, or explicitly rejected with data.
- [ ] Report doc and handoff updated with runtime results; memory update filed
      only when that workflow is explicitly requested by the operator.

## Open Questions

- Measured HVPs per Newton iteration under Eisenstat–Walker at production
  conditioning (κ≈6e6 LS-Hessian) — the Phase 3 gate input.
- Native cpp/CPU per-eval wall at the iota011_R0935 config on Perlmutter CPU
  nodes: no local record exists; needed for the definitive post-fix
  cpp-vs-GPU headline. (`single_stage_fair_compare_gpu.slurm` co-produces the
  reference; one fair-compare run answers it.)
- Does benchmark-mode's deferred reporting snapshot (`EX:18279-18294`)
  interact with the now-populated sync cache in any path that skips the
  final-sync reuse? Verify during Phase 1.
