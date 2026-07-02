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

- Phase 1 now has direct H100 runtime proof for the two cache contracts, with
  one important fidelity boundary. RunPod H100 run
  `/workspace/runpod-k1-runs/direct-jax-m10-h100-edea9ba10-20260702T030856Z`
  (`runpod_artifacts_2026-07-02/trace/`) completed with exit `0`, wrote
  `final_artifact_write_returned`, and in trace mode emitted 17 K1 returns, 9
  K2 returns, and objective-evaluation forward-result reuse events after K2 for
  the same candidates. Default trial `skip` runs intentionally keep final-sync
  solved-state reuse disabled when trial fidelity differs from final/reporting
  fidelity; the trace run recorded active trial overrides
  (`bfgs_maxiter=1500`, `newton_maxiter=40`, `newton_polish_policy=skip`) and
  final reporting therefore recorded `target_lane_reporting_forward_result_*`
  with `reused=false`. The same non-trace/default-skip boundary was reproduced
  in `/workspace/runpod-k1-runs/direct-jax-m10-nontrace-h100-edea9ba10-20260702T033009Z`
  (`runpod_artifacts_2026-07-02/nontrace/`), where final sync returned in about
  `27.82 s` but did not reuse the lower-fidelity trial solve.
- A same-fidelity non-trace H100 proof run completed at
  `/workspace/runpod-k1-runs/direct-jax-m10-nontrace-trialrun-nobudget-h100-edea9ba10-20260702T041302Z`
  (`runpod_artifacts_2026-07-02/nontrace_trialrun_nobudget/`, cell
  `mpol=10-ntor=10-f0530745`). It used
  `--target-lane-boozer-newton-polish-policy run` and
  `--target-lane-trial-boozer-newton-polish-policy run` with no explicit full
  BFGS/Newton budget overrides. Progress recorded
  `target_lane_trial_boozer_newton_polish_policy_override=null`, every
  `trial_boozer_overrides` value `null`, four K1 returns, three K2 returns, and
  final-sync reuse:
  `target_lane_reporting_forward_result_started reused=true`,
  `target_lane_reporting_forward_result_returned reused=true`,
  `target_lane_reporting_value_and_grad_started reused=true`,
  `target_lane_reporting_value_and_grad_returned reused=true`,
  `target_lane_final_sync_returned elapsed_s=1.1226`, and exit `0`.
  This proves the production non-trace path can consume the decomposed K1
  solved-state memo when trial and final solve fidelity match.
- The implementation now normalizes no-op trial polish policy overrides:
  an explicit trial `run` with full `run`, or default trial `skip` with full
  `skip`, resolves to no active trial-policy override. Remote H100 focused
  validation passed `7` policy/cache tests and `2` final-sync fallback tests
  after syncing the patched files to `/workspace/simopt-k1-current`.
- The K1 matrix launcher remains useful but is no longer the only Phase-1 proof
  vehicle. It has an
  opt-in progress gate (`MATRIX_ASSERT_K1_PROGRESS=1`) backed by
  `benchmarks/single_stage_k1_progress_assertions.py`. By default it checks K1
  return events, trial Newton skip semantics, and optional final-sync reuse; when
  `MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE=1`, it additionally checks
  exactly-one K1 return per `objective_evaluation` and optional trace-forward
  reuse. The launcher also emits `k1_matrix_report.json` via
  `python -m benchmarks.single_stage_k1_matrix_report`, a pure artifact reader that
  rolls up per-case exit status, assertion status, progress-file summaries,
  trace-reuse counts, final-sync reuse, and missing-artifact issues before the
  matrix failure gate. Remote Perlmutter validation passed for py-compile, the
  focused synthetic pytest slices, and CLI synthetic progress/report smokes.
  Fresh Perlmutter GPU submission initially failed when the account was omitted.
  The valid Perlmutter shape is `sbatch -A m4680_g -C gpu -q shared ...`.
  Source-commit Phase-1 Perlmutter jobs `55373498` and `55373499` were not used
  as proof: `55373498` failed during editable-install submodule setup, and
  `55373499` failed the matrix command because the high-resolution donor
  continuation path lacked `MATRIX_WARM_START_RUN_DIR`.
- Phase-1 Perlmutter GPU job `55380861` was submitted from staged clean
  executable source at `1d055547e`
  `/pscratch/sd/j/jungdae/simopt-jax-clean-local-1d055547ebbc-k1matrix-20260702T092557Z-src`
  with `MATRIX_CASES=baseline_dense_run_chunk8,dense_skip_chunk8`,
  `MATRIX_MAXITER=1`, `MATRIX_ASSERT_K1_PROGRESS=1`,
  `MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE=1`, and
  `MATRIX_REQUIRE_TRACE_REUSE=1`. It was still `PENDING (Priority)` at the
  time of this update, so it is a submitted validation run, not result proof.
- Phase 3 measurement plumbing is implemented and now has a real H100 artifact.
  `SIMSOPT_TRACEABLE_NEWTON_MATVEC_COUNTS=1` records actual operator matvec
  callback counts into `newton_trace_linear_solve_matvec_actual` and
  `newton_last_linear_solve_matvec_actual`; default behavior is unchanged.
  Remote RunPod H100 run
  `/workspace/runpod-k1-runs/phase3-phase4-iota011-R0935-run-matvec-autochunk-h100-20260702T072931Z-final`
  exited `0` and wrote progress for cell `mpol=10-ntor=10-45d88835`.
  The rejected K1 trial recorded `newton_iter=40`,
  `newton_attempted_iterations=40`, all 40 actual matvec entries positive at
  `1307`, and `newton_last_linear_solve_matvec_budget=1302`. The accepted K1
  solve recorded `newton_iter=3` and three positive actual entries, also
  `1307`. At n=663 this is about `1.97 * n` matvecs per Newton iteration,
  which resolves the measure-first Phase 3 gate in favor of a dense
  materialization comparator.
- Phase 3 Option A is now implemented as a default-off comparator, not a
  default path. `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER=dense_lu` switches the
  traceable Newton correction solve from operator-GMRES over HVPs to dense
  materialization plus on-device LU when the dense byte policy permits it;
  unset/default stays `operator_gmres`. Remote H100 focused validation passed
  after syncing the patch to `/workspace/simopt-k1-current`: py-compile for the
  touched files and
  `pytest -q tests/geo/test_boozersurface_jax_private.py -k "traceable_forward_result_packs_newton_linear_solver_backend_code or dense_lu_comparator or iteration_diagnostics or matvec_counts"`
  reported `6 passed, 126 deselected`.
- The first real H100 single-stage comparator smoke was correctness-clean but
  slower than the operator-GMRES baseline on this mpol10 case. Baseline
  `/workspace/runpod-k1-runs/phase5-cap300-finalreuse-h100-20260702T084258Z`
  exited `0` in `5:38.69` wall with K1 spans `33.0 s`, `26.0 s`, `34.6 s`,
  final J `7.164389974566755e-4`, final iota `0.11017554592247202`, final
  volume `0.04916423166118373`, and Boozer residual `4.573194822063472e-7`.
  Dense-LU comparator
  `/workspace/runpod-k1-runs/phase3-denselu-cap300-h100-20260702T090318Z`
  exited `0` in `6:32.27` wall with K1 spans `149.0 s`, `21.9 s`, `20.6 s`,
  final J `7.164393917615716e-4`, final iota `0.11017556043410176`, final
  volume `0.049164231668496566`, and Boozer residual
  `4.573198237229834e-7`. The dense path reduced the reported linear-solve
  matvec budget from `1302` to `663` and improved some later K1 solves, but the
  first dense materialization cost dominated. Conclusion: keep dense-LU
  experimental/debug-only until a higher-resolution or repeated-solve case
  proves a wall-time win.
- A follow-up short reporting smoke was launched after adding the explicit
  `newton_linear_solve_backend_code` packer field, but the RunPod endpoint
  disappeared mid-run (`ssh: connect ... Connection refused`) and provider-side
  `runpodctl pod list --all` returned no pods. The focused H100 test above
  proves the field packer and dispatch; a future real artifact should confirm
  the progress JSON carries backend code `1`/`2` without relying on the matvec
  budget as the discriminator.
- Phase 4 byte-budget chunk sizing now has a no-explicit-override H100 proof in
  the same run: `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE` and
  `SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU` were unset, preallocation stayed on,
  transfer guard stayed disallow, and K1 progress recorded
  `dense_operator_chunk_batch_size=8` with
  `max_dense_hessian_bytes=268435456`. This validates the default auto-sizing
  on the allocated H100 80GB target; a lower-memory A100-40GB validation remains
  useful when that hardware is available.
- The same H100 artifact exposed one remaining reporting-reuse gap:
  `target_lane_reporting_forward_result_started` at `858.75 s` returned at
  `1060.20 s`, so final reporting still paid about `201.4 s` for a K1 forward
  result in this run.
- That reporting-reuse gap is now closed for cap-limited trial solves whose
  cap did not bind. RunPod H100 smoke
  `/workspace/runpod-k1-runs/phase5-cap300-finalreuse-h100-20260702T084258Z`
  exited `0` with the same cap-300 final objective/physics as the earlier
  sweep, recorded `target_lane_reporting_forward_result_started reused=true`
  and `target_lane_reporting_forward_result_returned reused=true`, and reduced
  the final reporting sync span to `0.0101 s`. The cache still intentionally
  rejects failed/cap-bound trial solves and non-iteration fidelity overrides.
- Phase 5 trial-budget plumbing has landed in the child single-stage runner,
  the parent parity wrapper, and the K1 matrix launcher. The launcher now
  accepts `MATRIX_TRIAL_BOOZER_BFGS_TOL` and
  `MATRIX_TRIAL_BOOZER_BFGS_MAXITER`, records them in each case manifest, and
  forwards them as `--target-lane-trial-boozer-bfgs-*` to the child. RunPod
  H100 sweep
  `/workspace/runpod-k1-runs/phase5-iota011-R0935-bfgs-sweep-h100-20260702T075558Z`
  completed caps `200`, `300`, and `500` with exit `0` on the same
  `iota011_R0935` Phase 2 config (`maxiter=20`, full accepted/final BFGS
  budget `1500`, full Newton polish `run`, no explicit dense chunk override).
  All three stopped after one accepted iteration and reached indistinguishable
  physics, but cap `300` was fastest:

  | trial BFGS cap | final event elapsed | phase2 elapsed | K1 durations (s) | rejected-trial K1 | final J | final iota | final vol | Boozer residual |
  | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
  | 200 | 671.2 s | 298.3 s | 168.0, 25.8, 34.8 | 25.8 s (`pre_newton_iter=200`, `newton_iter=1`) | 7.16440496073957e-4 | 0.11017563996762356 | 0.04916423160514324 | 4.5732067736942573e-7 |
  | 300 | 532.2 s | 296.5 s | 166.8, 26.0, 34.6 | 26.0 s (`pre_newton_iter=300`, `newton_iter=1`) | 7.164389974566755e-4 | 0.11017554592247202 | 0.04916423166118373 | 4.573194822063472e-7 |
  | 500 | 612.9 s | 376.2 s | 165.8, 106.3, 34.8 | 106.3 s (`pre_newton_iter=500`, `newton_iter=9`) | 7.164368308119462e-4 | 0.11017540985669198 | 0.049164231742263885 | 4.5731775303772417e-7 |

  Follow-up cap-300 final-reuse smoke
  `/workspace/runpod-k1-runs/phase5-cap300-finalreuse-h100-20260702T084258Z`
  completed in about `5m38s` wall, with K1 durations `33.0 s`, `26.0 s`,
  `34.6 s`, phase2 elapsed `161.4 s`, final J
  `7.164389974566755e-4`, final iota `0.11017554592247202`, final volume
  `0.04916423166118373`, and Boozer residual `4.573194822063472e-7`.
  Final reporting reuse was `true` and the reporting forward-result span was
  `0.0101 s`, eliminating the earlier cap-300 `~65 s` final K1 recompute.

  Interpretation: cap `300` is the current best measured trial budget. It
  preserves the final objective/physics envelope of cap `500` while avoiding
  the expensive rejected-trial Newton work, and after the final-reuse policy
  patch it no longer pays a final-reporting K1 recompute when the accepted
  solve did not hit the trial cap. The trial-budget knob is validated as useful.
- Phase 6 same-resolution runtime seed-spec bypass landed at `651639189` and
  was validated remotely with a real CLI same-shape copy smoke
  (`payload_equal=true`, identical SHA-256 source/output).
- Phase 2 has a first accepted-result H100 A/B on the real `iota011_R0935`
  gallery seed, but it should be treated as a measured short-run result rather
  than a production quality pass. Both legs used the same seed, targets
  (`iota=0.11015671329581699`, `vol=0.04920000000000004`), resolution
  (`mpol=10`, `ntor=10`, `nphi=255`, `ntheta=64`), chunk batch `8`,
  preallocation enabled, `maxiter=20`, benchmark/minimal artifacts, and the
  cloned successful H100 launcher environment; only
  `--target-lane-trial-boozer-newton-polish-policy` differed.
  - `skip` artifact:
    `/workspace/runpod-k1-runs/phase2-iota011-R0935-skip-quality-h100-20260702T050023Z`
    (`results.json` in cell `mpol=10-ntor=10-d38a5a3d`) exited `0`,
    `OPTIMIZER_SUCCESS=True`, `iterations=1`, `OPTIMIZER_NFEV=19`,
    wall `16:21.59`, `INITIAL_OBJECTIVE=7.173839457696713e-4`,
    `FINAL_OBJECTIVE=7.171619766357395e-4`,
    `FINAL_IOTA=0.1102158156590774`,
    `FINAL_VOLUME=0.0491642076547637`,
    `FINAL_BOOZER_RESIDUAL=4.5783162720818724e-7`.
    It recorded 19 K1 returns, 13 K2 returns, and 6 baseline-gradient
    fallback returns. K1 durations were `24.8 s`, `8.1 s`, then mostly
    `4.9 s`; K2 was `228.3 s` for the first call and then about `15.8 s`.
    Phase 2 took `534.9 s`; final reporting/sync took `279.0 s`, including a
    `189.7 s` reporting forward-result step. This quality launcher did not
    emit a `reused=true` final-sync payload, so do not use it as proof of
    final-sync reuse.
  - `run` artifact:
    `/workspace/runpod-k1-runs/phase2-iota011-R0935-run-quality-h100-20260702T052815Z-clonedskipenv`
    (`results.json` in cell `mpol=10-ntor=10-22d2939f`) exited `0`,
    `OPTIMIZER_SUCCESS=True`, `iterations=1`, `OPTIMIZER_NFEV=3`,
    wall `15:28.43`, `INITIAL_OBJECTIVE=7.173839457696713e-4`,
    `FINAL_OBJECTIVE=7.164330117522752e-4`,
    `FINAL_IOTA=0.11017516973394964`,
    `FINAL_VOLUME=0.04916423188534984`,
    `FINAL_BOOZER_RESIDUAL=4.5731470149987164e-7`.
    It recorded 3 K1 returns, 2 K2 returns, and 1 baseline-gradient fallback.
    K1 durations were `168.1 s`, `403.5 s`, and `34.7 s`; K2 durations were
    `57.6 s` and `15.8 s`. Phase 2 took `679.8 s`; final reporting/sync took
    `79.8 s`, including a `68.5 s` reporting forward-result step.
  - Interpretation: the trial-polish cost mechanism is confirmed by the K1
    timings (`run` spends minutes in full K1 polish where `skip` returns in
    seconds), but the default-`skip` quality gate is not closed by this single
    short run. Both legs stopped after one accepted iteration on SciPy's loose
    relative-function-reduction criterion; `run` took far fewer evaluations and
    reached a slightly lower objective, while `skip` made many cheap rejected
    probes. Keep `skip` as the performance hypothesis, but require a
    longer/tighter quality gate or a trial-BFGS-budget sweep before declaring it
    production-quality superior.
- Two launcher/runtime traps were found during that A/B and are separate from
  the trial-policy comparison:
  - Strict JAX transfer guard (`JAX_TRANSFER_GUARD*`) fails during
    `SingleStageRuntimeSpecBiotSavartJAX` construction because
    `biotsavart_backend._spec_cache_key -> _array_cache_key` calls
    `np.asarray` on a device `jax.Array` (`shape=(9)`, `dtype=float64`,
    `device=cuda:0`). The completed quality runs used the existing
    `SIMSOPT_JAX_TRANSFER_GUARD=disallow` launcher variable, matching the
    successful skip leg, not the raw JAX guard.
  - An incomplete cloned launcher environment can fail Boozer init condition
    estimation with a CPU/GPU scalar mix in
    `_dense_matrix_condition_estimate` (`matrix_norm` on GPU times
    `inverse_norm` on CPU). The fair A/B run cloned the successful skip
    launcher environment exactly and changed only the trial policy.

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

1. **Phase 1 — GPU runtime proof of the committed fixes**
   - [x] Submit the K1 matrix at `0cf4230cb` (or current HEAD) with
         `MATRIX_CASES=baseline_dense_run_chunk8,dense_skip_chunk8`,
         `MATRIX_MAXITER=1`:
         `sbatch --export=ALL,REPO_ROOT=<checkout>,RUN_ROOT=<scratch>,MATRIX_CASES=... benchmarks/perlmutter/single_stage_k1_matrix_gpu.slurm`
         Perlmutter job `55380861` is submitted and pending from executable
         source `1d055547e`.
   - [x] Assert in progress events: exactly one
         `target_lane_decomposed_k1_forward_returned` per eval (no duplicate
         K1). The H100 trace/default-skip smoke emitted 17 K1 returns for the
         17 objective evaluations in `runpod_artifacts_2026-07-02/trace/`.
         Trial `skip` behavior was proven in the earlier Perlmutter
         solve-call-correction smoke (`55363238`) with
         `newton_polish_skipped=true` / `newton_iter=0`; the H100 trace run
         confirms the cache/reuse side of that path. Do **not** require
         `pre_newton_iter=0`: the `skip` policy bypasses Newton polish only;
         the pre-Newton L-BFGS stage still runs and must report
         `pre_newton_iter` / `pre_newton_nfev` / `pre_newton_ngev`.
   - [x] In trace-enabled runs, assert the wrapper emits
         `target_lane_optimizer_objective_eval_N_forward_result_reused`
         (not `..._started`) after K2 for the same candidate. Proven by the
         RunPod H100 trace artifact in `runpod_artifacts_2026-07-02/trace/`.
   - [x] Run one case **without** `--record-objective-evaluation-trace` to
         prove the production non-trace path populates the sync cache
         and final sync shows
         `target_lane_reporting_forward_result_started reused=true`, for a
         same-fidelity/no-active-trial-override run. Proven by
         `runpod_artifacts_2026-07-02/nontrace_trialrun_nobudget/`.
         Default trial-`skip` runs intentionally do **not** reuse the lower
         fidelity trial solve for final reporting.
   - [x] Record per-eval wall deltas against the 2026-06-29 A100 baseline
         (warm: 168s / 493s / 141s per eval). Treat the report's post-fix
         K1 33-63s figures as a partial signal only: that run proved final-sync
         reuse but still reported `newton_polish_skipped=false`, so the final
         trace/default-skip H100 smoke is the valid trial-skip + trace-reuse
         proof. Same-fidelity full-polish H100 eval 2 still cost about `203 s`
         (`event_elapsed_s` 166.77 -> 369.66) with 539 pre-Newton BFGS steps and
         27 Newton iterations, reinforcing why default trial `skip` remains the
         production performance policy.

2. **Phase 2 — Trial-skip trajectory quality gate**
   - [x] Same seed/targets (iota011_R0935: iota 0.11015671329581699,
         vol 0.04920000000000004), `maxiter` 20–50, A/B
         `--target-lane-trial-boozer-newton-polish-policy skip` vs `run`.
         Completed on RunPod H100 with `maxiter=20`:
         `phase2-iota011-R0935-skip-quality-h100-20260702T050023Z` vs
         `phase2-iota011-R0935-run-quality-h100-20260702T052815Z-clonedskipenv`.
   - [x] Compare: final J, FINAL_IOTA / FINAL_VOLUME / FINAL_BOOZER_RESIDUAL,
         accepted-iteration count, total line-search eval count,
         rejected-trial rate, wall time.
         Recorded in the execution-status section above.
   - [ ] Gate: `skip` reaches equal-or-better J at equal accepted iterations
         (tolerance: the two trajectories may diverge; judge by objective
         quality, not step-for-step parity), and the rejected-trial rate does
         not increase enough to erase the per-trial savings.
         First H100 A/B does **not** close this gate: both legs stopped after
         one accepted iteration, but `run` reached the lower final objective
         (`7.1643e-4` vs `7.1716e-4`) with fewer objective evaluations.
         The K1 timing mechanism is confirmed; the quality gate needs a
         longer/tighter run or the Phase 5 trial-BFGS-budget sweep.

3. **Phase 3 — Structural: cut HVPs per Newton iteration (measure first)**
   - [x] Extend the `1a9deabac`/`945a010b2` diagnostics to record the actual
         GMRES matvec count per Newton iteration (Eisenstat–Walker-adjusted),
         surfaced through the existing K1 subtimer events. Source and focused
         remote validation landed at `61d1cb99a`; RunPod H100 artifact
         `phase3-phase4-iota011-R0935-run-matvec-autochunk-h100-20260702T072931Z-final`
         closed the GPU measurement.
   - [x] Decision gate (write results into this file): if measured
         HVPs/iteration ≥ ~n (663), implement Option A; if well below,
         document and close this phase as not-a-win. The measured value was
         `1307` actual operator matvecs per Newton iteration at n=663, so
         Option A is the next implementation target.
   - [x] Option A: per-iteration dense materialization
         (`_materialize_dense_linear_operator`, `OPT:3669`) + on-device PLU
         solve inside the traceable Newton loop, reusing the factor-once
         dispatch machinery (`BZ:3527-3548`). Env-gated comparator first
         (mirror the `lsmr_j` precedent), default off. Implemented at
         `1d055547e` behind `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER=dense_lu`;
         default remains `operator_gmres`.
   - [x] Option B (fallback): keep GMRES, precondition with a stale PLU factor
         refreshed every k iterations. Only if Option A's per-iteration
         assembly cost regresses. Rejected for now: the default-off dense-LU
         comparator is correctness-clean but slower on the first H100 mpol10
         smoke, so no fallback/default implementation is justified yet.
   - [x] Equivalence gate: optimizer objective/gradient unchanged
         (bit-identical where the contract requires; otherwise the repo's
         ≤1e-12 equivalence harness) before any default flip. No default flip
         occurred; focused H100 regressions and a real H100 smoke show the
         comparator preserves final objective/physics for its current
         experimental scope.

4. **Phase 4 — Chunk-batch byte-budget auto-sizing**
   - [x] Derive the dense-operator chunk batch from the GPU byte budget
         (`SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU`, env name at
         `runtime.py:126`, default at `runtime.py:234`) instead of the
         hardcoded env default 8 (`OPT:3607`); keep
         `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE` as explicit override.
         Constraint: `lax.map` needs a static batch at import (`OPT:3601-3606`).
         Source and focused remote validation have landed; no-explicit-override
         GPU validation remains open below.
   - [x] Verify the auto value on the allocated H100 target with
         `XLA_PYTHON_CLIENT_PREALLOCATE=true`, transfer guard disallow, and no
         explicit chunk/budget env overrides. The H100 artifact above exited
         `0` and recorded auto `dense_operator_chunk_batch_size=8` with
         `max_dense_hessian_bytes=268435456`.
   - [ ] Verify the same auto value on a lower-memory target GPU, including the
         prior 40GB A100 diagnostic class when available. Historical handoff
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
   - [x] Run a measured sweep (e.g. 200/300/500) on the Phase 2 config.
         Completed on RunPod H100 at
         `/workspace/runpod-k1-runs/phase5-iota011-R0935-bfgs-sweep-h100-20260702T075558Z`.
         Cap `300` is fastest among the three clean exits with effectively
         unchanged final objective and physics.
   - [x] Guard: accepted/final/reference solves keep the full budget;
         policy recorded in progress metadata like `newton_polish_policy`.
         The sweep command set only `--target-lane-trial-boozer-bfgs-maxiter`
         per case; accepted/final remained at
         `--target-lane-boozer-bfgs-maxiter 1500`.

6. **Phase 6 (independent) — seed-spec projection bypass**
   - [x] Bypass same-resolution runtime-seed-spec projection when the
         serialized surface already matches the requested resolution (the
         staging hang noted in the report's Runtime boundary section). Not a
         perf-gap item; unblocks official warm-start staging for Phases 1–2.
         Landed at `651639189` and validated remotely with a real same-shape
         CLI copy smoke (`payload_equal=true`, identical SHA-256 output).

## Validation Plan

- [x] Targeted regressions (run remotely for this workflow; repo tests require
      the meta-path workaround from HANDOFF.md §4 — drop
      `ScikitBuildRedirectingFinder`, force `src/`):
      remote H100 validation passed
      `tests/integration/test_single_stage_newton_polish_policy.py -k "child_trial_override or trial_boozer_overrides_use_trial_policy_not_full_policy or trial_solve_cache or same_fidelity_trial_solve"`
      (`7 passed`) and
      `tests/integration/test_single_stage_jax_cpu_reference.py -k "adapter_final_sync_falls_back_to_decomposed_solve_cache or decomposed_host_objective_feeds_final_reporting_sync_cache"`
      (`2 passed`). The final-reuse policy patch later passed the focused H100
      slices
      `tests/integration/test_single_stage_newton_polish_policy.py -k "trial_solve_cache or same_fidelity or iteration_cap or non_iteration_override"`
      (`5 passed`) and the same final-sync adapter slice (`2 passed`). Earlier
      focused remote/Perlmutter validations covered the trace-wrapper and
      private optimizer slices listed here.
- [x] Phase 1 progress-event assertions (listed inline above) on actual RunPod
      H100 artifacts, not just exit codes. Perlmutter matrix artifacts remain a
      separate harness-validation item because the submitted jobs did not reach
      the intended K1 matrix run.
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
      `forward_result_reused` for the post-K2 same-candidate check, and
      final-sync `reused=true` on the same-fidelity non-trace path. The RunPod
      H100 artifacts prove these contract pieces except the same-A100 wall-time
      comparison; the same-A100 per-eval wall target remains open.
- [x] Cap-limited trial solve smoke shows final-sync `reused=true` when the
      accepted solve did not hit the iteration cap, while the cache still
      rejects failed/cap-bound trial solves and non-iteration fidelity
      overrides.
- [ ] Phase 2 quality gate passed and recorded (or `skip` default reverted
      with the evidence written into the report doc).
- [x] Phase 3 decision gate resolved with measured HVP counts in this file.
- [x] Phase 3 Option A implemented behind a comparator flag and smoke-tested on
      H100. The first real comparator smoke is correctness-clean but slower, so
      no default flip is justified.
- [ ] Phases 4–5 landed with regressions, or explicitly rejected with data.
- [ ] Report doc and handoff updated with runtime results; memory update filed
      only when that workflow is explicitly requested by the operator.

## Open Questions

- Phase 3 dense-LU comparator follow-up: the H100 smoke at mpol10 was
  correctness-clean but slower than operator-GMRES. Keep it default-off and
  only revisit for higher-resolution/repeated-solve cases where dense
  materialization can amortize its first-solve cost.
- Progress-artifact backend-code confirmation: focused H100 tests prove the new
  packer field, but the post-patch RunPod reporting smoke was interrupted when
  the pod disappeared. The pre-patch real artifact still distinguishes dense
  vs operator by matvec budget (`663` vs `1302`).
- Native cpp/CPU per-eval wall at the iota011_R0935 config on Perlmutter CPU
  nodes: no local record exists; needed for the definitive post-fix
  cpp-vs-GPU headline. (`single_stage_fair_compare_gpu.slurm` co-produces the
  reference; one fair-compare run answers it.)
- Benchmark-mode final-sync cache interaction is resolved for same-fidelity and
  non-binding iteration-cap trial solves. It remains intentionally disabled for
  true lower-fidelity or failed/cap-bound trial solves.
