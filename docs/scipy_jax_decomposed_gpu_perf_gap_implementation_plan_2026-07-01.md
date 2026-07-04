# scipy-jax-decomposed GPU Perf Gap Closure Plan

**Status:** In progress
**Last updated:** 2026-07-04

## Purpose

Execution plan for the follow-up work identified by the 2026-07-01 root-cause
deep-dive into why `lbfgs-scipy-jax-decomposed` on GPU ran slower than the
native cpp/CPU reference. Companion to
`docs/scipy_jax_decomposed_newton_polish_and_reporting_reuse_report_2026-07-01.md`
(the diagnosis/fix report). That report closed the workflow-redundancy layer in
code; this plan covers (1) the still-missing runtime proof of those fixes,
(2) the trial-policy quality gate, and (3) the remaining structural per-solve
gap. Historical file:line citations in the diagnosis section are anchored at
commit `0cf4230cb`; prefer the adjacent symbol/function names as durable anchors
when the actively edited files drift. The execution-status section reflects
later implementation commits and should be read as the current source-of-truth
for completion state.

Abbreviations: `EX` = `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`,
`OPT` = `src/simsopt_jax/geo/optimizers/optimizer.py`,
`BZ` = `src/simsopt_jax_adapters/geo/boozer_surface.py`.

## Goals

- Prove at GPU runtime that `0cf4230cb` (jax.Array memo-key normalization in the
  decomposed K1 memo/cache path: `last_solved_forward_result`,
  `cache_last_solved_payload`, and the `*_forward_result_reused` progress
  events) eliminates both the trace-lane duplicate K1 solve and the final-sync
  K1 re-solve on the production non-trace path.
- Resolve the trial-only Newton-polish policy: either prove `skip` does not
  degrade optimizer trajectory quality vs `run`, or keep lower-fidelity trial
  policies explicit. Current code can separate the first incumbent `x0` solve
  from later trial/probe solves; the remaining blocker is trajectory quality,
  not first-`x0` routing.
- Reduce the per-Newton-iteration HVP count in the traceable Newton polish
  (`newton_polish_traceable`), measured before designed.
- Replace the hardcoded dense-operator chunk batch default with an
  activation-aware byte-budget auto-sizer. The current implementation preserves
  the historical batch `8` default with the legacy `32 MiB` small-budget model,
  then uses a conservative `3072 MiB` activation footprint for larger budgets;
  live remat/chunk telemetry still has to validate or revise that safety factor.
- Evaluate whether the trial pre-Newton L-BFGS budget can be bounded safely.
  Do not default-enable a cap unless it is proven not to affect incumbent/full-
  fidelity objective evaluations.
- Un-clamp Eisenstat-Walker forcing so early Newton linear solves can stop at a
  genuinely loose tolerance while preserving the final tight tolerance.
- Reduce dense HVP assembly memory enough to A/B larger chunk batches
  (`8/16/32`) under normal preallocation and transfer-guard settings.
- Keep final reporting on the accepted solved state when a later rejected trial
  would otherwise evict the single-entry solve cache.
- Convert progress-event persistence from full-history rewrites to append-only
  or otherwise bounded-cost logging for long optimizer runs.
- Remove once-per-run duplicate traceable bundle/gradient construction where the
  current setup builds or executes unused gradient graphs, and avoid reporting
  recomputation when the same term values can be carried as value-and-grad aux
  data.
- Remove long-tail host/runtime overhead that becomes visible after the main
  K1/K2 reductions: repeated scalar progress transfers, duplicate solved-payload
  assembly, dense-factor telemetry dumps, and tiny-program cache churn.
- Extend the existing `_traceable_predict_warmstart_x` warm-start predictor
  (wrapped by `_warmstart_for` and exposed as `"warmstart_predict"` in
  `surface_objectives_traceable.py`) into a current-incumbent/factor-reused
  sensitivity predictor if it can reduce pre-Newton/Newton iterations without
  changing the converged solved state.
- Measure `lsmr_j` with a stabilization sweep and trajectory gate before any
  default decision; treat any projected speedup as unproven until the sweep
  collapses the iteration-count uncertainty.
- Treat mixed precision as a default-off Phase 8 experiment only after the
  condition-number reconciliation identifies which operator controls the error
  margin; require fp64 residual checks and trajectory parity before any policy
  change.
- Close the workflow only against a clean, instrumentation-free end-state target:
  same seed/resolution/hardware class, no K1 subtimer replay or objective trace,
  steady-state per-eval wall recorded separately from cold compile/setup, and no
  OOM under normal preallocation. Once the fair native cpp/CPU reference exists
  for the same config, the A100-tier closure criterion becomes numeric:
  steady-state GPU wall per objective evaluation must beat the recorded native
  cpp/CPU per-eval wall, with objective/physics within the accepted tolerances.

## Non-Goals

- Making `lsmr_j` the default adjoint solver (stays an experimental comparator
  behind `SIMSOPT_ADJOINT_LINEAR_SOLVER=lsmr_j`; current implementation is
  `_solve_regularized_normal_system_lsmr_j_with_status` and requires
  `newton_stab > 0`).
- Switching the outer optimizer to `lbfgs-ondevice` (monolithic-compile risk;
  decided against in the report).
- Any change to the native CPU/cpp reference path or to physics/objective math.
- Re-running the compile-cache operational work (persistent-cache dir handling
  is a known separate thread; see HANDOFF.md §9).
- Treating the arithmetic estimates in the performance audit as measured
  speedups. Timing claims require clean artifacts with
  `--trace-target-lane-k1-subtimers` and objective-evaluation trace disabled
  unless the claim is specifically about instrumentation overhead.
- Default-enabling `lsmr_j`, traceable dense-LU Newton, trial Newton `skip`, or
  trial BFGS caps without the trajectory/equivalence gates in this file.
- Making diagnostic trace/objective-trace lanes cheap. Production benchmark
  timing excludes those lanes; trace-mode affordability can be a separate
  tooling project.
- Cleaning up ALM/proximity host evaluators. Those are real inherited CPU paths,
  but they are outside the `scipy-jax-decomposed` target lane this plan closes.
- Parallel trial evaluation or outer-optimizer replacement. SciPy L-BFGS-B's
  line search remains serial here.
- Kernel-level Biot-Savart or surface-evaluation rewrites without a measured
  kernel bottleneck. The current evidence points at orchestration and
  linear-algebra routing, not the leaf kernels.

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
- Trial policy plumbing: the historical follow-up tried default trial `skip`,
  resolved it per K1 solve call via
  `wrap_target_lane_solved_pair_with_boozer_overrides`, and proved the
  mechanics. The later H100 quality A/B did **not** prove `skip` as a safe
  production default, so current HEAD keeps both full and trial Newton-polish
  defaults at `run`. A follow-up cap-300 A/B proved useful plumbing and exposed
  a first-`x0` routing bug; that bug is fixed in current code. Trial BFGS caps
  and `skip` still remain explicit experimental flags until accepted-trajectory
  quality gates prove them safe as production defaults.
- Structural cost: traceable Newton polish solves each correction with
  operator-GMRES over HVPs (restart 64 × maxiter 10 + 1 refinement pass;
  current symbols:
  `_solve_traceable_newton_operator_gmres_with_status`,
  `_traceable_forward_result_newton_polish_traceable`) — the current H100
  artifact records a budget of `1302` and actual telemetry of `1307` matvec
  callbacks per Newton iteration, each a full fwd+bwd BiotSavart pass over the
  255×64 half-period grid. The
  factor-once path materializes the dense n×n Hessian (n=663 at
  mpol10/ntor10 stellsym, run metadata) via `lax.map` chunks
  (`_materialize_dense_linear_operator`): batch 8 → 83 sequential chunks, batch
  4 (40GB A100) → 166.
  Native C++ Newton pays ONE analytic `sopp.boozer_residual_ds2` assembly +
  ONE LAPACK LU per iteration (`src/simsopt/geo/boozersurface.py:536-554`).
- Boozer init evidence: on-device L-BFGS ran 701 iterations, failed
  tol 1e-10, Newton converged in 6 iterations (m3 `boozer_init_progress.json`).
- A/B harness exists: `benchmarks/perlmutter/single_stage_k1_matrix_gpu.slurm`
  with cases `baseline_dense_run_chunk8`, `lbfgs_run_chunk8`,
  `dense_skip_chunk8`, `dense_run_lsmrj_chunk8`, `dense_run_chunk16`,
  selectable via `MATRIX_CASES`.

## Execution Status (2026-07-04 review-and-commit pass)

A verified delta review (Crucible at `d8d1a86d6` plus a same-day follow-up
re-verification at `3c7a9d787`) confirmed the checklist bookkeeping in this
file and converted the 2026-07-03 working-tree items into scoped commits.
Everything below the SHAs is local/source/test fact; the remote GPU gates in
the checklist remain open.

- The two Crucible majors are fixed and committed: `d3650596c` restored the
  trial Newton-polish `run` default in the three production launchers, and
  `139c05880` restored the dropped near-target iterative-refinement pass as an
  Eisenstat-Walker-gated step (details in the Phase 7 item below). `139c05880`
  also carries the hybrid comparator plumbing, the corrected
  `newton_polish_traceable` docstring, and the matvec-counter DRY cleanup.
- `3c7a9d787` committed the raw outer-term reporting aux and the
  explicit-anchor/current-incumbent warmstart predictor hooks in the traceable
  suite; `f83179c76` committed the remaining 2026-07-03 working-tree items
  (first-`x0` full-fidelity routing with the trial-budget scoping, full-vs-trial
  trace metadata, profiler current-incumbent instrumentation, and the
  reporting raw-term reuse in the example) with their focused regressions. The
  2026-07-03 section's "working tree" phrasing is superseded by these commits.
- The review found one further incomplete-fleet-revert instance beyond the
  launchers: `benchmarks/perlmutter/build_single_stage_matrix.py` still set
  `target_lane_trial_boozer_newton_polish_policy="skip"` in `BUDGETS`, exported
  non-empty per cell as `PROD_TRIAL_NEWTON_POLISH_POLICY` and therefore
  overriding the launcher `:-run` fallback on every headline matrix cell.
  Fixed at `986a144bc` with a per-cell manifest regression proven red against
  the old value. Grep target for future default reverts: launchers AND
  builders/submitters that export the same knob.
- Local validation for this pass: the plan's cited focused slices all
  reproduce exactly (90 passed total across the eight slices, including the
  full policy file at 40), the committed-HEAD state is self-contained (29
  focused tests pass in a clean worktree with no dirty files), and the matrix
  manifest (11) and fair-compare launcher contract (7) files pass after the
  builder fix.

## Execution Status (2026-07-03 local-only continuation)

The 2026-07-03 continuation stayed inside the operator's local-only boundary:
no RunPod, no Perlmutter, and no end-to-end runs. Local validation used the
repo bootstrap workaround from `HANDOFF.md` with the sibling JAX conda env,
dropping the `ScikitBuildRedirectingFinder` and forcing this checkout's `src/`.

- Current `HEAD` includes local implementations for the Phase 7 low-risk
  reductions that were already committed before this continuation:
  Eisenstat-Walker forcing uncap (`9bd9661b9`), accepted solved-state cache
  pinning (`a5a37997b`), append/compat progress logging (`6718968ef`), batched
  K1 progress scalar pulls (`c1965b1a8`), duplicate solved-payload elimination
  (`28bac01c0`), compact dense-factor telemetry (`3b48841ba`), persistent-cache
  threshold provenance (`4b7574d2f`), deferred setup-gradient construction
  (`3b24fa5f5` and `004684265`), deferred host baseline adjoint peels
  (`e690422a2`), fused explicit objective VJPs (`37970302a`), and HVP
  objective rematerialization with selectable checkpoint policy (`e830ed006`,
  `d8d1a86d6`). These remain local/source/test facts until clean remote
  artifacts close their timing and memory gates.
- The working tree now keeps the decomposed trial BFGS cap scoped to
  line-search/probe evaluations after the first SciPy `x0` candidate. The base
  solved pair is preserved for the exact first candidate, the trial-wrapped
  solved pair is used after that, final-sync cache policy still rejects
  failed/cap-bound/lower-fidelity trial solves, and trace metadata records full
  Boozer settings for the first full-fidelity solve before switching to trial
  metadata for later probes.
- The working tree also adds the default-off traceable Newton hybrid comparator
  `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER=hybrid_final_dense_lu`. It keeps
  operator-GMRES for loose early corrections and switches to dense-LU only when
  the same Eisenstat-Walker near-target predicate that applies the strict cap is
  true; if dense materialization is blocked, it falls back to operator-GMRES.
  The trace now records the actual per-iteration Newton linear-solve backend
  code so hybrid artifacts can separate loose-GMRES from final dense-LU solves.
- The working tree now carries raw single-stage outer-term scalar aux data in
  traceable forward results and passes those aux terms through final solved-state
  reporting when available. This removes the second raw outer-term evaluation on
  that path, with fallback recomputation preserved for older/missing payloads.
  It does not yet eliminate the separate final reporting full-surface field and
  distance evaluation; keep the remote timing/artifact gate open.
- The working tree also starts the current-incumbent predictor A/B path locally:
  `_traceable_predict_warmstart_x` now delegates to an explicit-anchor predictor,
  and the profile suite exposes `current_incumbent_warmstart_predict`. The
  callable can consume a supplied incumbent `(coil_dofs, x, factors)` when an
  eligibility flag is true and falls back to the baseline anchor otherwise. The
  target-lane profiler now times that callable, and memory analysis lowers it
  with the same solved payload and success-gated anchor; focused tests cover
  both successful and failed forward-result anchor flags. This is profile/A-B
  plumbing only; it does not flip the production predictor or prove
  iteration-count wins.
- Local validation passed:
  `python -m py_compile examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py tests/integration/test_single_stage_newton_polish_policy.py`;
  focused helper slice
  `tests/integration/test_single_stage_newton_polish_policy.py -k "first_x0 or decomposed_builder_rejects_full_state_lifting or trial_iteration_cap_solve_reuses_final_sync_cache_when_cap_does_not_bind or trial_iteration_cap_solve_rejects_final_sync_cache_when_cap_binds or trial_non_iteration_override_disables_final_sync_cache or same_fidelity_trial_solve_keeps_final_sync_cache"`
  (`7 passed, 33 deselected`);
  full policy helper file
  `tests/integration/test_single_stage_newton_polish_policy.py`
  (`40 passed`);
  and focused decomposed host-objective/reference helper slice
  `tests/integration/test_single_stage_jax_cpu_reference.py -k "(decomposed_host_objective or adapter_final_sync_falls_back_to_decomposed_solve_cache or decomposed_reporting_cache or decomposed_trace_reuse or trace_wrapper_uses_decomposed_k1_cache) and not runs_end_to_end"`
  (`14 passed, 194 deselected`). The hybrid comparator continuation added
  py-compile coverage for `OPT`, `BZ`,
  `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py`, and
  `tests/geo/test_boozersurface_jax_private.py`; the focused private optimizer
  slice
  `tests/geo/test_boozersurface_jax_private.py -k "traceable_dense_lu_comparator or traceable_newton_linear_solver_resolver_accepts_hybrid_alias or traceable_hybrid or materialized_policy_keeps_operator_step or reports_iteration_diagnostics or finite_descent_step or nonfinite_linear_step or backtracks_norm_increasing"`
  passed (`10 passed, 125 deselected`), and the focused single-stage
  progress/reporting slice
  `tests/integration/test_single_stage_jax_cpu_reference.py -k "traceable_forward_result_packs_newton_progress_fields or target_lane_final_reporting_reuses_forward_outer_raw_terms or decomposed_reporting_cache or decomposed_host_objective_feeds_final_reporting_sync_cache"`
  passed (`5 passed, 204 deselected`). The raw-term reporting boundary slice
  `tests/geo/test_surface_objectives_jax.py -k "traceable_runtime_public_boundaries_defers_reporting_metrics_until_used or traceable_reporting_metrics_from_solution_bundle_reuses_outer_raw_terms"`
  passed (`2 passed, 348 deselected`). The focused warmstart/predictor slice
  `tests/geo/test_surface_objectives_jax.py -k "warmstart or predictor"`
  passed (`9 passed, 341 deselected`). The final py-compile coverage also
  included `src/simsopt_jax_adapters/geo/surface_objectives.py`,
  `tests/geo/test_surface_objectives_jax.py`, and
  `tests/integration/test_single_stage_jax_cpu_reference.py`. The HVP remat
  correctness/config slice
  `tests/geo/test_boozersurface_jax_private.py -k "hessian_vector_product or hvp_objective_remat"`
  passed (`5 passed, 135 deselected`) after py-compile coverage for
  `src/simsopt_jax/geo/optimizers/optimizer.py` and
  `tests/geo/test_boozersurface_jax_private.py`. The profile-runner
  current-incumbent A/B instrumentation slice
  `tests/integration/test_single_stage_jax_cpu_reference.py -k "target_lane_profile_records_pre_newton_phase or target_lane_memory_analysis_profiles_current_incumbent_predictor"`
  passed (`2 passed, 208 deselected`) after py-compile coverage for
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  and `tests/integration/test_single_stage_jax_cpu_reference.py`; this slice
  now also asserts failed forward results pass `anchor_eligible=false` into the
  current-incumbent profiler. Earlier in this local
  continuation, the full local `tests/geo/test_boozersurface_jax_private.py`
  file passed (`130 passed, 5 skipped`; deprecation warnings only).
  `git diff --check` passed.

The open checklist items that require GPU timing, lower-memory validation,
trajectory quality, or production wall-time evidence remain open; local unit
tests are not evidence for those gates.

## Execution Status (2026-07-02)

Authoritative state as of the 2026-07-02 scoped implementation pass:

- Phase 1 now has direct H100 runtime proof for the two cache contracts, with
  one important fidelity boundary. RunPod H100 run
  `/workspace/runpod-k1-runs/direct-jax-m10-h100-edea9ba10-20260702T030856Z`
  (`runpod_artifacts_2026-07-02/trace/`) completed with exit `0`, wrote
  `final_artifact_write_returned`, and in trace mode emitted 17 K1 returns, 9
  K2 returns, and objective-evaluation forward-result reuse events after K2 for
  the same candidates. Trial-`skip` runs intentionally keep final-sync
  solved-state reuse disabled when trial fidelity differs from final/reporting
  fidelity; the trace run recorded active trial overrides
  (`bfgs_maxiter=1500`, `newton_maxiter=40`, `newton_polish_policy=skip`) and
  final reporting therefore recorded `target_lane_reporting_forward_result_*`
  with `reused=false`. The same non-trace/trial-skip boundary was reproduced
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
  default or explicit trial `run` with full `run`, or explicit trial `skip`
  with full `skip`, resolves to no active trial-policy override. Remote H100
  focused validation passed `7` policy/cache tests and `2` final-sync fallback
  tests after syncing the patched files to `/workspace/simopt-k1-current`.
- The K1 matrix launcher remains useful but is no longer the only Phase-1 proof
  vehicle. It has an
  opt-in progress gate (`MATRIX_ASSERT_K1_PROGRESS=1`) backed by
  `benchmarks/single_stage_k1_progress_assertions.py`. By default it checks K1
  return events, optional trial Newton skip semantics, and optional final-sync reuse; when
  `MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE=1`, it additionally checks
  exactly-one K1 return per `objective_evaluation` and optional trace-forward
  reuse. `MATRIX_TRACE_K1_SUBTIMERS=1` enables the expensive split-K1 replay
  diagnostics; timing comparisons should leave it unset. The launcher now runs
  K1 assertions over all reference/target progress files instead of the first
  file returned by `find`. It also emits `k1_matrix_report.json` via
  `python -m benchmarks.single_stage_k1_matrix_report`, a pure artifact reader that
  rolls up per-case exit status, assertion status, reference/target progress
  file summaries, trace-reuse counts, final-sync reuse, and missing-artifact
  issues before the matrix failure gate. Remote Perlmutter validation passed for
  py-compile, the focused synthetic pytest slices, and CLI synthetic
  progress/report smokes.
  Fresh Perlmutter GPU submission initially failed when the account was omitted.
  The valid Perlmutter shape is `sbatch -A m4680_g -C gpu -q shared ...`.
  Source-commit Phase-1 Perlmutter jobs `55373498` and `55373499` were not used
  as proof: `55373498` failed during editable-install submodule setup, and
  `55373499` failed the matrix command because the high-resolution donor
  continuation path lacked `MATRIX_WARM_START_RUN_DIR`.
- Phase-1 Perlmutter A100-40GB shared-interactive job `55381297` completed
  from executable source `e359bfd81d6f` at
  `/pscratch/sd/j/jungdae/simopt-jax-clean-local-k1matrix-e359bfd81d6f-20260702T100821Z-interactive/55381297`.
  It selected `MATRIX_CASES=baseline_dense_run_chunk8,dense_skip_chunk8`,
  `MATRIX_MAXITER=1`, `MATRIX_ASSERT_K1_PROGRESS=1`,
  `MATRIX_RECORD_OBJECTIVE_EVALUATION_TRACE=1`, and
  `MATRIX_REQUIRE_TRACE_REUSE=1`. Both selected cases wrote `summary.json`
  with no top-level runtime error, strict GPU provenance (`backend=gpu`,
  `devices=["cuda:0"]`, transfer guard `disallow`), and final metric parity
  (`final_iota_abs_diff=0.0`, `final_volume_rel_diff=2.8e-16`,
  `field_error_rel_diff=3.2e-15`). Both cases still recorded `passed=false`
  because neither the reference leg nor the target leg accepted an optimizer
  step (`optimizer_status=2` on both); treat this as K1/reuse harness evidence,
  not an accepted-trajectory quality pass. The target leg timing improved from
  baseline to dense-skip: total wall `617.6 s -> 300.8 s` and outer optimizer
  `277.6 s -> 132.5 s`. The reference leg remained about `38 min` because this
  run forced `--trace-target-lane-k1-subtimers`, whose replay dominated wall
  time.
- The same Perlmutter run exposed a report-aggregation bug: each matrix case
  now contains both `reference_outputs` and `target_outputs` progress files,
  but `benchmarks/single_stage_k1_matrix_report.py` assumed exactly one
  `outer_optimizer_progress.json`. Commit `0f6c94955` fixes the report reader
  to summarize all progress files while preserving `target_outputs` as the
  primary progress view. The patched reader successfully regenerated
  `k1_matrix_report.json` against the `55381297` artifacts.
- Follow-up Perlmutter job `55382657` is submitted from source
  `1beeec411151` at
  `/pscratch/sd/j/jungdae/simopt-jax-clean-local-1beeec411151-k1matrix-20260702T120245Z-src`
  with artifacts under
  `/pscratch/sd/j/jungdae/simopt-jax-clean-local-k1matrix-1beeec411151-20260702T120245Z-interactive`.
  It selects `MATRIX_CASES=baseline_dense_run_chunk8,dense_skip_chunk8`,
  `MATRIX_MAXITER=20`, the iota011_R0935 warm-start seed, m10/n10
  255x64 resolution, `MATRIX_REQUIRE_TRACE_REUSE=1`,
  `MATRIX_REQUIRE_FINAL_SYNC_REUSE=1`, and `MATRIX_TRACE_K1_SUBTIMERS=0`.
  It is pending at submission time, so this is execution status, not result
  proof.
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
- The first real H100 single-stage comparator smoke was correctness-clean but is
  not a final dense-vs-GMRES speed decision. Baseline
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
  first dense materialization cost dominated the one-smoke wall. Conclusion:
  keep dense-LU experimental/debug-only, but treat the "slower" result as
  pre-Eisenstat-Walker and inconclusive for steady-state policy. Re-test after
  the Eisenstat-Walker fix and include the hybrid candidate: loose operator-GMRES
  early in Newton, dense-LU for the final tight correction iterations.
- A follow-up short reporting smoke was launched after adding the explicit
  `newton_linear_solve_backend_code` packer field, but the RunPod endpoint
  disappeared mid-run (`ssh: connect ... Connection refused`) and provider-side
  `runpodctl pod list --all` returned no pods. The focused H100 test above
  proves the field packer and dispatch; a future real artifact should confirm
  the progress JSON carries backend code `1`/`2` without relying on the matvec
  budget as the discriminator.
- Phase 4 byte-budget chunk sizing now has a no-explicit-override H100 smoke in
  the same run: `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE` and
  `SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU` were unset, preallocation stayed on,
  transfer guard stayed disallow, and K1 progress recorded
  `dense_operator_chunk_batch_size=8` with
  `max_dense_hessian_bytes=268435456`. This proves the current default preserves
  the historically safe batch `8`; it does **not** validate the byte model. The
  auto-sizer now preserves the small/default-budget legacy behavior but uses a
  conservative activation-footprint constant (`3072 MiB` per parallel probe)
  when larger byte budgets try to scale beyond batch `8`. That prevents a
  "safe-looking" multi-GiB budget from jumping straight to the known-bad
  batch-32 shape, but Phase 4 remains open until the activation budget is tested
  on the target GPU class.
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

  Interpretation: cap `300` is the best measured value within this explicit
  cap-sweep artifact, and the trial-budget knob is validated as useful for
  controlled experiments. It is **not** validated as a production default. A
  later direct RunPod A100 same-fidelity smoke with
  `--target-lane-trial-boozer-bfgs-maxiter 300` showed why: the cap is applied
  inside the decomposed solved-pair wrapper before SciPy can distinguish the
  first incumbent `x0` evaluation from later line-search probes. That first K1
  evaluation spent `948.3 s` (`pre_newton_iter=300`, `newton_iter=50`) and
  still returned `primal_success=false`; the next rejected trial spent
  `304.8 s` (`pre_newton_iter=300`, `newton_iter=17`, stalled). This invalidates
  cap `300` as an implicit default even though explicit cap sweeps remain
  useful.
- The incumbent/probe separation bug from that A100 smoke is now fixed in the
  decomposed host value/grad wrapper. The first exact SciPy `x0` candidate is
  routed through the unwrapped full-fidelity solved pair and can populate the
  final-sync solved-state cache; later line-search/probe candidates use the
  trial-wrapped solved pair, and cap-bound trial solves still remain excluded
  from final-sync reuse. Target-lane bundle construction was also narrowed so
  temporary trial Boozer overrides are used only for the materialization probe,
  not for constructing the base solved pair. Remote RunPod A100 validation was
  run against a current tracked-file mirror on
  `/workspace/simopt-jax-clean-local-b31bbde4d96d-clean`: py-compile passed for
  the touched files,
  `tests/integration/test_single_stage_newton_polish_policy.py` reported
  `38 passed`, and
  `tests/integration/test_single_stage_jax_cpu_reference.py -k decomposed_host_objective`
  reported `9 passed, 195 deselected`. This closes the contract bug; cap `300`
  remains explicit-only until a longer accepted-trajectory quality gate says it
  is safe as a production default.
- Phase 6 same-resolution runtime seed-spec bypass landed at `651639189` and
  was validated remotely with a real CLI same-shape copy smoke
  (`payload_equal=true`, identical SHA-256 source/output).
- The follow-up whole-flow audit confirmed the remaining optimization work is
  concentrated in routing and orchestration, not the leaf Biot-Savart kernels:
  the K2 default still uses the dense/squared-system adjoint path; `lsmr_j`
  exists but is experimental and requires positive stabilization; K1 Newton
  still defaults to operator-GMRES over HVPs; the Eisenstat-Walker clamp is
  effectively pinned at the tight floor for `newton_tol=1e-11`; K1 subtimer
  tracing replays K1 pieces and is not a clean timing mode; final-sync reuse can
  still miss when a rejected trailing trial evicts the accepted solve; and
  progress JSON persistence rewrites the full event list on every event.
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
    seconds), but the trial-`skip` quality gate failed to close. Both legs
    stopped after one accepted iteration on SciPy's loose relative-function-
    reduction criterion; `run` took far fewer evaluations and reached a
    slightly lower objective, while `skip` made many cheap rejected probes.
    Therefore `skip` remains an explicit experimental flag, not the production
    default. The later cap-300 A100 direct smoke exposed a first-`x0` routing
    bug that is now fixed; BFGS caps still remain explicit-only until a longer
    accepted-trajectory quality gate proves the cap safe as a production
    default.
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

The redundancy fixes now have GPU proof for trace reuse and same-fidelity
non-trace final-sync reuse, but clean wall-time proof still needs a no-subtimer,
no-objective-trace benchmark. The trial-`skip` policy is a deliberate contract
change (line search sees L-BFGS-only merit values), so it needs a
trajectory-quality gate before production sign-off. That gate did not pass on
the first H100 A/B, so production defaults keep trial Newton polish at `run` and
trial BFGS caps explicit-only even though the first-`x0`/probe routing bug is
now fixed. The structural measurement already showed actual operator-GMRES
matvec counts near `2*n` at mpol10, while the later audit showed
Eisenstat-Walker forcing is effectively clamped to the tight floor at
`newton_tol=1e-11`. That splits the next work into two buckets: low-risk
reductions that preserve the accepted objective contract (Phase 7), and
behavior-changing linear-algebra candidates that must stay behind explicit
parity/trajectory gates (Phase 8).

## Assumptions

- The prior Perlmutter diagnostic GPU was A100-SXM4-40GB
  (`gpu_shared_interactive`). Use the available interactive GPU for validation,
  but compare wall time only against the same GPU class and record memory,
  chunk batch, and preallocation. On H100-class hardware the lane already beat
  cpp (1.14×, 2026-06-23 cross-GPU benchmark), so "GPU slower than cpp/cpu" is
  an A100-tier + workflow-multiplier statement, not universal.
- Condition estimates must be operator-qualified before they drive LSMR or
  mixed-precision decisions. Current artifacts include `ls_condition_estimate`
  near `6.1e6`, while code comments and prior structural analysis discuss
  residual-J and squared-system estimates separately (for example,
  `kappa(J)` vs `kappa(J^T J)`). Phase 8 must reconcile which operator,
  stabilization, and estimator each number describes.
- The dirty working tree stays under concurrent edits; every phase commits
  scoped (`git commit --only -- <paths>`).

## Implementation Plan

1. **Phase 1 — GPU runtime proof of the committed fixes**
   - [x] Submit the K1 matrix at `0cf4230cb` (or current HEAD) with
         `MATRIX_CASES=baseline_dense_run_chunk8,dense_skip_chunk8`,
         `MATRIX_MAXITER=1`:
         `sbatch --export=ALL,REPO_ROOT=<checkout>,RUN_ROOT=<scratch>,MATRIX_CASES=... benchmarks/perlmutter/single_stage_k1_matrix_gpu.slurm`
         Perlmutter shared-interactive job `55381297` completed on A100-40GB
         from executable source `e359bfd81d6f`. It produced per-case summaries
         and K1/reuse progress evidence, but not an accepted-step quality pass.
   - [x] Assert in progress events: exactly one
         `target_lane_decomposed_k1_forward_returned` per eval (no duplicate
         K1). The H100 trace/explicit-skip smoke emitted 17 K1 returns for the
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
         Explicit trial-`skip` runs intentionally do **not** reuse the lower
         fidelity trial solve for final reporting. The current production
         default is same-fidelity trial `run` with no implicit trial BFGS cap.
   - [x] Record per-eval wall deltas against the 2026-06-29 A100 baseline
         (warm: 168s / 493s / 141s per eval). Treat the report's post-fix
         K1 33-63s figures as a partial signal only: that run proved final-sync
         reuse but still reported `newton_polish_skipped=false`, so the
         explicit-skip H100 smoke is the valid trial-skip + trace-reuse proof.
         Same-fidelity full-polish H100 eval 2 still cost about `203 s`
         (`event_elapsed_s` 166.77 -> 369.66) with 539 pre-Newton BFGS steps and
         27 Newton iterations, motivating a future incumbent-aware trial budget
         design. The measured cap-300 path is explicit-only because the first
         accepted-trajectory gate has not proved it safe as a production
         default; the earlier first-`x0` routing bug is fixed in current code.

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
   - [x] First-attempt result recorded: `skip` did **not** reach
         equal-or-better J on the first H100 A/B. Both legs stopped after one
         accepted iteration, but `run` reached the lower final objective
         (`7.1643e-4` vs `7.1716e-4`) with fewer objective evaluations. Current
         HEAD therefore correctly keeps the trial Newton-polish default at
         `run`, and the cap-300 BFGS budget remains an explicit experiment.
   - [ ] Run a powered quality gate before hardening the policy verdict:
         require multiple accepted iterations by tightening `ftol` or by using
         an acceptance-count stop, then compare final objective/physics,
         accepted/rejected eval counts, invalid-state counts, and wall time.
         Also measure `J_trial(x) - J_full(x)` on the same candidate points so
         the trial-fidelity tradeoff is quantified instead of treated as binary.
         If `skip` is too low-fidelity but full `run` is too expensive, test the
         explicit middle policy: pre-Newton L-BFGS plus one or two
         Eisenstat-Walker-loose Newton iterations for trial probes, while
         incumbent/accepted/final solves stay full-fidelity.

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
         (`_materialize_dense_linear_operator`) + on-device PLU
         solve inside the traceable Newton loop, reusing the factor-once
         dispatch machinery. Env-gated comparator first
         (mirror the `lsmr_j` precedent), default off. Implemented at
         `1d055547e` behind `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER=dense_lu`;
         default remains `operator_gmres`.
   - [ ] Option B / hybrid follow-up: keep GMRES for loose early Newton
         corrections and use dense-LU only for tight final iterations, or
         precondition GMRES with a refreshed PLU factor. Do not decide this from
         the pre-Eisenstat-Walker dense-LU smoke: that run was correctness-clean
         but conflated first dense materialization cost with steady-state solve
         policy. Default-off hybrid comparator plumbing is committed at
         `139c05880`
         behind `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER=hybrid_final_dense_lu`,
         with focused local regressions for near-target dense-LU selection and
         dense-materialization fallback. Keep this item open until a clean
         remote timing/trajectory gate reruns after the Eisenstat-Walker fix.
   - [x] Equivalence gate: optimizer objective/gradient unchanged
         (bit-identical where the contract requires; otherwise the repo's
         ≤1e-12 equivalence harness) before any default flip. No default flip
         occurred; focused H100 regressions and a real H100 smoke show the
         comparator preserves final objective/physics for its current
         experimental scope.

4. **Phase 4 — Chunk-batch byte-budget auto-sizing**
   - [x] Preserve the historical default behavior when no explicit override is
         supplied. The current implementation derives
         `dense_operator_chunk_batch_size=8` from the default
         `SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU=256 MiB` and keeps
         `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE` as an explicit override.
         Constraint: `_materialize_dense_linear_operator` uses `lax.map`, so the
         chunk batch remains static inside compiled kernels.
   - [x] Verify the default-preservation smoke on the allocated H100 target with
         `XLA_PYTHON_CLIENT_PREALLOCATE=true`, transfer guard disallow, and no
         explicit chunk/budget env overrides. The H100 artifact above exited
         `0` and recorded auto `dense_operator_chunk_batch_size=8` with
         `max_dense_hessian_bytes=268435456`.
   - [ ] Recalibrate the auto-sizing model so the byte budget is an activation
         budget, not just a matrix-size budget. Local implementation now keeps
         the default `256 MiB -> batch 8` behavior but switches above the default
         budget to `_DENSE_OPERATOR_ACTIVATION_BYTES_PER_PARALLEL_COLUMN =
         3072 MiB`, a 2x safety factor over the historical batch-32 OOM scale
         and the stale `32 MiB` output-column model. Keep this item open until
         live remat/chunk telemetry validates or revises that constant.
   - [ ] Verify the recalibrated auto value on a lower-memory target GPU,
         including the prior 40GB A100 diagnostic class when available.
         Historical handoff evidence records batch=32 OOM around a 49.35 GiB
         effective allocation at mpol10/nphi255 (HANDOFF.md §5); the auto-sizer
         must be validated on the actual allocated GPU rather than assuming a
         nominal-memory class is safe.

5. **Phase 5 — Bounded trial pre-Newton budget**
   - [x] Add a trial-context `bfgs_maxiter`/`bfgs_tol` override alongside the
         trial polish policy (same plumbing:
         `build_target_lane_trial_boozer_overrides`) and expose it
         through the parent parity wrapper plus the K1 matrix launcher.
         Evidence for sizing: init used 701/1500 iterations and still failed
         1e-10.
   - [x] Run a measured sweep (e.g. 200/300/500) on the Phase 2 config.
         Completed on RunPod H100 at
         `/workspace/runpod-k1-runs/phase5-iota011-R0935-bfgs-sweep-h100-20260702T075558Z`.
         Cap `300` is fastest among the three clean exits with effectively
         unchanged final objective and physics.
   - [x] Guard: accepted/final/reference solves keep the full budget, and the
         first SciPy incumbent `x0` objective evaluation is not treated as a
         low-budget trial. The explicit cap sweep kept final/reference solves
         on the full budget, and the decomposed value/grad wrapper now routes
         the first exact `x0` candidate through the full-fidelity solved pair
         before switching later line-search/probe candidates to the trial
         solved pair. Remote A100 contract validation passed the full
         `test_single_stage_newton_polish_policy.py` file (`38 passed`) plus
         the decomposed host-objective regression slice (`9 passed`).

6. **Phase 6 (independent) — seed-spec projection bypass**
   - [x] Bypass same-resolution runtime-seed-spec projection when the
         serialized surface already matches the requested resolution (the
         staging hang noted in the report's Runtime boundary section). Not a
         perf-gap item; unblocks official warm-start staging for Phases 1–2.
         Landed at `651639189` and validated remotely with a real same-shape
         CLI copy smoke (`payload_equal=true`, identical SHA-256 output).

7. **Phase 7 — Low-risk runtime reductions**
   These are intended to reduce sequential device work or host overhead without
   changing the accepted objective contract. Each item still needs a remote
   artifact because local JAX timing is not representative.

   - [ ] Un-clamp Eisenstat-Walker forcing in
         `_eisenstat_walker_choice2_tolerance` (`OPT`) so the tight cap binds
         near convergence instead of forcing `1e-12` for every iteration at
         `newton_tol=1e-11`. Add a focused test that demonstrates an early
         far-from-converged residual gets a loose tolerance above the floor and
         a near-converged residual still gets the tight cap. Remote gate: K1
         matvec counts drop on early Newton iterations while final Boozer
         residual, final iota, final volume, and objective remain within the
         existing tolerances. Implementation and focused regression are in place;
         keep this open until the remote unit slice and K1 artifact gate pass.
         2026-07-04 Crucible correction: the original uncap commit
         (`9bd9661b9`) had also silently dropped the single iterative-refinement
         pass from the default operator-GMRES correction solve, which was the
         sole mechanism reaching the tight near-target tolerance (measured
         rel-residual 3.1e-14 -> 3.4e-10 at kappa=625, n=663; the removed
         tolerance floor was proven non-load-bearing). Restored at
         `139c05880` as a near-target-gated refinement: an unconverged
         single-pass correction in
         the Eisenstat-Walker strict-cap region now receives one bounded
         refinement pass (`_refine_traceable_newton_operator_gmres_solution`),
         so early loose solves keep the uncap's matvec win while the final
         tight tolerance is actually achieved. Regressions:
         `test_traceable_newton_gmres_refinement_recovers_tight_tolerance`,
         `test_newton_polish_traceable_refines_unconverged_gmres_near_target`.
   - [ ] Pin accepted solved states in the decomposed solve cache. Replace or
         extend the single-entry `last_solved_forward_result` cache with a small
         exact-DOF-keyed cache that preserves the last accepted/final solve even
         if a trailing rejected line-search probe is evaluated later. Preserve
         the current fidelity policy: failed, cap-bound, or lower-fidelity trial
         solves must not be reused for final reporting. Regression: a rejected
         trailing trial does not evict the accepted solve and final sync records
         `reused=true` for same-fidelity accepted DOFs. Implementation and
         focused regression are in place; keep this open until the remote unit
         slice and K1 artifact gate pass.
   - [ ] Convert progress-event persistence to append-only NDJSON or an
         equivalent bounded-cost format. Keep the current summary JSON available
         for existing artifact readers, but stop re-sanitizing and rewriting the
         full event list on every event. Regression: a synthetic long event
         stream scales O(N) in writes and the matrix-report/progress readers can
         consume the new artifact layout. Implementation and focused regression
         are in place; keep this open until the remote unit slice and progress
         artifact-reader gate pass.
   - [ ] Batch scalar progress materialization in
         `_summarize_k1_forward_result_for_progress` and related reporting
         helpers. The current path performs many small `host_array` /
         `jax.device_get` transfers per evaluation. Gate: one batched host
         materialization or a reduced transfer count, identical progress fields,
         and no transfer-guard violations. Implementation and focused regression
         are in place for the K1 forward-result progress summarizer; keep this
         open until remote transfer-count and artifact-reader gates pass.
   - [ ] Avoid duplicate `cache_last_solved_payload` work inside one objective
         evaluation. The current decomposed helper can build and store the
         solved payload before K2 and then rebuild it on the success/rejection
         branch. Gate: one authoritative solved-payload assembly per candidate
         while preserving final-sync reuse and primal-failure fallback behavior.
         Implementation and focused regression are in place; keep this open
         until the remote unit slice and K1 artifact gate pass.
   - [ ] Compact dense-factor telemetry in progress/reporting JSON. When
         dense-LU or exact-factor comparators are active, record shape, dtype,
         norm, condition/status, and small scalar diagnostics instead of dumping
         full `P`/`L`/`U` arrays with `.tolist()`. Gate: progress artifacts stay
         useful and bounded-size, and matrix readers do not require full factors.
         Implementation and focused regression are in place; keep this open
         until the remote unit slice and dense-factor artifact gate pass.
   - [ ] A/B the persistent-cache minimum compile threshold instead of forcing
         `jax_persistent_cache_min_compile_time_secs=0.0` for all programs.
         Runtime plumbing is available through
         `SIMSOPT_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`; the default
         remains `0.0` to preserve existing small-kernel cache behavior, and
         the measurement gate remains open. Runtime provenance now records both
         the resolved backend threshold and JAX config threshold so the A/B
         artifact can be interpreted after the fact. Gate: record cache
         hit/miss counts, cache directory growth, cold setup wall, and warm
         setup wall; keep the setting that improves warm reuse without
         persisting tiny eager programs that cost more to deserialize than to
         compile.
   - [ ] Defer or eliminate duplicate traceable gradient graph construction in
         setup. The current runtime can build both the primary compiled bundle
         and an optimizer-only bundle that each construct `_forward_result_for`
         and `_total_gradient_for`; `general_only_forward=True` does not by
         itself suppress gradient construction. Gate: fewer cold compile poles
         without changing value/gradient contracts. Local implementation note:
         optimizer-only `general_only_forward=True` bundles now preserve the
         same callable keys while deferring the total-gradient/value-and-grad JIT
         construction until those callables are used. The decomposed solved-pair
         also builds `value_grad_from_solved` from the optimizer compiled bundle,
         reusing that bundle's `compiled_total_gradient_for` instead of
         constructing a separate solved-state adjoint kernel. Focused
         regressions pin both routes; keep this item open until clean remote
         artifacts show the expected compile-pole reduction.
   - [ ] Remove the eager seeded baseline adjoint when it is not needed for the
         active optimization path, or make it share the same compiled graph used
         by the fused value-and-grad path. Current setup executes
         `compiled_total_gradient_for` at the baseline state before the optimizer
         needs a candidate gradient. Contract note: the public seeded optimizer
         helper still returns a concrete `(value, grad)` tuple and therefore
         remains eager; host-wrapper baseline peels now defer their baseline
         adjoint until the exact-baseline `host_value_and_grad` path is actually
         called, and memoize the host result while returning a fresh gradient
         copy. Gate: seeded setup wall drops where the public seed is not needed,
         and rejected primal-failure fallback still returns the validated
         baseline gradient.
   - [ ] Carry reporting term values as aux data where possible instead of
         recomputing all outer terms and an extra full-surface Biot-Savart in the
         reporting graph. Gate: final reporting uses the accepted solved state,
         preserves all reported fields, and avoids an additional surface-field
         evaluation when the value-and-grad path already computed the same data.
         Implementation note (`3c7a9d787`, example-side reuse `f83179c76`):
         packed traceable forward results now carry
         raw outer-term scalar aux data and final solved-state reporting consumes
         that aux data when available, falling back to recomputation for older or
         missing payloads. This removes the second raw outer-term evaluation on
         the accepted solved-state reporting path, but it does not yet remove the
         separate full-surface field/distance reporting evaluation; keep this
         item open until remote timing/artifact gates prove impact or more aux
         data covers those fields.
   - [ ] Investigate merging the separate `dJ_dx` and `direct_grad` reverse
         sweeps into a joint derivative where dependency flags allow it. Gate:
         HLO/op-count or timing evidence shows one backward pass replaces two,
         and term-level gradients remain within the existing parity tolerance.
         Local implementation note: the fully dependent objective path now uses
         one explicit two-primal `jax.vjp` seed to compute `dJ_dx` and
         `direct_grad` together, while the x-only and coil-only dependency
         branches keep their previous one-sided strict-VJP behavior. Keep this
         item open until the remote parity/timing gate confirms the compile and
         memory tradeoff is positive.
   - [x] Preserve the value-returning empty-candidate `cdist` fallback in
         `CurveCurveDistance.shortest_distance` and
         `CurveSurfaceDistance.shortest_distance`. Do **not** replace the empty
         fallback with `minimum_distance`: the exact clearance margin feeds
         hardware/status margins. Current code already hoists sampled point
         clouds before the brute-force fallback, applies `downsample` on the
         curve-surface path, and `tests/geo/test_curve_objectives.py` covers
         empty- and non-empty-candidate brute-force equality. No target-lane
         code change is required for this item.
   - [ ] Add objective-level rematerialization for HVP/dense-assembly paths
         where the residual/geometry tape dominates live memory. The existing
         leaf Biot-Savart kernel is already checkpointed; this task is about the
         higher-level `jvp(grad(fn))` HVP closure. Gate with a correctness test
         and remote memory telemetry; do not keep the change if it increases
         compile/runtime more than the larger-batch win it unlocks. Treat
         `jax.checkpoint` policies (dot/residual-saving variants) as a tuning
         axis alongside on/off; the leaf Biot-Savart kernel comment already
         anticipates policy selection. Local implementation note: the scalar
         objective feeding `_hessian_vector_product_fn` can now be wrapped with
         `jax.checkpoint` via `SIMSOPT_HVP_OBJECTIVE_REMAT=1`, with
         `SIMSOPT_HVP_OBJECTIVE_REMAT_POLICY` selecting the default or
         dot-saveable policy, default-off until the clean remote memory/timing
         gate proves the remat tradeoff positive. Focused local tests now prove
         the default-off selector leaves the objective callable untouched,
         default, `dots`, and dot-saveable remat policies preserve HVP values
         with extra objective arguments, and unknown policies fail loudly.
   - [ ] A/B dense operator chunk batches `8`, `16`, and `32` with
         `XLA_PYTHON_CLIENT_PREALLOCATE=true`, transfer guard disallow, and no
         K1 subtimer replay. Batch `32` only becomes a candidate if the remat
         path keeps memory within the allocated GPU budget. Record K2 wall,
         device memory high-water, and whether XLA emits any OOM or remat
         regressions.
   - [ ] A/B the XLA command-buffer / CUDA-graph flag family on a clean
         no-subtimer run. The steady-state cost profile is long sequences of
         small sequential device steps (dense assembly in 83 chunks at batch 8,
         GMRES budgets up to 1302 matvecs, on-device L-BFGS iterations), so
         per-launch overhead is a first-order term. The intended numerical
         program is unchanged, but bitwise equality is not assumed; dynamic
         control flow may simply prevent capture. Gate: record per-eval wall
         with the flags on vs off, compare objective/gradient outputs under the
         existing parity tolerances, and close the item with a recorded null
         result if capture does not engage or parity fails.
   - [ ] Revisit accepted-path L-BFGS handoff only after the Eisenstat-Walker
         fix. Design a progress-based handoff/cap that leaves first-incumbent
         and accepted/final solves full-fidelity unless a separate trajectory
         gate proves the cap safe as a production default.
   - [ ] Evaluate a grid-sequenced (coarse-to-fine) accepted-path warmup after
         the Eisenstat-Walker fix: run the pre-Newton L-BFGS stage on a reduced
         quadrature grid and hand off to the full-resolution Newton polish, so
         the converged solved state stays defined by the full-fidelity
         `newton_tol` solve. Same gate class as the handoff/predictor items:
         identical converged physics within tolerance, fewer full-resolution
         iterations overall, and the extra compiled resolution variant accounted
         against the cold-compile budget.
   - [ ] Extend the existing `_traceable_predict_warmstart_x` predictor (the
         baseline-anchored `_warmstart_for` / `"warmstart_predict"` closure) into
         a current-incumbent/factor-reused sensitivity predictor. The missing
         opportunity is not "add a predictor from scratch"; it is to reuse the
         current solved Jacobian/factor and batched RHS sensitivities so each
         trial starts closer to the implicit solved state. Gate: fewer
         pre-Newton L-BFGS/Newton iterations, identical converged solved-state
         physics within tolerance, and no reuse of failed or lower-fidelity
         factors. Implementation note (`3c7a9d787`, profiler instrumentation
         `f83179c76`): the baseline predictor has been
         factored through an explicit-anchor helper, and the profile suite now
         exposes `current_incumbent_warmstart_predict` for A/B probes. It selects
         a caller-supplied incumbent anchor only when the supplied eligibility
         flag is true; otherwise it falls back to the baseline anchor. The
         target-lane profile runner now records the current-incumbent predictor
         timing and memory-analysis shape when the suite exposes it, using the
         profiled solved payload and a success-gated anchor flag; focused tests
         cover both accepted and failed forward-result flags. Keep this item open
         until the live A/B records iteration-count and solved-physics evidence.
   - [ ] Re-run dense-LU vs operator-GMRES after the Eisenstat-Walker fix, and
         include the hybrid candidate: loose GMRES for early iterations, dense-LU
         for final tight correction solves. Keep all variants default-off until
         the clean timing and trajectory gates pass.
   - [ ] Opt-in multi-device sharding of dense-operator probe assembly on
         multi-GPU nodes (for example 4x A100 Perlmutter allocations). The dense
         columns are independent probes gathered without any cross-device
         reduction, so sharding the probe batch is an algebraically equivalent
         candidate rather than a changed solver; the factor solve stays on one
         device. Gate: prove identical assembled operator bytes, record
         assembly-wall scaling, and keep single-device behavior unchanged by
         default.

8. **Phase 8 — Behavior-changing linear-algebra experiments**
   These can plausibly produce the largest speed and memory wins, but they
   change the adjoint/Newton linear-algebra route. They stay env-gated until
   parity and trajectory gates pass.

   - [ ] Run an `lsmr_j` stabilization sweep on the iota011_R0935 config with
         the same resolution and chunk settings as the dense baseline. Record
         stabilization value, LSMR iterations/matvecs, K2 wall, peak memory,
         gradient norm, gradient parity against dense, final objective, final
         iota/volume, Boozer residual, accepted/rejected eval counts, and wall
         time. This is the measurement that decides whether `lsmr_j` is a speed
         path or only a memory path.
         Pre-register the interpretation before running: median iterations
         `<=150` is a speed-and-memory candidate if dense-gradient parity passes;
         `151..700` is a memory-path candidate only unless wall time also wins;
         `>700` rejects `lsmr_j` as a speed path for this configuration.
   - [ ] Define the `stab=0` contract before extending `lsmr_j` to the fully
         unstabilized case. The current code requires positive stabilization;
         unstabilized support needs a KKT or two-solve formulation with explicit
         success/failure semantics, not a silent fallback.
   - [ ] If the dense adjoint route remains a speed or parity candidate, compare
         the current `jnp.linalg.lstsq` dense solve with LU/QR plus the existing
         fp64 iterative-refinement residual check. Gate: same dense-gradient
         parity, same linear-solve success semantics, lower wall or compile cost,
         and no loss of robustness on ill-conditioned candidates.
   - [ ] Evaluate the rectangular residual-Jacobian direct route as the third
         sibling beside `lsmr_j` (iterative, unsquared) and dense factorization
         of the squared operator: materialize `J` with first-order JVP probes
         (cheaper per probe than HVP columns, conditioned at `kappa(J)` rather
         than its square) and solve the regularized least-squares system with
         QR. Same parity, trajectory, memory, and default-off gates as the other
         Phase 8 candidates.
   - [ ] Add a mixed-precision dense-factor experiment only after the
         condition-number reconciliation is complete. Candidate: fp32/TF32
         factorization with fp64 residual evaluation and iterative refinement.
         Gate: operator condition number within the registered margin, fp64
         residual below the dense baseline tolerance, dense-gradient parity,
         accepted-trajectory quality, and opt-in only on hardware where the
         memory/wall benefit is real.
   - [ ] Add a trajectory gate for any behavior-changing solver: compare dense
         default vs candidate over at least one short accepted-result run and one
         longer run. Required fields: final J, final iota, final volume, final
         Boozer residual, accepted iterations, total objective evals, rejected
         trial rate, invalid-state counts, and wall time.
   - [ ] Keep `SIMSOPT_ADJOINT_LINEAR_SOLVER=lsmr_j` and
         `SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER=dense_lu` default-off unless
         the candidate is equal or better on both physics/optimizer quality and
         wall/memory for the target production resolution.

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
      The first-`x0` separation patch was validated on RunPod A100 after
      overlaying current tracked source onto the remote tree: py-compile passed
      for the touched files,
      `tests/integration/test_single_stage_newton_polish_policy.py` reported
      `38 passed`, and
      `tests/integration/test_single_stage_jax_cpu_reference.py -k decomposed_host_objective`
      reported `9 passed, 195 deselected`.
- [x] Phase 1 progress-event assertions (listed inline above) on actual RunPod
      H100 artifacts, not just exit codes. Perlmutter job `55381297` also
      exercised the matrix harness on A100-40GB and wrote reference/target
      progress files for both selected cases; the cases themselves remain
      failed optimizer probes because no step was accepted.
- [ ] `/usr/bin/time` per-case wall comparison across matrix cases; GPU memory
      high-water from `nvidia-smi_before/after` snapshots the harness writes.
      Job `55381297` produced a partial comparison, but it forced
      `--trace-target-lane-k1-subtimers`; rerun with
      `MATRIX_TRACE_K1_SUBTIMERS=0` for fair wall timing. Job `55382657` is
      submitted for this rerun but has not produced runtime artifacts yet.
- [ ] Any Phase 3–5 change that can touch the objective path passes the
      equivalence gate before default-on; env-gated until then.
- [ ] Phase 7 Eisenstat-Walker tests prove the forcing tolerance is loose away
      from convergence and tight near convergence; remote K1 artifacts show
      lower early Newton matvec counts without worse final physics.
- [ ] Phase 7 accepted-cache regression proves final sync can reuse an accepted
      solve after a later rejected trial, and still refuses failed/cap-bound or
      lower-fidelity trial solves.
- [ ] Phase 7 progress-log regression proves bounded/O(N) writes and verifies
      current artifact readers or compatibility shims.
- [ ] Phase 7 host-progress regression records fewer device-to-host transfers
      for the same progress fields, and `cache_last_solved_payload` regression
      proves one solved-payload assembly per candidate without weakening final
      sync.
- [ ] Phase 7 telemetry regression proves dense-factor progress artifacts are
      compact and bounded-size under dense-LU/exact-factor comparators.
- [ ] Phase 7 persistent-cache threshold A/B records cache directory growth plus
      cold/warm setup wall before changing the default threshold.
- [ ] Phase 7 remat/chunk A/B runs without K1 subtimer replay and records K2
      wall plus GPU memory high-water for batches `8/16/32`.
- [ ] Phase 7 predictor A/B records pre-Newton L-BFGS iterations, Newton
      iterations, K1 wall, final residual/iota/volume, and factor-reuse metadata
      for baseline predictor vs current-incumbent/factor-reused predictor.
- [ ] Condition-number reconciliation artifact: for the same candidate and
      stabilization settings, record which operator each reported condition
      estimate describes (`J`, `J^T J`, Hessian-with-second-derivative terms,
      or regularized augmented `[J; sqrt(stab)I]`) before interpreting `lsmr_j`
      or mixed-precision margins.
- [ ] Phase 8 `lsmr_j` sweep produces a table of stabilization, iteration counts,
      K2 wall, peak memory, and dense-gradient parity before any default
      decision.
- [ ] Phase 8 dense-solve and mixed-precision experiments report operator
      condition number, fp64 residual after refinement, gradient parity, accepted
      trajectory quality, peak memory, and wall time before any default decision.
- [ ] Any wall-time claim excludes instrumentation replay:
      `MATRIX_TRACE_K1_SUBTIMERS=0`, `--trace-target-lane-k1-subtimers` absent,
      and `--record-objective-evaluation-trace` absent unless the benchmark is
      explicitly measuring trace overhead.

## Risks and Mitigations

- Risk: trial-`skip` changes line-search merit values enough to alter
  trajectories and mask a quality regression on longer runs.
  Mitigation: Phase 2 A/B gate on final objective/physics, not wall time alone;
  keep `run` reachable via flag for reference runs.
- Risk: trial BFGS caps are applied too broadly by the decomposed solved-pair
  wrapper and degrade incumbent/full-fidelity evaluations, not just risky
  line-search probes.
  Mitigation: current code routes first `x0`/accepted/final/reference solves
  through full-fidelity settings; leave caps explicit-only until an
  accepted-trajectory gate proves the capped trial path safe as a production
  default.
- Risk: Phase 3 Option A pays n HVPs of assembly on iterations where
  Eisenstat–Walker GMRES would have converged in far fewer matvecs.
  Mitigation: the measure-first decision gate; comparator stays env-gated.
- Risk: auto-sized chunk batch OOMs under XLA preallocation on lower-memory
  cards.
  Mitigation: treat the current batch `8` smoke as default preservation only;
  recalibrate the activation footprint before raising the budget-derived batch
  above `8`, and keep the explicit env override for emergency down-sizing.
- Risk: concurrent Codex/agent edits in the dirty tree clobber plan work
  (has happened before in this repo's dirty worktree).
  Mitigation: scoped commits per phase (`git commit --only -- <paths>`),
  verify guards present before committing.
- Risk: single-entry K1 memo is insufficient if a future optimizer evaluates
  value and gradient at different points in one step.
  Mitigation: L-BFGS-B calls `value_and_grad` at one x per eval today; add a
  regression asserting memo-hit on the final-sync DOFs so any change breaks
  loudly.
- Risk: loosening Eisenstat-Walker tolerances accepts poor Newton directions and
  increases backtracking or invalid candidates.
  Mitigation: keep the final tolerance tight, rely on existing line search, and
  gate on final Boozer residual plus accepted-trajectory metrics.
- Risk: HVP rematerialization trades memory for more compute/compile and can be
  a net slowdown at batch `8`.
  Mitigation: keep it behind a measured A/B; retain only if it unlocks a larger
  safe batch or materially lowers peak memory on constrained GPUs.
- Risk: a multi-entry solve cache reuses stale or lower-fidelity data.
  Mitigation: exact DOF key, explicit fidelity metadata, and existing rejection
  rules for failed/cap-bound/lower-fidelity trials.
- Risk: NDJSON progress output breaks downstream tooling that expects the legacy
  single JSON file.
  Mitigation: keep a compatibility summary JSON and update matrix-report readers
  in the same phase.
- Risk: current-incumbent sensitivity prediction reuses a stale or failed
  linearization and seeds a trial from the wrong implicit state.
  Mitigation: key the factor on exact incumbent DOFs and fidelity metadata,
  reject failed/lower-fidelity factors, and gate on converged solved-state
  physics rather than iteration count alone.
- Risk: `lsmr_j` speedup depends on spectrum and stabilization; worst-case
  iteration counts can exceed dense assembly.
  Mitigation: measure the stabilization sweep first and leave the solver
  experimental unless both parity and trajectory gates pass.
- Risk: mixed precision hides a linear-solve error that only appears in the
  final objective trajectory.
  Mitigation: require fp64 residual/iterative-refinement checks, condition-number
  margins, dense-gradient parity, and accepted-trajectory gates; keep the path
  default-off until all pass.

## Completion Criteria

- [x] GPU smoke shows: 1 K1 solve per eval, production trial
      `newton_polish_policy=run` with no implicit trial BFGS cap,
      trace-wrapper `forward_result_reused` for the post-K2 same-candidate
      check, and final-sync `reused=true` on the same-fidelity non-trace path.
      Historical explicit-`skip` smokes separately prove `newton_iter=0` when
      that experimental policy is requested. Historical explicit cap smokes
      prove the cap plumbing, and the first-`x0` routing regression is fixed;
      cap-300 remains non-default because the longer accepted-trajectory quality
      gate is still open.
- [x] Cap-limited trial solve smoke shows final-sync `reused=true` when the
      accepted solve did not hit the iteration cap, while the cache still
      rejects failed/cap-bound trial solves and non-iteration fidelity
      overrides.
- [ ] Phase 2 quality gate is powered by multiple accepted iterations or a
      tightened stopping rule. The first H100 A/B is recorded and correctly keeps
      trial Newton `skip` and trial BFGS caps explicit-only, but it is not a
      production trajectory verdict because both legs stopped after one accepted
      iteration.
- [x] Phase 3 decision gate resolved with measured HVP counts in this file.
- [x] Phase 3 Option A implemented behind a comparator flag and smoke-tested on
      H100. The first real comparator smoke is correctness-clean; it does not
      justify a default flip, and it also does not close the dense-LU decision
      because it predates the Eisenstat-Walker fix and conflates first
      materialization with steady-state behavior.
- [ ] Phases 4–5 landed with regressions, or explicitly rejected with data.
      Phase 5 plumbing and the first-`x0` full-fidelity guard are landed with
      remote regressions; trial caps remain explicit-only. Phase 4 still needs
      activation-budget recalibration and lower-memory target GPU validation.
- [ ] Phase 7 low-risk reductions either landed with focused regressions and
      remote clean-timing artifacts, or were explicitly rejected with artifact
      data. At minimum this covers Eisenstat-Walker forcing, accepted-solve cache
      pinning, progress logging, batched progress transfers, duplicate
      solved-payload assembly, compact dense-factor telemetry, persistent-cache
      threshold A/B, duplicate setup-gradient construction, seeded
      baseline-adjoint setup, reporting recomputation, joint-gradient
      investigation, value-preserving distance fallback contract,
      current-incumbent predictor A/B, and remat/chunk A/B.
- [ ] Clean no-subtimer benchmark artifacts exist for the accepted production
      path and separate cold compile/setup from steady-state per-eval timing.
- [ ] Phase 8 behavior-changing solver work remains clearly marked
      experimental unless `lsmr_j`/dense-LU/dense-solve/mixed-precision
      candidates pass dense-gradient parity, accepted-trajectory quality,
      wall-time, and memory gates.
- [ ] Report doc and handoff updated with runtime results; memory update filed
      only when that workflow is explicitly requested by the operator.
- [ ] End-state target defined and met for the A100-tier production question:
      clean no-subtimer/no-trace steady-state per-eval wall, cold compile/setup
      separated, no OOM with normal preallocation, and final objective/physics
      within the accepted tolerances on the iota011_R0935 mpol10/nphi255 config
      or the documented successor production config. After the fair cpp/CPU leg
      exists for the same config, replace this structural target with the numeric
      threshold: GPU steady-state wall per objective evaluation must be lower
      than the recorded native cpp/CPU per-eval wall.

## Open Questions

- Phase 3 dense-LU comparator follow-up: the H100 smoke at mpol10 was
  correctness-clean but inconclusive for policy. Keep it default-off, then
  re-evaluate after Eisenstat-Walker un-clamping and include the hybrid
  GMRES-early / dense-LU-final candidate.
- Progress-artifact backend-code confirmation: focused H100 tests prove the new
  packer field, but the post-patch RunPod reporting smoke was interrupted when
  the pod disappeared. The pre-patch real artifact still distinguishes dense
  vs operator by matvec budget (`663` vs `1302`).
- Native cpp/CPU per-eval wall at the iota011_R0935 config on Perlmutter CPU
  nodes: no local record exists; needed for the definitive post-fix
  cpp-vs-GPU headline. (`single_stage_fair_compare_gpu.slurm` co-produces the
  reference; one fair-compare run answers it.) When that artifact lands, promote
  its native cpp/CPU per-eval wall into the numeric A100-tier end-state target
  above.
- Benchmark-mode final-sync cache interaction is resolved for same-fidelity and
  non-binding iteration-cap trial solves. It remains intentionally disabled for
  true lower-fidelity or failed/cap-bound trial solves.
- What is the actual `lsmr_j` iteration count across stabilization values on the
  production seed, and does it behave as a speed path, memory path, or neither?
- Which condition estimate should govern LSMR and mixed-precision decisions:
  residual-J, squared normal operator, Hessian-with-second-derivative terms, or
  stabilized augmented operator?
- Does objective-level remat make batch `32` memory-safe on both 80GB H100 and
  40GB A100 with preallocation enabled, or is batch `16` the practical ceiling?
- After Eisenstat-Walker is fixed, how early can accepted-path L-BFGS hand off
  to Newton without degrading final objective/physics?
- Does a current-incumbent/factor-reused predictor reduce pre-Newton L-BFGS and
  Newton iterations beyond the existing baseline predictor without changing the
  converged solved state?
- Which persistent-cache minimum compile threshold avoids tiny-program cache
  churn while preserving the useful warm-cache wins?
- If the dense adjoint route survives the post-Eisenstat-Walker comparison, can
  LU/QR plus fp64 iterative refinement replace dense `lstsq` without weakening
  solve-status semantics?
- Does mixed precision pass the reconciled condition-number and fp64 residual
  gates strongly enough to justify a hardware-specific opt-in?
- How much wall time remains in host progress/reporting once progress logging is
  append-only and final reporting reuses the accepted solve?
