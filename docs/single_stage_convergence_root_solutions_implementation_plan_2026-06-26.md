# Single-Stage Convergence: Root-Cause → Root-Solution Implementation Plan

> Created: 2026-06-26 · Source: GPD orchestrated root-cause analysis (3 `gpd-debugger`/`gpd-verifier`
> agents + runtime evidence, conflicts reconciled). Repo HEAD at analysis: `4704d5671`.
> Review-fix update: live source was rechecked at `2c71021b1`; source-backed corrections below
> supersede stale analysis details where the two differ.
> External-docs check: official JAX docs confirm that `jax_enable_x64` controls 64-bit defaults and
> availability (`jax-ml/jax` `docs/default_dtypes.md`), and that benchmark/runtime claims must account
> for device transfer, JIT compilation, and `block_until_ready()` timing (`docs/benchmarking.md`).
> Optimizer-docs check: official SciPy L-BFGS-B docs define `ftol` as relative objective decrease,
> `gtol` as projected-gradient tolerance, and `maxls=20` as the default line-search step cap; Optax
> docs expose L-BFGS as a JAX gradient transformation with line searches such as
> `scale_by_zoom_linesearch`; COIN-OR Ipopt docs describe Ipopt as an interior-point filter-line-search
> NLP solver. These checks support keeping the current SciPy-driver diagnosis separate from optimizer
> library-swap proposals.
> Execution update: local partial execution with dirty-tree preservation. Earlier focused results
> were recorded at `458d06a3a`; later Phase 0 and Phase 5 support gates are current dirty-tree
> evidence unless a separate hash is named. Phase 6
> host-materialization fix is implemented locally; the Phase 5 callback-cache checker is implemented
> locally; Phase 4.1's default high-mpol `ftol` floor is implemented locally; the Phase 1
> production-FD gate/harness is implemented locally, including canonical JAX runtime-seed support
> as the preferred production path and explicit raw Stage-2 replay as a diagnostic path. A production
> mpol10 RunPod artifact now exists, but it is a failed/diagnostic FD ladder rather than a
> certificate: the data supports a coarse-FD/mis-instrumented-gate hypothesis while direction 2
> remains undecided and not certified. Phase 3's hard hardware target-lane filter is
> locally converted to an explicit residual contract plus smooth objective penalties, while
> self-intersection remains a hard filter. Local raw-seed mpol-8 execution fails before
> FD at Boozer initialization; local runtime-seed mpol-10 execution reaches the first FD evaluation but
> was stopped after high local CPU/RSS use. GPU convergence, production line-search/A-B evidence,
> and GPU compile breadth runs remain open. Phase 0 artifact evaluators are implemented locally for
> line-search-trace classification and fixed-candidate δJ comparison, but no production Phase 0
> line-search/noise-calibration artifact has been produced. The latest local mpol10 CPU runtime-seed
> attempts reuse the resolved startup state, materialize traceable hessian/operator linearization before
> target-bundle construction, and forward explicit `--outer-maxls 20`, but still time out before an outer
> optimizer `objective_evaluation` event. The current marker artifact
> `.artifacts/single_stage_convergence/phase0/runtime_seed_reuse_sync_marker_timeout_90_20260627/phase0_runtime_seed_reuse_linearized_newton_1e-11_maxls20_timeout_90_probe.json`
> reports `phase0_line_search_trace.classification: "missing_objective_evaluations"` with
> `outer_optimizer_progress.json.current_event: "target_lane_initial_objective_finite_check_started"`.
> The active local blocker is production-size CPU materialization of the initial target value/gradient
> for the host finite check, not startup Boozer replay. Phase 1/5 harness support gates were re-run
> locally on 2026-06-27:
> `tests/test_adjoint_fd_validation_contract.py tests/test_compile_breadth_probe_contract.py
> tests/test_check_cached_kernel_callback_compatibility.py -q` -> `43 passed in 2.54s`.
> The same repo interpreter exposes only the CPU JAX backend (`Unknown backend cuda. Available backends
> are ['cpu']`), so GPU convergence and GPU compile-breadth artifacts require an external CUDA runtime.
> Phase 4.2's
> rank-deficiency probe/fix is implemented locally with
> synthetic float64 near-singular coverage, but not yet exercised inside a production single-stage run.
> Phase 5 atomic compile-breadth checkpointing is implemented locally so late GPU failures preserve
> completed mpol records plus the active resolution in the output JSON. A failed H100 checkpoint now
> exists at
> `.artifacts/single_stage_convergence/phase5_compile/run_20260627T191118Z_fresh/compile_breadth_cuda_mpol6_8_10.json`,
> but it exited with no completed mpol records after mpol6 Boozer initialization failed and the JSON
> remains a stale `status: "running"` checkpoint; it is not a compile-breadth measurement. The
> `scipy-jax-decomposed` support lane is also hardened locally: the split K1/K2
> host objective mirrors fused failure handling, the zero-adjoint branch is limited to exactly-zero
> adjoint RHS values, malformed decomposed solved pairs must carry `primal_success`, and the
> decomposed route forwards the target-lane nonfinite-trial policy. Focused CPU
> support/parity/e2e gates pass, and the single-file CPU-reference integration parity run now
> passes locally (`tests/integration/test_single_stage_jax_cpu_reference.py -q` -> `171 passed,
> 15 skipped, 4 warnings in 1187.13s`). The path-based Stage-2 suite also passes locally:
> `tests/integration/test_stage2_jax.py tests/integration/test_stage2_target_lane_purity.py -q`
> -> `201 passed, 4 warnings in 265.27s`. This is not a production GPU convergence result or a
> full kernel parity-suite claim.
> Status: PARTIAL LOCAL EXECUTION (not production complete).

## Purpose

Convert the verified root-cause → root-solution synthesis for the **single-stage banana
optimization's non-convergence** into an executable, reviewable plan. The triggering symptom was a
production run that ran but did not converge (SciPy L-BFGS-B `ABNORMAL_TERMINATION_IN_LNSRCH`). The
analysis established the failure is a **finite line-search stall** (not a NaN, not a core-JAX bug)
concentrated in the single-stage *outer-optimization driver*, plus a separate GPU *cold-compile*
blocker. This file is the SSOT for fixing those.

## Goals

- [ ] A real **converged** single-stage result on GPU: J monotonically decreasing, SciPy
      `status == 0`, ≥1 accepted L-BFGS step, projected `||grad||` decreasing toward `gtol`,
      iota → target.
- [ ] The production single-stage gradient (operator/lstsq adjoint) **certified at production
      scale** (currently the mpol10 runtime seed; lower mpol is support/triage only), with any
      FD/Richardson evidence recorded from success-clean plus/minus solves.
- [ ] Full-space lanes (`scipy-jax-decomposed` / `-fullgraph`) converge from feasible-but-marginal
      and infeasible-at-seed configurations (no `ABNORMAL` from the constraint cliff).
- [ ] Robustness: outer `ftol` matched to the objective noise floor; no rank-deficiency NaNs.
- [x] The `newton_polish` host-materialization contract re-audited; stale contradiction wording retired
      if the current focused tests already coexist.

## Non-Goals

- Re-porting or "fixing" the **core JAX kernels** (`simsoptpp` C++ + `simsopt_jax*` BiotSavart/Boozer
  /adjoint) — verified machine-precision-faithful and broadly used (30 files); out of scope.
- Switching optimizer **library** (Optax / on-device / IPOPT) — the issue is not SciPy; the same
  SciPy L-BFGS-B converges on the reduced lane. A library swap would change the symptom, not the cause.
  Current repo docs/source keep `ondevice` as a SciPy-compatible L-BFGS-B state machine, keep
  `optax-lbfgs-ondevice` as a separate Optax gradient-transformation lane rather than a SciPy
  L-BFGS-B parity oracle, and Ipopt would require a constrained-NLP formulation/Jacobian path.
- Global multi-modality / local-minima search — separate concern, handled by continuation + multistart.
- fp32-on-GPU single-stage adjoint — physically unobtainable (κ·eps_f32), tracked elsewhere.

## Current Context (confirmed facts)

- **Empirical decisive datum** (rejected production run diagnostics, `prod_ss_fixval` rung-mpol10):
  `OPTIMIZER_FUN_FINITE=true`, `OPTIMIZER_JAC_FINITE=true`, `OPTIMIZER_INVALID_STATE=false`,
  `status=2` (`ABNORMAL_TERMINATION_IN_LNSRCH`), `nfev=21`. → L-BFGS-B evaluated 21 **finite**
  (value, grad) points and could not find an acceptable step. **Refutes the NaN hypothesis**; it is a
  genuine finite line-search failure.
- **Scope is single-stage-driver, not core JAX**: `success_filter` / `_traceable_rejected_objective_value`
  / `baseline_coil_gradient` appear only in the single-stage example + its adapter rejection path; the
  **Stage-2 JAX lane does not use any of it** (grep-confirmed). Core kernels are sound and broadly used.
- **The legacy reduced lane already converged on CPU** (`scipy-jax`, 153 iters → iota ≈ target;
  the `rc=1` was a dimension-mismatch parity fail, not a convergence fail —
  `HANDOFF-ss-11-51-matrix.md`). Plain single-stage `scipy-jax` is now deprecated; keep that
  result as historical evidence and use `scipy-jax-decomposed` for new production single-stage runs.
- **Prior high-mpol GPU parity evidence is external to this checkout.** Earlier run notes reported
  finite m18/m36 `dJ` and C++-matching forward iota to 16 digits, but no
  `HANDOFF-mpol-homotopy-ladder-push.md` file is present in this repo. Treat that as background
  evidence, not a current in-repo acceptance artifact.
- **The c=64 dimension floor binds on the production operator/lstsq adjoint** and clears κ≈3.9e5 with
  ~10³–10⁴× margin (empirical κ-oracle, Agent C). NaN risk from conditioning alone is low.
- The full-space `scipy-jax-fullgraph` lane reaches the **same** `ABNORMAL`/status 2 (it is not a
  decomposed-only issue; both are full-space SciPy L-BFGS-B).
- **Phase-1 production FD status is diagnostic, not certified.** The RunPod artifact
  `.artifacts/single_stage_convergence/phase1_fd/run_20260627T181848Z/adjoint_fd_cuda_runtime_seed_mpol10_ntor10_nphi255_ntheta64.json`
  has a finite traceable gradient (`gradient_norm≈1781`) and decreasing FD errors, but direction 2
  does not show clean central-difference order-2 behavior. Richardson extrapolation from the two
  finest points gives dir0 ≈1.0% relative error, dir1 ≈3.8%, and dir2 ≈13.8%; dir2 barely improves
  over its raw finest-eps error. This supports "FD gate likely mis-calibrated" and does **not**
  certify the production adjoint.
- **The FD harness now records the decisive solve-status channel, but production certification is
  still open.** `_pack_traceable_forward_result()` returns `value`, `x`, `sdofs`, `iota`, `G`,
  `success`, `primal_success`, and `adjoint_linear_solve_available`, and
  `compute_traceable_single_stage_fd_ladder()` now records plus/minus `forward_result(...)`
  diagnostics. The remaining Phase-1 blocker is the unresolved production mpol10 direction-2
  verdict, not missing FD solve-status telemetry.

> File:line anchors below are from the analysis at HEAD `4704d5671`; the example/adapter files are
> large and line numbers drift — **re-confirm with grep at implementation time** (anchors given as
> symbol + approximate line).

## Rationale

- **Smooth barrier is the true root fix for the `ABNORMAL` (1a).** The rejected-step plateau has
  *exactly zero* slope, so no `ftol` change can rescue it — only restoring a real gradient can. This
  also moves back toward original simsopt's smooth-penalty design (which the banana driver deviated
  from by layering a hard gate on top of an already-smooth objective).
- **Certify before trusting (3).** The production operator/lstsq adjoint has not yet been certified
  crossing the inner solve; "the optimizer ran and J dropped" is not proof the gradient is correct,
  and a monotone FD error decrease is not enough unless the accepted eps window shows clean order or
  Richardson agreement.
- **Reduced lane = fastest real converged GPU result.** It already converges on CPU and its GPU
  per-eval kernel is correct; the only blocker is a single slow cold compile, which is avoidable at
  reduced mpol.

## Assumptions (must hold; explicitly marked)

- **[A1 — UNVERIFIED for this seed]** The production-seed `ABNORMAL` is the same finite-stall
  mechanism (1a cliff and/or 1b noise) as the documented infeasible-seed cases. The 21 finite evals
  support this, but the per-trial line-search value/gradient trace for *this* seed has not been
  captured. Closing this is Phase 0.
- **[A2]** Objective J-value noise δJ ≈ 1e-10 worst-case (from homotopy floor data); per-seed value
  needs calibration before fixing `ftol` precisely.
- **[A3]** The smooth-barrier change can be made in the adapter rejection path + example filter
  without altering the core kernel numerics (the kernels are not touched).
- **[A4]** Runtime execution of production-scale JAX checks is RunPod/CUDA-only per user constraint.
  The eps ladder itself must be config/noise adaptive, not hardware-keyed.
- **[A5 — TO VERIFY]** A forward-mode JVP vs reverse-mode VJP check may be possible for the traceable
  runtime objective, but the public scalar objective is implemented with `jax.custom_vjp`. If forward
  mode through that callable is unsupported, use the dot-product test plus FD/Richardson diagnostics
  instead of treating JVP/VJP as a required gate.

## Implementation Plan

### Phase 0 — Confirm the mechanism for the production seed (closes A1)
1. Capture the per-trial line-search trace on the failing seed.
   - [x] Add an artifact evaluator for the required outer line-search trace. Local result:
         `benchmarks/single_stage_outer_loop_probe.py::evaluate_phase0_line_search_trace()` reads
         `outer_optimizer_progress.json`, classifies finite plateau vs finite descent vs nonfinite
         objective/gradient events, and preserves optimizer termination evidence (`message`,
         `status`, `ls_status`, `nfev`, and optional low-level `task`) from progress result events.
         It requires explicit `ABNORMAL_TERMINATION_IN_LNSRCH` message/task text for line-search
         proof; `status=2` alone is retained as metadata but does not count as proof. When
         `single_stage_outer_loop_probe.py --record-objective-evaluation-trace` is used, the probe
         payload now includes `phase0_line_search_trace`.
   - [ ] Re-run the rejected single-stage config with `--record-objective-evaluation-trace`
         (and without `--compact-objective-evaluation-trace` if replay-grade vectors are needed) →
         inspect `${OUT_DIR_ITER}/outer_optimizer_progress.json` objective-evaluation events for the
         21 outer trial `(x, f, g)` tuples, and preserve the SciPy `message`/low-level `task` text that
         identifies `ABNORMAL_TERMINATION_IN_LNSRCH`; `status=2` alone is not the line-search proof.
         `record_scipy_callback_trace` / `result.scipy_callback_trace` is Boozer/adapter metadata, not
         the outer single-stage line-search trace.
         Local runtime result: the outer-loop probe now exposes
         `--single-stage-case-timeout-seconds`, `--target-lane-boozer-newton-tol`,
         `--target-lane-boozer-newton-maxiter`, `--outer-maxls`, and
         `--reuse-jax-runtime-seed-solve`
         so Phase 0 production attempts can fail into structured JSON with partial artifact paths
         instead of relying on an external shell timeout. The latest maxls-20 local attempts used
         mpol10/ntor10/nphi255/ntheta64, the canonical runtime seed,
         `--record-objective-evaluation-trace`, `--target-lane-boozer-newton-tol 1e-11`, and
         explicit `--outer-maxls 20`. The 600s run
         `.artifacts/single_stage_convergence/phase0/runtime_seed_reuse_linearized_timeout_600_20260627/phase0_runtime_seed_reuse_linearized_newton_1e-11_maxls20_timeout_600_probe.json`
         failed closed with `status: "case-execution-failed"` and
         `phase0_line_search_trace.classification: "missing_objective_evaluations"`, still with no
         `objective_evaluation` events. After adding fail-closed sync-boundary progress markers, the
         90s marker run
         `.artifacts/single_stage_convergence/phase0/runtime_seed_reuse_sync_marker_timeout_90_20260627/phase0_runtime_seed_reuse_linearized_newton_1e-11_maxls20_timeout_90_probe.json`
         failed closed with `event_count: 38`, `objective_evaluation_count: 0`, and
         `outer_optimizer_progress.json.current_event:
         "target_lane_initial_objective_finite_check_started"`. The partial trace proves the child
         converts the reused resolved seed from `linearization_kind: "value_only"` to `"hessian"` and
         returns the lazy target value/grad, then stalls while materializing the initial value/grad for
         the host finite check. This is still not the requested line-search trace.
   - [ ] Classify: all-finite + tiny-decrease/flat-plateau → 1a/1b confirmed (expected); any NaN →
         re-open the adjoint-NaN path (3). (Pre-registered prediction: finite + plateau.)
2. Calibrate δJ for the seed (closes A2).
   - [x] Add an artifact evaluator for fixed-candidate δJ comparison. Local result:
         `evaluate_phase0_noise_calibration_pair()` compares two replay-grade
         `outer_optimizer_progress.json` traces, requires the selected candidate DOFs to match exactly,
         verifies recorded `boozer_solver_metadata.newton_tol` for both runs, and reports
         absolute/relative objective deltas for the `newton_tol=1e-11` vs `1e-13` pair.
   - [x] Wire the probe-level fixed-candidate replay/calibration route. Local result:
         `benchmarks/single_stage_outer_loop_probe.py` now exposes
         `--replay-objective-evaluation-trace`, `--enable-phase0-noise-calibration-gate`,
         `--phase0-noise-baseline-progress-json`, expected baseline/tightened Newton tolerances, and
         the selected objective-evaluation index. The gate forwards the baseline trace into the
         single-stage example's existing replay path, records the tightened replay-grade
         `outer_optimizer_progress.json`, compares it with `evaluate_phase0_noise_calibration_pair()`,
         records target-lane replay `boozer_solver_metadata.newton_tol`, and reports replay success as
         `measurement_passed` / `status: "replay-measurement-passed"` while keeping top-level `passed`
         reserved for real convergence proof.
   - [ ] Two inner solves at `newton_tol=1e-11` vs `1e-13`, hold coils fixed, diff J → measured δJ.
         Local support attempt:
         `.artifacts/single_stage_convergence/phase0/low_mpol_replay_rehearsal_20260627/baseline_newton_1e-11_probe.json`
         failed closed before startup because the JAX target lane requires an immutable runtime seed
         spec and this checkout only has the production-shape
         `benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json`; no
         low-mpol objective-evaluation trace was produced. This is a command-path/runtime-seed
         limitation, not production δJ evidence.

### Phase 1 — Certify the production gradient without conflating FD-window failure with adjoint failure
1. Record the corrected current status.
   - [x] Gate/harness added locally in `benchmarks/adjoint_fd_validation.py`: the probe now defaults
         to the Phase-1 production floor `mpol>=8`, rejects lower-mpol certification runs, and emits a
         `traceable_single_stage_fd` section for the fused traceable single-stage value/grad.
   - [x] Initial hard-window implementation used **[6e-4, 5e-3]** with default ladder
         `(3e-3, 1.5e-3, 7.5e-4)` and rejected single-`--eps` certification. That implementation
         is useful harness scaffolding but is now known to be too rigid for the mpol10 production
         artifact; the lower eps bound must be revised before Phase 1 can pass honestly.
         `benchmarks/tier5_performance_characterization.py` forwards the same ladder and defaults
         its Tier-4 shape to the checked runtime seed (`mpol=10`, `ntor=10`, `nphi=255`,
         `ntheta=64`).
   - [x] Confirm the gate uses the operator-backed traceable forward path (`linear_solve_factors=None`,
         `surface_objectives_traceable.py` ~607-615 → `optimizer.py` `_solve_dense_square_operator_least_squares_system_with_status` ~4792), NOT the eager dense-PLU sibling the toy tests use.
         Local result: the gate asserts `runtime_bundle["forward_result"]` metadata reports
         `linear_solve_backend=="operator"`, `linear_solve_factors is None`, and
         `dense_linear_solve_factors_available is False`; the fused `value_and_grad` path is then
         directionally FD-checked by the production run.
   - [x] Add canonical JAX runtime-seed support for production-scale certification. Local result:
         `--jax-runtime-seed-spec` accepts the checked
         `benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json`, installs
         the resolved seed state, derives coil layout (`num_tf_coils`, `banana_curve_index`) from
         that runtime spec, promotes the value-only state to
         traceable hessian/operator metadata with
         `install_traceable_hessian_linearization_for_value_only_state()`, and runs only the fused
         traceable single-stage FD certificate that is valid for that resolved seed state. The seed path
         is intentionally explicit, not defaulted: a certificate must state exactly what it certifies.
         Tier 5 forwards the same runtime-seed option; `--raw-stage2-seed` opts into legacy raw replay.
   - [x] Execute the first production mpol10 RunPod FD ladder and preserve the failed/diagnostic JSON.
         Artifact:
         `.artifacts/single_stage_convergence/phase1_fd/run_20260627T181848Z/adjoint_fd_cuda_runtime_seed_mpol10_ntor10_nphi255_ntheta64.json`.
         It does not certify the adjoint: dir0 Richardson is ≈1.0%, dir1 ≈3.8%, and dir2 ≈13.8%.
         The result supports a coarse-FD/mis-instrumented-gate hypothesis and leaves dir2 undecided.
         Prior local support-gate refresh (2026-06-27, before the status-aware FD patch):
         `PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
         /Users/suhjungdae/code/columbia/.venv-simsopt-uv/bin/python -m pytest
         tests/test_adjoint_fd_validation_contract.py tests/test_compile_breadth_probe_contract.py
         tests/test_check_cached_kernel_callback_compatibility.py -q` -> `43 passed in 2.54s`.
         This validates harness contracts only; it is not a production certificate. After the
         status-aware FD patch, only static checks have been run locally; runtime/tests should run on
         RunPod per the current execution constraint.
2. Preserve plus/minus forward-solve diagnostics in the FD ladder.
   - [x] Change `compute_traceable_single_stage_fd_ladder()` to accept the runtime bundle's
         `forward_result` callable and `value_and_grad`.
   - [x] For every central-FD plus/minus evaluation, call `forward_result(x0 ± eps*d)` and record
         `value`, `success`, `primal_success`, `adjoint_linear_solve_available`, `iota`, and `G`.
         Use `forward_result["value"]` for the FD numerator so diagnostics and measured value are
         tied to the exact same traceable solve.
   - [x] Classify any eps record with rejected or primal-failed plus/minus sides as
         `invalid_fd_window` or `inner_solve_failed`, not `gradient_failed`.
   - [x] Add contract tests proving rejected plus/minus evaluations are surfaced in the JSON and do
         not satisfy the gradient-certification gate.
3. Replace the weak coarse-to-finest gate with a real asymptotic-window gate.
   - [x] Compute per-direction observed order from adjacent accepted eps intervals, using only eps
         records whose plus/minus `success` and `primal_success` flags are true.
   - [x] Compute Richardson extrapolation from the two finest accepted eps points and compare the
         extrapolated directional derivative to the adjoint directional derivative.
   - [x] Pass a direction only when either the finest accepted raw FD already satisfies tolerance or
         Richardson/order evidence satisfies tolerance. A monotone decrease alone must not pass.
   - [x] Emit a stable per-direction status such as `accepted`, `needs_smaller_eps`,
         `invalid_fd_window`, or `gradient_mismatch`, plus the numeric reason.
4. Make the eps ladder config/noise adaptive.
   - [x] Replace the hard lower window floor `6e-4` with a scale/noise-aware rule. The default ladder
         may still start at `3e-3`, but the gate must be able to descend below `7.5e-4` when
         forward-result success flags are clean and roundoff/noise has not appeared.
   - [x] Add or update a RunPod command path for the next production ladder, for example
         `7.5e-4 3.75e-4 1.875e-4 9.375e-5`, while documenting that RunPod is the execution
         environment and not the reason those eps values are mathematically valid.
   - [x] Store the selected ladder, objective scale, gradient norm, and any measured noise/roundoff
         floor in the artifact so the eps choice is reviewable.
5. Add eps-free supplementary checks.
   - [x] Add an operator dot-product/transpose test using the runtime adjoint state:
         verify `<A v, w> == <v, A^T w>` for representative random vectors through
         `apply_forward` / `apply_transpose`. This certifies the linear transpose wiring and should
         be reported separately from the nonlinear gradient FD gate.
   - [x] Investigate JVP-vs-VJP on the exact runtime objective path. If the public custom-VJP scalar
         objective cannot support forward-mode JVP, record that limitation and do not block Phase 1
         on an inapplicable check. If it is supported or a suitable exact-AD internal path exists,
         add a directional JVP/VJP agreement check to decide whether dir2 is a real adjoint
         discrepancy or only an FD-window/non-smoothness issue.
6. Re-run Phase 1 on RunPod only.
   - [ ] Run the smaller/adaptive ladder on the existing mpol10 runtime seed.
   - [ ] Preserve the JSON artifact and explicitly classify dir2 as certified, needs-smaller-eps,
         invalid-FD-window, or gradient-mismatch.
   - [ ] Do not mark Phase 1 green until every sampled direction has either an accepted asymptotic FD
         window or an explicit non-FD certificate that covers the same derivative path.

### Phase 2 — Get a real converged GPU result via the decomposed production lane (ranked #1)
1. Run `--optimizer-backend scipy-jax-decomposed` on GPU at **mpol ≤ 6** (compiles feasibly).
   - [ ] Persistent compile cache on a **network volume** (`JAX_COMPILATION_CACHE_DIR=/workspace/...`,
         not `/tmp`); confirm warm-cache hit on a second run.
   - [ ] Record the convergence table (see Validation). Gate acceptance on the Phase-1 gradient
         certificate.
   - [ ] (No code change expected for this step — config/run only.)

### Phase 3 — Smooth the hardware penalty (Root Cause 1a) — TRUE ROOT FIX
1. Convert the hard hardware feasibility gate to a constraint-residual contract.
   - [x] Add a single-stage hardware constraint evaluator beside `success_filter`
         (`single_stage_banana_example.py` ~7728-7800) that returns the four positive-when-violating
         residuals:
         `cc_dist - curve_curve_min_dist`, `cs_dist - curve_surface_min_dist`,
         `ss_dist - surface_vessel_min_dist`, and `max_curvature - curvature_threshold`.
         These signs are load-bearing: distance constraints are lower bounds, so threshold minus
         measured distance is positive when unsafe; curvature is an upper bound, so measured
         curvature minus threshold is positive when unsafe.
         Keep the existing boolean predicate as a derived feasibility value, or update every
         `success_filter` callsite/cache/test in one coherent contract change. Do not silently make a
         bool-typed `success_filter` return an array.
         Local result: `build_single_stage_target_lane_hardware_constraint_evaluator()` returns
         `residuals`, `positive_residuals`, metrics, and derived `success`; the legacy hardware
         `success_filter` delegates to that evaluator. Standard target-lane optimization no longer
         passes the hardware bool into the hard filter; it relies on the existing weighted smooth
         curve-curve, curve-surface, surface-vessel, and curvature objective terms.
   - [x] Keep self-intersection as a separate hard predicate unless a differentiable surrogate is
         explicitly designed; do not fold it into the smooth hardware barrier by accident.
         Local result: `build_single_stage_target_lane_success_filter()` now returns only the
         self-intersection filter. The ALM lane already used the same self-intersection-only hard
         filter.
2. Replace the flat plateau + frozen gradient with a smooth exterior barrier.
   - [x] `_traceable_rejected_objective_value` (`surface_objectives.py` ~1038-1048): drop the
         `stop_gradient` on finite candidate values and do not route hardware-margin violations
         through the rejected plateau. The smooth exterior hardware barrier must use
         positive-when-violating residuals without sign inversion or double-counting. Distance
         constraints are lower bounds, curvature is an upper bound, so a uniform
         `metric - threshold` margin would penalize safe distance slack and miss unsafe distance
         violations.
         Local result: hard hardware rejects no longer route through this helper in the standard
         target lane; the candidate value is no longer `stop_gradient`ed for remaining hard-filter
         rejection paths. The smooth hardware exterior penalty is the existing weighted objective
         terms plus the explicit residual contract above, not a second double-counted penalty.
   - [x] Consumer (`surface_objectives_traceable.py` ~718-739): feed margins into the smooth penalty
         instead of the boolean `lax.cond`.
         Local result: the standard target lane does not feed hardware into `success_filter`, so
         hardware margins remain in the smooth objective instead of the boolean `lax.cond`.
   - [x] Return the **true point-dependent gradient** on rejection, not the frozen baseline:
         `surface_objectives_traceable.py` ~3640-3641 and the decomposed fallback
         `single_stage_banana_example.py` ~9714-9722.
         Local result: fused/custom-VJP rejected candidates with `primal_success=True` now use the
         candidate state gradient; true primal failures still use the baseline fallback.
   - [x] Update the existing rejection-path tests that currently pin baseline-gradient behavior
         (e.g. `tests/geo/test_surface_objectives_jax.py` and
         `tests/integration/test_single_stage_jax_cpu_reference.py`) to assert the new
         point-dependent hardware-barrier gradient.
         Local result: focused tests now cover candidate-state gradient on filter rejection,
         baseline fallback on true primal failure, and
         hardware residuals deriving feasibility from cached immutable state.
   - [x] Add an artifact evaluator for the Phase-3 accepted-step proof. Local result:
         `benchmarks/single_stage_outer_loop_probe.py::evaluate_phase3_accepted_step_gradient_trace()`
         reads replay-grade `outer_optimizer_progress.json` objective-evaluation groups, requires
         candidate DOF and optimizer-gradient vectors, compares the last successful event in each
         accepted-step group against the baseline group, and can require a configured hardware-margin
         threshold for constraint-marginal evidence. When that margin threshold is configured, the
         same accepted-step comparison must be both constraint-marginal and point-dependent. The probe
         exposes
         `--enable-phase3-gradient-proof-gate` and `--phase3-constraint-margin-abs-tol`. This is a
         support gate only; the production Phase-3 accepted-step artifact remains open until a real
         progress JSON passes it.
3. Preserve genuine inner-solve-failure handling.
   - [x] Keep a (smooth, informative-gradient) penalty for true Boozer-solve failure (distinct from
         hardware-margin violation); do NOT reintroduce a flat plateau.
         Local result: `primal_success` is carried through the custom-VJP state so true primal/solve
         failure remains the baseline fallback while filter rejection after a successful primal solve
         receives the candidate-state gradient. The decomposed host split now fails loudly if
         `forward_result["primal_success"]` is missing instead of conflating filtered rejection with
         solve failure.

### Phase 4 — Robustness hardening (Root Causes 1b + 3 residual)
1. Match outer `ftol` to the noise floor (1b).
   - [x] Floor `ftol_by_mpol` (`single_stage_banana_example.py` ~14328-14340) at ≥ 1e-8 for high mpol
         (`ftol ≥ ~100·δJ`), or clamp at the resolution site (~15629-15633). Optionally tighten inner
         `newton_tol` 1e-11→1e-13 (knob already plumbed: `--target-lane-boozer-newton-tol`) to lower δJ.
         Local result at `458d06a3a`: defaults for `mpol>=8` resolve through
         `resolve_single_stage_outer_ftol()` and never fall below `1e-8`; explicit `--outer-ftol`
         remains a user override. Focused gate `tests/integration/test_single_stage_progress_diagnostics.py`
         passes (`12 passed`).
2. Guard the rank-deficiency NaN (3 residual).
   - [x] First add a regression/probe proving the current `_dense_matrix_solve_numerically_safe`
         misses a near-singular float64 production iterate. Local result: a deterministic consistent
         float64 system with estimated condition ≈2.1e13 produced machine-small residual but
         1e-4–1e-3 relative solution error while the old LU/lstsq status reported success.
   - [x] Apply the smallest fail-closed fix without extending the float64 forward-error bound.
         Local result: `_dense_matrix_nonsingular_threshold()` now caps float64 dense-solve condition
         estimates at 1e12, leaving the existing fp32 forward-error gate unchanged. Focused regression
         `tests/geo/test_adjoint_cg_solver.py::test_float64_dense_status_fails_closed_on_near_singular_forward_error`
         covers both LU and lstsq dense-square status paths; adjacent singular/fp32 gates still pass.
   - [x] Keep tiny roundoff-scale adjoints from being reported as NaN when the residual is clean.
         Local result: dense LU/lstsq status now has a narrow small-solution exception:
         condition-unsafe solves remain rejected unless the residual gate passes and the returned
         solution is itself below `100 * effective_linear_solve_tolerance`. This preserves the O(1)
         near-singular wrong-solution fail-closed tests while allowing the local decomposed baseline
         seed's finite O(1e-9) adjoint instead of the old RHS-based zero shortcut.

### Phase 5 — Production-scale GPU compile (Root Cause 2, mpol10)
1. Measure before narrowing.
   - [ ] Run `benchmarks/compile_breadth_probe.py` at mpol 6/8/10 under `JAX_LOG_COMPILES=1` +
         `XLA_FLAGS=--xla_dump_to=...` → identify the dominant sub-kernel (K1 forward vs K2 adjoint).
         Current local blocker: the repo interpreter exposes only the CPU JAX backend; a direct check
         with `JAX_PLATFORMS=cuda,cpu` reports `Unknown backend cuda. Available backends are ['cpu']`.
         A failed checkpoint exists at
         `.artifacts/single_stage_convergence/phase5_compile/run_20260627T191118Z_fresh/compile_breadth_cuda_mpol6_8_10.json`;
         it confirms H100/CUDA provenance but has `completed_mpol: []`, `results_by_mpol: {}`, and a
         stale `status: "running"` after the process exited 1 during mpol6 Boozer initialization.
         It is therefore not a compile-breadth result.
2. Narrow the dominant kernel (only what the probe implicates).
   - [x] Promote the decomposed host-split support path (removes the optimizer macro-step need to
         enclose K1 forward solve and K2 value/grad in one callable). Current dirty-tree local result:
         `scipy-jax-decomposed` has the split solved-pair host objective, exact-zero adjoint RHS
         coverage, nonzero-below-tolerance adjoint RHS coverage, fused/decomposed finite gradient
         parity, host-sync budget coverage, compile-trace K1/K2 separation coverage, e2e
         SciPy-control parity, and nonfinite-trial policy forwarding. Focused CPU gate passes as part
         of the local `33 passed`
         matrix; production GPU convergence remains unchecked.
   - [ ] If the compile-breadth probe implicates the dense-adjoint gate, make it breadth-aware
         (`optimizer.py` `_dense_square_operator_materialization_allowed` ~4746) so surface-sized
         adjoints default to operator-GMRES without an env var.
   - [x] Add the fused `_value_and_grad_for` to `CALLBACK_FREE_TARGETS` (persistent-cache coverage),
         confirm via `benchmarks/check_cached_kernel_callback_compatibility.py`.
         Local result: `benchmarks/check_cached_kernel_callback_compatibility.py`
         exists, scans `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py` target
         `_value_and_grad_for`, follows direct same-module helper calls, and reports
         `passed=true`; focused unit gate `tests/test_check_cached_kernel_callback_compatibility.py`
         passes (`21 passed`).
   - [x] Record compile-environment provenance before trusting GPU artifacts. Local result:
         `benchmarks/compile_breadth_probe.py` records `JAX_LOG_COMPILES` and `XLA_FLAGS` under
         `compile_environment`. The actual Phase-5 proof still requires an emitted GPU JSON artifact;
         no separate static literal-pinning test is kept for the probe.
   - [x] Preserve partial compile-breadth artifacts during the sweep. Prior dirty-tree local result:
         `benchmarks/compile_breadth_probe.py` atomically checkpoints the output JSON before and after
         each resolution with `status`, `active_resolution`, `completed_mpol`, `compile_environment`,
         and `results_by_mpol`. The focused gate
         `tests/test_compile_breadth_probe_contract.py` previously proved an in-progress mpol8 sweep
         preserves the completed mpol6 record and that a failed temp-file write leaves the previous
         checkpoint JSON intact; tests have not been rerun locally after the fail-closed patch.
   - [x] Add fail-closed exception finalization around each `_probe_resolution()` call so a runtime
         failure rewrites the checkpoint with a terminal failure status and the exception class/message.
         The prior H100 JSON with `status: "running"` is stale historical evidence from before this
         source fix; it is not a successful compile-breadth measurement.

### Phase 6 — Re-audit the `newton_polish` host-materialization contract (Root Cause 4)
1. Verify whether the historical contradiction still exists.
   - [x] Run the focused contract slice:
         `PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /Users/suhjungdae/code/columbia/.venv-simsopt-uv/bin/python -m pytest tests/geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_host_control_uses_host_backtracking tests/geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_host_control_accepts_dynamic_objective_args tests/geo/test_boozersurface_jax.py::TestOptimizerAdapter::test_newton_polish_host_control_uses_host_dense_materialization tests/geo/test_optimizer_jax_item19.py::test_item19_host_dense_hessian_agrees_with_device_materializer -q`
         -> `4 passed in 3.49s` on the current dirty tree.
   - [x] Historical review result at `2c71021b1`: the failing node was
         `test_newton_polish_host_control_uses_host_dense_materialization`; it monkeypatches
         `_materialize_dense_hessian` to prove host control uses an independent host materializer, but
         `_materialize_dense_hessian_host` delegated back into `_materialize_dense_hessian`.
         The adjacent host-control routing and item19 dense-materializer comparison nodes passed, so the
         live contradiction was specifically about whether `allow_host_control=True` is allowed to reuse
         the chunked device materializer.
   - [x] Product decision: keep a true host materializer for `allow_host_control`. Local fix changes
         `_materialize_dense_hessian_host` to materialize HVP columns through a host loop, leaving the
         chunked device materializer unchanged. Current focused Phase-6 contract slice passes as shown
         above.

## Validation Plan

- [ ] **Convergence table** (the primary acceptance artifact), from the run progress JSON:
      | outer iter | J | ‖proj g‖∞ / ‖g‖∞ | accepted? | status |
      Pass = J monotone ↓, ≥1 accepted step, `status==0`, reported projected-gradient norm ↓ to
      `gtol`.
- [ ] **A/B control**: same seed pre-fix (`status=2, nfev=21, 0 accepted`) vs post-fix (`status=0`).
- [ ] **Phase-3 proof**: on a constraint-marginal seed, the accepted-step gradient is
      point-dependent (differs from the frozen `baseline_coil_gradient`).
- [x] **Phase-3 proof support gate**: synthetic progress artifacts validate that
      `evaluate_phase3_accepted_step_gradient_trace()` accepts replay-grade point-dependent accepted
      gradients, rejects frozen accepted-step gradients, rejects compact traces without vector values,
      rejects split evidence where the marginal step is not also point-dependent, and preserves
      optional hardware-margin evidence. This is artifact-tooling validation, not production Phase-3
      evidence.
- [x] **Phase-0 support gates**: synthetic progress artifacts validate finite plateau/nonfinite trace
      classification, fixed-candidate δJ comparison, and probe-level replay/noise-calibration wiring.
      This is only artifact-tooling validation, not production Phase-0 evidence.
- [ ] **Phase-1 gradient certificate** passes on RunPod at production mpol10 runtime-seed scale with
      success-clean plus/minus `forward_result` records for every accepted eps, observed-order or
      Richardson acceptance for every sampled direction, and no direction left in `needs_smaller_eps`.
- [x] **FD diagnostic classification** reports rejected or primal-failed plus/minus probes as
      `invalid_fd_window` / `inner_solve_failed`, not as a gradient mismatch.
- [ ] **Supplemental transpose certificate** reports dot-product agreement at solver tolerance for
      the runtime adjoint linear operator path. Source support is installed, but the first RunPod
      adaptive runtime-seed attempt showed `BoozerSurfaceJAX.get_adjoint_runtime_state()` is
      unavailable for that value-only seed path; the artifact now records this as non-blocking
      `unsupported` instead of failing before the FD ladder.
- [x] **Exact-AD feasibility check** either adds a JVP/VJP agreement check on the exact runtime
      objective path or records why the public `jax.custom_vjp` boundary makes that check
      inapplicable.
- [x] **FD cert harness** (Phase 1 support gate) defaults to mpol 8, rejects lower-mpol certification
      runs, and asserts forward-result operator metadata from the runtime-bundle forward-result path.
      It also supports the canonical mpol10/ntor10/nphi255/ntheta64 JAX runtime seed spec for
      resolved-seed fused traceable FD certification when the seed path is passed explicitly. The
      initial production eps ladder is now diagnostic scaffolding, not the final acceptance rule.
- [x] **Focused regression — core untouched**: focused Stage-2 objective/gradient, Stage-2
      BiotSavart value/VJP, and Boozer host-materialization selectors pass locally. This is not a
      full-suite claim.
- [x] **Path-based Stage-2 suite** now passes locally:
      `tests/integration/test_stage2_jax.py tests/integration/test_stage2_target_lane_purity.py -q`
      -> `201 passed, 4 warnings in 265.27s`. The local cleanup removes one brittle exact-final-DOF
      assertion from an already objective-matched L-BFGS-B trajectory test, makes the repo-fixture
      fallback test independent of machine-local `DATABASE/EQUILIBRIA` contents, and fixes
      `SpecBackedBiotSavartJAX` restart graph reconstruction so saved JAX specs reload banana
      symmetry replicas as a shared base curve/current plus rotated/scaled wrappers.
- [ ] **Compile probe** (Phase 5) emits a successful GPU results JSON before any narrowing PR.
      The local harness now atomically checkpoint-writes partial JSON artifacts, and a failed H100
      checkpoint exists, but no completed GPU measurement artifact has been produced in this checkout.
- [x] **CPU-reference single-stage integration parity** now passes locally:
      `tests/integration/test_single_stage_jax_cpu_reference.py -q` -> `171 passed, 15 skipped,
      4 warnings in 1187.13s`. The local cleanup removes/narrows stale assertions that expected
      finite gradients from condition-rejected shifted target states; the current traceable adjoint
      is fail-closed under the dense-condition cap, so those fixtures are not production gradient
      certificates.
- [x] **Decomposed host-split support gate** (Phase 5 support path) passes locally on CPU:
      finite baseline seed/parity, K1/K2 compile separation, SciPy-control e2e parity, and
      nonfinite-trial policy forwarding are covered in the focused `33 passed` matrix. This does not
      replace the GPU compile/convergence artifact.
- [x] **Callback-cache checker** (Phase 5 support gate) exists and the static source guard reports
      no forbidden host/debug callback primitives in fused `_value_and_grad_for` or direct same-module
      helper calls.
- [ ] Full JAX-port test suite green under the repo interpreter
      (`/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest`, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`).
      Current collection attempts over broad marker/kernel buckets are not green: marker-based
      Stage-2 collection still imports unrelated missing/stale modules before filtering, and the
      broad kernel path collection currently hits missing `diffrax` tracing tests. Use the path-based
      Stage-2 evidence above as the Stage-2 proof, not as a full-suite substitute.

## Risks and Mitigations

- Risk: Smooth-barrier weights `w_k` distort the physical optimum near the constraint boundary.
  Mitigation: short weight sweep; compare converged config + constraint activity to the CPU-converged
  reference; require constraints satisfied at the optimum.
- Risk: The 1a edit lives in the **core adapter** (`surface_objectives*.py`) shared by the traceable
  machinery, so it could perturb Stage-2 / other traceable objectives.
  Mitigation: gate behind the single-stage rejection path; run the Stage-2 + parity regression
  (Validation) before/after; the path is grep-confirmed single-stage-only today — keep it that way.
- Risk: Tightening inner `newton_tol` (1b) raises per-eval cost.
  Mitigation: prefer raising `ftol` first (cheap); tighten inner tol only if mpol18-level precision is
  needed; note the bfgs override is hard-capped at 1e-8 (`single_stage_banana_example.py` ~11170/11221).
- Risk: `newton_stab > 0` perturbs the adjoint solution.
  Mitigation: use a tiny *relative* floor only as a NaN guard; verify the production gradient
  certificate still passes with it on.
- Risk: Phase-5 narrowing rewrites the non-dominant kernel (wasted effort / regression).
  Mitigation: measurement-first — do not narrow without the `compile_breadth_probe.py` result.

## Completion Criteria

- [ ] Converged GPU single-stage result with `status==0` and a production-scale Phase-1 gradient
      certificate. Phase 1 is not complete while direction 2 remains only "likely mis-calibrated";
      it must be classified as certified, needs-smaller-eps, invalid-FD-window, or gradient-mismatch
      by the revised RunPod ladder/checks.
- [ ] A full-space lane (`scipy-jax-decomposed`) converges from a marginal/infeasible seed after the
      smooth-barrier fix (Phase 3).
- [x] `ftol` floor + rank-deficiency guard + tiny-adjoint dense-solve gate landed and locally
      validated (Phase 4).
- [ ] Per-seed δJ/noise calibration from Phase 0.2 is still required before claiming the production
      noise budget is fully measured.
- [x] `newton_polish` host-materialization contract audited; focused tests agree (Phase 6).
- [ ] Full suite green after Phase 6 and the later Phase 1/3/5 changes.
- [x] Focused Stage-2 / kernel parity regression slice passed; core JAX kernels remain untouched.
- [x] Path-based Stage-2 suite passes locally:
      `tests/integration/test_stage2_jax.py tests/integration/test_stage2_target_lane_purity.py -q`
      -> `201 passed, 4 warnings in 265.27s`.
- [x] Single-file CPU-reference integration parity passes locally:
      `tests/integration/test_single_stage_jax_cpu_reference.py -q` -> `171 passed, 15 skipped,
      4 warnings in 1187.13s`.
- [ ] Full kernel parity suite has not been run cleanly in this local continuation; broad collection is
      currently blocked by unrelated missing/stale modules, including missing `diffrax` tracing tests.

## Open Questions

- **[Closed locally]** Phase 6 product decision: keep a true host Hessian path for
  `allow_host_control`; focused contract tests now pass after making
  `_materialize_dense_hessian_host` independent from `_materialize_dense_hessian`.
- **[SSOT]** Should the smooth penalty live in the example driver or be promoted into the adapter as
  the canonical constraint handling? (Affects whether other future single-stage drivers inherit it.)
- **[Data]** Does the revised success-clean adaptive ladder close production dir2, or does it expose
  a real gradient mismatch/non-smooth direction?
- **[Design]** Is JVP/VJP available on an exact internal runtime objective path despite the public
  scalar objective using `jax.custom_vjp`, or should the certificate rely on dot-product plus
  FD/Richardson diagnostics only?
- **[Data]** Per-seed δJ (Phase 0.2) and the mpol10 dominant-kernel breadth (Phase 5.1) — both require
  one short runtime measurement each before the dependent fix is finalized.
- Is a converged result wanted at **production mpol10** (needs Phase 5) or is mpol≤6 sufficient for
  the immediate milestone (Phase 2 only)?
