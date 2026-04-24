# Banana Optimization Todos

Scope: Stage 2 and single-stage banana optimization, plus the shared SIMSOPT
library paths reached by those workflows.

Status as of 2026-04-24: this file is the active SSOT for the banana
optimization issue backlog. `PERF_ISSUES.md` is historical/detail context; do
not treat it as a competing priority list.

Upstream boundary: `hiddenSymmetries/simsopt` master does not contain
`examples/single_stage_optimization`. Items under `src/simsopt` can be checked
against upstream. Banana workflow items are fork-local.

Work rule: every patch must land with a regression proof and an impact
measurement. For correctness items, impact can be "old path fails, new path
passes." For performance and memory items, record wall time and peak RSS or
allocation deltas on a fixed fixture.

## Tracking Template

Use this gate for each item before marking it done:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

## Correctness

### C1. Fix `SquaredFlux` surface invalidation

Category: correctness
Scope: shared upstream SIMSOPT path
Impact: high
Effort: small

Current anchors:

- `src/simsopt/_core/optimizable.py` :: `Optimizable.add_recompute_dependency`
- `src/simsopt/objectives/fluxobjective.py` :: `SquaredFlux.__init__`
- `src/simsopt/objectives/fluxobjective.py` :: `SquaredFlux.recompute_bell`
- `src/simsopt/objectives/fluxobjective.py` :: `SquaredFlux.J`
- `src/simsopt/objectives/fluxobjective.py` :: `SquaredFlux.dJ`
- `tests/core/test_optimizable.py`
- `tests/objectives/test_fluxobjective.py`

Problem:

`SquaredFlux` sets field evaluation points from `surface.gamma()` at
construction time, but declares only `depends_on=[field]`. If the surface moves,
the objective can combine stale `B(x_old)` with current surface normals.

Fix direction:

- Register the surface as an invalidation source without exposing surface DOFs
  through `SquaredFlux.x`, using `Optimizable.add_recompute_dependency`; the
  class still differentiates only through the magnetic field/coils.
- Refresh field points from the current surface geometry when the surface
  invalidates the objective.
- Add a test that mutates only surface DOFs and compares against a freshly
  constructed `SquaredFlux`.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Validation:

- `python3 -m pytest tests/objectives/test_fluxobjective.py -q`
- `python3 -m pytest tests/core/test_optimizable.py tests/objectives/test_fluxobjective.py tests/geo/test_curve_objectives.py tests/geo/test_banana_impact_benchmark.py tests/geo/test_single_stage_example.py -q`
- `python3 examples/single_stage_optimization/benchmark_banana_impact.py --fixture squared-flux --repeat 1 --warmup 0 --output /tmp/banana_impact_squared_flux_after_c1.json`

### C2. Fix `CurveSurfaceDistance` surface invalidation

Category: correctness
Scope: shared upstream SIMSOPT path
Impact: high
Effort: small

Current anchors:

- `src/simsopt/_core/optimizable.py` :: `Optimizable.add_recompute_dependency`
- `src/simsopt/geo/curveobjectives.py` :: `CurveSurfaceDistance.__init__`
- `src/simsopt/geo/curveobjectives.py` :: `CurveSurfaceDistance.recompute_bell`
- `src/simsopt/geo/curveobjectives.py` :: `CurveSurfaceDistance.compute_candidates`
- `tests/core/test_optimizable.py`
- `tests/geo/test_curve_objectives.py`

Problem:

`CurveSurfaceDistance` caches candidate curve/surface point pairs, but declares
only the curves as dependencies. If the surface moves and the curves do not,
the candidate cache can remain stale and miss newly close curve/surface pairs.

Fix direction:

- Register the surface as an invalidation source without exposing surface DOFs
  through `CurveSurfaceDistance.x`, using
  `Optimizable.add_recompute_dependency`; the class still differentiates only
  through curve DOFs.
- Keep candidate invalidation centralized in `recompute_bell`.
- Add a test that moves only the surface and verifies `J()` /
  `shortest_distance()` match a fresh object.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Validation:

- `python3 -m pytest tests/geo/test_curve_objectives.py -q`
- `python3 -m pytest tests/core/test_optimizable.py tests/objectives/test_fluxobjective.py tests/geo/test_curve_objectives.py tests/geo/test_banana_impact_benchmark.py tests/geo/test_single_stage_example.py -q`
- `python3 examples/single_stage_optimization/benchmark_banana_impact.py --fixture curve-surface-distance --repeat 1 --warmup 0 --output /tmp/banana_impact_curve_surface_after_c2.json`

### C3. Tighten Boozer first-use lifecycle

Category: correctness
Scope: shared upstream SIMSOPT path plus banana usage
Impact: high
Effort: small to medium

Current anchors:

- `src/simsopt/geo/boozersurface.py` :: `BoozerSurface.__init__`
- `src/simsopt/geo/boozersurface.py` :: `BoozerSurface.run_code_from_last_solution`
- `src/simsopt/geo/boozersurface.py` :: `BoozerSurface.run_code`
- `src/simsopt/geo/surfaceobjectives.py` :: `MajorRadius.compute`
- `src/simsopt/geo/surfaceobjectives.py` :: `NonQuasiSymmetricRatio.compute`
- `src/simsopt/geo/surfaceobjectives.py` :: `Iotas.compute`
- `src/simsopt/geo/surfaceobjectives.py` :: `BoozerResidual.compute`
- `tests/geo/test_boozersurface.py`
- `tests/geo/test_surface_objectives.py`

Problem:

Fresh `BoozerSurface` objects set `need_to_run_code=True`, but `res` is only
created after a solve. Several consumers read `boozer_surface.res` before
calling `run_code()` when `need_to_run_code` is true.

Fix direction:

- Make first-use state explicit: either require a declared initial `iota/G`
  source or fail fast with a precise contract error.
- Do not silently guess missing Boozer solve seeds.
- Cover fresh-object behavior for `NonQuasiSymmetricRatio`, `Iotas`, and
  `BoozerResidual`.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Validation:

- `python3 -m pytest tests/geo/test_surface_objectives.py::IotasTests::test_unsolved_boozer_surface_fails_with_contract_error tests/geo/test_boozersurface.py::BoozerSurfaceTests::test_run_code -q`
- `python3 -m pytest tests/geo/test_surface_objectives.py::IotasTests tests/geo/test_surface_objectives.py::NonQSRatioTests tests/geo/test_surface_objectives.py::BoozerResidualTests tests/geo/test_boozersurface.py::BoozerSurfaceTests::test_run_code -q`
- Downstream smoke: `single_stage_banana_example.py --init-only` against
  `/Users/suhjungdae/code/columbia/autoresearch/harvested_seeds/R_nv2_iota305_hbtclean_2026-04-23/biot_savart_opt.json`
  plus matching `surf_opt.json`, native `mpol=10`, `ntor=10`, reduced
  `nphi=31`, `ntheta=16`. Result: Boozer Newton success with
  `iota=0.3048386265857189`, `volume=0.039921036663101706`, optimizer skipped
  by `--init-only`. Legacy missing-`STAGE2_BS_SHA256` warning observed.
- Downstream e2e: non-`--init-only` run against the same harvested seed pair,
  `wout_nfp5ginsburg_000_014417_iota15.nc`, native `mpol=10`, `ntor=10`,
  reduced `nphi=31`, `ntheta=16`, and `--maxiter 1` completed with exit code
  0. Result artifact:
  `/tmp/simsopt_surrogate_full_e2e_R_nv2/mpol=10-ntor=10-b0f93213/results.json`.
  Software path succeeded: Boozer init completed, optimizer iteration 1 ran,
  an unsafe self-intersecting trial was rejected, and final artifacts were
  written. Optimizer convergence was intentionally not expected because
  `maxiter=1`; final state preserved the feasible start with
  `FINAL_IOTA=0.30483862658571914`, `FINAL_VOLUME=0.03992103666310177`,
  `HARDWARE_CONSTRAINTS_OK=true`, and `FINAL_FEASIBILITY_OK=true`.

### C4. De-risk Boozer second-order residual semantics

Category: correctness risk
Scope: shared upstream SIMSOPT path
Impact: medium
Effort: small

Current anchors:

- `src/simsopt/geo/surfaceobjectives.py` :: `boozer_surface_residual`
- `tests/geo/test_surface_objectives.py`

Problem:

The second-order residual path has confusing local reuse/shadowing around
`d2B2_dcdc` and dense Hessian assembly. This is primarily a maintainability
trap, but it sits in a correctness-sensitive Newton/LS path.

Fix direction:

- Split shadowed locals into semantic names.
- Add weighted and unweighted `derivatives=2` regression coverage before
  changing allocation strategy.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Validation:

- `python3 -m pytest tests/geo/test_boozersurface.py::BoozerSurfaceTests::test_boozer_penalty_constraints_cpp_notcpp -q`

## Performance

### P1. Split optimizer hot path from diagnostics

Category: performance
Scope: fork-local banana workflow
Impact: very high
Effort: small to medium

Current anchors:

- `examples/single_stage_optimization/banana_opt/stage2_objectives.py` ::
  `build_stage2_optimization_fun`
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py` ::
  `evaluate_stage2_alm_problem`
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py` ::
  `evaluate_total_objective`
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py` ::
  `evaluate_base_objective`
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py` ::
  `evaluate_alm_objective`
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  :: `evaluate_search_step`

Problem:

Search evaluation still computes report-only diagnostics and per-term
summaries during line-search probes.

Fix direction:

- Create a strict fast-path evaluator that returns only scalar value, gradient,
  and mandatory reject data.
- Move exact distances, curvature summaries, field snapshots, per-term
  diagnostics, and logging to accepted-step callbacks or final reports.

Impact measure:

- Wall time per objective evaluation.
- Number of `shortest_distance`, per-term `dJ`, and Boozer diagnostic calls per
  SciPy function evaluation.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Impact:

- Target-mode `evaluate_search_step` now calls the compact objective payload
  path; accepted-step, preserved-timeout, and solver-checkpoint artifacts
  recompute rich diagnostics only when artifact fields require them.
- Resumed legacy checkpoints normalize compact best-incumbent payloads before
  carrying them forward into new checkpoint or preserved-timeout artifacts.
- Final and preserved-timeout payloads now emit `SEARCH_STEP_*` counters and
  timers for surface solve, objective evaluation, topology gate, hardware
  snapshot, fast-vs-diagnostic objective calls, rejected-after-surface-solve
  trials, hardware rejects, and curvature rejects.

Validation:

- `python -m pytest tests/geo/test_single_stage_example.py -k "resume_incumbent_normalization or best_accepted_incumbent or solver_checkpoint_accepted_incumbent or best_feasible_incumbent or search_step_metrics or evaluate_search_objective_uses_fast_payload_outside_frontier_mode"`
- `python -m pytest tests/geo/test_single_stage_example.py tests/geo/test_banana_objective_modules.py -q`
- `python -m ruff check examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py tests/geo/test_single_stage_example.py`

### P2. Remove exact sampled distances from optimizer hot loops

Category: performance
Scope: fork-local banana callers; shared SIMSOPT distance APIs reviewed and left unchanged
Impact: very high
Effort: small

Current anchors:

- `src/simsopt/geo/curveobjectives.py` :: `CurveCurveDistance.shortest_distance`
- `src/simsopt/geo/curveobjectives.py` :: `CurveSurfaceDistance.shortest_distance`
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py`
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py`

Problem:

Exact sampled distance diagnostics use full `cdist` scans and are still called
from optimizer evaluation paths.

Fix direction:

- Use smooth constraint/objective surrogates during optimization.
- Compute exact sampled distances only for accepted-step diagnostics and final
  reports.

Impact measure:

- `cdist` call count inside search evaluation should go to zero.
- Wall time before/after on the same Stage 2 ALM and single-stage fixture.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Impact:

- Stage 2 ALM fast-path feasibility now uses smooth distance constraint
  payloads and keeps exact `shortest_distance()` calls behind diagnostic
  emission.
- Single-stage `evaluate_search_step` now builds search hardware status from
  surrogate constraint payloads; exact sampled distance snapshots remain on
  accepted-step, preserved-timeout, initial, and final artifact paths.
- Penalty-mode compact objective payloads now include the small geometry
  constraint signal needed by the search gate, without restoring per-term
  diagnostic derivatives, and frontier hardware penalties consume those values
  as explicit objective-space violation ratios merged with physical current
  violation ratios rather than fake physical distances.
- Stage 2 parity now distinguishes the fast path, which uses smooth constraint
  feasibility, from the diagnostic path, which still emits exact sampled
  shortest-distance values for reports.

Validation:

- `python3 -m pytest tests/geo/test_banana_objective_modules.py -k "evaluate_total_objective_fast_path or stage2_alm_problem or search_hardware_snapshot" -q`
- `python3 -m pytest tests/geo/test_single_stage_example.py -k "evaluate_search_step_frontier_trust_excess_remains_search_penalty or evaluate_search_step_frontier_topology_reject_becomes_penalty or evaluate_search_step_frontier_hardware_reject_becomes_penalty or evaluate_search_step_repair_phase1_keeps_valid_hardware_bad_candidate_live" -q`
- `python3 -m pytest tests/geo/test_frontier_constraints.py -q`
- `python3 -m pytest tests/geo/test_frontier_constraints.py tests/geo/test_banana_objective_modules.py tests/geo/test_single_stage_example.py -q`
- `python3 -m pytest tests/geo/test_banana_modularization_parity.py tests/geo/test_single_stage_alm_integration.py tests/geo/test_stage2_single_stage_handoff.py tests/geo/test_single_stage_workflow_helpers.py -q`
- `python3 -m pytest tests/geo/test_alm_utils.py -q`
- `python3 -m ruff check examples/single_stage_optimization/banana_opt/stage2_objectives.py examples/single_stage_optimization/banana_opt/single_stage_geometry.py examples/single_stage_optimization/banana_opt/single_stage_objectives.py examples/single_stage_optimization/banana_opt/frontier_constraints.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py tests/geo/test_banana_objective_modules.py tests/geo/test_single_stage_example.py tests/geo/test_frontier_constraints.py tests/geo/test_banana_modularization_parity.py`

### P3. Tune L-BFGS-B `maxcor`

Category: performance
Scope: fork-local banana workflow plus Boozer solver options
Impact: medium to high
Effort: small

Current anchors:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  :: `--maxcor`
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py` ::
  optimizer options
- `examples/single_stage_optimization/banana_opt/single_stage_phase1.py` ::
  phase 1 optimizer options
- `src/simsopt/geo/boozersurface.py` :: L-BFGS-B solver options

Problem:

`maxcor=300` is large for these problem sizes. SciPy defines `maxcor` as the
number of limited-memory correction pairs, so high values increase memory and
linear algebra cost.

Fix direction:

- Benchmark `20`, `40`, `60`, and `300` on the same fixture.
- Lower defaults only where objective progress is not materially worse.
- Preserve explicit CLI override.

Impact measure:

- Peak RSS.
- Wall time.
- Accepted iterations.
- Final objective and gradient norm for a fixed short run.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Impact:

- Added `benchmark_lbfgsb_maxcor.py` and measured the P3 comparison set on a
  deterministic 160-DOF, 30-iteration L-BFGS-B fixture.
- Selected `maxcor=40` as the shared banana default. `20` used less Python heap
  but ended the fixed short run with a worse gradient norm. `40`, `60`, and `300`
  reached the same final objective and gradient on the fixture, while `40`
  avoided most of the allocation growth from larger histories.
- Benchmark output from the isolated per-`maxcor` process run
  `/tmp/banana_lbfgsb_maxcor_p3_remaining.json`:
  - `20`: median 0.006907 s, Python peak 257629 bytes, process peak RSS
    79953920 bytes, final objective `1.111026e+01`, gradient infinity norm
    `8.814162e+00`.
  - `40`: median 0.006795 s, Python peak 574832 bytes, process peak RSS
    79642624 bytes, final objective `1.109831e+01`, gradient infinity norm
    `4.840545e+00`.
  - `60`: median 0.007010 s, Python peak 1029686 bytes, process peak RSS
    80543744 bytes, final objective `1.109831e+01`, gradient infinity norm
    `4.840545e+00`.
  - `300`: median 0.006869 s, Python peak 17496539 bytes, process peak RSS
    87310336 bytes, final objective `1.109831e+01`, gradient infinity norm
    `4.840545e+00`.
- Single-stage, Stage 2, and the goal-mode wrapper now share the measured
  banana default through `banana_opt.lbfgsb_defaults.DEFAULT_LBFGSB_MAXCOR`;
  Stage 2 now exposes `--maxcor`, so explicit CLI/env overrides remain
  available.
- `BoozerSurface` limited-memory LS was reviewed and left at its prior core
  default because the banana quadratic fixture is not a Boozer residual
  convergence proof.

Validation:

- `npx ctx7@latest library SciPy "P3 maxcor L-BFGS-B SciPy minimize option maxcor limited memory correction pairs"`
- `npx ctx7@latest docs /scipy/scipy "L-BFGS-B maxcor option number of correction pairs minimize options memory"`
- `python examples/single_stage_optimization/benchmark_lbfgsb_maxcor.py --maxcor 20 --maxcor 40 --maxcor 60 --maxcor 300 --dimension 160 --maxiter 30 --repeat 3 --warmup 1 --output /tmp/banana_lbfgsb_maxcor_p3_remaining.json`
- `python -m ruff check examples/single_stage_optimization/banana_opt/lbfgsb_defaults.py examples/single_stage_optimization/benchmark_lbfgsb_maxcor.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py examples/single_stage_optimization/STAGE_2/banana_coil_solver.py examples/single_stage_optimization/run_single_stage_goal_mode_comparison.py tests/geo/test_banana_impact_benchmark.py tests/geo/test_single_stage_example.py tests/geo/test_single_stage_workflow_helpers.py src/simsopt/geo/boozersurface.py`
- `python -m py_compile examples/single_stage_optimization/banana_opt/lbfgsb_defaults.py examples/single_stage_optimization/benchmark_lbfgsb_maxcor.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py examples/single_stage_optimization/STAGE_2/banana_coil_solver.py examples/single_stage_optimization/run_single_stage_goal_mode_comparison.py src/simsopt/geo/boozersurface.py`
- `python -m pytest tests/geo/test_banana_impact_benchmark.py -q`
- `python -m pytest tests/geo/test_single_stage_example.py -k "maxcor or hardware_search_flags or curvature_traversal" -q`
- `python -m pytest tests/geo/test_single_stage_workflow_helpers.py -k "goal_mode_comparison_wrapper_defaults_match_single_stage_entrypoint" -q`
- `python -m pytest tests/geo/test_boozersurface.py -q`

### P4. Reuse exact Boozer Newton factorization

Category: performance
Scope: shared upstream SIMSOPT path
Impact: high
Effort: small to medium

Current anchors:

- `src/simsopt/geo/boozersurface.py` ::
  `solve_residual_equation_exactly_newton`
- `src/simsopt/objectives/utilities.py` :: `forward_backward`
- `tests/geo/test_boozersurface.py`

Problem:

The exact Newton path performs repeated solves against the same Jacobian and
then stores separate `(P, L, U)` factors for adjoints.

Fix direction:

- Reuse one factorization across Newton correction, iterative refinement, and
  adjoint solve data.
- Do not treat `lu_factor` / `lu_solve` as a drop-in replacement unless the
  `forward_backward` contract is updated too.

Impact measure:

- Wall time per exact Newton solve.
- Number of factorizations per Newton iteration.
- Residual convergence parity.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### P5. Share Boozer-derived objective evaluation state

Category: performance
Scope: fork-local banana workflow plus shared objective classes
Impact: high
Effort: medium

Current anchors:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  :: `build_single_stage_objective_bundle`
- `src/simsopt/geo/surfaceobjectives.py` :: `NonQuasiSymmetricRatio`
- `src/simsopt/geo/surfaceobjectives.py` :: `Iotas`
- `src/simsopt/geo/surfaceobjectives.py` :: `BoozerResidual`

Problem:

`Iotas`, `NonQuasiSymmetricRatio`, and `BoozerResidual` redo avoidable field,
surface, and adjoint work. The banana bundle also builds multiple
`BiotSavart(coils)` instances unless isolation is required.

Fix direction:

- Design one per-surface evaluation bundle with explicit invalidation.
- Share Boozer solve state, field points, residuals, and adjoint factors.
- Keep state isolation only where required by mutable field-point ownership.

Impact measure:

- Number of Boozer solves and field `set_points` calls per accepted step.
- Wall time for one full single-stage objective/gradient evaluation.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Impact:

- `BoozerResidual.compute()` and banana `RefinedBoozerResidual.compute()` now
  obtain residuals, residual-by-B, surface Jacobian, and LS adjoint inputs from
  one `boozer_surface_residual_dB(..., derivatives=1)` evaluation instead of
  recomputing the residual-by-B path for `dJ_by_dB()` and the LS adjoint VJP.
- A focused LS `BoozerResidual` regression verifies one residual-by-B kernel
  call for `J(); dJ()` on an already solved Boozer surface.
- `build_single_stage_objective_bundle()` now builds one `BiotSavart(coils)`
  per Boozer surface for the Boozer-derived QS and residual terms, rather than
  separate mutable field objects for `NonQuasiSymmetricRatio` and
  `BoozerResidual`.
- `measure_frontier_reference_metrics()` uses the same per-surface Boozer
  objective term builder, so frontier reference diagnostics keep the same field
  object sharing contract as the optimization bundle.

Validation:

- `python3 -m pytest tests/geo/test_surface_objectives.py::BoozerResidualTests::test_boozerresidual_compute_uses_one_field_point_update -q`
- `python3 -m pytest tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_boozer_derived_objective_terms_share_one_biotsavart_per_surface -q`
- `python3 -m pytest tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_frontier_reference_metrics_share_boozer_objective_biot_savarts -q`
- `python3 -m pytest tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_boozer_residual_exact_threads_fixed_current_into_example_adjoint_path -q`
- `python3 -m pytest tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_refined_boozer_residual_ls_uses_cached_adjoint_state -q`
- `python3 -m pytest tests/geo/test_surface_objectives.py::BoozerResidualTests::test_boozerresidual_derivative -q`
- `python3 -m pytest tests/geo/test_surface_objectives.py::BoozerResidualTests -q`
- `python3 -m pytest tests/geo/test_single_stage_example.py -q`

### P6. Optimize candidate distance objective loops only after hot-loop cleanup

Category: performance
Scope: shared upstream SIMSOPT path
Impact: medium
Effort: medium

Current anchors:

- `src/simsopt/geo/curveobjectives.py` :: `CurveCurveDistance.J`
- `src/simsopt/geo/curveobjectives.py` :: `CurveCurveDistance.dJ`
- `src/simsopt/geo/curveobjectives.py` :: `CurveSurfaceDistance.J`
- `src/simsopt/geo/curveobjectives.py` :: `CurveSurfaceDistance.dJ`
- `src/simsoptpp/python_distance.cpp`

Problem:

Candidate pruning exists, but Python still loops through candidate pairs and
invokes JAX work per pair.

Fix direction:

- Consider batched candidate evaluation once hot-loop exact diagnostics have
  been removed.
- Remove duplicate surface `gamma()` fetches.
- Do not rewrite C++ pruning before profile evidence says it matters.

Impact measure:

- Wall time for `J()` and `dJ()` across controlled candidate counts.
- Python call count per candidate batch.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### P7. Vectorize `Curve.dkappadash_by_dcoeff`

Category: performance
Scope: shared upstream SIMSOPT path
Impact: medium
Effort: small

Current anchors:

- `src/simsopt/geo/curve.py` :: `Curve.dkappadash_by_dcoeff`
- `tests/geo/test_curve.py`

Problem:

The method loops in Python over every curve DOF while doing broadcastable array
operations.

Fix direction:

- Add output-equivalence coverage against the current implementation.
- Replace the per-DOF loop with one broadcasted NumPy expression over the last
  axis.

Impact measure:

- Wall time and allocation count for representative banana coil orders.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### P8. Keep `CurveCWSFourierCPP` and low-level kernel rewrites last

Category: performance
Scope: shared upstream SIMSOPT path
Impact: unknown until reprofiled
Effort: large

Current anchors:

- `src/simsopt/geo/curvecwsfourier.py`
- `src/simsoptpp/magneticfield_biotsavart.cpp`

Problem:

These paths are plausible hotspots, but earlier evidence points to higher
return in Python orchestration, duplicated diagnostics, invalidation bugs, and
allocation-heavy aggregation.

Fix direction:

- Re-profile after C1-C3, P1-P5, and M1-M4.
- Only then decide whether a compiled or deep geometry rewrite is justified.

Impact measure:

- Profile evidence before design.
- Wall time and peak RSS after any rewrite.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

## Memory

### M1. Add baseline impact harness

Category: memory/performance measurement
Scope: fork-local banana workflow
Impact: enabling
Effort: small

Current anchors:

- `examples/single_stage_optimization/benchmark_banana_impact.py`
- `tests/geo/test_banana_impact_benchmark.py`
- `tests/geo/test_curve_objectives.py`
- `tests/objectives/test_fluxobjective.py`
- `tests/geo/test_surface_objectives.py`
- `examples/single_stage_optimization/run_stage2_iota_decision_gate.py`

Problem:

Several TODOs are plausible but not measurable yet in a consistent way.

Fix direction:

- Add a lightweight benchmark script or test utility that records wall seconds,
  Python peak allocations, and process peak RSS for fixed low-resolution
  fixtures.
- Keep it outside normal fast tests unless explicitly requested.

Impact measure:

- The harness itself should produce stable JSON/markdown output for before/after
  comparisons.
- The JSON schema now includes `process_peak_rss_bytes`; markdown output renders
  the same fixture timing, Python allocation, RSS, and checksum fields as a
  comparison table.
- Default fixture reports now measure each built-in fixture in a fresh subprocess
  so `process_peak_rss_bytes` is attributable to that fixture instead of a
  previous fixture in the same process.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Validation:

- `python3 -m pytest tests/geo/test_banana_impact_benchmark.py -q`
- `python3 -m py_compile examples/single_stage_optimization/benchmark_banana_impact.py tests/geo/test_banana_impact_benchmark.py`
- `python3 -m ruff check examples/single_stage_optimization/benchmark_banana_impact.py tests/geo/test_banana_impact_benchmark.py`
- `python3 examples/single_stage_optimization/benchmark_banana_impact.py --repeat 1 --warmup 0 --output /tmp/banana_impact_sample.json`
- `python3 examples/single_stage_optimization/benchmark_banana_impact.py --fixture biot-savart --repeat 1 --warmup 0 --format markdown --output /tmp/banana_impact_sample.md`
- `/opt/homebrew/Caskroom/miniforge/base/bin/python3 /Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/benchmark_banana_impact.py --fixture biot-savart --repeat 1 --warmup 0 --format json`

### M2. Reduce `Derivative` accumulation copies

Category: memory
Scope: shared upstream SIMSOPT path
Impact: high
Effort: medium

Current anchors:

- `src/simsopt/_core/derivative.py` :: `Derivative.__add__`
- `src/simsopt/_core/optimizable.py` :: `OptimizableSum`
- `src/simsopt/objectives/utilities.py` :: `MPIObjective.dJ`
- `src/simsopt/field/biotsavart.py` :: VJP returns
- `src/simsopt/field/magneticfield.py` :: `MagneticFieldSum.B_vjp`
- `tests/core/test_derivative.py`

Problem:

Python `sum(...)` over `Derivative` objects repeatedly copies dictionaries and
arrays.

Fix direction:

- Add a single-pass accumulation helper with unchanged derivative-key
  semantics.
- Move hot aggregation sites off Python `sum(...)`.

Impact measure:

- Allocation count and copied bytes for realistic derivative collections.
- Gradient parity with old aggregation.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### M3. Accumulate `MagneticFieldSum` outputs in place

Category: memory
Scope: shared upstream SIMSOPT path
Impact: medium
Effort: small

Current anchors:

- `src/simsopt/field/magneticfield.py` :: `MagneticFieldSum`
- `src/simsopt/objectives/utilities.py` :: `sum_across_comm`
- `tests/field/test_magneticfields.py`
- `tests/objectives/test_utilities.py`

Problem:

`MagneticFieldSum` builds full-array temporary lists and then calls `np.sum`.
MPI helper paths can similarly duplicate arrays through gathered Python sums.

Fix direction:

- Accumulate directly into the destination output buffer.
- Use an in-place or collective reduction path for ndarray sums where possible.

Impact measure:

- Peak RSS and allocation count for multiple field components and point counts.
- Numeric parity for `B`, `A`, and spatial derivatives.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### M4. Add a lower-memory `BiotSavart` compute mode

Category: memory
Scope: shared upstream SIMSOPT path
Impact: medium to high
Effort: large

Current anchors:

- `src/simsoptpp/magneticfield_biotsavart.cpp` :: `BiotSavart::compute`
- `src/simsopt/field/biotsavart.py` :: current-derivative accessors
- `tests/field/test_biotsavart.py`

Problem:

The C++ compute path stores per-coil `B_i`, `dB_i`, and `ddB_i` caches even
when callers only need total fields.

Fix direction:

- Add an explicit total-only compute mode.
- Preserve the current per-coil cache path for current-derivative and VJP
  callers.

Impact measure:

- `/usr/bin/time -l` peak RSS for `compute(0)`, `compute(1)`, and `compute(2)`.
- Numeric parity for total `B`, `dB`, and `ddB`.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### M5. Make Boozer LS second-order path lower-memory

Category: memory
Scope: shared upstream SIMSOPT path
Impact: very high when LS second-order path is active
Effort: large

Current anchors:

- `src/simsopt/geo/boozersurface.py` :: LS Newton path
- `src/simsopt/geo/surfaceobjectives.py` :: `boozer_surface_residual`
- `tests/geo/test_boozersurface.py`
- `tests/geo/test_surface_objectives.py`

Problem:

The `derivatives=2` LS path materializes large second-order tensors and a dense
`H`.

Fix direction:

- Prefer matrix-free or reduced second-order contractions.
- Remove unnecessary `.copy()` allocations while preserving weighted and
  unweighted behavior.

Impact measure:

- Peak RSS and wall time for low-resolution and banana-representative LS
  fixtures.
- Newton residual trajectory parity.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### M6. Reduce curve-distance VJP accumulator over-allocation

Category: memory
Scope: shared upstream SIMSOPT path
Impact: medium
Effort: small

Current anchors:

- `src/simsopt/geo/curveobjectives.py` :: `CurveCurveDistance.dJ`
- `src/simsopt/geo/curveobjectives.py` :: `CurveSurfaceDistance.dJ`
- `tests/geo/test_curve_objectives.py`

Problem:

Distance gradients allocate zero VJP buffers for every curve even when only a
small candidate subset is active.

Fix direction:

- Allocate accumulators only for touched curve indices, or reuse persistent
  buffers reset through `recompute_bell`.

Impact measure:

- Allocation count and wall time for candidate-sparse banana fixtures.
- Gradient parity with current implementation.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### M7. Clean up ALM copy discipline

Category: memory
Scope: fork-local banana workflow
Impact: medium
Effort: small to medium

Current anchors:

- `examples/single_stage_optimization/alm_utils.py`
- `tests/geo/test_alm_utils.py`

Problem:

The ALM driver copies arrays and dictionaries at many inner-loop boundaries,
including full history snapshots on callbacks.

Fix direction:

- Copy at persistence/checkpoint boundaries, not every inner-loop call.
- Replace full history cloning with latest-entry snapshots or explicit
  immutable summary state.
- Document which evaluation-dict fields are owned versus borrowed.

Impact measure:

- Peak allocation and callback time per ALM iteration.
- Regression coverage for callback/history ownership.

Completion:

- [x] Repro or baseline added
- [x] Fix implemented
- [x] Impact measured
- [x] Validation command recorded

Resolution notes:

- `minimize_alm` history callbacks now receive the ALM history list as
  borrowed/read-only state, plus an owned latest-entry snapshot and multiplier
  snapshot for checkpoint writers.
- Non-finite inner-evaluation rejection now shallow-copies the evaluation dict
  and owns only mutable gradient arrays (`grad`, `metric_grad`, `base_grad`);
  other evaluation metadata stays borrowed until result/checkpoint persistence.
- Regression coverage asserts callback/history ownership and owned gradient
  array isolation.
- Remaining cleanup removes the duplicate post-solve `result.x` ownership copy
  by carrying the accepted candidate vector through the inner-attempt loop, and
  removes the duplicate restored-state copy before failure-result construction.
- Impact measurement on a 10,000-entry ALM history callback proxy:
  old full-history clone median 3481.8 us / peak 2,805,668 bytes;
  borrowed-history plus latest-entry snapshot median 2.2 us / peak 636 bytes.
- Validation: `python3 -m py_compile examples/single_stage_optimization/alm_utils.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py tests/geo/test_alm_utils.py`
- Validation: `python3 -m ruff check examples/single_stage_optimization/alm_utils.py examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py tests/geo/test_alm_utils.py`
- Validation: `python3 -m pytest tests/geo/test_alm_utils.py -q`
- Validation: `python3 -m pytest tests/geo/test_single_stage_alm_integration.py -q`

## Other / Needs Benchmark

### O1. Reword `BoozerResidualExact` usage policy

Category: other
Scope: fork-local banana workflow
Impact: clarity
Effort: small

Current anchors:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
  :: `BoozerResidualExact` selection
- `examples/single_stage_optimization/banana_opt/boozer_residuals.py`
- `tests/geo/test_single_stage_example.py`

Problem:

The old TODO said to gate `BoozerResidualExact` to true final-only usage. The
current code gates it to `stage == "final"`, but final-stage refinement can
still be a search loop. The doc should say exactly that.

Fix direction:

- Make final-stage/refinement semantics explicit.
- Keep exact residual out of ordinary initial-stage search unless deliberately
  selected.
- Add coverage for the intended stage selection contract.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### O2. Enable frontier-lane parallelism only within independent groups

Category: other/performance
Scope: fork-local banana workflow
Impact: medium to high wall time
Effort: medium

Current anchors:

- `examples/single_stage_optimization/run_single_stage_frontier_campaign.py` ::
  `resolve_frontier_lane_warm_start`
- `examples/single_stage_optimization/run_single_stage_frontier_campaign.py` ::
  lane execution loop
- `examples/single_stage_optimization/banana_opt/frontier_engine_base.py`
- `examples/single_stage_optimization/banana_opt/frontier_campaign_reporting.py`
- `tests/geo/test_frontier_contracts.py`

Problem:

The current campaign loop is serial. Naive parallelism is unsafe when
`reuse_latest_certified` makes later lanes depend on earlier certified lanes,
but lanes with independent warm-start bases can run concurrently.

Fix direction:

- Partition lanes by warm-start dependency.
- Parallelize only independent groups.
- Merge archive/progress files at group boundaries and preserve atomic writes.

Impact measure:

- Wall time for a fixed small campaign.
- Archive/progress parity against serial execution.

Completion:

- [ ] Repro or baseline added
- [ ] Fix implemented
- [ ] Impact measured
- [ ] Validation command recorded

### O3. Keep excluded claims excluded unless new evidence appears

Category: other
Scope: documentation guardrail

Do not re-open these without current code evidence:

- Frontier campaign writes being corrupt-prone. Current paths use atomic
  `mkstemp` plus `os.replace` patterns.
- `Curve.kappa()` and `Curve.torsion()` lacking caching. Both are cached in
  `src/simsoptpp/curve.h`.
- `BiotSavart` returning stale fields after coil current changes. Existing
  cache invalidation covers current-change paths checked in the prior review.
- Frontier Pareto dominance lacking early exit. The per-candidate check already
  returns on first failing metric.
- Treating `scipy.linalg.lu_factor` / `lu_solve` as a drop-in replacement for
  exact Boozer adjoint factors. `forward_backward(...)` consumes `(P, L, U)`;
  changing factor representation is a contract change.

## Suggested Order

1. M1: add baseline impact harness.
2. C1: fix `SquaredFlux` invalidation.
3. C2: fix `CurveSurfaceDistance` invalidation.
4. C3: tighten Boozer first-use lifecycle.
5. P1: split optimizer hot path from diagnostics.
6. P2: remove exact sampled distances from search evaluation.
7. P3: tune `maxcor` with measured defaults.
8. M7: clean ALM history/copy discipline.
9. P4/P5: reuse Boozer solve/evaluation state.
10. M2/M3/M6: reduce generic derivative and field aggregation allocations.
11. M4/M5/P7/P8/O2: larger work only after fresh measurements justify it.

## Validation Commands To Record Per Patch

Minimum expected commands depend on the touched surface:

- Correctness in flux/distance objectives:
  `python3 -m pytest tests/objectives/test_fluxobjective.py tests/geo/test_curve_objectives.py -q`
- Boozer lifecycle or residual changes:
  `python3 -m pytest tests/geo/test_boozersurface.py tests/geo/test_surface_objectives.py -q`
- Banana ALM or single-stage workflow changes:
  `python3 -m pytest tests/geo/test_alm_utils.py tests/geo/test_single_stage_example.py -q`
- Frontier campaign changes:
  `python3 -m pytest tests/geo/test_frontier_contracts.py tests/geo/test_frontier_archive.py -q`
- Generic derivative/field aggregation changes:
  `python3 -m pytest tests/core/test_derivative.py tests/field/test_magneticfields.py tests/field/test_biotsavart.py -q`

Always also run `git diff --check` for documentation and code patches.
