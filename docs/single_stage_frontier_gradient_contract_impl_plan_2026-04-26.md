# Single-Stage Frontier Gradient Contract Implementation Plan

Date: 2026-04-26
Status: proposal, validated against current tree at `1f78c5b71`
Scope: `examples/single_stage_optimization/` frontier objective assembly, search rejection, campaign reporting, evaluator cache, and lane isolation

## Review Verdict

Approve the implementation direction with revisions.

The core diagnosis is correct: current `frontier` mode can add hardware and topology contract penalties to the scalar objective returned to L-BFGS-B without adding corresponding gradient terms. Since the optimizer path uses `jac=True`, this can feed L-BFGS-B an `(f, g)` pair where `f` and `g` describe different local functions.

The plan should be revised in nine ways:

1. Do not describe explicit rejected trials as true gradient-consistent objective evaluations.
2. Split optimizer-return fields from archive/ranking/certification fields.
3. Use signed constraint values and their gradients for differentiable hardware penalties; do not use `dual_update_values` unless a test proves it has the same sign and scale for the exact constraint class.
4. Treat threaded campaign lane execution as a possible shared-state correctness risk, not only a CPU oversubscription risk.
5. Make `apply_frontier_search_contract_penalties(...)` leave optimizer fields alone, or replace it with APIs whose names and payload shapes separate smooth optimizer mutation from artifact annotation.
6. Avoid modern `scipy.optimize.check_grad(direction="random", rng=...)` requirements unless the repo minimum SciPy version is raised; the current package metadata allows `scipy>=1.5.4`.
7. Add rejected-trial artifact tests so rejected topology/hardware trials cannot update incumbents, checkpoints, or archive-eligible records.
8. Make differentiable hardware penalties unit-aware and smoothing-scale-aware.
9. Split cache/atomic-write work from NSGA3 argument-validation work.

## Current Code Evidence

### Value-only contract penalties

`examples/single_stage_optimization/banana_opt/frontier_constraints.py` currently contains:

- `evaluate_frontier_hardware_search_penalty(...)`, which computes a scalar penalty from hardware violation ratios.
- `evaluate_frontier_topology_search_penalty(...)`, which computes a scalar penalty from topology deficit.
- `apply_frontier_search_contract_penalties(...)`, which adds those penalties into `frontier_rank_total` and then overwrites `total`.

That helper does not add a matching gradient.

### L-BFGS-B objective path

`examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` calls SciPy `minimize(..., jac=True, method='L-BFGS-B', ...)`. In the same search path, frontier topology and hardware failures call the value-only penalty helpers and then return `total` plus `grad` to the optimizer.

SciPy documents that `jac=True` means the objective callable returns both objective value and gradient, and L-BFGS-B uses limited-memory correction terms through `maxcor`.[^scipy-minimize][^scipy-lbfgsb]

### Existing tests lock in the wrong behavior

The current tests in `tests/geo/test_single_stage_example.py` assert that a frontier topology or hardware rejection changes `total` while `grad` remains unchanged. Those tests should be replaced with contract tests that forbid this behavior on accepted smooth objective paths.

### SciPy version constraint

The repo metadata currently allows `scipy>=1.5.4` in `pyproject.toml` and `requirements.txt`. Modern SciPy docs show `check_grad(direction="random", rng=...)`, but the plan must not require those newer keywords in tests unless the minimum SciPy version is raised.[^scipy-check-grad]

### Boozer trust is already the good pattern

`evaluate_frontier_trust_penalty(...)` returns both `penalty` and `grad`, and `annotate_frontier_search_eval(...)` adds both to the objective evaluation. This should remain the model for any smooth frontier penalty.

## Correct Optimizer Contract

There are two valid L-BFGS-B interaction modes.

### Mode 1: Smooth accepted objective evaluation

If a candidate is accepted as a smooth objective evaluation, the returned value and gradient must match:

```text
optimizer_total = F(x)
optimizer_grad  = grad F(x)
```

Any penalty included in `optimizer_total` must also contribute to `optimizer_grad`.

### Mode 2: Explicit rejected trial

If a candidate enters a nonsmooth, invalid, or topology/hardware-rejected state, the code may use the existing rejection/backtracking pattern:

```text
restore accepted state
return elevated rejection value
return last accepted gradient
record rejection metadata
```

This is not a true gradient of the elevated rejection value. It is a line-search rejection oracle. It is acceptable only when the trial is explicitly marked rejected and the accepted model state is restored before returning to the optimizer.

## Design Invariants

1. No accepted smooth L-BFGS-B objective path may mutate `total` without mutating `grad`.
2. Nonsmooth failures must be explicit rejected trials with restored accepted state and rejection metadata.
3. Topology and field-line survival are gate/certification signals, not differentiable objective terms.
4. Hardware penalties may enter the optimizer only when they come from signed residuals with matching gradients.
5. Optimizer return values must be separate from archive ranking and certification values.
6. Hardware constraint semantics must follow `banana_opt/hardware_constraint_schema.py`.
7. Campaign artifacts should preserve diagnostics without implying diagnostics were part of the differentiable objective.

## Field Contract

Keep `total` and `grad` as the canonical values returned to SciPy, but make the semantics explicit:

```text
total                  optimizer objective returned to SciPy
grad                   optimizer gradient returned to SciPy
optimizer_total        explicit alias of total for artifacts/debugging
optimizer_grad_norm    norm of grad for artifacts/debugging
archive_rank_total     archive/ranking scalar, never returned to SciPy
frontier_rank_total    compatibility alias for archive_rank_total during migration
frontier_trust_penalty smooth optimizer penalty, if enabled
frontier_contract_status diagnostic envelope, not an optimizer objective term
```

Do not write `archive_rank_total` or `frontier_rank_total` back into `total`.

Optimizer mutation APIs must make the same split:

```python
apply_smooth_frontier_optimizer_penalty(search_eval, penalty_payload)
annotate_frontier_contract_status(search_eval, contract_status)
compute_archive_rank_total(search_eval, diagnostics)
```

`apply_smooth_frontier_optimizer_penalty(...)` accepts only payloads that include both `penalty` and `grad`. `annotate_frontier_contract_status(...)` may attach value-only hardware/topology diagnostics, but it must not change `total` or `grad`. `compute_archive_rank_total(...)` may produce archive and recommendation scores, but those scores must not masquerade as optimizer objectives.

For final artifacts, separate:

```text
optimizer_feasible
hardware_feasible
topology_certified
boozer_trust_ok
archive_eligible
recommendation_eligible
```

## Impact Measurement Plan

Measure before/after impact in layers. The first correctness patch should not be judged primarily by frontier hypervolume or final iota gain; its main outcome is that L-BFGS-B no longer receives accepted smooth objective evaluations whose scalar value and gradient describe different functions.

### Measurement workloads

Use the same workload set before and after each patch:

1. Fake analytic objective harness
   - exercises weighted, Chebyshev, epsilon, and Boozer-trust frontier scalarizations
   - has cheap exact objective/gradient calls
   - used for strict finite-difference contract tests
2. Short live-style single-seed smoke run
   - uses the real frontier objective assembly path
   - keeps iteration/evaluation budget small
   - used to catch state rollback, callback, and artifact regressions
3. Canonical frontier campaign
   - uses the same seed artifacts, scalarization mode, lane budget, and worker policy before and after
   - used for archive, hypervolume, certification, wall-clock, memory, and resume behavior

Every run should record:

```text
git_sha
command
input_seed_artifacts
scalarization_mode
frontier_reference_mode
maxiter_or_eval_budget
lane_count
worker_policy
thread_environment
output_root
```

For live runs with multiple workers, set and record per-lane thread controls before NumPy/SciPy/native kernels initialize, for example `OMP_NUM_THREADS=1` unless the run intentionally allocates more cores per lane.

### Primary metrics

| Layer | Metric | Expected after Patch 1 |
| --- | --- | --- |
| Gradient contract | accepted smooth directional finite-difference relative error, p50/p95/max | finite and within the configured tolerance |
| Gradient contract | accepted evaluations where `total` changed without matching `grad` change | exactly `0` |
| Rejection safety | rejected trials that update accepted surface/model state | exactly `0` |
| Rejection safety | rejected trials that update `last_successful_eval`, incumbent, checkpoint, best-feasible state, or optimizer-feasible archive records | exactly `0` |
| Optimizer behavior | L-BFGS-B abnormal terminations and line-search failures | not worse, ideally lower |
| Optimizer behavior | `nfev / nit`, final gradient norm, accepted-step ratio | diagnostic; compare to baseline |
| Search accounting | reject counts by reason: topology, hardware, invalid surface, Boozer trust | explicit and stable enough to audit |
| Archive quality | nondominated feasible count, certified archive count, hypervolume, recommendation stability | tracked, but not a hard Patch 1 gate |
| Physics metrics | raw `iota`, `boozer_surface_volume`, `J_QS`, `J_Boozer`, hardware margins, topology survival | no silent semantic drift |
| Runtime and memory | wall time, objective-eval time, peak RSS, cache hits/misses, cache size, resume success | tracked; hard gates start in cache/lane patches |

### Summary row schema

Add or generate a summary row per run with at least:

```text
git_sha
run_id
mode
scalarization_mode
frontier_reference_mode
success
status_message
nit
nfev
njev
nfev_per_iteration
line_search_failure_count
accepted_step_count
objective_eval_count
accepted_step_ratio
rejected_topology_count
rejected_hardware_count
rejected_invalid_surface_count
rejected_boozer_trust_count
contract_violation_count
rejected_state_leak_count
fd_error_p50
fd_error_p95
fd_error_max
final_grad_norm
archive_member_count
nondominated_feasible_count
certified_archive_count
hypervolume
recommendation_count
final_iota
final_boozer_surface_volume
final_j_qs
final_j_boozer
min_hardware_margin
topology_survival_fraction
wall_seconds
peak_rss_mb
cache_hit_count
cache_miss_count
cache_eviction_count
resume_success
corrupt_json_artifact_count
```

The extraction script should treat missing fields as missing data, not as zero, unless the field is explicitly defined as a count emitted by the run. Zero and missing have different meanings for rejected trials, cache counters, and archive certification.

### Patch-specific gates

Patch 1 correctness gates:

- `contract_violation_count == 0`
- `rejected_state_leak_count == 0`
- rejected topology/hardware trials are labeled rejected in callbacks and artifacts
- `frontier_rank_total` or `archive_rank_total` is never returned to SciPy as `total`
- fake objective finite-difference errors meet strict tolerances
- live-style smoke finite-difference errors meet the looser configured tolerance
- line-search failure count and abnormal termination count are not materially worse than baseline

Patch 2 differentiable hardware gates:

- hardware penalty helper returns both penalty and gradient
- finite differences match the analytic hardware penalty gradient
- missing signed residuals or gradients cause helper failure or hard rejection, never value-only fallback
- hardware feasibility improves or remains explainably unchanged at comparable optimization budget

Patch 3 reporting gates:

- `optimizer_total`, `archive_rank_total`, certification status, and recommendation eligibility are separately emitted
- `archive_rank_total != optimizer_total` is allowed and tested
- final reports distinguish optimizer acceptance from archive ranking and final certification

Patch 4 scalarization/normalization gates:

- compare nondominated feasible count, certified archive count, hypervolume, and recommendation stability
- require raw metric reporting alongside normalized metric reporting
- do not claim scalarization improvement from a single lane; use the canonical campaign workload

Patch 5 cache/atomic-write gates:

- peak RSS scales with cache capacity, not total evaluation count
- cache hit/miss/eviction counters are emitted
- interrupted-write tests leave either old valid JSON or new valid JSON
- resume succeeds from the written artifacts

Patch 6 NSGA3 gates:

- invalid reference mode, invalid objective count, invalid reference-direction shape, and unsupported banana-current mode fail before evaluator construction
- population size and reference-direction count policy is visible in the manifest

Patch 7 lane-isolation gates:

- worker policy is recorded
- live multi-lane runs use subprocess/process isolation by default
- fake lanes that mutate lane-local state produce distinct lane artifacts with no cross-lane leakage
- multi-worker and single-worker runs with the same seed contract produce comparable archive accounting

### Interpretation rules

Some metrics may move in the "wrong" direction after the first correctness patch and still indicate an improvement. Accepted-step count may decrease if topology or hardware failures stop being treated as accepted penalized smooth evaluations. That is acceptable when the rejected-trial counters increase, accepted state is restored, and optimizer-feasible archive records remain clean.

Do not require immediate hypervolume improvement for Patch 1. The expected benefit is optimizer trustworthiness: fewer inconsistent curvature updates, cleaner line-search decisions, and artifacts that state why a candidate was accepted, rejected, ranked, or certified.

## Implementation Plan

### Phase 0: Lock the current defect with tests

Update the frontier rejection tests before changing production code.

Replace tests that assert:

```text
total changed
grad unchanged
```

with tests that assert:

```text
accepted smooth path: total and grad both change when a smooth penalty is active
rejected trial path: rejection metadata is present and accepted state is restored
```

Required test cases:

- Boozer trust penalty changes both `total` and `grad`.
- Frontier topology failure no longer applies a value-only optimizer penalty.
- Frontier hardware failure no longer applies a value-only optimizer penalty.
- `apply_frontier_search_contract_penalties` cannot mutate optimizer `total`.
- Rejected topology trial restores surface/model state.
- Rejected hardware trial restores surface/model state.
- Rejected trial does not update `last_successful_eval`.
- Rejected trial does not update incumbent, checkpoint, or best-feasible state.
- Rejected trial does not enter the Pareto archive as optimizer-feasible.
- Callback display labels rejected trials as rejected, not as accepted objective evaluations.
- Fake weighted frontier objective passes finite-difference gradient check.
- Fake Chebyshev frontier objective passes finite-difference gradient check.
- Fake epsilon frontier objective passes finite-difference gradient check.
- Fake Boozer-trust objective passes finite-difference gradient check.

Use full-coordinate finite differences for fake analytic objectives. Use a repo-local central directional finite-difference helper for expensive live-style objectives. Do not require SciPy `check_grad(direction="random", rng=...)` while the package minimum remains `scipy>=1.5.4`.

Suggested helper:

```python
def directional_fd_error(fun, grad, x, direction, eps):
    direction = direction / np.linalg.norm(direction)
    fd = (fun(x + eps * direction) - fun(x - eps * direction)) / (2.0 * eps)
    ad = float(np.dot(grad(x), direction))
    return abs(fd - ad) / max(1.0, abs(fd), abs(ad))
```

Target tolerances:

- fake analytic objectives: relative/directional error around `1e-6` to `1e-8`
- live scientific objective smoke tests: looser, usually `1e-4` to `1e-3`, depending on objective scale and solver noise

### Phase 1: Stop topology from entering optimizer total

Change the frontier topology failure branch in `evaluate_search_step`.

Current behavior:

```text
topology failure
-> evaluate_frontier_topology_search_penalty
-> apply_frontier_search_contract_penalties
-> total changes, grad unchanged
```

New behavior:

```text
topology broken
-> hard invalidation, as today

topology failed
-> explicit rejected trial
-> restore accepted surface/model state
-> return rejection value plus last accepted gradient
-> record topology status and rejection metadata
```

Topology status should remain available for:

- callbacks
- final results
- frontier archive certification
- recommendation filtering
- campaign progress summaries

Topology status must not enter `total` unless a future differentiable surrogate exists and has tested gradients.

### Phase 2: Stop hardware from using value-only optimizer penalties

Change the frontier hardware failure branch in `evaluate_search_step`.

First correctness patch:

```text
if hardware reject and no differentiable hardware penalty payload is available:
    explicit rejected trial
```

Do not try to synthesize a gradient from violation ratios. Violation ratios are useful diagnostics, but they are not enough to define a differentiable objective.

Continue to record:

- `frontier_hardware_violation_ratios`
- `frontier_hardware_max_violation_ratio`
- hardware rejection reason
- hardware feasible/infeasible status

These should feed archive/ranking/certification only, not the L-BFGS-B return objective.

### Phase 3: Add differentiable hardware penalty helper

Add a narrow helper for smooth hardware penalties after Phase 2 is green.

Suggested module:

```text
examples/single_stage_optimization/banana_opt/frontier_hardware_penalties.py
```

Suggested function:

```python
def frontier_hardware_penalty_from_signed_constraints(
    search_eval,
    *,
    weights,
    smoothing,
):
    ...
```

Input contract:

- `constraint_names`
- `constraint_values`
- `constraint_grads`
- optional `feasibility_values` for reporting only

Do not use `dual_update_values` as the source of signed residuals unless a test proves it matches `constraint_values` for that exact formulation.

Normalize each constraint explicitly. Coil spacing, surface spacing, curvature, and current limits have different units and magnitudes, so the helper must not silently combine raw meters, inverse meters, and amperes.

Preferred internal payload:

```python
{
    "name": "coil_coil_spacing",
    "value": signed_value,
    "grad": signed_grad,
    "unit": "m",
    "scale": smoothing_scale,
    "weight": penalty_weight,
}
```

Constraint sign convention:

```text
constraint_value <= 0.0 means feasible
constraint_value >  0.0 means violated
```

Penalty:

```text
P_i = w_i * smooth_hinge(constraint_value_i / smoothing_i)^2
dP = sum_i dP_i/dconstraint_i * constraint_grad_i
```

Pick one hinge family and document the semantics.

Option A:

```text
P(v) = w * max(0, v / s)^2
```

This is simple and exactly zero when feasible, but it is not twice differentiable at the threshold. L-BFGS-B usually tolerates this piecewise-smooth behavior, but finite-difference tests should avoid `v == 0`.

Option B:

```text
P(v) = w * softplus(v / s)^2
```

This is smooth, but it gives small nonzero penalty and gradient even for feasible constraints. That behavior must be intentional because it nudges feasible candidates away from the boundary.

Required checks through tests:

- penalty is zero or near zero for feasible constraints
- penalty increases when a fake signed constraint crosses from feasible to violated
- penalty gradient shape matches `grad`
- penalty gradient is finite
- finite differences agree with the analytic penalty gradient
- unsupported constraint names fail the test/helper contract, not silently ignored
- missing signed gradients cause helper failure or hard rejection, never violation-ratio fallback

### Phase 4: Respect hardware schema exactly

Use `banana_opt/hardware_constraint_schema.py` as the SSOT.

Current schema categories:

- `coil_coil_spacing`: penalty, ALM, artifact
- `coil_surface_spacing`: penalty, ALM, artifact
- `surface_vessel_spacing`: penalty, ALM, artifact
- `max_curvature`: penalty, ALM, artifact
- `coil_length`: ALM, artifact
- `banana_current`: penalty, ALM, artifact, traversal forbidden
- `tf_current`: artifact only

Implementation rules:

- Do not promote `coil_length` into frontier search penalties unless the schema changes.
- Do not promote `tf_current` into frontier search penalties.
- Treat banana current carefully because traversal is forbidden; prefer bounds or ALM semantics over post-hoc frontier penalty injection.
- Do not add broad formula terms just because gradients are available.

### Phase 5: Split reporting and optimizer semantics

Refactor frontier result assembly so optimizer fields and archive/report fields cannot be confused.

Optimizer-facing fields:

```text
total
grad
optimizer_total
optimizer_grad_norm
```

Archive/ranking fields:

```text
archive_rank_total
frontier_rank_total
pareto_objective_vector
dominance_rank
hypervolume_contribution
```

Certification fields:

```text
frontier_hardware_status
frontier_topology_status
frontier_boozer_trust_status
frontier_certification_status
optimizer_feasible
archive_eligible
recommendation_eligible
```

Migration rule:

- Preserve `frontier_rank_total` as an artifact/report compatibility field only if current repo consumers still read it.
- Redefine it as an archive/ranking scalar, not as the optimizer return value.
- Add tests for any consumer that previously read `frontier_rank_total` to decide optimizer behavior.

### Phase 6: Update callbacks, final results, and campaign archives

Search callbacks and results currently read `frontier_rank_total` in several places. Update these call sites to choose the correct field:

- optimizer convergence and SciPy callback displays: `optimizer_total`
- frontier archive ranking: `archive_rank_total`
- final recommendation: certified Pareto metrics plus recommendation policy
- diagnostic summaries: include both optimizer and archive fields with labels

Final artifacts should make it impossible to confuse:

```text
why the optimizer accepted a step
why the archive ranked a candidate
why certification accepted or rejected a candidate
```

### Phase 7: Scalarization defaults after correctness

Do not change campaign scalarization defaults in the same patch as gradient-contract correction.

After Phases 0-6 pass, promote better frontier exploration:

1. Keep `shared_seed_relative_frontier_v2` as an explicit legacy/debug mode.
2. Make achievement Chebyshev the recommended local campaign mode.
3. Use full-simplex 4-objective directions over:
   - iota
   - Boozer-surface volume
   - QA error
   - Boozer residual
4. Keep epsilon-constraint campaigns for targeted physics envelopes.

`pymoo` documents NSGA-III as reference-direction based, and its reference-direction docs cover unit-simplex directions, Das-Dennis structured directions, and the combinatorial growth of reference points.[^pymoo-nsga3][^pymoo-refdirs]

### Phase 8: Robust normalization artifact

Keep seed-relative metrics in artifacts, but add an explicit normalization manifest for campaigns that opt into broader comparison.

Add:

```text
frontier_normalization_manifest.json
```

Fields:

```text
schema_version
normalization_kind
seed_metrics_raw
pilot_metrics_raw
ideal_values
nadir_values
median_values
floors
caps
physical_scales
created_from_lane_ids
```

Allowed kinds:

- `seed_relative`
- `explicit_ideal_nadir`
- `pilot_robust`

Rules:

- Normalization must be frozen before lane optimization begins.
- Do not derive normalization adaptively from the live archive.
- Archive members record both raw and normalized metrics.

### Phase 9: Evaluator cache and atomic writes

Refactor evaluator and NSGA3 artifact writes to use one shared atomic JSON writer.

Suggested helper:

```text
examples/single_stage_optimization/banana_opt/artifact_io.py
```

Suggested API:

```python
def write_json_atomic(path, payload):
    ...
```

Implementation requirements:

- create temp file in the same directory as the target
- write JSON with deterministic formatting, including `sort_keys=True`
- flush the file object
- `os.fsync(f.fileno())` for important checkpoints and cache entries
- close the file object
- replace with `os.replace(temp, path)`
- optionally `fsync` the parent directory on POSIX for crash durability after the replace
- remove temp path on failure if it still exists

Python documents `os.replace` as overwriting an existing file and making the rename atomic when successful, with failure possible across filesystems.[^python-os-replace]

Apply the helper to:

- evaluator spec writes
- evaluator disk cache writes
- NSGA3 population checkpoint writes
- NSGA3 generation history writes
- any frontier campaign writer still using direct `write_text`

Add an LRU cap to `SingleStageFrontierEvaluator._cache`.

Required tests:

- cache evicts oldest entries after capacity is reached
- cache hit/miss counters remain correct
- atomic writer leaves valid old or valid new JSON after simulated interrupted write
- disk cache schema mismatch is ignored without mutating in-memory cache

### Phase 10: Early NSGA3 validation

Move unsupported engine-argument checks before evaluator/runtime allocation.

Validate before launch:

- `frontier_engine == "nsga3"` requires `frontier_reference_mode == "achievement_chebyshev_full_simplex_v1"`
- `--single-stage-banana-current-mode=independent` is unsupported for evaluator-backed engines until the evaluator contract supports vector banana-current state
- `pymoo` availability
- population size and reference direction count consistency
- `n_objectives == 4`
- `ref_dirs.shape[1] == 4`
- `population_size >= len(ref_dirs)` unless a different policy is intentionally configured and documented
- reference direction count is capped or warned when accidentally huge

Required tests:

- invalid NSGA3 reference mode fails before evaluator construction
- independent banana-current mode fails before evaluator construction
- missing `pymoo` reports the dependency clearly
- reference-direction shape mismatch fails before evaluator construction
- accidental huge reference-direction count fails or requires an explicit override

### Phase 11: Lane isolation and thread policy

Current campaign lane parallelism uses `ThreadPoolExecutor`. Python documents that `ThreadPoolExecutor` runs calls in worker threads, while `ProcessPoolExecutor` uses separate processes and requires picklable callables/arguments.[^python-concurrent-futures] That distinction matters here: live single-stage optimization uses module globals and mutable SIMSOPT objects, so threaded lane execution risks cross-lane state contamination unless isolation is proven.

Concrete decision:

- Do not run live optimization lanes concurrently in threads by default.
- For `frontier_lane_workers > 1`, run lanes in isolated subprocesses with one output directory per lane.
- Prefer explicit subprocess-per-lane execution over `ProcessPoolExecutor` for the live optimizer path, because SIMSOPT objects and module-level runtime state may not be cleanly picklable.
- Keep threaded execution only for pure post-processing/reporting tasks that do not touch live optimizer module state.

Subprocess lane requirements:

- per-lane environment is created before NumPy/SciPy/native kernels initialize
- default `OMP_NUM_THREADS=1` and related BLAS/OpenMP knobs when workers > 1, unless explicitly overridden
- effective thread settings are recorded in the campaign manifest
- lane stdout/stderr/logs remain lane-local
- failures are captured as lane result records, not shared exceptions that corrupt other lanes

Acceptance test:

- two fake lanes mutate lane-local state and produce distinct artifacts with no cross-lane leakage
- worker thread/process policy is recorded in campaign progress

### Phase 12: Physics naming and objective clarity

Rename or add explicit metadata for volume:

```text
boozer_surface_volume
```

Do not call this plasma volume or LCFS volume unless the code actually computes that quantity.

For iota:

- keep higher-iota reward available
- add optional iota band constraints only after gradient tests exist
- add resonance-avoidance terms only if the objective is differentiable or explicitly handled as certification/gating

This phase is documentation and artifact semantics unless new differentiable physics terms are introduced.

## Patch Order

### Patch 1: Correctness only

Files likely touched:

- `banana_opt/frontier_constraints.py`
- `SINGLE_STAGE/single_stage_banana_example.py`
- `tests/geo/test_frontier_constraints.py`
- `tests/geo/test_single_stage_example.py`

Exit criteria:

- no accepted smooth frontier hardware/topology path can change optimizer `total` without changing `grad`
- topology failures are explicit rejected trials
- hardware failures without differentiable payload are explicit rejected trials
- Boozer trust still changes both value and gradient
- rejected trials restore state and cannot update incumbent/checkpoint/archive-feasible state
- focused gradient tests pass

### Patch 2: Differentiable hardware helper

Files likely touched:

- new `banana_opt/frontier_hardware_penalties.py`
- `banana_opt/single_stage_objectives.py`
- `banana_opt/hardware_constraint_schema.py` only if schema changes are truly needed
- focused tests

Exit criteria:

- hardware signed residual penalties return penalty and gradient
- finite differences match the analytic penalty gradient
- schema categories are respected
- no use of `dual_update_values` without explicit test coverage

### Patch 3: Reporting separation

Files likely touched:

- `SINGLE_STAGE/single_stage_banana_example.py`
- `banana_opt/frontier_campaign_reporting.py`
- `banana_opt/frontier_archive.py`
- `banana_opt/frontier_recommendation.py`
- campaign/report tests

Exit criteria:

- optimizer, archive, and certification fields are separate in artifacts
- old `frontier_rank_total` consumers either move to new fields or are tested as compatibility-only report consumers
- final reports explain optimizer feasibility separately from archive eligibility

### Patch 4: Scalarization and normalization

Files likely touched:

- `banana_opt/frontier_scalarization.py`
- `run_single_stage_frontier_campaign.py`
- `banana_opt/frontier_engine_multilane_local.py`
- docs/tests

Exit criteria:

- achievement Chebyshev is the recommended campaign default
- full-simplex 4-objective directions are easy to invoke
- normalization manifest is emitted and frozen before lane search
- seed-relative metrics remain available for matched comparisons

### Patch 5: Cache and atomic writes

Files likely touched:

- `banana_opt/frontier_evaluator.py`
- new or existing artifact I/O helper
- evaluator/cache tests

Exit criteria:

- evaluator memory cache is bounded
- JSON writes are atomic
- important checkpoint writes use flush and `fsync`
- cache schema and counters are tested

### Patch 6: NSGA3 validation

Files likely touched:

- `banana_opt/frontier_engine_nsga3.py`
- `run_single_stage_frontier_campaign.py`
- evaluator/NSGA3 tests

Exit criteria:

- unsupported NSGA3 args fail before evaluator construction
- reference-direction objective shape is validated
- population size and reference-direction count policy is validated
- missing `pymoo` reports the dependency clearly

### Patch 7: Lane isolation

Files likely touched:

- `run_single_stage_frontier_campaign.py`
- `banana_opt/frontier_engine_base.py`
- campaign execution tests

Exit criteria:

- live parallel lanes use process/subprocess isolation by default
- thread policy is recorded
- no shared module-global optimizer state is used concurrently by default

### Patch 8: Physics naming

Files likely touched:

- result/artifact writers
- campaign reports
- docs/tests

Exit criteria:

- `boozer_surface_volume` is explicit
- reports do not imply LCFS/plasma volume unless computed
- iota reward semantics are documented as reward, band, or certification depending on selected mode

## Validation Commands

Use the repo environment with all scientific dependencies installed.

```bash
python3 -m py_compile \
  examples/single_stage_optimization/banana_opt/frontier_constraints.py \
  examples/single_stage_optimization/banana_opt/single_stage_objectives.py \
  examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py
```

```bash
python3 -m pytest \
  tests/geo/test_frontier_constraints.py \
  tests/geo/test_single_stage_example.py \
  -k "frontier and (trust or hardware or topology or gradient or scalarization)" \
  -q
```

```bash
python3 -m pytest \
  tests/geo/test_single_stage_example.py \
  -k "maxcor or frontier_rank_total or frontier_archive or frontier_campaign" \
  -q
```

```bash
git diff --check
```

Local note from this validation pass: the current shell cannot run the targeted pytest slice because importing `simsopt` fails on missing `monty`.

## Acceptance Criteria

The correctness work is accepted only when:

- no accepted smooth frontier path can mutate optimizer `total` without mutating `grad`
- rejected topology/hardware trials restore accepted state and record rejection metadata
- rejected topology/hardware trials do not update `last_successful_eval`, incumbent, checkpoint, best-feasible state, or optimizer-feasible archive membership
- Boozer trust remains smooth and gradient-carrying
- hardware gradients come only from signed residuals with matching `constraint_grads`
- topology remains gate/certification unless a differentiable surrogate is introduced
- optimizer fields and archive/certification fields are separate
- current tests no longer encode value-only objective penalties as expected behavior
- `frontier_rank_total` is never used as the SciPy return value
- `archive_rank_total != optimizer_total` is allowed and tested
- missing signed hardware gradients cause hard rejection or helper failure, never value-only fallback
- tests avoid modern `check_grad` keyword assumptions unless the minimum SciPy version is raised

The broader frontier hardening is accepted only when:

- campaign reporting preserves hardware/topology diagnostics without confusing them with optimizer objective terms
- scalarization defaults are changed only after the correctness patch is green
- cache writes are atomic through one shared writer
- evaluator cache memory is bounded
- NSGA3 invalid argument combinations and invalid reference-direction shapes fail before evaluator construction
- live parallel lane execution is process/subprocess isolated by default
- final reports explicitly label `boozer_surface_volume` and do not imply plasma or LCFS volume unless that quantity is computed

## Main Risks

### Risk 1: Overbundling

Correctness, scalarization, cache, concurrency, and physics naming should not land in one patch. The first patch should only remove value-only optimizer penalties and add contract tests.

### Risk 2: Misusing ALM fields

`dual_update_values` may not always be the physical signed residual the frontier penalty wants. Use `constraint_values` plus `constraint_grads` unless a test proves equivalence for the exact formulation.

### Risk 3: Threaded lane cross-talk

Thread limiting does not prove optimizer state isolation. Live optimization lanes should be process/subprocess isolated unless a separate audit proves module globals and SIMSOPT objects are lane-local.

### Risk 4: Artifact compatibility drift

Downstream reports may read `frontier_rank_total`. Keep it as a reporting/archive field during migration, but do not let it control optimizer returns.

### Risk 5: Minimum-version drift in tests

The plan should not require modern SciPy `check_grad` keyword behavior while repo metadata allows `scipy>=1.5.4`. Use local finite-difference helpers or raise the minimum version explicitly.

## References

[^scipy-minimize]: SciPy `minimize` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
[^scipy-lbfgsb]: SciPy L-BFGS-B options documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
[^scipy-check-grad]: SciPy `check_grad` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html
[^python-concurrent-futures]: Python `concurrent.futures` documentation: https://docs.python.org/3/library/concurrent.futures.html
[^python-os-replace]: Python `os.replace` documentation: https://docs.python.org/3/library/os.html#os.replace
[^pymoo-nsga3]: pymoo NSGA-III documentation: https://pymoo.org/algorithms/moo/nsga3.html
[^pymoo-refdirs]: pymoo reference directions documentation: https://pymoo.org/misc/reference_directions.html
