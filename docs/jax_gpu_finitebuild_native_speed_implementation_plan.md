# JAX GPU Finite-Build Stage-II Native-Speed Implementation Plan

**Status:** Draft
**Last updated:** 2026-08-17

## Purpose

Make the native-scale JAX GPU finite-build Stage-II example faster than the
best measured, properly configured SIMSOPT/simsoptpp CPU lane while preserving
the FP64 objective, discretization, initial state, penalty terms, and endpoint
physics. This plan targets one credible workload and closes the route if its
bounded crossover is negative. Any win is narrowly a warm, repeated-workload
claim; fresh-cache startup remains a separately reported result.

## Goals

- Make `examples/jax/3_Advanced/stage_two_optimization_finitebuild.py` faster
  than the fastest qualifying native CPU lane on both synchronized warm solve
  time and warm persistent-cache subprocess wall time.
- Preserve the native 32x32 surface, four base coils, six filaments per pack,
  75 curve quadrature points, objective scale, penalty weights, solver
  tolerances, and FP64 math.
- Keep every repeated objective/gradient evaluation and the fused L-BFGS loop
  on the GPU. Permit only the optimizer's explicit terminal-result publication
  and the final packed diagnostic publication; prohibit per-step host
  observations.
- Compare no-callback runs at the smallest independently selected iteration
  budget that clears one frozen native-derived quality contract.
- Produce one same-round, hash-bound native/GPU receipt. A speed claim is valid
  only for the exact workload, host, GPU, source, runtime, and cache policy in
  that receipt.

## Non-Goals

- Rewriting the general optimizer stack or adding Optax, Optimistix, Lineax,
  Triton, or another dependency.
- Changing the shipped native example, physical objective, resolution, coil
  topology, tolerances, or scientific gates to manufacture a speedup.
- Adding a public API or a runtime knob for the selected L-BFGS history.
- Changing `serial_solve_jax`; the finite-build workflow will use the existing
  typed `simsopt_jax.solve.dispatch.minimize()` entry point so its numerical
  result and example publication remain separate from `serial_solve_jax`'s
  mandatory objective-log side effect.
- Retrying `stage_two_optimization_minimal.py`: its GPU value/gradient kernel
  was 2.415x faster, but tested LM and device-BFGS workflows still lost to a
  properly configured native solve.
- Optimizing stochastic Stage II, tracing, QFM, RCLS, VMEC, or single-stage
  Boozer in this change. Stochastic Stage II is a follow-on only after this
  plan produces a qualified win.

## Current Context

- Source modules under `src/`, `examples/`, and `benchmarks/` are unchanged
  since commit `41dfda284e3ebdf79d9bb32582c21a9a5dc11d9d`; later commits are
  docs-only.
- The matched finite-build workload evaluates 96 symmetry-expanded filaments
  against 1024 surface points with 75 quadrature points, approximately 7.37
  million source-target interactions per field evaluation.
- The JAX example already selects the fused device-resident
  `SIMSOPT_LBFGSB` driver, but it requests up to 400 correction pairs.
- `make_finite_build_stage_two_objective()` calls
  `_finite_build_penalties()`, which computes `minimum_clearance` during every
  objective evaluation even though the objective discards it. Minimum
  clearance is an endpoint diagnostic; the distance penalty remains an
  objective term and must not change.
- `serial_solve_jax` explicitly publishes its terminal result and writes an
  initial/final objective log. An outer `jax.transfer_guard("disallow")` rejects
  implicit transfers but does not mean zero explicit transfers. The relevant
  performance contract is therefore zero host observations during optimizer
  advance, followed by bounded endpoint publication.
- The public example and parity case independently assemble the same JAX solve,
  so optimizer policy can drift. Existing tests also inspect source/AST shape;
  they do not behaviorally prove workflow routing or transfer placement.
- Existing bounded parity proves scientific agreement at reduced scale, but no
  native-default finite-build CPU/GPU timing receipt exists.

## Design

This is a Tier-2 change: it modifies one objective module, adds one internal
example-workflow module, and updates two callers. Three designs were evaluated:

1. Patch the public example only. Rejected because the parity JAX lane would
   retain a different optimizer policy.
2. Wrap `serial_solve_jax`. Rejected because its required objective-log
   publication obscures the numerical timing boundary and performs transfers
   unrelated to the finite-build result.
3. Add a finite-build-only internal workflow over the existing typed
   `simsopt_jax.solve.dispatch.minimize()` entry point. Selected because it
   centralizes fixed optimizer policy, preserves the general solver API,
   exposes a stable prepared program for warm reuse, and keeps publication
   explicit.

The internal module will contain only two frozen, typed records and two
functions:

- `prepare_finite_build_stage_two(...) -> FiniteBuildStageTwoProgram` constructs
  stable objective/value-gradient and diagnostic callables once for a fixed
  shape.
- `solve_finite_build_stage_two(program, initial_parameters, *, driver,
  max_steps) -> FiniteBuildStageTwoResult` runs the typed optimizer with no
  callback and returns the optimizer metadata plus packed initial/final device
  diagnostics needed by the callers.

The module is imported directly by the public example and parity case; it is
not re-exported from `simsopt_jax.examples.__init__`. The selected L-BFGS
history is one private constant. BFGS parity mode retains its existing line
search and tolerance policy.

## Frozen Scientific and Timing Contracts

Before performance tuning, the benchmark writes an immutable gate-definition
artifact from an untimed native reference run using the shipped 400-step,
400-history formulation. The truncated reference anchor is captured from the
same run's own trajectory: the stopping callback records the first accepted
iterate whose objective clears the target, and the anchor's full endpoint
state is evaluated at that captured iterate. (An earlier same-day revision
derived the anchor from a separate truncated replay; the measured ~1%
cross-process OpenMP-reduction fork makes any replay a different trajectory,
so the anchor must come from the reference run itself.) Its hash is included
in every later leg.

*Amended 2026-08-17, before any timed configuration was ranked or selected:
the original two-sided endpoint bands and converged-norm gradient cap were
revised after review showed they false-reject endpoints that converge better
than the reference and truncated endpoints whose gradient norms are
legitimately above the fully converged norm (the V260/rho-floor false-reject
class). No selection or timed evidence existed under the earlier clauses.*

The quality contract contains:

- the exact input/configuration/source fingerprints (git commit plus SHA-256
  of the objective module, the parity case, and the benchmark) and the
  initial parameter vector;
- a target objective equal to `1.001` times the converged native reference
  endpoint;
- the truncated reference endpoint — the converged reference formulation
  replayed to its own first qualifying iteration — as the like-for-like
  comparison anchor for every measured, budget-truncated lane;
- finite solution, objective, full gradient, and diagnostic arrays;
- objective improvement from the common initial state;
- one-sided quality caps: endpoint objective no larger than the target;
  squared flux, length penalty, and distance penalty each no larger than
  `atol=1e-9` plus `1.05` times the truncated reference value (converging
  better than the reference is never a failure);
- two-sided geometry bands: minimum clearance and each coil length within
  `rtol=5e-2, atol=1e-9` of the truncated reference endpoint (a different
  geometry regime is a failure in either direction);
- endpoint gradient infinity norm no larger than `1.05` times the truncated
  reference norm (with a `1e-12` denominator floor); and
- positive minimum clearance.

Every GPU endpoint is additionally re-evaluated through the independent
native SIMSOPT/simsoptpp evaluator at the published solution vector, and the
gate clauses are applied to that native re-evaluation; equivalence is never
mediated by the JAX lane's own evaluator.

Gate derivation is complete before any timed configuration is ranked. A timed
leg that misses any clause is ineligible, regardless of speed.

Timing regions are named and non-overlapping:

- `construction_seconds`: host surface/coil construction and immutable device
  staging;
- `first_execute_seconds`: first synchronized objective/gradient or solve,
  including compilation;
- `warm_value_grad_seconds`: repeated synchronized calls of the same prepared
  value/gradient program;
- `warm_solve_seconds`: no-callback optimizer call after the exact shape and
  program identity have been warmed, including its terminal result read-back;
- `endpoint_publication_seconds`: final diagnostic evaluation, synchronization,
  and packed host publication; and
- `process_wall_seconds`: subprocess launch through validated JSON publication.

JAX dispatch is asynchronous, so every timed JAX boundary ends in
`jax.block_until_ready()` or the typed optimizer's synchronized terminal-result
conversion. Warm same-process timing and warm persistent-cache subprocess
timing are different metrics and are never relabeled as each other.

## Implementation Plan

1. Freeze the independent baseline and benchmark contract.
   - [ ] Add `benchmarks/stage_two_finitebuild_native_gpu.py`. Reuse the parity
     case's frozen input bundle and native evaluator construction instead of
     creating a second physics specification.
   - [ ] Write each run beneath
     `.artifacts/stage_two_finitebuild_native_gpu/<run-id>/`, with an atomic
     terminal manifest that enumerates every expected raw leg and its SHA-256.
   - [ ] Record value, full gradient, and all diagnostics at the initial state
     and two deterministic perturbed states for the independent native and JAX
     evaluators before changing the hot path.
   - [ ] Emit raw JSON rows for every phase and repetition. Bind git status and
     commit, changed-file hashes, interpreter and package paths/versions,
     `simsoptpp` binary hash, CPU affinity, OpenMP environment, JAX flags,
     platform/device identity, FP64 state, cache identity/state, peak host RSS,
     peak GPU memory, timestamps, and competing GPU processes.
   - [ ] Configure the JAX persistent compilation cache before the first JAX
     operation. Use a fresh isolated cache root for cold legs; prime one
     source/runtime/shape-bound root before warm persistent-cache subprocess
     legs and record its before/after digest.
   - [ ] Make the validator recompute gate eligibility, medians, paired ratios,
     and verdict from raw rows. Missing, nonfinite, mismatched, or partial rows
     make the result `NOT_PRODUCED`, not a win or loss.
   - [ ] Add
     `tests/benchmarks/test_stage_two_finitebuild_native_gpu.py` with inline
     temporary raw rows that exercise WIN, `CLOSED_BOUNDED_NEGATIVE`, and
     `NOT_PRODUCED`; do not add a second benchmark implementation or a golden
     timing fixture.

2. Remove only the discarded hot-path calculation and prove identity.
   - [ ] Before editing the objective, lower the full-scale
     `jax.jit(jax.value_and_grad(objective))` at the frozen initial state and
     retain both `lowered.as_text()` (StableHLO) and
     `lowered.compile().as_text()` (optimized executable HLO), plus
     `compiled.cost_analysis()`. Run this under the strict CUDA/FP64 environment
     defined in Validation, hash the artifacts, and bind them to the exact
     pre-refactor source snapshot.
   - [ ] Refactor
     `src/simsopt_jax_adapters/objectives/finite_build_stage_two.py` so the
     repeated objective computes squared flux, length penalty, and distance
     penalty but not `pairwise_min_distance_pure`. Keep minimum clearance in
     `finite_build_stage_two_diagnostics()`.
   - [ ] Repeat the same lowering after the refactor and diff the StableHLO,
     optimized HLO operation/fusion census, and cost analysis before running a
     timing canary. Bind the post-refactor artifacts to their own source
     snapshot while holding runtime, configuration, and inputs fixed. The null
     hypothesis is that XLA already dead-code-eliminates the discarded clearance
     branch. If the source/StableHLO changes but the normalized optimized-HLO
     operation/fusion census and cost analysis do not, classify this lever
     `DCE_NULL`; raw HLO text identity is not the equality predicate. Do not
     credit a null lever with a speedup; the measured canary below still decides
     whether the wider workflow has enough headroom to proceed.
   - [ ] Parameterize
     `tests/jax/objectives/test_finite_build_stage_two.py` over the frozen
     initial and two perturbed states. Compare the refactored JAX objective and
     full autodiff gradient directly with the independent native
     SIMSOPT/simsoptpp evaluator; do not use values generated only by the new
     JAX implementation as the oracle.
   - [ ] Compare JAX endpoint minimum clearance with native
     `CurveCurveDistance.shortest_distance()` and retain the distance-penalty
     value/gradient parity checks.
   - [ ] Run the warm value/gradient canary against native at
     `OMP_NUM_THREADS=2,4,8,16,32,48`. Stop with `CLOSED_BOUNDED_NEGATIVE` before
     workflow changes unless the refactored GPU kernel is at least `1.10x`
     faster than the best native kernel and all scientific checks pass.

3. Select one fixed optimizer policy with a bounded canary.
   - [ ] Measure the full native matrix of OpenMP counts `2,4,8,16,32,48` and
     L-BFGS histories `10,20,40,400`; measure JAX histories `10,20,40`. Record
     one additional untimed shipped-default (`OMP_NUM_THREADS` unset)
     disclosure lane, reported separately and never used as the denominator. Use three
     round-robin repetitions for selection and reserve five fresh repetitions
     for the final verdict.
   - [ ] *Amended 2026-08-17, before any selection evidence existed: the
     original trace-calibrate-then-replay protocol is unsound on this host.
     Measured with same-environment, same-affinity A/B pairs: single-threaded
     native solves are bitwise reproducible over 200 iterations, while
     `OMP_NUM_THREADS=8` solves fork by ~1% at 400 iterations — OpenMP
     reduction combination order in `sopp.integral_BdotN` (the squared-flux
     term) is arrival-order and cannot be pinned by any environment variable.
     A no-callback replay therefore lands ~1% from its calibration trace,
     swamping the 0.1% target rung.* Instead, measure native time-to-quality
     directly: each timed native solve runs with a stopping callback that
     terminates via `StopIteration` at the first accepted iterate whose
     scaled objective clears the frozen target, and its solve wall time is
     the measurement. The callback does not recompute physics; its cost is
     microseconds against ~150 ms iterations and is charged to the native
     lane. The stop iterate's full endpoint state must clear the frozen gate
     for the repetition to qualify; iteration counts are recorded and
     reported, never forced. Stop legs run under a preregistered iteration
     cap of twice the reference formulation's budget: the rung sits at the
     end of a 400-iteration reference trajectory while sibling trajectories
     fork ~1% two-sided, so a cap equal to the reference budget would fail
     roughly half of all repetitions on noise rather than speed. The cap is
     not a compared quantity — the stop rule decides the work — and every
     verdict publishes it alongside per-configuration qualifying counts, so
     rung-unreachability and repetition attrition stay auditable.
   - [ ] Because the fused GPU loop has no host trajectory, sweep the
     preregistered budgets `40,80,160,240,400`. The first
     qualifying GPU budget is an upper bound on its true crossing iteration;
     use that exact budget for the no-callback final measurements.
   - [ ] Choose the fastest qualifying native configuration as the denominator.
     Choose the JAX history with the lowest qualifying no-callback warm time;
     bake it into the internal workflow rather than adding a configuration
     knob. Freeze both selections before the final five-pair run.
   - [ ] Close the route if no JAX history reaches the frozen endpoint contract.

4. Add the internal workflow and route both JAX callers through it.
   - [ ] Before moving solve assembly, run
     `tests/test_host_boundary_ssot_ratchet.py::test_only_boundary_owners_call_jax_transfer_and_readiness_primitives`
     and inspect its exact `path::function::call` allowlist. Keep the new module
     free of direct JAX transfer/readiness primitives and use the repository
     host-boundary SSOT. Change the allowlist only if the implementation truly
     creates a new deliberate owner, and then name that exact owner in the
     review.
   - [ ] Add `src/simsopt_jax/examples/stage_two_finitebuild.py` with the frozen
     program/result records, preparation function, solve function, and one
     private fixed-history constant.
   - [ ] Use `simsopt_jax.solve.dispatch.minimize()` with typed
     `SimsoptLBFGSBOptions`/`SimsoptBFGSOptions`. Keep callbacks disabled in the
     shipped workflow so the fused L-BFGS path has no accepted-step host
     observations. Do not duplicate or call private optimizer internals.
   - [ ] Keep the prepared callable identity stable across warm benchmark
     repetitions; a repeated call must reuse the compiled executable for the
     same shape instead of creating a fresh jitted lambda or partial.
   - [ ] Replace optimizer assembly in
     `examples/jax/3_Advanced/stage_two_optimization_finitebuild.py` with the
     internal workflow. Preserve construction, execution-mode driver selection,
     tolerances, status semantics, and every published observable name.
   - [ ] Replace only the JAX solve portion of
     `examples/jax/parity/cases/native_stage_two_optimization_finitebuild.py`.
     Preserve its independent native evaluator, input fingerprints, Taylor
     test, and workflow-stage record. Add minimum clearance to both native and
     JAX parity observations so the new endpoint gate has an independent
     oracle.

5. Replace source-shape tests with behavioral contracts.
   - [ ] Update `tests/jax/examples/test_stage_two_finitebuild_example.py` to
     execute the bounded public `solve()` entry point and validate status,
     complete observable schema, finite values, objective improvement, and
     positive clearance. Remove AST/import-name assertions.
   - [ ] Remove the source-string assertion from
     `tests/integration/test_jax_mirror_stage_two_finitebuild_parity.py`; retain
     and extend the executable native/JAX comparison to minimum clearance.
   - [ ] Add
     `tests/integration/test_jax_stage_two_finitebuild_strict_transfer.py`.
     Stage inputs explicitly, run the prepared workflow under
     `jax.transfer_guard("disallow")` and `host_transfer_audit()`, then publish
     results outside the guarded numerical call. Require a positive
     `final_result` transfer-audit control, zero `advance`, `callback`, and
     `unclassified` transfers, and no callback-configured stepwise fallback.
     This proves no per-step round trip; it does not falsely claim that the
     typed optimizer's explicit terminal read-back is absent.

6. Prove or reject the native-speed claim and publish the terminal verdict.
   - [ ] If a preregistered Step-2 or Step-3 kill criterion closes the route,
     skip the later selection/final-pair work but still publish the completed
     canary evidence, terminal receipt, and scoreboard amendment. Do not create
     the unused workflow module after an early close.
   - [ ] Run five fresh paired native/JAX repetitions after selection. Alternate
     pair order, pin CPU affinity and OpenMP settings, serialize GPU use, and
     reject a round with a competing GPU process or changed source/runtime/cache
     identity.
   - [ ] Use the frozen no-callback budgets and require every endpoint to pass
     the scientific gate. Do not substitute equal iteration counts or a
     post-hoc objective rung.
   - [ ] Require the median paired `native_seconds / gpu_seconds` to be at least
     `1.10` for both `warm_solve_seconds` and warm persistent-cache
     `process_wall_seconds`, with every paired ratio greater than `1.00`.
   - [ ] Report fresh-empty-cache compile and process time separately. Make no
     cold-start win claim unless the same five-pair rule independently passes.
   - [ ] Write the result and raw-artifact hashes to
     `docs/receipts/stage_two_finitebuild_native_gpu.md`. If either required
     speed gate fails, mark the route `CLOSED_BOUNDED_NEGATIVE` and do not
     extend these changes to stochastic Stage II.
   - [ ] In the same measured-verdict commit, amend the
     `native-stage-two-optimization-finitebuild` row, summary counts, and dated
     amendment log in `docs/jax_example_device_assignment.md`. A qualified win
     moves the row to `gpu` with the evidence class supported by the receipt. A
     bounded negative moves it to `cpu`/measured only when the completed
     native-default evidence actually establishes that the fastest qualifying
     native lane is faster; otherwise the row remains `unmeasured` but its
     reason and amendment log cite the bounded result. Any `cpu` row must open
     with a mechanism-family prefix supported by the evidence. `NOT_PRODUCED`
     is not a measured verdict and leaves the assignment unchanged. If the row
     becomes `gpu`, stage the new receipt before running
     `tests/test_jax_example_device_assignment.py`, because that gate requires
     `git ls-files` to see a tracked regular file under `docs/receipts/`.
   - [ ] Keep the receipt headline explicit: a qualifying result establishes a
     warm same-process and warm persistent-cache repeated-workload win only.
     Never omit or relabel the separately measured fresh-cache result.

7. Close the execution-source pin cascade in the implementation commit.
   - [ ] The manifest membership rule selects every Python file under
     `benchmarks/`, `examples/`, and `src/`. Therefore a completed win-path diff
     adds two members, not one:
     `benchmarks/stage_two_finitebuild_native_gpu.py` and
     `src/simsopt_jax/examples/stage_two_finitebuild.py`. Bump
     `DIAG5_EXECUTION_SOURCE_ENTRY_COUNT` in
     `benchmarks/single_stage_native_equivalent_quality_successor_authority.py`
     and `_DIAG5_BOOTSTRAP_EXECUTION_ENTRY_COUNT` in
     `benchmarks/run_single_stage_native_equivalent_quality_campaign.py` from
     `614` to `616`.
   - [ ] Regenerate from the repository root with the tree's GPU interpreter:

     ```bash
     PYTHONPATH=.:src .venv-qn-gpu/bin/python \
       benchmarks/regenerate_execution_source_manifest.py \
       --admit benchmarks/stage_two_finitebuild_native_gpu.py \
       --admit src/simsopt_jax/examples/stage_two_finitebuild.py \
       --expect-count 616
     ```

   - [ ] If the route closes before Step 4 and the workflow module is never
     created, admit only the new benchmark and freeze both count twins at
     `615`. In either branch, refuse any undeclared entering/leaving member,
     inspect every regenerated digest, and commit the manifest, both count
     twins, and the source changes together.

## Validation Plan

Define the strict fast-GPU environment once in the shell used for the HLO
capture, strict-transfer test, and native-default example:

```bash
STRICT_GPU_ENV=(
  MPI4PY_RC_INITIALIZE=false
  SIMSOPT_BACKEND_MODE=jax_gpu_fast
  SIMSOPT_BACKEND_STRICT=1
  SIMSOPT_JAX_TRANSFER_GUARD=disallow
  JAX_TRANSFER_GUARD=disallow
  SIMSOPT_PRECISION=fp64
  JAX_PLATFORMS=cuda
  JAX_ENABLE_X64=1
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
)
```

- [ ] Run focused objective and bounded example tests:

  ```bash
  MPI4PY_RC_INITIALIZE=false JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv-qn-gpu/bin/python -m pytest -q \
    tests/jax/objectives/test_finite_build_stage_two.py \
    tests/jax/examples/test_stage_two_finitebuild_example.py
  ```

- [ ] Run CPU parity and, if the canaries reach Steps 4-5, the new
  strict-transfer integration test:

  ```bash
  MPI4PY_RC_INITIALIZE=false JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv-qn-gpu/bin/python -m pytest -q \
    tests/integration/test_jax_mirror_stage_two_finitebuild_parity.py

  env "${STRICT_GPU_ENV[@]}" .venv-qn-gpu/bin/python -m pytest -q \
    tests/integration/test_jax_stage_two_finitebuild_strict_transfer.py
  ```

  On the 615-member early-close branch, the second command and its planned-new
  test file do not exist. That branch instead runs the focused objective and
  existing example tests, CPU parity, the new benchmark validator,
  execution-source membership/rehearsal gates, host-boundary and scoreboard
  ratchets, Ruff, and diff checks.

- [ ] Run the native-default JAX GPU example in FP64 fast mode and require
  `status=ok`, finite endpoint arrays, a lower final objective, and positive
  minimum clearance:

  ```bash
  env "${STRICT_GPU_ENV[@]}" .venv-qn-gpu/bin/python \
    examples/jax/3_Advanced/stage_two_optimization_finitebuild.py --json
  ```

- [ ] Run `tests/benchmarks/test_stage_two_finitebuild_native_gpu.py`; its
  temporary synthetic rows test reduction logic and make no timing claim.
- [ ] Run the execution-source membership and rehearsal gates after manifest
  regeneration, in the same staged implementation state:

  ```bash
  PYTHONPATH=.:src .venv-qn-gpu/bin/python -m pytest -q \
    tests/benchmarks/test_regenerate_execution_source_manifest.py \
    tests/benchmarks/test_rehearse_single_stage_projected_route_cpu.py
  ```

- [ ] Run `tests/test_host_boundary_ssot_ratchet.py` and
  `tests/test_jax_example_device_assignment.py`. The latter runs only after a
  winning receipt has been staged so its tracked-file gate sees the intended
  same-commit state.
- [ ] Run `ruff check` and `ruff format --check` on every changed Python file.
- [ ] Run `git diff --check` and confirm unrelated files remain untouched.
- [ ] Verify the receipt recomputes every published speed ratio from raw timing
  rows and binds the exact source/runtime/device/cache configuration.

## Risks and Mitigations

- Risk: Reducing L-BFGS history makes each step cheaper but needs more steps or
  reaches a worse endpoint.
  Mitigation: Select by no-callback time to the same frozen gate and reject any
  history that weakens endpoint quality.

- Risk: Removing minimum-clearance evaluation changes the distance penalty or
  geometry ownership.
  Mitigation: Test the XLA-DCE null hypothesis first, then compare value/full
  gradient at three frozen states against the independent native evaluator and
  compare endpoint clearance separately.

- Risk: A temporary callable is retraced, making a claimed warm solve include
  compilation or making cache behavior non-reproducible.
  Mitigation: Prepare stable callables once, synchronize every timing boundary,
  record cache state, and keep cold, same-process warm, and persistent-cache
  subprocess metrics separate.

- Risk: A slow or oversubscribed native denominator creates a false GPU win.
  Mitigation: Sweep the full bounded thread/history matrix, replay selected
  budgets without callbacks, interleave final pairs, and use the fastest
  qualifying native lane.

- Risk: Transfer-guard success is misreported as zero transfers.
  Mitigation: combine the implicit-transfer guard with the repository's explicit
  host-transfer audit and require endpoint-readback positive control plus zero
  optimizer-advance transfers.

- Risk: The GPU solve wins while compilation or host construction makes the
  example slower overall.
  Mitigation: require both warm solve and warm persistent-cache subprocess wins,
  and report fresh-cache process time separately.

- Risk: The workload still cannot amortize the device optimizer.
  Mitigation: close the route after this bounded experiment; do not add another
  solver framework or widen the project.

## Completion Criteria

- [ ] Objective, discretization, topology, FP64 precision, initial state,
  penalty weights, tolerances, and scientific gate fingerprints are unchanged.
- [ ] If the route passes the canaries, the public example and parity JAX lane
  use the same internal finite-build workflow and fixed fast-mode history; the
  native parity evaluator remains independent. An early bounded close adds no
  unused workflow module.
- [ ] Objective, bounded example, CPU parity, benchmark validator,
  execution-source membership/rehearsal, host-boundary ratchet, scoreboard,
  Ruff, and diff checks pass. Strict-transfer validation also passes whenever
  the route reaches the workflow implementation; it is absent by design on the
  615-member early-close branch.
- [ ] A win has five interleaved pairs satisfying every endpoint gate, paired
  median speedup at least `1.10x` for warm solve and warm persistent-cache
  subprocess wall, and no individual paired regression. Otherwise the first
  preregistered failed kill criterion terminates the route truthfully.
- [ ] A hash-bound receipt records either the qualified, narrowly scoped win or
  a truthful `CLOSED_BOUNDED_NEGATIVE`; incomplete evidence is `NOT_PRODUCED`.
- [ ] The device-assignment row, summary counts, and amendment log match the
  measured verdict; any `gpu` row cites the same-commit tracked receipt.
- [ ] The execution-source manifest, its two frozen count twins, and the
  realized new-member count agree in the same implementation commit.

## Open Questions

- Which JAX history among `10`, `20`, and `40` is the fastest qualifying one?
  The bounded canary decides before the workflow constant is frozen.
- Does XLA already eliminate the discarded minimum-clearance reduction? The
  pre-canary StableHLO/optimized-HLO diff answers this before timing; the canary
  then decides whether the complete route proceeds.
