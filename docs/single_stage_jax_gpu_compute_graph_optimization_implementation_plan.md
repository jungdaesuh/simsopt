# Single-Stage JAX GPU Compute-Graph Optimization Implementation Plan

**Status:** Blocked — faithful nested route stopped as engineering bounded-negative; formal shared closure remains incomplete
**Last updated:** 2026-08-09

## Purpose

Make the native-default custom-JAX single-stage Boozer value-and-gradient path
materially faster on GPU without weakening its FP64 scientific contract. The
work targets the device computation identified by code review: exact Boozer
Newton, exact-adjoint matrix construction and factorization, and the direct and
implicit coil pullbacks.

This is a numerical-performance plan. It is separate from the completed r5
speed-campaign closeout and from
`docs/single_stage_changed_state_gpu_timeline_implementation_plan.md`, which
owns measurement instrumentation. Neither document is amended by this work.

## Execution Disposition — 2026-08-09

The faithful nested-route implementation work is stopped and non-promoting.
This is an engineering bounded-negative result, not a formal campaign `LOSS`
and not proof that the campaign `WIN` target is unreachable. Production remains
the `C0` incremental-GMRES route.

The decisive measured dispositions are:

- A100 dense-direct `C1` and `C2` passed numerical, p95, and memory gates but
  reached only 1.1289x and 1.0854x isolated-process p50 speedup, below the
  required 1.25x gate. Dense-branch disposition SHA-256:
  `bc454d3344be7587b9f7867b8ca95d564d4a8b2f14e1451f15534ac30afe4289`.
- The one-jitted accepted-incumbent boundary reached 1.0224x versus its 1.15x
  phase gate. Disposition SHA-256:
  `46f34aaa659198e8e7de69596bf689e59c96362a650b2b11fa3e8229bdb1eaf0`.
- The fused scalar Lagrangian pullback reached 0.9731x and increased peak
  self-RSS by 11.17%, failing both its speed and 10% RSS gates. Disposition
  SHA-256:
  `ceef25206fd90e39d91b01b4f915542390bfae49907c80f51c36eda711946437`.
- The RTX profile attributes 98.354% of active device time. Device activity is
  44.731% of the evaluation envelope, inter-launch gaps are 55.005%, and
  command-buffer graph work covers 73.639% of the classified device union.
  The nested graph remains launch-fragmented, but neither the dense route nor
  the measured executable-boundary fusion closed enough of the full gap.

The matched current-source complete-path receipt and formal gap budget remain
`NOT_PRODUCED`. After the original desktop-memory preflight blocker was
resolved, a corrected immutable RTX snapshot passed its first-evaluation and
ten-sample warm gates. The live orchestrator then reported `wall_time_limit` in
`native_cpu` after 383 of 1000 iterations; the partial artifact has no persisted
termination receipt, and `C0` and Optax did not start. The persisted evidence
therefore establishes only a nonterminal partial trajectory, not a speed
verdict, and does not complete the formal pivot rule.

No further native-CPU replay is selected as performance-development work.
The existing 287–351 s native baseline may be used only as a clearly labelled
historical engineering reference. A future formal receipt must either validate
an identity-compatible baseline-import contract or run a separately authorized
matched native measurement under a separately planned certification task.

The next performance effort is a separately contracted DESC-style
coupled/fullspace, device-resident formulation. It may reuse the attribution,
dense-linearization, retained-factor, parity, and provenance primitives from
this tranche, but it owns a new mathematical-equivalence, trajectory, endpoint,
memory, and timing contract. It must not silently change this stopped plan.

## Goals

- Reduce warm changed-state value-and-gradient time while retaining the same
  physical objective, FP64 state, grids, weights, tolerances, failure policy,
  accepted-incumbent semantics, and final endpoint certificate.
- Compare the current matrix-free incremental-GMRES Newton against a bounded,
  dense-direct 255-by-255 Newton canary with one LU factorization and native-
  style refinement.
- Reuse a dense factorization only when it belongs to exactly the returned
  solved state and the same residual graph, dtype, grid, and weights.
- Replace separate direct and implicit coil reverse traversals with one
  stopped-gradient Lagrangian pullback if value, gradient, residual, and memory
  validation pass.
- Remove demonstrated kernel-launch fragmentation in dense-operator tails and
  Biot-Savart batching without padding physical points, coils, or quadrature
  nodes.
- Produce a complete, provenance-bound performance receipt before any variant
  is promoted.
- Quantify the complete-path performance gap and a measured phase-share speedup
  budget before treating any canary as capable of reaching the formal campaign
  `WIN` target. Pivot cleanly if the faithful nested route cannot close that
  budget.
- Use the available local RTX 5090 and Landau A100 as separate early decision
  lanes. In particular, determine whether the dense FP64 route changes the
  bottleneck on A100 before selecting a primary campaign environment.

## Non-Goals

- Porting DESC's optimizer, proximal/trust-region formulation, QR/SVD choices,
  objective blocking API, or trajectory semantics into SIMSOPT.
- Replacing the host L-BFGS-B state machine in this faithful optimization
  tranche. The current profile reports negligible explicit transfer time and
  attributes 98.354% of active device time; a device-resident outer state
  machine belongs to the separately contracted coupled/fullspace work.
- Using reduced-precision state, results, residuals, gradients, tolerances, or
  endpoint certificates; loosening tolerances; changing quadrature or surface
  resolution; accepting a different endpoint; or hiding failures with damping,
  pseudoinverses, or fallback solvers. A compensated reduced-precision internal
  canary, if ever justified by an FP64 compute-bound profile, belongs to a
  separate non-promoting plan and is not part of this tranche.
- Speculatively evaluating all line-search candidates as a batch in the
  native-faithful lane. It executes candidates that the sequential state
  machine would not visit and can change memory, failure exposure, reduction
  order, and acceptance decisions. Any such experiment requires a separate
  tolerance-parity contract.
- Reusing a Jacobian or LU across different states, coil vectors, residual
  graphs, grids, weights, or dtypes.
- Changing the closed r5 protocol, validator, frozen files, or historical
  `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING / NOT_PRODUCED` record.
- Treating source-level loop counts, GPU utilization, HLO size, or a partial
  trajectory as a speed verdict.
- Pooling RTX 5090 and Landau A100 samples, or treating a cross-host ratio with
  different source, runtime, CPU, timing boundaries, or profiler policy as a
  formal performance comparison.

## Current Context

- Plan baseline is commit
  `320e5cba814414a43e48cb5b6e53f4ad356a9925` with a dirty worktree. All
  pre-existing modifications and untracked files are user-owned and must be
  preserved. Implementation starts from a separately recorded source-state
  hash, not from the plan's creation hash alone.
- The user confirmed current access to a Landau A100. Historical single-stage
  runs used physical A100 UUID `GPU-250014ca-8cb3-bdcd-ad1d-2f6f64529b8d`
  through Landau's Slurm environment. That identity is historical evidence, not
  a substitute for a current allocation and device preflight.
- The last validated Landau recipe required a pinned CUDA 12.6 compatibility
  library path on its CUDA-11.4-era driver plus the complete dependency overlay,
  including `lineax==0.1.1`. CUDA 12.8/12.9 compatibility stacks failed on that
  host. Phase 0 must revalidate the current driver, compatibility path, package
  overlay, source snapshot, and physical device before using A100 evidence.
- The native exact solver in `src/simsopt/geo/boozersurface.py` constructs a
  dense Jacobian, applies `solve(J, b)` plus one refinement solve, and returns a
  factorization of the final-state Jacobian.
- The production traceable JAX solver in
  `src/simsopt_jax/geo/optimizers/optimizer.py` is device-resident but applies
  the Jacobian through sequential JVPs inside incremental GMRES and uses
  backtracking. It does not materialize the Newton Jacobian.
- For the native-default exact system, the solved-state dimension is 255. One
  FP64 dense matrix therefore occupies about 0.50 MiB
  (`255 * 255 * 8` bytes), excluding AD activations and compiler workspaces.
- The configured exact-Newton GMRES budget permits two 255-vector Krylov
  cycles. Including the residual check, one exhausted solve can require about
  513 operator applications; the optional correction solve can approximately
  double that count. A dense materialization requires 255 basis JVPs. These
  counts motivate a canary but do not predict wall-clock speed.
- The direct exact adjoint already materializes the 255-by-255 transpose
  operator in bounded batches, factors it once, performs an initial solve and
  one refinement solve, and reuses the factors for ten condition-estimator
  solves. The optimization opportunity is avoiding a second materialization at
  the same final state, not removing those twelve solve operations.
- With the default dense batch width of eight, a 255-column materialization is
  31 full batches plus a seven-column remainder. The full batches are vmapped;
  the remainder is currently scanned one column at a time. Increasing the
  batch width is not free: the current GPU policy estimates roughly 3 GiB of AD
  activation memory per parallel column above the legacy budget.
- The scalar objective gradient is already computed with one adjoint. The
  remaining duplication is two coil reverse traversals: a direct objective VJP
  and an implicit residual/stationarity VJP.
- One successful production candidate is also split across an anchored-forward
  executable, a gradient/evidence executable, and eager failure-mask and
  eligibility operations. Source inspection suggests roughly eleven JAX
  program submissions before acceptance, but this is not a CUDA-kernel count;
  Phase 0 must measure PJRT executes and kernels before selecting this work.
- The production shape has 461 coil degrees of freedom, 18 physical coil
  contributions, 250 quadrature nodes, and surface point counts of 169 for the
  inner residual and 1600 for the non-QS term. Point and coil chunking are
  inactive at current thresholds. Quadrature alone is split exactly as
  128 plus 122.
- The earlier segmented timeline was diagnostic: it attributed about 99.953%
  of its measured window to the device, 0.0467% to host work, and 0.0000316%
  to explicit transfers, but left 93.034% of device time unattributed. The
  subsequent RTX profile closes that hole at 98.354% attribution of active
  device time and measures a 55.005% inter-launch-gap share. Together they rule
  out a transfer-first plan without establishing a winning faithful-route
  canary.
- The closed r5 record contains a diagnostic custom-GPU/native trajectory ratio
  of `7541.455 / 287.304 = 26.249`. It is not a formal speed verdict: no complete
  four-lane `campaign.json` exists, so the protocol result remains
  `NOT_PRODUCED`. The frozen protocol defines three distinct timing gates:
  custom time-to-quality no greater than 90% of native time-to-quality; custom
  time-to-quality no greater than 90% of Optax time-to-quality; and custom
  fixed-budget warm median no greater than 90% of native with the quality
  clause satisfied by both fixed-budget lanes. Cold compile time is reported
  separately. Phase thresholds in this plan are minimum engineering gates, not
  maximum savings and not proof that the formal target is reachable.
- Coupled/fullspace diagnostics belong to a separate formulation-research path.
  They are not evidence that a coupled formulation has already won this
  campaign or that any diagnostic speedup transfers to the faithful nested
  lane. Only a separately contracted, provenance-valid time-to-target campaign
  can support that claim.
- `jax.linearize` can construct one reusable fixed-state linearization instead
  of repeatedly re-linearizing the primal for basis JVPs. It also retains
  linearization residuals whose memory scales with the computation. It is a
  measured dense-assembly canary, not a selected default, until compile time,
  peak memory, HLO, and warm timing are compared against checkpointed
  `vmap(jvp)`.
- XLA command buffers may already be enabled by the runtime default. Source
  configuration alone cannot establish capture participation. Phase 0 must
  record the resolved setting and trace actual captured versus uncaptured work
  before command-buffer tuning is treated as an optimization lever.
- DESC supplies useful structural ideas—batched/chunked derivatives, blocked
  objective evaluation, and factor reuse inside a step—but its optimizer and
  objective contracts differ. Only locally parity-tested execution patterns
  are candidates here.

## Rationale

The target graph is small in linear-algebra dimension but expensive in each
residual/JVP because it differentiates Biot-Savart and Boozer geometry. A GPU
can lose when this expensive graph is launched sequentially hundreds of times.
The plan therefore reduces graph traversals in this order:

1. replace a potentially exhausted sequential Krylov loop with one bounded
   dense sweep and one direct factorization;
2. carry the exact final-state factors into the transpose adjoint instead of
   rebuilding the same operator;
3. combine the direct and implicit coil derivatives into one scalar pullback;
4. vectorize only measured exact tails and retile only measured Biot-Savart
   work.

This ordering also controls risk. Each step has an isolated canary, a numerical
oracle, and a performance stop gate. No broad rewrite is needed to learn
whether the architecture is viable.

### Performance-gap budget and pivot rule

This plan does not infer a complete-path ceiling by multiplying its 20%, 10%,
or 15% phase promotion thresholds. Those thresholds cover overlapping scopes
and are minima, not caps. After Phase 0, let `p_i` be each disjoint candidate
phase's measured share of the complete warm value-and-gradient path and let
`r_i` be a bounded fractional reduction supported by a canary or an explicitly
labelled optimistic limit. The non-overlapping projection is

`S = 1 / (1 - sum_i(p_i * r_i))`.

The receipt must report conservative and optimistic projections, the remaining
unattributed share, overlap dispositions, and the matched complete-path time
required by the formal campaign gate. A projection is routing evidence, not a
speed result.

After Phase 0 and after every passing phase, recompute the budget. Close the
faithful nested promotion branch when both conditions hold:

1. at least 90% of the warm device interval is attributed and the remaining
   unattributed share, even under the declared optimistic bound, cannot bridge
   the matched formal timing target; and
2. every still-authorized faithful lever is either measured, bounded by the
   phase trace, or stopped by a named parity, memory, or provenance gate.

The pivot rule cannot fire without matched current-source native and Optax
timing sufficient to define the formal target. If either reference is missing,
the tranche may still close on its engineering gates, but it cannot claim that
the formal campaign target is unreachable.

As of 2026-08-09, the formal pivot rule has not fired because the matched
complete-path receipt is `NOT_PRODUCED`. The project nevertheless stops this
faithful implementation branch on its engineering gates: all selected high-
value canaries failed their own promotion thresholds, and another CPU replay
would add certification evidence rather than improve the GPU execution graph.

That closure is an engineering bounded-negative, not a campaign `LOSS`. The
dense linearization, factor handoff, fused pullback, and trace primitives remain
eligible inputs to a separately contracted coupled/fullspace campaign. That
campaign must own its own trajectory, endpoint, provenance, and time-to-target
receipts.

### Native-faithfulness boundary

The optimization may change GPU execution order, batching, and factor storage
without changing the mathematical problem. Bitwise agreement with CPU LAPACK
is not expected because pivoting and floating-point reduction order can differ.
Scientific parity instead requires unchanged equations and tolerances plus the
existing objective, gradient, residual, trajectory, and endpoint gates.

Three variants keep the comparison honest:

| Variant | Linear algebra | Step control | Purpose |
|---|---|---|---|
| `C0` | Current JVP/incremental GMRES | Current JAX backtracking | Frozen baseline |
| `C1` | Dense JAX LU plus refinement | Current JAX backtracking | Isolate the linear-solver effect |
| `C2` | Dense JAX LU plus refinement | Native exact-Newton update/stop semantics | Candidate for C++/native-faithful promotion |

`C1` is not called native-trajectory faithful. `C2` is not promoted merely
because it is faster; its accepted-state sequence must pass the native oracle.

The trajectory contracts are separate:

- `C1` must preserve `C0`'s backtracking and accepted-incumbent decisions,
  statuses, and counters on the frozen replay, with values/states compared under
  the existing JAX parity tolerances.
- `C2` must match the native dense update order, accepted states, stopping
  decisions, statuses, and counters under the native-oracle tolerances.
- Both variants must satisfy the shared cross-lane scientific objective,
  gradient, residual, and endpoint tolerances. `C1` is never required to match
  the native inner trajectory.

## Assumptions

- Two optimization hosts are available: the pinned local RTX 5090 configuration
  used by the changed-state timeline and the Landau A100 confirmed by the user.
  Phase 0 qualifies both, keeps their samples and caches separate, and selects
  the primary campaign environment from complete-path evidence rather than
  assuming RTX 5090 first and A100 only as a later portability run.
- `.venv-qn-gpu/bin/python` remains the intended complete runtime. Phase 0 must
  prove its imports resolve to this checkout before any result is trusted.
- The existing trace annotations and benchmark runner can be extended in a new
  schema without changing the numerical graph. If less than 90% of warm device
  time can be assigned to the candidate phases, microbenchmark wins remain
  diagnostic until a phase-complete trace is available.
- Ten warm changed-state samples after compilation are sufficient for a canary
  decision. The final campaign uses its own frozen sampling and validator
  contract.
- A 20% warm-median value-and-gradient improvement over `C0`, with no material
  p95 or memory regression, is the minimum engineering gate for composing a
  candidate. The ultimate project goal remains a complete validated GPU-vs-
  native `WIN`, not merely this internal improvement.
- The current `26.249x` diagnostic ratio is planning context, not a stable
  baseline or formal target. Phase 0 must measure matched current-source lanes
  before calculating the gap budget.

## Sequencing and Rough Effort

| Work | Rough effort | Dependency |
|---|---:|---|
| Phase-complete dual-GPU baseline | 1–1.5 days | Current timeline tooling; Landau preflight |
| Dense-Newton canaries | 2–3 days | Baseline |
| Adjoint assembly/final-state handoff | 1–2 days | Baseline; reuse needs dense route |
| Scalar Lagrangian pullback | 1–2 days | Baseline; composes after handoff |
| Launch/tail/Biot-Savart canaries | 2–3 days | Baseline; partly parallel |
| Integration and receipt | 1–2 days | Passing canaries |

The rough critical path is 9–14 engineering days, plus GPU queue and full-
campaign wall time. A failed canary stops its branch early rather than consuming
the whole estimate.

## Implementation Plan

### Phase 0 — Freeze a phase-complete baseline (0.5–1 day)

**Disposition (2026-08-09):** attribution, command-buffer, RTX first-evaluation,
RTX warm-baseline, and A100 qualification evidence exist. The matched
complete-path/gap budget and dependent A100 Phase-0 receipt remain incomplete.

- [ ] Record HEAD, `git status --short`, tracked-diff hash, untracked manifest,
  exact source hashes, Python/JAX/jaxlib, CUDA runtime/driver, GPU UUID, CPU
  affinity, XLA/JAX environment, dense-batch policy, and all point/coil/
  quadrature chunk settings.
- [ ] Expand provenance into a role/path/size/SHA-256 manifest for every tracked
  or untracked execution-bearing source, configuration, benchmark, test, and
  native extension. Materialize an immutable copied worktree/tree artifact and
  bind every child receipt to that manifest; HEAD plus a path-only dirty-file
  list is not a reproducible snapshot.
- [ ] Prove `.venv-qn-gpu/bin/python` imports `simsopt`, `simsopt_jax`, and
  `simsopt_jax_adapters` from the immutable snapshot and resolves the
  manifest-bound `simsoptpp` binary.
- [ ] Qualify Landau before any canary: record Slurm allocation/job identity,
  hostname, physical A100 UUID and memory, driver, resolved CUDA libraries,
  pinned CUDA 12.6 compatibility path, JAX/jaxlib backend and devices, FP64/x64
  policy, `lineax` and overlay versions, source/import/binary hashes, CPU
  affinity, and a finite strict-transfer smoke evaluation. Reject CUDA 12.8/
  12.9 compatibility or any missing/mixed overlay rather than repairing it
  inside a timed child.
- [ ] Add a new benchmark schema and artifact root for compute-graph canaries;
  do not append to the closed r5 receipt or the instrumentation-only timeline
  artifact.
- [ ] Reuse the canonical native-default input builder and one frozen changed-
  state candidate. Bind its full parameter SHA-256, state dimension, coil DOF
  count, grids, weights, tolerances, and solver graph to every sample.
- [ ] Before any compilation-excluded timing or profiler sample for a variant,
  run a canonical first-evaluation fail-fast gate with a 900-second wall-time
  limit including compilation: finite FP64 scalar objective; a nonempty,
  all-finite FP64 461-component gradient; successful inner Newton and adjoint
  solves; finite residual certificates; and native-versus-variant initial-point
  objective and gradient parity for each of `C0`, `C1`, and `C2` that proceeds.
  A timeout or failure stops that variant before expensive replay, profiling,
  or compilation-excluded timing.
- [ ] Measure `C0` in fresh processes on both qualified GPUs: one cold compile,
  at least ten warm value-and-gradient samples, p50/p95, peak process-tree RSS,
  peak GPU memory, HLO executable identities, kernel/launch counts, and profiler
  interval unions. Use separate source/runtime/device receipts and compilation
  caches; do not pool samples.
- [ ] Record the resolved XLA command-buffer configuration and measure actual
  capture participation, graph-launched device time, uncaptured launches,
  kernel-duration distribution, device-active share inside the evaluation
  envelope, and inter-launch gap share. Run one matched enable/disable A/B
  control outside promotion timing; do not infer capture state from absent
  repository flags.
- [ ] Add opt-in exact-route Newton telemetry for residual evaluations and
  actual linear-operator applications. Collect it outside timed samples and
  quantify its observer effect. Do not reuse similarly named polish/Hessian
  telemetry as evidence for the production exact-Newton route, and do not use
  operator counts alone to select dense over GMRES.
- [x] Close the former 93.034% attribution hole. The current RTX artifact
  attributes 98.354% of active device time and records the remaining
  unattributed share without treating utilization as speed evidence.
- [ ] Measure matched current-source native, `C0`, and available Optax
  complete-path timing under the applicable protocol boundaries. Record the
  formal target time separately from the historical 26.249x diagnostic ratio.
- [ ] Construct the first conservative and optimistic performance-gap budget
  from disjoint measured phase shares. Record assumed reductions, overlap
  handling, unattributed-time bounds, projected complete-path ratios, and the
  exact evidence that would trigger the faithful-route pivot rule.

**Exit gate:** a reproducible RTX 5090 `C0` artifact and a machine-readable
Landau qualification outcome exist. If Landau qualifies, a separately
provenance-bound A100 `C0` artifact is also required. Every measured artifact
has phase attribution, correct source/runtime/device identity, passing first-
evaluation/value/gradient/adjoint evidence, resolved command-buffer
participation, exact-route operator telemetry, and a reviewable per-device
performance-gap budget. If Landau qualification fails, record the specific
blocker and continue only with the RTX artifact; do not represent the A100 lane
as measured. If measurement ownership fails, fix it before changing numerics.
If the pivot rule is already proven, close the faithful implementation branch
and retain only reusable, separately contracted research primitives.

### Phase 1 — Dense-direct exact-Newton canaries (2–3 days)

**Disposition (2026-08-09):** complete and engineering bounded-negative. `C1`
and `C2` failed the required isolated-process p50 gate; retain `C0` incremental
GMRES and stop the dense branch.

- [ ] Write one-step CPU oracle tests from the native dense Jacobian and
  refinement equations in `src/simsopt/geo/boozersurface.py`. Compare residual,
  step, refined residual, next state, convergence status, and failure behavior.
- [ ] Add an internal dense-direct runner beside
  `_build_traceable_exact_newton_runner` in
  `src/simsopt_jax/geo/optimizers/optimizer.py`. Reuse the dense operator
  materialization and LU safety owners in
  `src/simsopt_jax/geo/optimizers/linear_solve.py`; do not duplicate a second
  assembler, factorization policy, or condition estimator.
- [ ] Implement two internal dense-materialization canaries behind the same
  assembler contract: the current checkpointed batched `jax.jvp` construction
  and a fixed-state `jax.linearize(residual_fn, x)` followed by batched tangent
  applications. Compare primal execution count, HLO, compile time, viable batch
  width, peak GPU/RSS memory, matrix/step parity, and warm time. Retain neither
  as the production default solely from source-level expectations.
- [ ] Implement `C1` first: replace only GMRES/correction with dense LU plus one
  refinement while retaining the current residual scaling, retry,
  backtracking, convergence, and fail-closed semantics. This isolates the
  performance effect of the linear solve.
- [ ] Implement `C2` as a separate canary matching the native dense update and
  stopping order. Do not hide the distinction behind a permanent public solver
  flag; retain separate internal callables during comparison and remove the
  losing route before promotion.
- [ ] Return a typed final-state linearization payload containing the final
  solved state, exact coil DOFs/coil-set dynamic inputs, dense Jacobian, packed
  LU/pivots, residual identity, and numerical status. If the solver rolls back
  or returns a different incumbent, rebuild at that returned state or return no
  reusable factors.
- [ ] Preserve one LU factorization per Newton matrix, one refinement solve,
  the existing condition screen, strict-transfer behavior, and fail-to-NaN
  propagation.
- [ ] Benchmark `C0`, `C1`, and `C2` on the same changed state with identical
  compile/warm boundaries on each qualified GPU. Record executed Newton
  iterations, materializations, factorizations, LU solves, residual evaluations,
  kernel launches, and memory. Keep the RTX 5090 and A100 decisions separate;
  an A100 dense-route win can select A100 as the primary campaign environment
  without becoming a pooled cross-device speed claim.
- [ ] Recompute the conservative and optimistic complete-path gap budget with
  the measured dense result. A Newton-phase win that cannot materially change
  the formal complete-path projection remains a reusable diagnostic primitive,
  not promotion evidence.

**Exit gate:** select at most one dense route. It must pass the one-step oracle,
its variant-specific short multi-step trajectory contract, and the changed-
state numerical certificate. It must improve warm Newton time by at least 20%
without a p95 or peak-memory regression above 10%. If neither dense route
passes, retain GMRES and stop the dense branch.

### Phase 2 — Adjoint assembly and exact final-state factor handoff (1–2 days)

**Disposition (2026-08-09):** stopped by the Phase 1 gate. Retained-factor and
transpose-adjoint primitives remain reusable research components; no production
factor handoff was selected.

- [ ] Benchmark the current reverse-mode `J^T` construction against forward-
  mode `J` construction followed by a transpose solve. Both require 255 basis
  directions, but their activation memory and viable batch widths can differ.
  Compare matrix orientation/order, LU pivots, solution, live residual,
  gradient, compile time, warm time, and peak memory at the same effective
  batch width before changing the production adjoint.
- [ ] Use an ephemeral, device-resident `ExactFinalLinearization` pytree as the
  only handoff for the production canary. It contains the returned solved state,
  exact coil DOFs/coil-set dynamic inputs, final-state `J`, packed LU/pivots,
  orientation, residual configuration, and solve status.
- [ ] Keep that pytree local to one combined jitted producer-consumer candidate
  evaluation. The adjoint, objective, residual, and coil VJP consume its state
  and coil fields; no separately callable gradient executable can pair its `J`
  or LU with replacement state/coil inputs. Return only the candidate state,
  value, gradient, status, and evidence needed by the host controller—not the
  reusable factors themselves.
- [ ] Keep `factor_handoff_identity.py` as the SSOT only for host-persisted or
  cross-evaluation factors. Do not call its NumPy/JSON/SHA host seal from the
  compiled path. If a later design selects host sealing, treat it as a separate
  variant and include factor/state D2H hashing plus H2D restaging in all timing,
  transfer, and memory receipts.
- [ ] Construct and factor `J` only after the solver has selected the returned
  state, or prove that a retained matrix was evaluated at exactly that state.
  The internal producer and consumer are one compiled closed dataflow; no
  public or environment-controlled API accepts arbitrary factors.
- [ ] Change `_traceable_result_linear_solve_factors` and the exact branch of
  `_traceable_solve_linearization` in
  `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py` to route the
  local ephemeral payload without host materialization. Add device-resident
  factorization, live-operator, and orientation certificates; do not describe
  the existing host identity gate as JIT-compatible.
- [ ] Solve the transpose adjoint with the same packed LU using
  `lu_solve(..., trans=1)`. Keep the initial solve, refinement, backward-error
  and forward-error gates, condition estimate, zero-RHS fast path, and all
  execution telemetry.
- [ ] Make the condition certificate orientation-aware without refactorizing:
  for the adjoint operator `J^T`, compute `||J^T||_1` from the retained `J`,
  estimate `||(J^T)^-1||_1` by swapping the packed-LU forward and transpose
  solve callbacks, and retain the same ten Hager-Higham factor applications.
  Do not pass factors of `J` to a helper that assumes they factor `J^T`.
- [ ] Certify the live adjoint residual as `b - J^T lambda`. Add deliberately
  nonsymmetric test matrices for which `cond_1(J) != cond_1(J^T)`, and compare
  both orientation-specific condition estimates against NumPy. An orientation
  tag rejection by itself is not a numerical certificate.
- [ ] Tie packed LU/pivots numerically to retained `J` by reconstructing the
  factored matrix with the existing LU/pivot owner and checking a normalized
  factorization residual. Benchmark this certificate as part of the complete
  path. It proves numerical factor validity, not packed-byte identity.
- [ ] Prove the successful path executes no second dense materialization and no
  second LU factorization. Retain the current twelve LU solve operations when
  the condition estimator runs; factor reuse does not authorize deleting its
  numerical safety certificate.
- [ ] Prove the production consumer cannot receive state or coils separately
  from its local payload. In internal helper tests, mismatched grid/weights,
  residual graph, dtype, orientation, rollback, failed Newton, or numerically
  invalid factors must fail a certificate or take the selected rebuild path.
  Do not require a mathematically equivalent packed representation to fail, and
  do not claim byte identity. Keep SHA-based stale-byte tests for the separate
  host-persisted handoff contract.

**Exit gate:** objective and gradient match the no-handoff exact route within
the existing FP64 tolerances; adjoint residual and success telemetry match; the
trace proves one matrix construction and one factorization at the returned
state. If no dense Newton route survived Phase 1, the adjoint-only forward-
assembly canary may still proceed, but it must not claim Newton-to-adjoint
factor reuse. A matched complete value-and-gradient comparison must include
final-state `J`/LU production, compile time, warm p50/p95, peak device/RSS
memory, transfers, and launch counts; handoff is selected only if it improves
the complete path by at least 10% without violating the 10% p95/memory gates.

### Phase 3 — Fuse the scalar coil pullback (1–2 days)

**Disposition (2026-08-09):** complete and stopped by phase-speed and RSS gates.
The split production pullback remains authoritative.

- [ ] Inventory every caller and test of
  `_traceable_objective_gradient_parts`. Preserve that callable's seven-value
  diagnostic return contract and split direct/implicit reporting behavior.
- [ ] Add a separate internal production-total gradient callable around the
  exact scalar Lagrangian
  `L(c) = Phi(stop_gradient(x*), c) -
  stop_gradient(lambda)^T F(stop_gradient(x*), c)`.
- [ ] Compute `b = dPhi/dx`, solve `J^T lambda = b` once, then obtain the total
  coil gradient from one pullback of `L`. Preserve the current sign convention
  `direct_grad - implicit_grad`. Route only the timed total-objective caller to
  the fused callable; diagnostics and per-part tests retain the existing split
  implementation. Do not add a public algorithm flag.
- [ ] Apply `stop_gradient` to the solved state and adjoint. Do not differentiate
  through Newton, LU, refinement, condition estimation, factor-identity gates,
  or host acceptance logic.
- [ ] Keep the exact zero-RHS branch, failure-to-NaN behavior, adjoint execution
  counts, residual evidence, and inactive-term handling unchanged.
- [ ] Validate the fused result against the existing two-pullback implementation
  with coordinate finite differences, random directional derivatives,
  off-stationary solved states, zero-weight terms, zero RHS, and forced adjoint
  failure.
- [ ] Compare HLO size, compile time, warm time, kernel launches, peak GPU
  memory, and rematerialization. Reject fusion if a larger live set erases the
  warm saving or violates the memory gate.

**Exit gate:** the fused scalar pullback passes all gradient and failure-mode
tests and improves the measured coil-VJP phase by at least 15% or the complete
value-and-gradient path by at least 10%, without more than 10% peak-memory
growth.

### Phase 4 — Remove measured launch fragmentation (2–3 days)

**Disposition (2026-08-09):** measured and conditionally stopped. Command
buffers already cover most classified device work, and the one-jitted
accepted-incumbent boundary failed its phase gate. Unmeasured speculative
options are deferred to a separately contracted formulation rather than added
to this faithful route.

- [ ] Count PJRT executes, CUDA kernels, and inter-launch gaps for the current
  anchored-forward, gradient, and eager postprocessing sequence. Treat the
  source-level estimate of eleven submissions as a hypothesis, not a receipt.
- [ ] Use the Phase 0 command-buffer control to decide whether capture mechanics
  own measurable time. Tune capture boundaries only when the trace shows an
  uncaptured or fragmented region with a bounded complete-path opportunity;
  otherwise record command-buffer tuning as a stopped branch.
- [ ] First combine the eager failure mask and eligibility calculations into
  one jitted postprocessor and reuse the pending evaluation's materialized
  eligibility during acceptance. Preserve generation/hash checks and prove
  acceptance adds no objective evaluation.
- [ ] If the two major executable boundary remains material, prototype one
  jitted accepted-incumbent candidate program returning value, gradient,
  solved candidate state, status, and optional evidence. Reuse the combined
  producer-consumer envelope already required by the dense-factor canary rather
  than creating a second owner. Keep Wolfe decisions, accepted-state promotion,
  counters, callbacks, and the outer L-BFGS-B state machine on the host. Compare
  every trial value/gradient/status and the exact seven-step accept/reject
  sequence before promotion.
- [ ] Vectorize the exact seven-column dense-operator remainder as one
  seven-wide `vmap`/scan body instead of seven scalar bodies. Do not pad it to
  eight: the extra zero-direction JVP is unnecessary work and would make
  execution counts and activation memory depend on a synthetic basis column.
- [ ] Benchmark dense widths 4 and 8 under the existing memory policy. Widths
  above eight are out of scope unless measured activation memory proves they
  fit the unchanged budget; the policy and effective width must be receipt-
  bound.
- [ ] Benchmark the current exact `128 + 122` quadrature split against other
  exact two-block sizes. Preserve physical node order, normalization, the
  pairwise reduction operator, and an unpadded integrand tail. Treat any change
  in global floating-point association as tolerance-parity work, not byte
  parity.
- [ ] Inspect HLO/CUPTI for the actual grouped `B` plus reverse path. Add a
  custom-VJP/tiled contraction only if the trace proves saved reverse traversals
  or launches; source-level Python loop removal alone is insufficient.
- [ ] Defer coil symmetry compression (`18` contributions to `3` base coils)
  unless an explicit expansion/VJP oracle proves identical current signs,
  transformations, ordering, field, and coil gradient.
- [ ] Revisit one-JIT candidate evaluation or a device-resident outer optimizer
  only if the new phase-complete trace shows material PJRT/host gaps after the
  device graph improvements.
- [ ] Do not batch speculative line-search trials in `C1` or `C2`. If a later
  divergent experiment is authorized, require proof of identical selected
  alpha/status under tolerance, exact no-accepted-candidate behavior, bounded
  memory, finite evaluation of otherwise skipped candidates, and its own
  non-promoting receipt.

**Exit gate:** promote only an individually benchmarked change that improves
its owned phase by at least 15%, passes the end-to-end warm gate when composed,
and preserves the exact-tail and reduction contracts.

### Phase 5 — Compose, replay, and freeze a candidate (1–2 days plus GPU time)

**Disposition (2026-08-09):** not entered because no candidate passed its
individual promotion gate. No production candidate or four-lane campaign was
frozen.

- [ ] Compose only variants that passed their individual gates. Re-run ablation
  samples after each addition to detect negative interactions and compile-cache
  identity changes.
- [ ] Run the canonical initial point, the frozen changed state, a seven-
  accepted-step replay including rejected line-search trials, and the full
  native-default endpoint. Compare native CPU, `C0`, and the candidate.
- [ ] Apply the correct trajectory oracle: `C1` must reproduce `C0` JAX
  backtracking/acceptance decisions within JAX tolerances; `C2` must reproduce
  native update/stop decisions within native-oracle tolerances. Do not require
  `C1` to match the native inner trajectory. A `C1` accept/reject or
  backtracking decision flip fails `C1`; endpoint agreement may qualify a
  separately named divergent candidate but cannot restore C1 causal identity.
- [ ] Require unchanged objective/gradient tolerances, Boozer residual and
  adjoint certificates, accepted-incumbent semantics, iteration budget,
  reporting fields, and endpoint eligibility.
- [ ] Run at least ten independent warm samples on each qualified GPU with
  alternating baseline/candidate order. Report cold compile separately and
  include p50, p95, peak RSS, peak GPU memory, kernel/launch counts, and source/
  runtime/device identity. Keep per-device artifacts and statistical decisions
  separate.
- [ ] Select one primary campaign environment from complete-path, parity,
  provenance, memory, and stability evidence; freeze its source/runtime/device
  contract and run the complete four-lane speed campaign in fresh processes.
  The other GPU remains a separately reported corroboration/portability artifact
  and is never pooled into the primary verdict.
- [ ] If no composed candidate clears the internal 20% warm gate, record a
  bounded-negative engineering result with raw receipts for every attempted
  variant, parity evidence, and proof that production routing is unchanged. If
  it clears that gate but the complete campaign does not validate `WIN`, do not
  claim the project speed goal is complete.

After the base canary exists, run the post-canary evidence chain through the
single orchestrated CLI; do not hand-author a finalizer JSON or call its Python
API directly. A `C1` invocation supplies both `--c0-raw` and
`--c0-trajectory-receipt`. A `C2` invocation omits both flags and the workflow
binds the same native raw file as its one-step and trajectory reference.

```bash
.venv-qn-gpu/bin/python -m \
  benchmarks.single_stage_compute_graph_canary_workflow \
  --variant C1 \
  --canary-spec "$CANARY_SPEC" \
  --base-canary-artifact "$BASE_CANARY" \
  --c0-receipt "$PHASE0_RECEIPT" \
  --trajectory-artifact-root "$TRAJECTORY_ROOT" \
  --native-raw "$TRAJECTORY_ROOT/native.json" \
  --native-trajectory-receipt "$TRAJECTORY_ROOT/native-receipt.json" \
  --c0-raw "$TRAJECTORY_ROOT/c0.json" \
  --c0-trajectory-receipt "$TRAJECTORY_ROOT/c0-receipt.json" \
  --variant-raw "$TRAJECTORY_ROOT/c1.json" \
  --variant-trajectory-receipt "$TRAJECTORY_ROOT/c1-receipt.json" \
  --profile-count "$TRAJECTORY_ROOT/profile-counts.json" \
  --trajectory-oracle "$TRAJECTORY_ROOT/oracle.json" \
  --profile-output-root "$PROFILE_ROOT" \
  --nsys-binary "$NSYS_BINARY" \
  --nvtx-library "$NVTX_LIBRARY" \
  --nsys-version "$NSYS_VERSION" \
  --finalizer-spec "$FINALIZER_SPEC" \
  --promotion-destination "$PROMOTION_ARTIFACT"
```

## Validation Plan

Run JAX-focused test files in separate processes because import-time platform
and x64 settings can mutate global runtime state. Phase 0 must substitute the
proven qualified interpreter if `.venv-qn-gpu/bin/python` is not the captured
runtime.

```bash
env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/geo/test_adjoint_cg_solver.py

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/geo/test_traceable_adjoint_zero_rhs.py

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/geo/test_surface_objectives_jax.py \
  -k 'traceable_objective_gradient_parts or exact_batched_adjoint'

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/geo/test_factor_handoff_identity.py

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/geo/test_factor_handoff_routing.py

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/integration/test_factor_once_adjoint_phase2.py

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/field/test_biotsavart_jax.py

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/integration/test_jax_mirror_single_stage_boozer_vacuum_parity.py

env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-gpu/bin/python -m pytest -q \
  tests/benchmarks/test_single_stage_changed_state_gpu_timeline.py
```

- [ ] Add focused dense-Newton unit tests covering one-step parity, refinement,
  convergence, backtracking (`C1`), native update semantics (`C2`), rollback,
  singular/nonfinite matrices, strict transfer guard, and telemetry counts.
- [ ] Add first-evaluation regression tests requiring completion within the
  900-second limit including compilation, a finite FP64 value, exactly 461
  finite FP64 gradient entries, successful inner/adjoint telemetry, finite
  residual certificates, native-versus-variant initial value/gradient parity,
  and fail-closed rejection before compilation-excluded timing.
- [ ] Add dense-assembly parity tests comparing checkpointed batched
  `vmap(jvp)` with `jax.linearize` across the production 255-dimensional system
  and representative smaller nonsymmetric systems. Measure retained-memory
  growth rather than asserting that either implementation is memory-free.
- [ ] Add command-buffer provenance and trace-schema checks that distinguish
  resolved configuration from observed capture participation and keep the
  enable/disable control outside promotion samples.
- [ ] Add a Landau preflight receipt test covering Slurm/job identity, physical
  device UUID, CUDA 12.6 compatibility resolution, complete dependency overlay,
  JAX backend/x64 policy, immutable source/import/binary identity, and explicit
  rejection of mixed or unsupported compatibility stacks.
- [ ] Add factor-handoff tests proving exact final-state identity and rejecting
  every stale/cross-state case, plus nonsymmetric orientation tests whose
  `cond_1(J)` and `cond_1(J^T)` differ.
- [ ] Add fused-Lagrangian tests for scalar, directional, coordinate, zero-RHS,
  off-stationary, and failure-to-NaN behavior.
- [ ] Add a caller-inventory ratchet for the production-total and diagnostic-
  parts gradient callables so the seven-value diagnostic contract cannot be
  silently routed through the fused total-only path.
- [ ] Add exact-tail tests at dense dimensions 7, 8, 15, 16, 255, and 256 and
  quadrature counts 127, 128, 129, 250, 256, and 257.
- [ ] Compare compiled and eager values/gradients, CPU and GPU results, repeated
  cache hits, and changed-state cache misses.
- [ ] Run Ruff, formatter check, `compileall`, `git diff --check`, and a static
  check that closed r5 files remain byte-identical.
- [ ] Validate the final receipt from raw children; do not hand-edit a
  `campaign.json` or substitute a synthetic receipt for a real campaign.

## Risks and Mitigations

- **Dense materialization is slower than GMRES on easy states.** Compare on
  states spanning the observed 2/2/3/4 Newton-iteration behavior and keep the
  current route if the 20% gate is not met.
- **Dense batching exceeds GPU memory because AD activations dominate the
  0.50-MiB matrix.** Keep width bounded by the current memory policy, measure
  peak allocation, and vectorize the seven-wide tail without increasing the
  full-batch width.
- **`jax.linearize` removes repeated primal linearization but retains a large
  fixed-state linearization.** Compare it against checkpointed batched JVPs at
  equal numerical work, report peak device/RSS memory and compile time, and
  reject it if the retained live set violates the memory gate.
- **Phase thresholds are mistaken for a complete-path speedup ceiling or
  guarantee.** Use disjoint measured phase shares, publish both conservative
  and optimistic projections, disposition overlap explicitly, and call the
  result a routing budget rather than timing evidence.
- **Command buffers are assumed absent because no repository flag enables
  them.** Record the resolved runtime setting and observed trace participation;
  keep the matched disable control diagnostic and stop tuning if capture is
  already effective.
- **Landau's A100 is available but its historical runtime recipe silently
  drifts.** Requalify the allocation, physical UUID, driver, CUDA 12.6
  compatibility libraries, dependency overlay, JAX backend, and source/import
  identity before every artifact root; fail the A100 lane closed on drift.
- **A faster A100 canary is pooled with RTX 5090 or compared against a mismatched
  native host.** Keep device caches, samples, timing boundaries, receipts, and
  verdicts separate, and select exactly one matched environment for the formal
  campaign.
- **A factor belongs to a pre-step rather than the returned state.** Produce the
  ephemeral payload only after the final state is selected; rebuild or reject
  on rollback, failure, or any live-certificate mismatch.
- **Lagrangian fusion changes signs or accidentally differentiates through the
  solve.** Preserve `direct - implicit`, use explicit stopped gradients, and
  gate against directional derivatives and the current two-pullback oracle.
- **Fusion saves launches but increases compile time or live memory.** Report
  cold compile separately and reject a warm improvement that violates p95 or
  memory gates.
- **Reduction or padding changes numerical behavior.** Keep exact physical
  shapes, never pad geometry before inverse-distance evaluation, and keep the
  explicit pairwise reduction operator. Record association changes and test
  every relevant boundary size under the existing FP64 tolerances.
- **A microbenchmark win does not transfer to the optimizer.** Require the
  seven-step replay and complete endpoint before freezing a candidate.
- **The dirty tree invalidates provenance.** Execute from an immutable copied
  snapshot with a role/path/size/SHA-256 manifest for every execution-bearing
  tracked and untracked byte, verify import/binary provenance inside each child,
  and reject mixed-source artifacts.

## Completion Criteria

The technical tranche is complete only when the shared evidence below and
exactly one closure branch pass.

### Shared closure evidence

- [x] At least 90% of warm device time is phase-attributed on the selected
  changed-state artifact.
- [ ] A conservative and optimistic performance-gap budget reports disjoint
  measured phase shares, overlap handling, unattributed-time bounds, the
  matched formal timing target, and every pivot-rule input. It does not treat
  phase promotion thresholds or projected savings as measured end-to-end time.
- [ ] Every timed variant first passes the compile-inclusive 900-second gate:
  finite FP64 scalar value; exactly 461 finite FP64 gradient entries; successful
  inner/adjoint statuses; finite residual certificates; and native-versus-
  variant initial objective/gradient parity. Any timeout or failure prevents
  compilation-excluded timing and profiling for that variant.
- [x] Runtime command-buffer configuration and observed capture participation
  are both receipt-bound; an enable/disable control, if run, remains outside
  promotion timing.
- [ ] The Landau A100 either has a passing current preflight and a separate
  `C0` receipt, or has a machine-readable qualification blocker and contributes
  no timing claim. RTX 5090 and A100 samples, caches, and verdict inputs are not
  pooled.
- [x] One dense-Newton route has either passed all gates and been selected, or
  both dense routes have a documented bounded-negative stop result.
- [ ] Any selected factor handoff proves same-graph final-state/coil binding and
  numerical LU-to-`J` plus adjoint certificates, and removes the duplicate dense
  materialization/factorization without removing safety solves.
- [x] No scalar-adjoint fusion was selected: the measured fused route passed
  its gradient oracle but was stopped by its phase-speed and RSS gates.
- [ ] Every launch-fragmentation change has HLO/CUPTI evidence and exact-tail
  tests; unmeasured options remain deferred.
- [ ] Every attempted canary has raw timing, parity, memory, and provenance
  evidence from the immutable source manifest.

### Promotion branch

- [ ] The composed candidate improves warm p50 value-and-gradient time by at
  least 20% versus `C0`, with no more than 10% regression in p95, peak RSS, or
  peak GPU memory.
- [ ] `C1` or `C2` passes its own trajectory oracle, and native CPU/JAX CPU/JAX
  GPU objective, gradient, residual, and endpoint parity pass unchanged cross-
  lane tolerances.
- [ ] Production routing selects only the winning internal implementation and
  removes benchmark-only losing routes without adding public environment
  toggles.

### Bounded-negative branch

- [ ] Every planned variant was either measured or stopped by a named earlier
  gate, with the reason and raw evidence recorded.
- [ ] No candidate satisfied the 20% complete-path gate; production routing and
  numerical defaults are proven unchanged.
- [x] The authoritative result is explicitly engineering-bounded-negative and
  non-promoting; it is not represented as a campaign `LOSS` or `WIN`.
- [ ] If the performance-gap pivot rule fired, its attribution coverage,
  complete-path projection, remaining-lever bounds, and closure rationale are
  machine-readable. Reusable primitives are handed to a separately contracted
  formulation campaign rather than silently changing this lane's trajectory.

### Project speed goal complete

- [ ] A complete, raw, provenance-valid four-lane campaign receipt exists for
  the frozen candidate.
- [ ] The unchanged campaign validator reports `WIN`; no partial trajectory,
  diagnostic trace, or manually constructed receipt is used as the verdict.
- [ ] The authoritative results document records the winning outcome and
  identifies the exact source/runtime/device snapshot.

## Resolved Decisions and Handoff Questions

- `C1` and `C2` are both non-promoting; neither becomes the production
  trajectory contract.
- The attribution question is resolved at 98.354% of active device time.
- Command buffers already cover 73.639% of the classified device union; capture
  tuning is not the primary remaining lever for this route.
- `jax.linearize` is retained as a reusable dense-linearization primitive, not
  selected as the production Newton route.
- Production routing remains `C0` incremental GMRES. No adaptive selector or
  public algorithm flag is authorized.
- The formal complete-path/gap receipt remains an optional certification
  follow-up, not a prerequisite for starting separately contracted coupled-
  formulation research.
- The new coupled/fullspace plan must define which native baseline is eligible,
  its short-canary stop gate, exact mathematical and endpoint parity criteria,
  and the device-resident optimizer/state-machine boundary before code changes.
- Any future formal closure requires a superseding machine-readable completion
  audit bound to the v20-or-later evidence; `plan-completion-audit-v1.json`
  remains historical and must not be silently rewritten as current.
