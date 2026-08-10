# GPU-Native Coupled/Fullspace Single-Stage JAX Implementation Plan

**Status:** Ready for gated execution
**Last updated:** 2026-08-09
**SSOT:** This document is the sole execution plan for the coupled/fullspace
single-stage GPU work. The stopped faithful nested-route plan is historical
evidence, not an executable dependency.

## Goal command

```text
/goal Execute docs/single_stage_jax_gpu_coupled_fullspace_implementation_plan.md as the SSOT through all gated completion criteria. Implement and validate a GPU-native JAX single-stage formulation that intentionally departs from the sequential C++ execution structure and trajectory through batching, fusion, parallelism, and device-resident control, while preserving the same physics, mathematical problem, DOFs, constraints, gradients, and certified endpoint/KKT conditions. Done means a provenance-valid GPU solve faster than the 287.30421751597896 s native baseline, or a fully evidenced bounded-negative closure.
```

## Objective

Build a production-grade JAX single-stage solver that uses GPU strengths rather
than reproducing the sequential C++ execution schedule. The JAX implementation
may change the algorithm, optimizer trajectory, batching, derivative strategy,
and control flow. It must not change the physical or mathematical problem.

The performance target is the fastest recorded complete native engineering
trajectory: `287.30421751597896 s`. An engineering speed success requires a
complete, scientifically certified GPU result with objective
`<= 4.4822247e-08` in less wall time. Iteration counts are diagnostic only
because the new algorithm is intentionally allowed to take a different path.

This plan has two terminal dispositions:

1. **ENGINEERING_SPEED_GOAL_ACHIEVED:** a provenance-valid, complete GPU solve
   reaches the frozen certified-quality target faster than the historical
   engineering threshold of `287.30421751597896 s`. This is not by itself a
   formal campaign `WIN`.
2. **CLOSED_BOUNDED_NEGATIVE:** all authorized coupled/fullspace routes have
   been evaluated through their stated gates and none has a credible path to
   the target. This completes the investigation but does not achieve the speed
   goal and is not proof that every possible GPU formulation is impossible.

No partial trajectory, utilization percentage, microbenchmark, profiler trace,
or projected runtime is a speed verdict.

### Historical engineering target boundary

The `287.30421751597896 s` value is bound to the current-byte authoritative
record `docs/single_stage_speed_campaign_results.md`, SHA-256
`e4a95e54423c67369d5996f59b6f5a160f2a050c66aba2e216bd892b62e1763d`,
lines 113-125 as of 2026-08-09. Its boundary is the timestamp of the 1,000th
accepted native iteration. That native run reported an objective of
`4.4822247e-08`, but it was iteration-limited, was not terminal-stationary, and
did not produce a promoted endpoint certificate or complete campaign receipt.
The underlying claim-compatible raw baseline receipt is absent.

Consequently, this value is an engineering route-selection target only. The GPU
endpoint must independently pass this plan's full feasibility and KKT
certificate and must have objective `<= 4.4822247e-08`. A formal comparative
`WIN` additionally requires a validated identity-compatible baseline import or
a separately authorized matched native measurement. Until then, the formal
campaign verdict remains `NOT_PRODUCED`, even if the engineering speed goal is
achieved.

## Why this formulation is necessary

The stopped faithful route retained the nested C++ structure: an outer coil
step repeatedly invokes an inner Boozer Newton solve and implicit adjoint. On
GPU this becomes a staircase of dependent residuals, JVPs, Krylov iterations,
line-search decisions, and reverse traversals. The measured RTX profile
attributed 98.354% of active device time, but only 44.731% of the evaluation
envelope was device-active and 55.005% was inter-launch gap. Dense Newton,
single-JIT, and fused scalar-pullback canaries produced only small or negative
gains. Production therefore remains the `C0_INCREMENTAL_GMRES` nested route.

The coupled/fullspace route exposes coil and Boozer-surface variables together
and drives them with a compiled device-resident state machine. This creates the
opportunity to batch independent work, fuse objective and constraint
derivatives, keep optimizer state on device, and replace repeated nested solves
with parallel work over a joint state.

The historical gap is approximately `7541.455 / 287.30421751597896 = 26.25x`.
This plan therefore requires early time-to-certified-quality projections and
stops variants that offer only isolated constant-factor improvements with no
credible route to the complete target.

## Mathematical contract

Let `c` be the exact free coil DOFs and let `y` contain the Boozer-surface DOFs,
`iota`, and `G`, using the existing state ordering and conventions. The current
reduced problem is conceptually

```text
minimize_c  Phi(c, y*(c))
subject to  F_Boozer(c, y*(c)) = 0
            V(y*(c)) - V_target = 0,
```

where `y*(c)` is obtained through an inner nonlinear solve. The fullspace
problem is

```text
minimize_{c,y}  Phi(c, y)
subject to      F_Boozer(c, y) = 0
                V(y) - V_target = 0.
```

These formulations describe the same mathematical problem only when the same
Boozer equation and volume-label equality are satisfied on the same admissible
root branch. A finite penalty minimum with a nonzero Boozer or label residual is
not an equivalent solution. No new physical DOF is introduced: variables that
were solved inside `y*(c)` are exposed as constrained optimization coordinates.

### Frozen native term ledger

Phase 0 must convert this source-level ledger into a machine-readable contract
with source path, expression, sign, target, weight, dtype, reduction, and
tolerance for every row:

| Classification | Native term | Fullspace treatment |
| --- | --- | --- |
| Objective | `NonQuasiSymmetricRatio` | `Phi_non_qs`, weight 1 |
| Objective | `BoozerResidual` | `Phi_residual`, weight 1, retained exactly even though exact feasibility drives its residual to zero |
| Objective penalty | `0.5 * (iota - iota_target)^2` with identity mode | `Phi_iota`, weight 1; not a hard constraint |
| Objective penalty | `0.5 * (major_radius - major_radius_target)^2` with identity mode | `Phi_major_radius`, weight 1; not a hard constraint |
| Objective penalty | `0.5 * max(total_length - length_target, 0)^2` with max mode | `Phi_length`, weight 1; one-sided penalty, not an inequality constraint |
| Equality | exact Boozer residual equation | `F_Boozer(c, y) = 0` with the existing sign, normalization, grid, and state ordering |
| Equality | `Volume(surface) - initial_volume` label | `F_volume(y) = 0` with the exact signed-volume convention and target |
| Fixed state | first base current and every currently fixed coil/surface component | encoded outside `z`; never represented as an optimizable coordinate |
| Inactive terms | curvature, curve-curve, curve-surface, and surface-vessel weights equal to zero | remain exactly zero and are not silently promoted to constraints |

The authoritative source expressions are the native objective construction in
`examples/3_Advanced/single_stage_boozer_vacuum_optimization.py` and the
same-state explicit JAX terms that Phase 1 extracts and parity-tests. If the
machine-readable ledger disagrees with either source, execution stops before an
optimizer is built.

The admissible inner root is the branch reached by the authoritative exact
bootstrap and continued from that state. A candidate endpoint must be
reproduced by the authoritative exact solve seeded at candidate `y` without a
material branch switch. A different valid coil-design basin may be reported,
but it may not use a different inner root to broaden the reduced problem.

At a promotable endpoint, the formulation-specific certificate must establish
the equality-constrained KKT system. The frozen native problem has no hard
inequality constraint; its one-sided length term is part of `Phi`. In schematic
form,

```text
F_Boozer(c, y) = 0
F_volume(y) = 0
grad Phi + J_F^T lambda + J_volume^T nu = 0.
```

Augmented-Lagrangian, merit, penalty, trust-region, projection, or
preconditioning methods may steer the solve. None may replace the endpoint
equations, native penalty definitions, branch policy, or certificate. If a
future revision introduces a genuine hard inequality, its dual feasibility and
complementarity contract must be added explicitly before execution.

## Invariants and intentional freedoms

| Must remain invariant | May intentionally differ |
| --- | --- |
| NCSX input, currents, symmetry, grids, quadrature, and physical constants | Joint instead of nested coordinates |
| Exact free/fixed coil-DOF mapping; the first current remains fixed | Optimizer, line search, trust-region, or augmented-Lagrangian policy |
| Boozer residual equation, sign, normalization, state ordering, and `G` convention | Iteration count, accepted trajectory, and intermediate states |
| Objective terms, targets, weights, constraints, and scaling definitions unless a separately approved variable preconditioner leaves the mathematical problem unchanged | Static batching, `vmap`, `scan`, fusion, recomputation, and derivative blocking |
| FP64 authoritative state, residuals, gradients, KKT values, and endpoint observables | Device-resident optimizer and multiplier state |
| Correct derivatives of the implemented fullspace equations | Direct joint AD rather than a nested implicit adjoint |
| Finite endpoint, physical feasibility, cross-evaluator agreement, and field-line checks | Endpoint may lie in a different valid basin if it is separately classified and meets all quality gates |
| Source, runtime, device, environment, and artifact provenance | RTX 5090 or Landau A100 as the selected GPU, based on measured results |

Trajectory parity with C++ is neither required nor meaningful for this lane.
The existing strict-parity JAX example and nested production route remain
unchanged until this lane independently passes every promotion gate.

## DESC findings and reuse boundary

DESC demonstrates useful implementation patterns:

- packed multi-object state and explicit dependency routing;
- batched and blocked JVP/VJP evaluation with bounded-memory chunks;
- `vmap` plus `scan` handling for exact remainders;
- compiled inner numerical loops and explicit synchronization boundaries;
- reusable linear-constraint factorizations; and
- scan-based coil-distance evaluation and custom VJPs where justified.

DESC is not performance evidence for this problem and is not an implementation
of the target lane. Its standard stage-two QuadraticFlux optimization holds the
equilibrium fixed. Its equilibrium-constrained ProximalProjection path remains
nested, re-solves the equilibrium after changed outer states, disables JIT in
that wrapper, and uses Python outer optimizer loops. Uncommitted experimental
DESC worktree changes are excluded from the design basis.

We may reproduce independently validated execution patterns. We will not port
DESC code, infer a speedup from its source, or claim that DESC has already
solved the coupled device-resident problem.

## Proposed production architecture

### Pure fullspace objective core

Create `src/simsopt_jax/objectives/single_stage_fullspace.py` with immutable,
typed structures such as:

- `FullSpaceLayout`: canonical offsets, shapes, and pack/unpack operations for
  free coil DOFs, surface DOFs, `iota`, and `G`;
- `FullSpaceObjectiveConfig`: authoritative targets, weights, constraints,
  quadrature, scaling, and FP64 policy;
- `FullSpaceProblem`: frozen arrays and physical inputs; and
- `FullSpaceEvaluation`: named raw terms, weighted total, Boozer residual
  metrics, constraint metrics, and endpoint observables.

The public hot functions are pure JAX functions:

```text
evaluate(z, problem) -> FullSpaceEvaluation
value(z, problem) -> scalar
value_and_grad(z, problem) -> scalar, grad
constraints(z, problem) -> residual pytree
```

Promote or extract existing pure explicit-state term functions instead of
duplicating formulas or introducing a core-to-adapter dependency. Dense
Jacobian construction is excluded from the hot optimization graph and reserved
for bounded postsolve diagnostics when necessary.

### Adapter and feasible bootstrap

Create `src/simsopt_jax_adapters/geo/single_stage_fullspace.py` to:

- freeze the exact coil layout and free/fixed mapping;
- freeze the surface layout, quadrature, targets, weights, and constraints;
- obtain one authoritative feasible initial Boozer state from the existing
  exact solve; and
- transfer the complete immutable initial state to the GPU once.

The adapter owns Python object conversion only. It must not contain the hot
optimization loop or become a second definition of the physics.

### Device-resident constrained solve

Expose a typed public route under `src/simsopt_jax/solve/`. The initial canary
may reuse the existing fixed-shape `fused_stepwise` L-BFGS engine, but private
optimizer modules must not become public dependencies.

The promoting route must have:

- one initial host-to-device state transfer;
- a fixed-shape compiled `lax.scan` or `lax.while_loop` state machine;
- optimizer, multiplier, penalty/trust, and convergence state resident on
  device;
- no Python callback, progress callback, scalar extraction, or device-to-host
  transfer in the hot loop;
- static or masked device-side globalization decisions;
- batched evaluation of independent candidate or derivative work only where
  the mathematical acceptance rule is preserved;
- one final synchronization and device-to-host endpoint transfer; and
- a separate non-promoting diagnostic mode when iteration telemetry is needed.

`CFS-P0` is the weighted-penalty L-BFGS graph canary and is always
non-promoting. `CFS-AL1` is the frozen base coupled equality-constrained
augmented-Lagrangian route. `CFS-AL1-B` changes only its globalization schedule.
`CFS-AL2` is the separately identified inner-accuracy escalation authorized by
the sealed `CFS-AL1` result below.

The frozen `CFS-P0` coordinates and merit are explicit. Let `z0` be the
feasible bootstrap, let `s` be the elementwise variable scale in the route
policy, and optimize centered dimensionless coordinates
`u = (z - z0) / s`, so `u0 = 0` and `z = z0 + s * u`. Let `q(z)` contain the
254 masked Boozer equalities followed by the signed volume-label equality, and
let `D = diag(1/sqrt(254), ..., 1/sqrt(254), 1/abs(volume_target))`. The merit is
exactly `M0(u) = Phi(z(u)) + (10/2) * ||D q(z(u))||_2^2`. One fused
`value_and_grad(..., has_aux=True)` evaluation must compute `Phi`, `q`, `M0`,
and `grad_u M0` from one primal traversal.

For this route, optimizer stationarity means `||grad_u M0||_inf <= 1e-7`;
final physical KKT stationarity is separately certified and is not replaced by
this diagnostic stop. Canary feasibility means `||D q||_inf`; final named raw
residual gates still apply. L-BFGS uses memory 10, `ftol=0`, `gtol=1e-7`,
`maxfun=15000`, and `maxls=30`. The sequential route's `(1.0,)` records its
initial line-search step, not a restriction to only `alpha=1`. A changed-state
canary makes progress only when its final finite `M0` and final finite
`||D q||_inf` are both strictly below their initial values. Every accepted
solver state and the terminal state must be finite; any evaluated nonfinite
trial must be rejected on device and recorded by the solver's invalid-step
state rather than hidden by host control. The fused state carries a cumulative
nonfinite-evaluation count and an all-accepted-states-finite latch; both must
pass at the solve boundary.

The frozen 10-step changed state is defined in optimizer space, not by a
near-roundoff physical perturbation. For 716 coordinates let
`d = linspace(-1, 1, 716) / ||linspace(-1, 1, 716)||_2`; the canary starts at
`u_changed = 1e-3 * d`, hence `||u_changed||_2 = 1e-3`, and maps it back through
`z_changed = z0 + s * u_changed`. This changed-state definition is fixed before
the gate-valid canary and is identical across the RTX 5090 and A100 lanes.

For `CFS-AL1`, use the same centered `u`, scaled equality vector `c(u)=Dq(z(u))`,
and scaled multipliers. At inner stage `k`, optimize exactly
`L_k(u) = Phi(z(u)) + lambda_k^T c(u) + (rho_k/2)||c(u)||_2^2`. Initialize
`lambda_0=0` and `rho_0=10`. After every completed inner stage, including the
last, update `lambda_{k+1}=lambda_k+rho_k*c(u_k)` and
`rho_{k+1}=min(10*rho_k,250)`. Thus the stage penalty sequence is
`10,100,250,...`. Restart L-BFGS history at every stage because the objective
changes with `lambda` and `rho`; a normal 100-iteration inner limit advances to
the next AL stage, while abnormal/nonfinite status is fatal and masks remaining
stages on device. `maxfun=15000` is a per-inner-stage limit. Final raw-constraint
KKT evaluation converts scaled multipliers by
`lambda_raw = D^T lambda_scaled` and uses the post-update multiplier. The full
10-stage outer loop must be one compiled device dispatch with no Python stage
loop or mutable closure-constant substitution.

The sealed RTX `CFS-AL1` result on 2026-08-09 is a frozen nonconverged route,
not a tuning input that may silently alter its identity. It completed all ten
100-step stages in `13.829276024 s`, but every stage ended at its iteration
limit with augmented-gradient infinity norm between `2.0399956e-03` and
`1.8381740e-02`, compared with `1e-7`. Final scaled feasibility was
`1.3160195e-05`, raw KKT stationarity was `1.0417892e-01`, and objective was
`7.3369881e-05`; no sign, scaling, derivative, transfer, or nonfinite defect was
found. `CFS-AL1-B` is not selected because no required >=20% globalization
fragmentation evidence exists and only 8--13 evaluations beyond the 100 inner
iterations occurred per stage.

`CFS-AL2` is therefore a new route identity. It changes only the maximum inner
accuracy budget from 100 to 1000 iterations per stage (10000 total); it does
not change the mathematical problem or any multiplier, penalty, scaling,
line-search, physics, or certificate rule. The sealed AL1 timing projects a
worst-case synchronized solve time of approximately `138.3 s`, leaving about a
2.08x margin to the `287.30421751597896 s` native engineering threshold. Run
one immutable cold AL2 sample first. Proceed to endpoint audits and three warm
samples only if that sample passes feasibility, KKT, objective, finite-state,
transfer, and memory gates. Otherwise close AL2 bounded-negative without a
replay.

### Derivatives and GPU parallel work

Use the joint fullspace graph to eliminate the nested implicit-adjoint hot path.
Optimize in this order, driven by changed-state profiles:

1. compute the primal once and form the joint scalar gradient once;
2. batch coil, point, quadrature, objective-block, and constraint-block work
   within measured memory limits;
3. use blockwise JVP/VJP operations for KKT or constraint products;
4. fuse compatible raw terms before reverse mode to reduce graph traversals;
5. use exact-tail `vmap`/`scan` decomposition without padding physical data;
6. retain state-bound factorizations only within a mathematically identical
   state and residual graph; and
7. consider a custom VJP only after a profile identifies a repeated reverse
   traversal and same-state derivative tests prove equivalence.

Buffer donation, compilation cache, command-buffer participation, and bounded
recomputation are implementation levers, not scientific changes. Mixed
precision is excluded from this plan.

### Endpoint certification

Create a formulation-specific fullspace certificate rather than fabricating
the nested schema's `inner_success`. It must include:

- normalized optimizer termination state;
- finite FP64 state, objective, constraints, multipliers, and gradients;
- Boozer residual norm and volume-label residual;
- KKT stationarity and primal feasibility for the Boozer and volume-label
  equalities;
- iota, major radius, one-sided length penalty, and all inactive hardware-term
  metrics;
- fixed-current and fixed-DOF preservation;
- objective and quality comparison against the native reference;
- native evaluation of the JAX endpoint and JAX evaluation of the native
  endpoint;
- optimizer-independent Poincare closure and traced-iota spot checks; and
- basin classification when endpoints are materially different.

A final exact projection may be used for certification only if the projected
state remains within the frozen state/quality tolerances and does not conceal a
failed fullspace solve. Record both pre- and post-projection values. A material
projection change fails the route.

### Separate lane and receipts

Add a new example, suggested as
`examples/jax/3_Advanced/single_stage_boozer_vacuum_fullspace.py`, with a
distinct example ID, route ID, immutable snapshot, receipt schema, and
validator. Do not place this behind a flag in the strict-parity example and do
not reuse the nested campaign receipt unchanged.

The receipt declares `trajectory_equivalence_required: false` and binds the
mathematical contract, source snapshot, executed-source hashes, runtime,
backend, physical GPU identity, environment policy, inputs, timing boundary,
endpoint artifacts, and certificate.

### Finite authorized route matrix

The authorized implementation set is exactly the following. A new algorithm
requires an explicit revision of this SSOT before code or measurements begin.

| Route ID | Algorithm and purpose | Promotion status | Escalation/terminal rule |
| --- | --- | --- | --- |
| `CFS-P0` | Joint weighted-penalty objective on the existing public fused fixed-shape L-BFGS route | Diagnostic only | Establishes same-state physics, derivative, compile, transfer, memory, and time-to-progress viability. It cannot produce a speed success, even if feasible. Failure of a shared physics/derivative/transfer gate stops the tranche and emits `NOT_SELECTED_BY_GATE` receipts for both downstream routes. A performance failure proceeds no further unless `CFS-AL1` has independent justification in the Phase-0 budget; otherwise both downstream routes receive `NOT_SELECTED_BY_GATE`. |
| `CFS-AL1` | Device-resident equality-constrained augmented Lagrangian with fixed-shape multiplier/penalty stages and fused L-BFGS inner steps | Sole base promoting route | Proceeds only after `CFS-P0` correctness gates. A feasibility, KKT, memory, or projected-time failure rejects the route. A complete certified speed success ends execution. |
| `CFS-AL1-B` | Same equations and augmented-Lagrangian state as `CFS-AL1`, with static batched/masked globalization candidates | Conditional promoting escalation | Selected only when `CFS-AL1` passes correctness and feasibility progress, the changed-state trace attributes at least 20% of the solve envelope to globalization launch/gap fragmentation, and the frozen memory budget permits the batch. Otherwise record `NOT_SELECTED_BY_GATE`. |
| `CFS-AL2` | Same equations, scaling, multiplier/penalty schedule, and one-dispatch graph as `CFS-AL1`, with inner solves capped at 1000 iterations per stage | Evidence-selected promoting accuracy escalation | Selected because all ten sealed AL1 inner solves exhausted 100 iterations far above inner stationarity while the measured 10000-iteration projection remains below the native threshold. One cold sample decides whether endpoint audits and warm samples proceed. |

Phase 0 froze every numerical knob for the original routes; the dated AL1
evidence paragraph above is the explicit pre-implementation authorization and
freeze record for `CFS-AL2`. Frozen knobs include centered variable scaling,
multiplier initialization/update, penalty schedule and ceiling, L-BFGS memory,
`ftol`, `gtol`, `maxfun`, `maxls`, iteration/stage budgets, globalization
candidates, batch widths, convergence tolerances, progress rule, and failure
policy. No executed route may be silently overwritten or reinterpreted.

Projected SQP, interior-point, trust-region, mixed-precision, and alternative
penalty schedules are outside this tranche. Bounded-negative closure is
achievable when every route has either an executed terminal disposition or a
validated `NOT_SELECTED_BY_GATE` receipt that names the upstream gate and raw
evidence making execution invalid or unnecessary. The results must list these
excluded designs as untested rather than ruled out.

### File ownership and dependency order

| Owner | Planned path | Responsibility and dependencies |
| --- | --- | --- |
| Physics ledger and pure objective | `src/simsopt_jax/objectives/single_stage_fullspace.py` | Typed layout, raw terms, equations, pack/unpack, value/gradient, and constraint products. Extracts public pure functions from existing adapter-private term code; contains no Python object adapters. |
| Native-object adapter/bootstrap | `src/simsopt_jax_adapters/geo/single_stage_fullspace.py` | Freezes native coils, surface, targets, grids, and one exact feasible initial state; depends on the pure core and existing Boozer adapter only. |
| Public solve contract | `src/simsopt_jax/solve/fullspace.py` | Typed route/options/results and public dispatch for `CFS-P0`, `CFS-AL1`, and `CFS-AL1-B`; reuses a promoted public seam over fixed-shape fused L-BFGS without importing private optimizer APIs. |
| Endpoint certificate | `src/simsopt_jax/solve/fullspace_certificate.py` | Fullspace feasibility, KKT, branch, finite-state, and fixed-DOF certificate; does not fabricate nested `inner_success`. |
| User-facing example | `examples/jax/3_Advanced/single_stage_boozer_vacuum_fullspace.py` | Separate fullspace example/ID; strict-parity example remains unchanged. |
| GPU runner and artifact producer | `benchmarks/run_single_stage_fullspace_gpu.py` | Snapshot-bound first/10/100/full phases, synchronized timing, transfer audit, immutable raw artifacts, and no native execution. |
| Receipt and validator | `benchmarks/single_stage_fullspace_receipt.py`, `benchmarks/validate_single_stage_fullspace_campaign.py` | Canonical schema, artifact digests, live runtime/device binding, baseline classification, terminal disposition, and fail-closed validation. |
| Cross-evaluator/field-line audit | `benchmarks/single_stage_fullspace_endpoint_audit.py` | Native-on-JAX, JAX-on-native, root-branch reproduction, Poincare closure, and traced-iota evidence. |
| Results SSOT | `docs/single_stage_jax_gpu_coupled_fullspace_results.md` | Phase dispositions, exact commands/hashes, selected device/route, final outcome, and untested routes. |
| Focused tests | `tests/jax/objectives/test_single_stage_fullspace.py`, `tests/jax/solve/test_fullspace_solver.py`, `tests/jax/solve/test_fullspace_certificate.py`, `tests/benchmarks/test_single_stage_fullspace_campaign.py` | Same-state terms/derivatives, device state machine, certificate, receipt, provenance, and routing boundaries. |

Dependency order is ledger/core, adapter/bootstrap, derivatives/certificate,
public solver, runner/receipts, short canaries, endpoint audit, then complete
performance. The existing strict-parity lane and campaign files are consumers
of none of these modules until a separate promotion change.

## Hardware and measurement policy

- Use the local RTX 5090 and Landau A100 as separate decision lanes. Never pool
  samples across devices or hosts.
- Revalidate the A100 driver, CUDA compatibility stack, dependency overlay,
  physical GPU UUID, and source/runtime identity before using its results.
- Use GPU-only short development canaries. Do not rerun the native CPU solve as
  routine development work; the historical baseline is sufficient for route
  selection.
- Select the primary GPU by measured time-to-certified-progress after both
  devices pass the same-state correctness and transfer gates. The A100's FP64
  hardware makes it a decision input, not merely a portability run.
- Report compilation separately from warm execution. A speed claim must also
  disclose cold end-to-end time.
- Time with explicit synchronization at the start and end of the solve. No
  asynchronous dispatch timing is valid.
- The primary metric is time to a certified endpoint of equal-or-better
  quality. A fixed 1000-iteration result is secondary because algorithms and
  iteration semantics differ.
- A historical engineering comparison may use `287.30421751597896 s`. A formal
  campaign claim requires either a validated identity-compatible baseline
  import or a separately authorized matched native measurement.

Development canaries run once per route/device/phase. The selected complete
route runs once cold and exactly three warm times in fresh isolated processes,
with no outlier deletion or replacement. Each warm run must independently pass
the full endpoint certificate, each synchronized solve time must be below
`287.30421751597896 s`, and the median is reported with the full three-sample
range. A warm solve has a 360 s solve-boundary timeout; the containing process
has a 900 s timeout to accommodate compilation and artifact writing. A timeout
is a failed sample, not an outlier.

Warm timing starts after the compiled executable and initial device state are
ready and synchronized, immediately before the device-resident solve dispatch.
It ends after the final state is blocked ready and before endpoint artifact
serialization. Cold process, compilation/cache-load, endpoint audit, and total
end-to-end times are reported separately. A 10% margin
(`<= 258.57379576438106 s` for all three warm solves) is the preferred formal
promotion target, but it does not cure the missing claim-compatible native
baseline.

The mandatory artifact chain is:

```text
immutable snapshot
-> same-state correctness
-> first-evaluation transfer audit
-> 10-step tuning canary
-> frozen route knobs
-> 100-step projection canary
-> one complete cold solve
-> endpoint/cross-evaluator/field-line audit
-> three complete warm solves
-> receipt validator
-> independent audit
```

The runner implemented in this plan must make these commands live and record
their exact argv in every receipt:

```text
.venv-qn-gpu/bin/python benchmarks/run_single_stage_fullspace_gpu.py --phase first-eval --route CFS-P0 --device <rtx5090|a100> --output <artifact-dir>
.venv-qn-gpu/bin/python benchmarks/run_single_stage_fullspace_gpu.py --phase canary --steps 10 --route <route> --device <rtx5090|a100> --output <artifact-dir>
.venv-qn-gpu/bin/python benchmarks/run_single_stage_fullspace_gpu.py --phase canary --steps 100 --route <route> --device <selected> --output <artifact-dir>
.venv-qn-gpu/bin/python benchmarks/run_single_stage_fullspace_gpu.py --phase complete --sample <cold|warm-1|warm-2|warm-3> --route <selected> --device <selected> --output <artifact-dir>
.venv-qn-gpu/bin/python benchmarks/single_stage_fullspace_endpoint_audit.py --campaign <artifact-dir>
.venv-qn-gpu/bin/python benchmarks/validate_single_stage_fullspace_campaign.py --campaign <artifact-dir>
```

## Implementation phases and gates

Each phase is fail-closed. A later phase may not begin until the listed outputs
and tests exist. Gate failures are recorded; they are not patched around by
weakening tolerances or changing the problem.

### Phase 0 — Freeze the contract and evidence boundary

Tasks:

- [x] Record the exact source-state hash and dirty/untracked scope before edits.
- [x] Materialize and validate the frozen native term ledger, joint layout,
      equations, penalties, authoritative FP64 policy, branch rule, endpoint
      tolerances, and historical engineering target classification.
- [x] Freeze the three-route matrix and every route knob; define the example ID,
      receipt schema, validator, snapshot manifest, and live runtime/device
      provenance contract.
- [x] Encode the protocol's divergent-lane rules: trajectory comparisons are
      void, accepted endpoint physics is authoritative, and cross-evaluator
      plus field-line checks are mandatory.
- [x] Define memory, compilation, transfer, and time-to-progress canary budgets
      before collecting performance data.
- [x] Publish a baseline route-budget artifact showing the required 26.25x gap
      and expected graph-traversal/launch reductions for each candidate.
- [x] Implement and test the exact runner/validator CLI contract and artifact
      chain before any GPU timing is collected.

Gate:

- The schema and validator reject missing source/runtime/device identity,
  partial runs, noncanonical artifacts, failed endpoint checks, and any attempt
  to substitute trajectory parity for endpoint certification. The ledger,
  finite route matrix, replay counts, timeouts, and timing boundaries are
  machine-readable and immutable for later phases.

### Phase 1 — Pure joint state and same-state physics parity

Tasks:

- [x] Implement the typed immutable joint layout and exact pack/unpack path.
- [x] Extract/promote pure explicit-state objective and constraint terms into
      the core without formula duplication or dependency inversion.
- [x] Build the adapter and one-time feasible bootstrap.
- [x] Implement pure joint evaluation and named raw-term reporting.

Gate:

- Layout round-trips exactly; fixed DOFs are absent from the optimized vector;
  state ordering and `G` convention are exact. At identical `(c, y)`, every raw
  term, weighted total, Boozer residual, normalization, label error, and
  observable matches the existing authoritative evaluator within frozen FP64
  tolerances.

### Phase 2 — Joint derivatives and certificate primitives

Tasks:

- [x] Implement joint value-and-gradient and constraint JVP/VJP products.
- [x] Add directional finite-difference checks and JVP/VJP transpose tests on
      nonsymmetric, changed-state cases.
- [x] Implement fullspace feasibility and KKT certificate primitives.
- [ ] Add cross-evaluator endpoint and field-line audit interfaces.

Gate:

- All derivative checks pass in FP64 on CPU reference tests and both selected
  GPU environments. First-evaluation value and gradient are finite. No nested
  implicit adjoint is executed in the fullspace hot gradient.

### Phase 3 — Device-resident solver canary

Tasks:

- [x] Expose a public typed fused solve route with immutable options.
- [x] Implement the fixed-shape coupled constrained state machine.
- [x] Keep primal, optimizer, multiplier, penalty/trust, and convergence state
      on device for the full hot loop.
- [x] Add diagnostic and promoting modes with an explicit non-promotion marker
      on callback/telemetry mode.
- [x] Add transfer instrumentation around the true solve boundary.

Gate:

- A changed-state canary proves exactly one initial H2D boundary, zero D2H
  transfers or callbacks inside the compiled solve, and one final synchronized
  D2H boundary. The canary makes certified feasibility/objective progress and
  has finite state throughout.

### Phase 4 — Parallelism, fusion, and memory tuning

Tasks:

- [ ] Profile kernel duration, device-busy share, inter-launch gaps, command-
      buffer participation, graph submissions, compilation, and peak memory.
- [ ] Tune bounded derivative/objective/coil/point/quadrature blocks on RTX 5090
      and A100 independently.
- [ ] Fuse compatible primal and reverse traversals.
- [ ] Add buffer donation and state-bound reuse where live-buffer and aliasing
      tests prove safety.
- [ ] Compare static masked globalization with batched candidate evaluation only
      if both preserve the selected mathematical acceptance rule.

Gate:

- Same-state and endpoint canaries remain within all scientific tolerances, no
  physical padding is introduced, memory stays below the frozen device budget,
  and the measured launch/traversal reduction is large enough for the route's
  complete-path projection to remain credible.

Kill rule:

- Stop a variant if its measured 100-step time-to-certified-progress projection
  cannot reach `287.30421751597896 s` after applying the Phase-0 frozen
  conservative projection margin and remaining certification-cost estimate. Do
  not run a full campaign to confirm an already-decisive negative projection.

### Phase 5 — Short correctness and scaling canaries

Run only the minimum staged evidence needed:

- [x] first changed-state value/gradient and transfer audit;
- [x] 10-step compile/finite/progress canary;
- [x] 100-step projected-time and feasibility canary; and
- [ ] one problem-scale memory/compile canary without completing the solve.

Gate:

- The route is finite, improves the frozen merit and feasibility measures,
  preserves all fixed state, remains within memory/compile budgets, and has a
  credible measured path to a certified endpoint below the native target.

Only a passing route proceeds to a complete GPU solve. This phase must not
multiply replays merely to improve confidence in a failing route.

### Phase 6 — Complete certified GPU solve

Tasks:

- [x] Run the selected route to its frozen termination rule on the selected
      GPU from an immutable snapshot.
- [ ] Produce the complete raw timing, endpoint, KKT, provenance, memory, and
      environment artifacts.
- [ ] Run native-on-JAX and JAX-on-native endpoint evaluation.
- [ ] Run Poincare closure and traced-iota spot checks.
- [ ] Classify basin equivalence or valid different-basin quality.
- [ ] Run exactly one cold and three warm isolated complete solves, with no
      sample deletion or replacement, after route and device selection.

Gate:

- Every scientific and provenance gate passes; every one of the three complete
  warm solve times is below `287.30421751597896 s`; and their median/range is
  reported. Otherwise the engineering route is not a speed success.

### Phase 7 — Promotion or bounded-negative closure

For `ENGINEERING_SPEED_GOAL_ACHIEVED`:

- [ ] Make the fullspace route an explicit opt-in production route.
- [ ] Keep `C0_INCREMENTAL_GMRES` as rollback until broader regression and
      portability validation passes.
- [ ] Produce an engineering receipt and results document that distinguish the
      historical target from any later formally matched baseline and keep the
      formal campaign verdict `NOT_PRODUCED` until that separate gate passes.
- [ ] Promote only after independent receipt, source/runtime, math, and endpoint
      audit.

For `CLOSED_BOUNDED_NEGATIVE`:

- [x] Record every attempted route, frozen gate, raw artifact, and exact failure
      mechanism.
- [x] Prove production remained unchanged and no partial evidence was promoted.
- [x] State which fullspace designs were ruled out and which remain untested.
- [x] Use `NOT_PRODUCED` when a complete claim-bearing comparison does not exist;
      do not relabel it `LOSS`.

## Required tests

Implement focused one-file test processes for at least:

- joint layout, pack/unpack, state ordering, and fixed-DOF preservation;
- every raw fullspace term versus the authoritative explicit-state evaluator;
- Boozer residual equation, normalization, label equation, `iota`, and `G`;
- directional gradients and JVP/VJP transpose identity;
- CPU/GPU same-state FP64 parity;
- constrained optimizer state transitions and masked termination;
- zero hot-loop callbacks and device-to-host transfers;
- endpoint feasibility, equality-constrained KKT, branch reproduction, and
  failure classification;
- cross-evaluator endpoint audits and field-line checks;
- receipt schema, canonical serialization, artifact digests, snapshot identity,
  live runtime/device identity, and incomplete-run rejection; and
- promotion/rollback routing so the strict-parity example and `C0` default are
  unchanged before promotion.

Use the repository's validated environment and run each pytest file in a fresh
process to control JAX memory accumulation. Required post-flight checks are
Ruff, formatting, compileall, `git diff --check`, source/runtime provenance,
and a scoped dirty-tree audit. No test count alone establishes scientific or
performance completion.

## Fail-closed stop conditions

Immediately stop and mark the current route non-promoting on any of:

- a raw-term, state-layout, sign, normalization, target, weight, or constraint
  mismatch;
- optimization of a fixed current or fixed geometry DOF;
- nonfinite value, gradient, state, multiplier, residual, or observable;
- a hot-loop host callback, scalar extraction, synchronization, or D2H transfer;
- a penalty-stationary but infeasible endpoint;
- failed KKT, field-line, cross-evaluator, branch, or frozen objective-term
  certificate;
- material endpoint change during final exact projection;
- dense or retained workspace exceeding the frozen memory budget;
- incomplete or noncanonical raw artifacts;
- source, snapshot, runtime, environment, or physical-device mismatch;
- a complete-path projection that cannot credibly beat the native target; or
- a performance claim based on partial iterations, utilization, or unmatched
  hardware rather than a complete certified result.

An optimizer failure rules out that configured route. It does not prove the
coupled mathematical formulation or all GPU approaches impossible.

## Risks and mitigations

- **Mixed-unit conditioning:** define an invertible variable preconditioner and
  prove that it leaves the physical equations and endpoint invariant.
- **Penalty compromise:** use KKT-consistent multiplier updates and reject any
  endpoint that misses primal feasibility, regardless of objective value.
- **Compile or memory explosion:** block derivatives and objectives using
  profile-driven static chunks; stop before full solves when the frozen budget
  is exceeded.
- **Dynamic control fragments capture:** use fixed-shape masked device state and
  verify command-buffer participation from a trace rather than configuration.
- **Reduction-order drift:** require frozen FP64 same-state and endpoint
  tolerances; do not require bitwise trajectory identity.
- **Different coil-design basin:** classify it explicitly, preserve the
  continuation-connected inner root, and require all physics, KKT,
  cross-evaluator, field-line, and objective-quality gates.
- **Stale factorization:** bind every retained factor or linearization to the
  exact state, residual graph, dtype, grid, and weights; never accept public
  factor injection.
- **Misleading telemetry:** keep diagnostic callback runs non-promoting and use
  synchronized callback-free runs for timing.
- **DESC overinterpretation:** reuse only independently tested patterns and make
  no DESC-derived speed claim.
- **Dirty worktree/provenance:** preserve unrelated work, isolate scoped files,
  and publish immutable source/runtime identity before claim-bearing runs.

## Explicit non-goals

- Line-by-line or trajectory-faithful reproduction of the C++ optimizer.
- Weakening physics, equations, constraints, endpoint tolerances, or FP64
  authority to obtain speed.
- Treating a finite penalty objective as equivalent without feasibility/KKT.
- Replacing the strict-parity example or production `C0` route before promotion.
- Importing DESC code or claiming DESC performance transfers to SIMSOPT.
- Mixed-precision state or results in this tranche.
- Routine native CPU reruns during GPU development.
- Full campaigns for variants already rejected by short, decisive gates.
- A formal speed claim from the historical baseline without a valid baseline-
  import contract or separately authorized matched native evidence.

## Definition of done

The SSOT is complete only when one terminal disposition is recorded with exact
artifacts and an independent audit.

### Speed objective achieved

All of the following are required:

- [ ] Same mathematical problem, physical inputs, DOFs, equations, objective
      terms, weights, and constraints are demonstrated by tests and receipts.
- [ ] Joint gradients and constraint products pass frozen derivative checks.
- [ ] The promoting hot solve is device-resident and callback-free.
- [ ] The endpoint passes feasibility, equality-constrained KKT, branch,
      cross-evaluator, field-line, finite-state, and frozen objective-quality
      gates.
- [ ] The source, runtime, environment, physical GPU, timing boundary, and all
      result artifacts are provenance-bound and independently validated.
- [ ] Exactly three complete warm GPU solves all finish below
      `287.30421751597896 s`; their median and range are reported, and cold
      compile/end-to-end time is reported separately.
- [ ] The result is labelled an engineering speed achievement; a formal `WIN`
      remains `NOT_PRODUCED` until the baseline-import or matched-native
      contract is independently satisfied.
- [ ] Production promotion is explicit, reversible, and independently audited.

### Bounded-negative investigation complete

All of the following are required:

- [x] Each of `CFS-P0`, `CFS-AL1`, `CFS-AL1-B`, and `CFS-AL2` has either an
      executed predefined terminal disposition or a `NOT_SELECTED_BY_GATE`
      disposition in the authoritative results record bound to the decisive
      upstream gate and raw evidence.
- [x] Raw artifacts establish the scientific, transfer, compilation,
      and timing failure mechanism without extrapolating beyond the evidence.
- [x] No incomplete or non-promoting result is described as a speed verdict.
- [x] Production remains on the last validated route.
- [x] The closure identifies untested formulations and does not claim universal
      impossibility.

A bounded-negative closure satisfies execution of this plan but does **not**
mean the speed objective was achieved.
