# JAX solver production architecture: blindspot report

Date: 2026-07-28  
Live source snapshot: `9fbb5697098120f426f0d3c7be30653f9e7d5684`

## Verdict

The clean production design is **not** to expose Optax, Optimistix, Lineax,
SciPy, and SIMSOPT as peer user-selectable drivers.

Expose mathematical problem types and backend-neutral solve policy. Internally,
assign one default implementation to each problem family:

| Problem family | Production owner | Reference or fallback |
|---|---|---|
| Smooth, deterministic, unconstrained scalar minimization | Optax L-BFGS | SciPy CPU reference |
| Stochastic or deliberately fixed-step first-order minimization | Optax Adam | SIMSOPT differential fixtures during migration |
| Box-constrained scalar minimization | SciPy L-BFGS-B on CPU | No silent GPU fallback |
| Dense or modest least squares | Optimistix LM with Lineax QR | SciPy LM reference |
| Large matrix-free least squares | Optimistix LM with a certified Lineax iterative solve | QR/FP64 or CPU reference after an explicit policy decision |
| Nonlinear roots | Optimistix Newton with Lineax | SIMSOPT acceptance and certification |
| Linear systems | Lineax, selected from declared operator structure | Certified higher-precision or direct solve |
| QFM ALM, MwPGP/GPMO, and GSCO | SIMSOPT-owned outer algorithms | Existing native/reference lanes |
| Mixed-precision acceptance and fallback | SIMSOPT-owned policy | FP64 authority |

This is not a claim that these libraries reproduce C++ or SciPy trajectories.
They do not. The production guarantee should be a backend-neutral scientific
certificate, not identical internal iterations.

## Evaluation criteria

A reliable upstream choice must satisfy all of these:

1. Users specify the mathematical problem, tolerances, budgets, and placement;
   they do not need to know the implementation library.
2. A backend upgrade cannot silently change the public result schema.
3. `success` means the same thing across implementations and is derived from a
   SIMSOPT-owned certificate.
4. Unsupported combinations fail explicitly. They never transfer to CPU or
   change algorithms silently.
5. Device-resident workflows do not materialize through NumPy between inner
   solver stages.
6. Exact evaluation counts are reported only when they are actually measured.
7. Dependency upgrades are qualified against a fixed acceptance matrix before
   becoming the supported stack.
8. Domain algorithms retain their SIMSOPT-specific state transitions and
   acceptance rules.

## What the live source currently exposes

The current typed API has 15 backend-specific `Driver` members in
`src/simsopt_jax/solve/driver.py`. Public contracts then expose backend-specific
option classes, callback event types, raw status domains, and result metadata in
`src/simsopt_jax/solve/contracts.py` and `src/simsopt_jax/solve/__init__.py`.
That makes dependency choices part of the long-term public API.

The current `OptimizerResult` is a host result: `x`, gradients, residuals, and
Jacobians are NumPy arrays. `src/simsopt_jax/solve/dispatch.py` always
materializes the backend result on the host. The higher-level serial solve then
uses `jax.device_put(result.x, ...)`, creating a device-to-host-to-device
round-trip.

The current Optax wrapper JIT-compiles an individual update but runs the outer
iteration in Python. Each iteration blocks and transfers a gradient norm to the
host. It therefore is not a fully staged device-resident solve.

The high-level serial solvers also mutate `problem.x` and unconditionally write
a timestamped `simsopt_*.dat` file. Those are workflow policies, not properties
of a numerical solve, and should not occur inside the functional core.

The public scalar and least-squares problem types do not carry bounds. The
publicly named `SIMSOPT_LBFGSB` route therefore does not constitute a supported
box-constrained JAX API even though private bounded kernels exist.

Adam remains used by legacy optimizer routes, including Boozer-surface paths.
It must be migrated at those call sites, but it should not become the default
deterministic scalar solver. Adam is appropriate only when the problem contract
declares stochastic/noisy gradients or deliberately requests a fixed-step
first-order schedule.

## Recommended public architecture

```text
user problem + neutral policy
            |
            v
  SIMSOPT semantic dispatcher
  - validates capabilities
  - selects one internal owner
  - never silently changes placement
            |
   +--------+---------+----------+
   |                  |          |
 Optax          Optimistix    SciPy CPU
 Adam/L-BFGS      + Lineax    bounds/reference
   |                  |          |
   +--------+---------+----------+
            |
            v
 SIMSOPT certificate + normalized termination
            |
      device result or
      explicit host result
```

### Stable problem contracts

Use separate typed problem contracts:

- `SmoothScalarProblem`
- `StochasticScalarProblem`
- `BoxConstrainedScalarProblem`
- `LeastSquaresProblem`
- `RootProblem`
- `LinearProblem`

Bounds, residual structure, operator structure, and stochastic semantics belong
to the problem type. They must not be inferred from a driver name.

### Stable policy

The stable policy should contain only externally meaningful controls:

- absolute and relative scientific tolerances;
- maximum accepted steps and, where measurable, evaluation budgets;
- precision and certification policy;
- device placement;
- memory budget;
- explicit progress or trace policy;
- whether a failed certificate raises or returns a result.

Do not expose upstream option objects. Do not make the internally selected
algorithm a general configuration knob. If expert override is temporarily
needed for qualification, place it in an explicitly unstable or test-only API.

### Two result layers

Separate:

1. an internal device `Solution` PyTree, consumed without host transfer by
   domain algorithms and nested solves; and
2. a public host `SolveResult`, materialized once when the caller requests a
   host boundary.

The public result should have a SIMSOPT-owned `TerminationReason`, for example:

- `CONVERGED_GRADIENT`
- `CONVERGED_RESIDUAL`
- `CONVERGED_STEP`
- `MAX_STEPS`
- `MAX_EVALUATIONS`
- `LINE_SEARCH_FAILED`
- `NUMERICAL_FAILURE`
- `CALLBACK_STOP`
- `CERTIFICATION_FAILED`

`success` is derived from the normalized reason plus the problem-family
certificate. Raw upstream diagnostics may be retained in a nested diagnostic
object, but they are not the public contract.

Counters must be semantically named and optional when unavailable:

- accepted steps;
- objective evaluations;
- gradient or Jacobian evaluations;
- line-search evaluations;
- linear solves and inner iterations.

Do not synthesize exact counts such as `nfev = nit + 1` when a library performs
unobserved line-search evaluations.

### Progress and logging

Use one backend-neutral progress event. Default execution has no per-step host
callback. Device traces use fixed-shape, bounded telemetry with an explicit
sampling interval. A host callback is an explicit host-controlled mode.

The solve core returns a value and does not mutate the input problem or write
files. State publication and objective-log writing belong in optional workflow
adapters.

## Algorithm-selection rules

Selection follows declared mathematical structure, never a label such as
`fast`, `parity`, `cpu`, or `gpu`.

### Scalar minimization

- Deterministic, smooth, unconstrained: Optax L-BFGS.
- Stochastic/noisy or deliberate fixed-step schedule: Optax Adam.
- Box constrained on CPU: SciPy L-BFGS-B.
- Box constrained on GPU: reject as unsupported in the first production
  contract unless a real user requirement justifies completing and certifying
  a bounded JAX implementation.

Optax and Optimistix L-BFGS belong to the same broad algorithm family but have
different line searches, history policies, stopping tests, and counters. The
current empirical comparison found the same broad minimizer basin, not general
trajectory or stopping parity. Optax is the better default here because it is
already the natural owner of Adam, its L-BFGS lane performed well in the local
fixtures, and it has the stronger institutional maintenance base. This is not a
claim that Optax is universally superior to Optimistix.

### Least squares and roots

- Prefer QR for dense, modest, ill-conditioned, or rank-sensitive least-squares
  problems.
- Use an iterative Lineax solve only when the operator is too large to
  materialize and the output passes a condition-aware certificate.
- Root solves require more than a small residual. Ill-conditioned systems also
  need a forward-error bound or a domain invariant insensitive to
  non-identifiable parameter directions.

The local parity experiment found Optimistix LM plus QR matched the SciPy
reference closely on the tested ill-conditioned fixture, while the LSMR route
did not. Therefore LSMR must not be the unconditional LM default.

### Linear solves

Choose from declared operator properties:

- positive-definite: CG;
- general square matrix-free: GMRES;
- rectangular least squares: LSMR;
- dense/modest or certification fallback: QR or a direct factorization.

Lineax should own the generic kernels. SIMSOPT should own selection,
certification, iteration budgets, precision escalation, and failure policy.

### Domain algorithms

QFM ALM, MwPGP/GPMO, and GSCO are not commodity replacements for a library
solver. Keep their outer state machines in SIMSOPT. Replace only their generic
inner linear/nonlinear kernels after differential tests.

## Dependency and release policy

Optax, Optimistix, and Lineax should all sit behind SIMSOPT adapters.
Optimistix and Lineax remain pre-1.0 projects and have shipped breaking changes
in recent minor releases. The current package and CI pin JAX exactly but leave
those libraries on open-ended lower bounds. That is not a reproducible
production stack.

Use:

1. compatible ranges in package metadata where ecosystem interoperability
   requires them;
2. one exact, published constraints file for the blessed JAX stack used by
   release and merge-blocking CI;
3. a latest-compatible canary lane that detects upstream changes without
   changing the blessed stack;
4. an explicit qualification PR to update the constraints file;
5. no upstream library types in SIMSOPT's stable public API.

The JAX extra also needs an explicit Python 3.11+ contract. The project-wide
metadata currently advertises Python 3.8+, while JAX 0.10 and the selected
solver stack require newer Python.

Current primary-source maintenance evidence:

- [Optax releases](https://github.com/google-deepmind/optax/releases) show an
  actively maintained DeepMind project and the 0.2.8 release.
- [Optimistix releases](https://github.com/patrick-kidger/optimistix/releases)
  show the 0.1.0 production dependency and its breaking changes.
- [Lineax releases](https://github.com/patrick-kidger/lineax/releases) show the
  0.1.1 production dependency, breaking changes, and LSMR fixes.
- [SciPy releases](https://github.com/scipy/scipy/releases) provide the stable
  CPU reference lifecycle.

## Prioritized blindspots and forced decisions

### P0: Does v1 promise GPU-resident box constraints?

**Recommendation: no.** Support SciPy L-BFGS-B as the explicit CPU bounded
lane. Reject GPU-bounded requests. Do not preserve or finish a full custom JAX
L-BFGS-B speculatively.

How to evaluate a future change: require at least one real production workload
where host execution is unacceptable, then certify active bounds, projected
gradient/KKT conditions, line-search failure semantics, CPU/GPU behavior, and
nonfinite handling.

### P0: What exactly makes a result successful?

Define a certificate per problem family before choosing defaults. A library
success flag is insufficient. This is the most important missing contract after
the bounds decision.

How to evaluate:

- scalar minimization: finite objective and gradient/KKT conditions;
- least squares: finite residual plus rank/conditioning-aware error criteria;
- root solving: residual plus forward-error or domain-invariant criteria;
- mixed precision: agreement with the FP64 authority within declared error
  budgets;
- domain algorithms: their physical and state-machine invariants.

### P0: Is the inner solve required to remain on device?

**Recommendation: yes for the JAX production lane.** Any host callback,
materialization, or CPU reference call must be explicitly selected and visible
in the result. This forces the device/public result split and removal of the
current NumPy round-trip.

### P1: Is Adam part of the stable high-level API?

**Recommendation: only through `StochasticScalarProblem` or a clearly named
first-order policy.** Migrate legacy Boozer call sites to Optax Adam, but do not
offer Adam as an interchangeable default for deterministic minimization.

### P1: How is dense versus matrix-free LM selected?

Use declared structure and a user-specified memory budget. Keep the concrete
threshold internal and deterministic. Publish it in diagnostics, not as an
algorithm knob.

### P1: Are SciPy-shaped statuses and counters a compatibility requirement?

**Recommendation: no.** Keep familiar result fields where they have stable
meaning, but replace raw integer statuses with normalized termination reasons
and make unavailable counters optional. The user has explicitly allowed
breaking changes, so this is the right time to remove the accidental contract.

### P2: How long do predecessor generic solvers remain?

Keep them only as private differential-test fixtures for one qualification
cycle. Remove their public drivers immediately in the breaking API change.
Delete the private predecessors once the blessed-stack acceptance suite and
reference lanes cover their failure cases.

## Acceptance gates

Replacement is complete only when all gates pass:

1. Contract tests cover each problem family, normalized termination reason,
   nonfinite input, budget exhaustion, callback stop, unsupported placement,
   and certificate failure.
2. Differential tests compare final scientific quantities against the native
   and SciPy references. Exact trajectories are recorded diagnostically but
   are not a universal pass criterion.
3. Bound tests include active lower and upper bounds, fixed variables, and KKT
   checks.
4. Least-squares/root tests include rank deficiency, ill conditioning,
   non-identifiable parameters, and failed iterative inner solves.
5. CPU FP64, GPU FP64, mixed precision, and pure-accelerator topology are tested
   separately.
6. Device-residency tests detect unintended host transfers and callbacks.
7. Counter tests prove every reported exact count is actually observed.
8. The blessed exact dependency stack is merge-blocking; the newest-compatible
   stack is a canary.
9. Domain algorithms pass matched-state transition and physical-invariant
   tests, not only final residual checks.
10. Every silent fallback path is absent; unsupported capability combinations
    produce a stable, explicit error.

## Migration shape

1. Introduce the stable problem, policy, device-solution, host-result,
   termination, and certification contracts with RED tests.
2. Implement the semantic dispatcher and explicit unsupported-capability
   failures.
3. Move Optax Adam and L-BFGS behind the scalar adapters.
4. Move Optimistix LM/Newton and Lineax behind least-squares, root, and linear
   adapters.
5. Keep SciPy/native reference runners separate from production dispatch.
6. Migrate every legacy caller, including Boozer and domain workflows.
7. Remove the 15-driver public enum, backend-specific options/events/status
   maps, implicit logging, problem mutation, and host round-trips.
8. Port domain outer algorithms last and replace only their generic inner
   kernels.
9. Qualify the exact dependency stack on CPU and GPU.
10. Delete private generic predecessors after one complete qualification cycle.

## Rewritten implementation prompt

> Replace SIMSOPT's backend-named JAX solver API with a production-grade,
> problem-semantic solver architecture using red-green TDD. Define stable typed
> contracts for deterministic scalar, stochastic scalar, box-constrained
> scalar, least-squares, root, and linear problems; a backend-neutral solve
> policy; a device-resident `Solution`; a materialized host `SolveResult`; and
> SIMSOPT-owned termination/certification semantics. Internally use Optax for
> Adam and unconstrained L-BFGS, Optimistix plus Lineax for LM and Newton/root
> solves, Lineax for generic linear solves, and SciPy L-BFGS-B for the explicit
> CPU bounded/reference lane. Reject GPU box constraints until a real
> requirement and certified implementation exist. Keep QFM ALM, MwPGP/GPMO,
> GSCO, mixed-precision certification, and fallback policy SIMSOPT-owned.
> Eliminate backend-specific public options/events/statuses, silent fallback,
> implicit file output, problem mutation, synthesized counters, and
> device-host-device result transfers. Add a blessed exact dependency
> constraints file, a latest-compatible canary, full caller migration,
> differential reference tests, failure-mode tests, pure-device residency
> tests, conditioning-aware certification, and a one-cycle deletion gate for
> predecessor generic solvers.

## Open questions

1. Confirm the recommended v1 promise: **GPU-resident box constraints are not
   supported; bounded solves use the explicit SciPy CPU lane.**
2. Define the authoritative scientific certificate and tolerances for each
   production problem family.
3. Identify any public downstream user who requires raw SciPy integer statuses,
   exact `nfev`/`njev`, or per-step host callbacks.
4. Identify the largest expected dense least-squares problem and the memory
   budget used to select QR versus matrix-free iterative solves.
