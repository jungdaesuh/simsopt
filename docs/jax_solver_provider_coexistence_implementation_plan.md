# JAX solver-provider coexistence implementation plan

**Status:** Draft
**Last updated:** 2026-07-30

## Purpose

Keep the custom SIMSOPT BFGS/L-BFGS algorithms as production providers while
adding an additive, green qualification stack for explicit third-party
comparisons. Separate implementation availability from public solver policy.

## Goals

- Synchronize the integration base with current `hiddenSymmetries/simsopt`
  before restructuring solver code.
- Expose problem semantics and normalized SIMSOPT results, not third-party
  library names, as the durable solver-selection API. Preserve the executed
  provider and version as result provenance.
- Keep SIMSOPT custom and SciPy reference lanes available while independently
  adding Lineax, Optax, and Optimistix providers.
- Do not remove or silently replace custom BFGS/L-BFGS; Optax is an explicit
  comparator until a separate provider decision is approved.
- Make provider availability, default selection, call-site migration, and
  predecessor deletion separate commits.
- Permit clean provider-specific cherry-picks and policy-only reverts.
- Qualify numerical behavior, failure behavior, counters, precision, and device
  placement before changing defaults or deleting predecessors.

## Non-Goals

- Bitwise identity between mathematically related solvers.
- A speculative JAX implementation of bounded L-BFGS-B.
- Replacement of QFM ALM, MwPGP/GPMO, GSCO, mixed-precision certification, or
  SIMSOPT acceptance/continuation policy.
- Migration of ODE integration or event localization; those require a separate
  Diffrax/Optimistix audit.
- Deleting custom algorithms merely because a third-party provider is present.
- Preserving the current provider-named `Driver` enum as the permanent public
  policy model.

## Current Context

### Confirmed facts at `2026-07-30`

- Current branch: `pr/jax-port-squashed`.
- Current branch HEAD: `6d205e8f25634db15f40fede7dad544d8bbeff77`.
- Fetched canonical upstream: `upstream_hss/master` at
  `24fa9b4b88c3f30c199aa8d3309eb9e64b4943ad`.
- Divergence is 542 branch-only commits and 11 upstream-only commits; the merge
  base is `377cf665158f47a9bed4a8b03a00352457ea27c8`.
- A fresh `git merge-tree` check found no conflict markers, and the branch has
  no changes to the eight paths changed only by upstream since the merge base.
  This is preflight evidence, not permission to skip a fresh execution-time
  fetch.
- The current working tree contains unrelated modified and untracked files.
  Reconstruction must occur in a separate clean worktree.
- Commit `c0134753d96c5772a6de8ae228a6a8ac6e3fe637` introduced the solver
  slice in one coupled change: 41 selected solver/config/test files, 21,048
  insertions, and 4 deletions. Selectively reverting it does not provide
  provider-level independence.
- Canonical upstream contains no `src/simsopt_jax` or `tests/jax` paths. The
  provider-named `Driver` API has therefore not shipped upstream and does not
  require a public deprecation cycle for this PR.
- `src/simsopt_jax/solve/driver.py` currently exposes 15 provider-specific
  drivers.
- `src/simsopt_jax/solve/contracts.py` includes provider identity and
  provider-specific callback/result types in public contracts.
- `src/simsopt_jax/solve/dispatch.py` imports SciPy, Lineax, Optimistix, Optax
  runtime adapters, and custom implementations into one dispatch layer.
- `tests/jax/solve/test_import_boundaries.py` already protects lightweight
  imports and must remain green.
- `pyproject.toml` already declares Optax, Optimistix, Lineax, and Equinox in
  JAX extras, but uses open lower bounds. Its `scipy>=1.13` floor is below the
  SciPy floor required by the pinned JAX 0.10.0 metadata.
- The current Optax adapter has a host-controlled outer loop and does not count
  line-search evaluations. The current Optimistix adapters synthesize
  `nfev`/`njev`, materialize final results on the host, and carry a
  version-scoped strict-GPU transfer-guard xfail for Optimistix 0.1.0.
- Current Optimistix LM options default to Lineax LSMR, while current official
  Optimistix LM defaults to Lineax QR. Lineax QR requires full rank; SVD or
  LSMR is needed for declared rank-deficient regimes.
- The public generic JAX constrained-minimization route is unsupported.
- Native CPU solve paths in `src/simsopt/solve/serial.py` pass bounds to SciPy;
  the current `simsopt_jax.solve` SciPy adapter exposes no bounds argument.
- The bounded local example campaign recorded by
  `examples/jax/authority_evidence.json` (artifact SHA-256
  `13d5843438a93c748234e7cb8a52eb75f7627d832fe8dd3658eea5759e59a521`)
  ran as `20260729T005942Z-5ade9aee` at
  `11340c829690fdc0652e47588f5da549829c056a`: 26 cases, 78 lane
  receipts, and 1,248 declared comparisons passed its predicate;
  native-default scale was `not_run`. This is local workflow/physics
  regression evidence, not solver-interchangeability certification.

### Target ownership

| Layer | Owner |
| --- | --- |
| Adam, AdamW, and unconstrained L-BFGS update kernels | Optax |
| Custom BFGS/L-BFGS line search, SciPy-compatible state, callbacks, counters, and statuses | SIMSOPT |
| LM and Newton/root internal algorithm state and step kernels | Optimistix |
| QR, SVD, LSMR, CG, GMRES, and related linear kernels | Lineax |
| CPU L-BFGS-B, general constraints, and reference solves | SciPy |
| Problem classification and provider-selection policy | SIMSOPT |
| Result/status/counter normalization and scientific certificates | SIMSOPT |
| Executed-provider/version/dtype/device provenance | SIMSOPT |
| Outer acceptance/continuation, mixed precision, FP64 fallback, and placement policy | SIMSOPT |
| QFM ALM, MwPGP/GPMO, and GSCO state machines | SIMSOPT |

### External authority reviewed on `2026-07-30`

| Claim | Primary authority |
| --- | --- |
| Optax provides Adam/AdamW, unconstrained L-BFGS, and projections but not L-BFGS-B | [Optax optimizer API](https://optax.readthedocs.io/en/latest/api/optimizers.html); [Optax projections](https://optax.readthedocs.io/en/latest/api/projections.html) |
| Optimistix provides LM and root solvers; LM accepts a Lineax solver and defaults to QR | [Optimistix least squares](https://docs.kidger.site/optimistix/api/least_squares/); [solver selection](https://docs.kidger.site/optimistix/how-to-choose/) |
| Lineax provides QR, SVD, LSMR, CG, and GMRES with solver-specific structure/rank contracts | [Lineax solver API](https://docs.kidger.site/lineax/api/solvers/) |
| JAX 0.10.0 requires SciPy 1.14 or newer | [JAX 0.10.0 package metadata](https://pypi.org/pypi/jax/0.10.0/json) |
| `pip-compile` supports `pyproject.toml`, extras, and hash generation | [`pip-compile` reference](https://pip-tools.readthedocs.io/en/stable/reference/pip-compile/) |

## Rationale

Two designs were considered:

| Design | Advantage | Cost | Decision |
| --- | --- | --- | --- |
| Keep provider-named `Driver` public and split implementations | Smaller initial refactor | Provider details remain API policy; default changes and provider removal stay coupled | Reject as the target architecture |
| Semantic public contracts plus internal provider adapters | Stable API, information hiding, independent qualification and reversibility | Requires a contract-first refactor | Adopt |

Coexistence is a qualification mechanism, not a public promise that all
providers are peers. A provider can be installed and selected by internal
qualification tooling without becoming public selection policy or the
production default. Default changes are small policy commits; predecessor
deletion is a later, evidence-gated decision.

The final upstream PR should present a linear series of independently green
commits. Architecture and interfaces are designed before tests. After that
design is fixed, behavior-changing migrations use local RED/GREEN regression
cycles where the old behavior provides a meaningful failing baseline. No
intentionally failing RED commit is published.

## Assumptions

- The canonical integration target remains `hiddenSymmetries/simsopt:master`.
- The 11 upstream-only commits observed on 2026-07-30 can be integrated without
  semantic conflict; this must be revalidated immediately before
  reconstruction.
- A clean branch can be rebuilt from synchronized upstream while the existing
  branch remains an immutable source/reference lane.
- Existing call sites can be classified by semantic problem type and required
  certificate.
- GPU-resident active box bounds are not a release requirement unless a
  concrete call site demonstrates otherwise.
- Exact supported dependency versions can be published for certification while
  a separate latest-compatible lane detects ecosystem drift.

## Commit Architecture

`B0` is the recorded upstream branch point, not a synthetic commit. The final
PR is linear, but provider and policy dependencies are the partial order below:

```text
B0 upstream_hss/master@<fresh SHA>  [anchor, not a commit]
 └─ N1..Nk non-solver JAX prerequisites
     └─ C1 semantic contracts and API-evolution artifact
         └─ C2 custom + SciPy reference providers
             └─ D0 base dependency governance and hash-verified locks
                 ├─ PL Lineax dependency + adapter + tests ── QL
                 │    └─ PX Optimistix dependency + adapter + tests ── QX
                 └─ PO Optax dependency + adapter + tests ── QO

C2 ── M1..Mn behavior-preserving semantic call-site migrations

{QO, M_adam}   ── S_adam   ── X_adam?
{QO, M_lbfgs}  ── S_lbfgs  ── X_lbfgs?
{QL, QX, M_lm} ── S_lmroot ── X_lmroot?
{QL, M_linear} ── S_linear ── X_linear?
```

| Unit | Required content | Must not contain |
| --- | --- | --- |
| `B0` | Fresh upstream SHA and provenance receipt | A Git commit |
| `N1`–`Nk` | Manifest-selected non-solver prerequisites, one concern per commit | Solver/provider code |
| `C1` | Semantic problem/result/policy contracts and API-evolution artifact | Third-party implementation |
| `C2` | Existing custom and SciPy providers behind contracts | New third-party provider |
| `D0` | Dependency policy, review format, and hash-verified base CPU/CUDA locks | Provider implementation or default |
| `PL` | Lineax declaration, lock delta, adapter, and provider tests | Optimistix or default policy |
| `PO` | Optax declaration, lock delta, Adam/L-BFGS adapter, and tests | Default policy |
| `PX` | Optimistix/Equinox declaration, lock delta, LM/root adapter, and tests; depends on `PL` | Default policy |
| `QL`, `QO`, `QX` | Provider-specific differential/failure/device receipts | Production default changes |
| `M1`–`Mn` | One call-site family routed through semantic policy with old behavior preserved | Provider default change |
| `S_*` | One qualified default-selection change for one problem family | Provider implementation, call-site refactor, or deletion |
| `X_*` | One certified predecessor deletion after its complete `M`/`Q`/`S` cone | Other policy changes or deletions |

## Implementation Plan

1. Freeze the donor manifest and establish upstream anchor `B0`.
   - [ ] Fetch canonical upstream and record the remote URL, target SHA, current
     branch SHA, merge base, and ahead/behind counts in a tracked receipt.
   - [ ] Run `git merge-tree` against the freshly fetched upstream target and
     record any real semantic conflicts.
   - [ ] Inventory the complete donor diff before replaying anything; classify
     every file and hunk as non-solver prerequisite, contract, reference
     provider, third-party provider, policy, call site, test, benchmark, or
     documentation.
   - [ ] Produce a tracked replay manifest containing each selected donor SHA or
     hunk, its dependencies, owning reconstruction commit, and validation.
   - [ ] Create a separate worktree; do not modify or clean the current dirty
     worktree.
   - [ ] Preserve `pr/jax-port-squashed` as the source/reference branch.
   - [ ] Create the reconstruction branch from the synchronized upstream base;
     do not use a `codex/` branch prefix.
   - [ ] Record `B0` as the exact branch-point SHA; do not create an empty or
     synthetic synchronization commit.
   - [ ] Replay manifest-selected non-solver prerequisites as `N1`–`Nk`, one
     concern per green commit, without importing the coupled solver diff.
   - [ ] Verify the recorded `B0` SHA is an ancestor and every `N` commit matches
     the replay manifest before solver work begins.

2. Complete solver and API inventories before changing contracts.
   - [ ] Enumerate every solver file and hunk introduced by `c0134753d`.
   - [ ] Map each hunk to exactly one planned commit; flag cross-cutting hunks
     for contract-first decomposition.
   - [ ] Produce a call-site table containing problem type, constraints,
     precision, device, current provider, fallback, and certificate.
   - [ ] Inventory every `Driver`, provider-option, callback, result,
     package-root export, serialized value, example, benchmark, and compatibility
     reader/writer.
   - [ ] Record the observable behavior delta: defaults, accepted inputs,
     callback ordering, statuses/messages, counter meanings, result fields,
     imports, host synchronization, and error behavior.
   - [ ] Provide concrete donor-to-semantic migration examples, compatibility
     tests, and an atomic rollback procedure.
   - [ ] Since `Driver` has never shipped upstream, choose removal before the
     final PR rather than a public deprecation timeline. Any temporary donor
     shim must be private, migration-only, and deleted before completion.
   - [ ] Mark bounded, constrained, domain-specific, or mixed-precision call
     sites as SIMSOPT/SciPy-owned unless qualification changes that decision.
   - [ ] Confirm that each new-provider commit can be omitted without breaking
     the custom/SciPy reference baseline.

3. Introduce backend-neutral public contracts (`C1`).
   - [ ] Write interface comments and compare at least two contract designs
     before implementation; do not use tests to discover the abstraction.
   - [ ] Define the sole owners of problem semantics, provider capability,
     selection policy, normalized finalization, and execution provenance.
   - [ ] Add immutable problem descriptors and solve policy types for
     constraints, precision, device, budgets, fallback, and required
     certificate.
   - [ ] Define certificate units/nondimensionalization, parameter and residual
     scaling, norm, absolute/relative formulas, finite-value policy, dtype, and
     independent oracle. Do not expose one provider-neutral `tol` with
     provider-dependent meaning.
   - [ ] Define one normalized result contract for solution, objective or
     residual, derivatives/certificate, measured counters, status, message,
     precision, placement, and immutable execution provenance.
   - [ ] Expose executed provider, provider version, algorithm, dtype, and
     device as read-only provenance; do not use provider identity as public
     selection policy or scientific success.
   - [ ] Add an explicit internal provider capability protocol and one static
     resolver; do not expose a public registry or use dynamic imports.
   - [ ] Keep package-root imports lightweight.
   - [ ] Add behavioral tests after the interface is fixed. Prove that adding
     or removing a provider changes only its adapter, resolver entry,
     dependency slice, and qualification—not public contracts, call sites,
     serialized requests, or package-root imports.
   - [ ] Commit only after contract, typing, and import-boundary tests pass.

4. Isolate existing custom and SciPy reference providers (`C2`).
   - [ ] After the adapter contract is fixed, add a failing dispatch regression
     proving semantic requests resolve to reference providers without importing
     Optax, Optimistix, or Lineax.
   - [ ] Move custom and SciPy execution behind explicit provider adapters.
   - [ ] Reuse the native CPU SciPy boundary for L-BFGS-B and general
     constraints. Add bounds to the semantic request and CPU adapter; do not
     claim the current JAX dispatch already carries them.
   - [ ] Certify active lower/upper bounds, fixed variables, infeasible input,
     projected-gradient/KKT termination, and general-constraint failure paths
     before marking the semantic CPU provider bound/constrained-capable.
   - [ ] Preserve custom Adam, BFGS/L-BFGS, LM/Newton, linear wrappers,
     mixed-precision policy, and domain algorithms as qualification baselines.
   - [ ] Centralize finalization: recompute success from finite required metrics
     and the SIMSOPT certificate; distinguish requested-budget completion,
     library termination, and scientific convergence.
   - [ ] Normalize exceptions, statuses, callbacks, and counters once at the
     SIMSOPT boundary. Report unavailable counters as unavailable; never
     synthesize evaluation counts from iteration counts.
   - [ ] Add public-path NaN/Inf, callback-stop, budget-exhaustion, and counter
     regressions; routing-only monkeypatch tests are insufficient.
   - [ ] Fail explicitly for unsupported JAX bounds or constraints; do not
     silently fall back or clip.
   - [ ] Prove the reference-only build and focused suite pass with third-party
     provider modules unavailable.

5. Define dependency governance and immutable base resolution (`D0`).
   - [ ] Raise the declared SciPy floor to satisfy the pinned JAX metadata.
   - [ ] Add `pip-tools==7.6.0` to the development toolchain and use
     `pip-compile --generate-hashes --extra <extra> pyproject.toml` to commit
     per-Python-lane CPU and CUDA12 locks under `requirements/locks/`.
   - [ ] Establish provider-specific extras plus one aggregate production extra
     so reference, Lineax, Optax, and Optimistix installation closures can be
     resolved and tested independently.
   - [ ] For every direct provider dependency, record necessity, maintenance,
     license compatibility, security/advisory status, transitive exposure,
     public-versus-internal API exposure, and removal owner.
   - [ ] Require explicit maintainer acknowledgement of the dependency review
     before landing a provider commit.
   - [ ] Keep dependency declarations single-source and shared by CPU/GPU
     extras where requirements are identical.
   - [ ] Add clean-environment resolver/import smoke tests for the base and each
     provider subset.
   - [ ] Add a non-authoritative latest-compatible canary lane.
   - [ ] Keep base governance separate from providers; each provider commit
     owns its direct declaration and corresponding hash-lock delta.

6. Add the Lineax provider slice (`PL`).
   - [ ] Add the direct Lineax declaration and hash-lock delta in the same
     commit as the provider adapter and its tests.
   - [ ] After the provider interface is fixed, add failing behavioral tests for
     QR, SVD, LSMR, CG, and GMRES capability, structural errors, convergence
     failure, and available counters.
   - [ ] Implement one typed Lineax solver-object factory for nested consumers
     and a separate standalone linear-solve adapter returning normalized
     SIMSOPT results and post-solve certificates.
   - [ ] Make the factory record exact solver class, operator tags, tolerances,
     restart/iteration limits, adjoint selection, and expected result
     representative.
   - [ ] Retain SIMSOPT solver selection, refinement, conditioning checks,
     forward-error checks, precision escalation, and fallback.
   - [ ] Route declared full-rank dense rectangular problems to QR,
     rank-deficient/pseudoinverse problems to SVD, and large matrix-free
     least-squares problems to LSMR only after conditioning qualification.
   - [ ] Restrict CG to correctly tagged definite operators; verify GMRES
     restart and non-convergence semantics for general square operators.
   - [ ] Test dense, matrix-free, nonsquare, singular/rank-deficient,
     ill-conditioned, non-convergent, FP32, and FP64 cases.
   - [ ] Under no-callback strict mode, verify CPU/GPU placement and no
     unintended host callbacks or transfers inside the solve.
   - [ ] Do not change LM/Newton defaults in this commit.

7. Qualify the Lineax slice (`QL`).
   - [ ] Run differential, failure, placement, precision, and concurrency tests
     against SciPy/custom reference lanes for each declared solver regime.
   - [ ] Record source SHA, exact lock digest, provider version, device, dtype,
     matrix structure/shape/conditioning, tolerances, and raw results.
   - [ ] Do not promote any semantic default in this commit.

8. Add the Optax provider slice (`PO`).
   - [ ] Add the direct Optax declaration and hash-lock delta in the same commit
     as the provider adapter and its tests.
   - [ ] After the adapter contract is fixed, add failing Adam behavioral tests
     for update equations, bias correction, schedules, termination, callbacks,
     NaN/Inf behavior, available counters, and pytree state.
   - [ ] Implement Adam and AdamW as distinct semantic policies. Define update
     equation, schedule indexing, decoupled decay semantics and mask,
     accumulator dtype, `eps`/`eps_root`, Nesterov mode, and termination; do not
     infer the algorithm solely from a positive decay scalar.
   - [ ] Add failing unconstrained L-BFGS behavioral tests for line-search
     acceptance, curvature/history updates, termination, invalid steps,
     available counters, and callback ordering.
   - [ ] Implement Optax L-BFGS only for declared unconstrained scalar problems.
   - [ ] Reject bound-bearing requests at capability resolution.
   - [ ] Treat line-search trial counts as separate diagnostics; do not report
     outer iterations as objective/gradient evaluation counts.
   - [ ] Keep custom Adam and L-BFGS available and keep defaults unchanged.

9. Qualify the Optax slice (`QO`).
   - [ ] Run differential, failure, placement, precision, and concurrency tests
     against custom/SciPy references for Adam and unconstrained L-BFGS.
   - [ ] Separate final-solution/scientific agreement from trajectories,
     termination, line-search behavior, statuses, and counters.
   - [ ] For stochastic Adam/AdamW claims, bind the PRNG keys, batch stream,
     schedule, and initialization; retain all seeds and require a declared
     multi-seed scientific acceptance predicate. Otherwise qualify only
     deterministic full-batch/fixed-step use.
   - [ ] Record source SHA, exact lock digest, provider version, device, dtype,
     objective fixture, tolerances, and raw results.
   - [ ] Do not promote a default in this commit.

10. Add the Optimistix nonlinear provider slice (`PX`).
    - [ ] Require `PL`; add direct Optimistix/Equinox declarations and the
     hash-lock delta in the same commit as the adapter and tests.
    - [ ] Construct Optimistix LM/root solvers with the typed Lineax
     solver-object factory from `PL`. Do not pass the standalone result adapter
     where Optimistix requires `lineax.AbstractLinearSolver`.
    - [ ] Record exact nested solver class, tags, tolerances, adjoint, and
     diagnostics. Do not claim inner counters/certificates that Optimistix does
     not expose; post-certify the final residual/root instead.
    - [ ] After the adapter contract is fixed, add failing LM behavioral tests
     for damping, acceptance/rejection, rank deficiency, QR/SVD/LSMR selection,
     termination, exposed statistics, and callback capability.
    - [ ] Implement Optimistix LM behind the least-squares contract.
    - [ ] Add failing Newton/root behavioral tests for convergence,
     non-convergence, singular Jacobians, continuation, and refinement.
    - [ ] Delegate generic steps to Optimistix while retaining SIMSOPT outer
     acceptance, continuation, certificate, precision, and fallback policy.
    - [ ] Declare callbacks or counters unsupported when the provider cannot
     supply their contract; do not synthesize them.
    - [ ] Keep custom LM/Newton implementations available and defaults
     unchanged.

11. Qualify the Optimistix/Lineax nonlinear slice (`QX`).
    - [ ] Define per-problem acceptance contracts before running comparisons;
     do not use bitwise identity as the general criterion.
    - [ ] Compare final parameters, objective/residual, gradient or KKT/root
     certificate, status class, failure mode, counters, and callback trace as
     separate fields.
    - [ ] Include deterministic well-conditioned, ill-conditioned,
     rank-deficient, invalid-input, NaN/Inf, budget-exhaustion, and
     non-convergence cases.
    - [ ] Run reference/custom versus third-party comparisons in FP64 CPU.
    - [ ] Run strict RTX 5090 FP64 lanes with explicit device assertions and no
     CPU fallback.
    - [ ] Launch fresh child processes configured before JAX import with exact
     `JAX_PLATFORMS`, X64, and transfer-guard policy; require pure-CUDA topology,
     runtime-provided inputs, leaf-wise input/state/output device and dtype
     assertions, child exit status, and raw effective configuration.
    - [ ] Separate explicitly permitted setup/finalization transfers from a
     strict no-callback solve-core transfer guard and record executable/cache
     identity.
    - [ ] Run mixed-precision lanes and verify deterministic FP64 escalation or
     declared failure.
    - [ ] Verify JIT/eager and forward/reverse differentiation contracts where
     solver differentiation is public behavior.
    - [ ] Store machine-readable receipts with source SHA, dependency versions,
     device topology, tolerances, and raw outcomes.
    - [ ] Treat provider disagreement as a triage input, not automatic proof
     that either implementation is wrong.
    - [ ] Resolve the current Optimistix 0.1.0 strict-GPU transfer-guard xfail;
     an xfail or relaxed guard cannot satisfy promotion.

12. Migrate production call sites without changing algorithms (`M1`–`Mn`).
    - [ ] For each call site, add a behavioral regression that fails if the
      semantic migration changes the existing provider, options, result,
      callback, status, counter, placement, or certificate contract.
    - [ ] Route the call site through semantic policy rather than importing a
      provider adapter.
    - [ ] Configure the semantic policy to select the pre-migration
      custom/SciPy behavior in this commit.
    - [ ] Preserve acceptance, continuation, precision, fallback, callback,
      budget, and failure semantics.
    - [ ] Run its native/CPU reference, JAX CPU, and strict JAX GPU lanes where
      supported.
    - [ ] Commit one problem family per change.
    - [ ] Keep QFM ALM, MwPGP/GPMO, and GSCO outer state machines
      SIMSOPT-owned; replace only independently qualified inner kernels.

13. Change defaults in problem-family policy-only commits (`S_*`).
    - [ ] `S_adam`: after `QO` and `M_adam`, select Optax Adam for qualified
      unconstrained deterministic full-batch/fixed-schedule problems. Include
      stochastic problems only after the matched multi-seed gate passes.
    - [ ] `S_lbfgs`: defer any Optax default decision until the custom
      BFGS/L-BFGS structure plan has passed its compatibility, science, and
      performance gates; Optax remains an explicit comparator meanwhile.
    - [ ] `S_lmroot`: after `QL`, `QX`, and the corresponding `M` commits,
      select Optimistix plus Lineax for qualified LM/root regimes.
    - [ ] `S_linear`: after `QL` and `M_linear`, select qualified Lineax
      regimes.
    - [ ] Include one decision table and one public-path policy regression per
      commit.
    - [ ] Bind each decision to a qualification-manifest cell keyed by problem
      family, dimensions, structure, rank/conditioning band, scale, dtype,
      device, initialization/basin, stochasticity, and certificate version.
      Fail closed outside authenticated cells.
    - [ ] Make each commit revertible without removing provider code or
      invalidating request/result provenance.
    - [ ] Leave bounded/general constrained and all unqualified
      shape/conditioning/precision regimes on SciPy/custom reference lanes.

14. Remove temporary donor-only provider policy.
    - [ ] Verify all 31 current `Driver`-using source/test/example files have
      been migrated or intentionally replaced.
    - [ ] Remove the private reconstruction shim, provider-specific public
      options/callback exports, and donor-only serialized values before the
      upstream PR.
    - [ ] Confirm the new upstream API has never emitted a deprecation promise
      for the unshipped `Driver` surface.

15. Delete predecessors only after certification (`X_*`).
    - [ ] Require tracked clean-revision qualification across every migrated
      call site before proposing a deletion.
    - [ ] Delete only duplicated generic equations/state machinery; retain
      SIMSOPT policy, certificates, mixed precision, and domain algorithms.
    - [ ] Use one deletion cone and commit per predecessor:
      `{M_adam,QO,S_adam}->X_adam`,
      `{M_lbfgs,QO,S_lbfgs}->X_lbfgs`,
      `{M_linear,QL,S_linear}->X_linear`, or
      `{M_lmroot,QL,QX,S_lmroot}->X_lmroot`.
    - [ ] Verify no remaining consumer, fallback, benchmark, or compatibility
      path imports the deleted code.
    - [ ] Keep deletions optional: the upstream PR may ship coexistence without
      any predecessor deletion.

16. Prove commit-stack reversibility and prepare upstream review.
    - [ ] Verify every commit is green when checked out independently.
    - [ ] In disposable worktrees, install and cherry-pick `PL`, `PO`, and
      `PL+PX` with their lock deltas in every dependency-valid subset/order.
    - [ ] Revert each `S_*` policy commit and prove behavior returns to the
      pre-migration provider while both implementations remain available.
    - [ ] Revert provider slices only after reverting/removing their dependent
      `S`, `M`, `Q`, and `X` consumers; arbitrary provider-first reverts are not
      supported.
    - [ ] Prove provider commits can be omitted from a reference-only build.
    - [ ] Use `git range-diff` to confirm reconstruction did not silently drop
      required non-solver port behavior.
    - [ ] Update `docs/jax_solver_algorithm_matrix.md` with implemented,
      qualified, default, deprecated, and deleted states as separate columns.
    - [ ] Submit the upstream PR as the clean additive series rooted at `B0`,
      not as selective reverts of `c0134753d`.

## Validation Plan

### Per-commit gate

- [ ] `git status --short` contains only the intended commit slice.
- [ ] `git diff --check` passes.
- [ ] Ruff lint and formatting checks pass for touched Python files.
- [ ] Pyright reports zero errors and warnings for the typed solver surface.
- [ ] Focused solver, contract, import-boundary, and compatibility tests pass.
- [ ] No production module imports benchmark code.
- [ ] No public package-root import eagerly loads Optax, Optimistix, or Lineax.
- [ ] No intentionally failing RED commit appears in published history.

### Contract and dispatch gate

- [ ] Semantic request types contain no third-party library objects.
- [ ] Normalized results distinguish solver termination from SIMSOPT scientific
  acceptance.
- [ ] Result provenance records executed provider/version, algorithm, dtype,
  device, source revision, and lock digest without making those fields
  selection inputs.
- [ ] Counters are measured or explicitly unavailable, never synthesized.
- [ ] Counter fields are semantically distinct: accepted nonlinear steps,
  objective/residual evaluations, gradient/Jacobian evaluations, line-search
  trials, linear solves, Krylov iterations, and post-certification evaluations.
  Each field records its instrumentation source.
- [ ] Terminal notification, bounded device trace, and host-progress callback
  are separate capabilities; exact comparison is required only when semantics
  and instrumentation match.
- [ ] Unsupported bounds, constraints, device pairs, and precision regimes fail
  explicitly.
- [ ] Provider selection is deterministic and covered by a decision table.
- [ ] Adding/removing a provider does not change public contracts, semantic call
  sites, serialized requests, or package-root imports.
- [ ] The donor-only `Driver` surface is absent from the final upstream API.

### Numerical qualification gate

- [ ] Adam trajectories match the declared equation/schedule contract within
  dtype-specific tolerances.
- [ ] L-BFGS comparisons certify solution/stationarity and line-search behavior,
  not identical iteration trajectories.
- [ ] LM/root comparisons certify residual/root norm, derivative certificate,
  termination class, and failure behavior.
- [ ] Linear solvers certify residual and forward error where condition
  estimates permit it.
- [ ] Every certificate declares units/scaling, norm, absolute and relative
  formulas, dtype, finite-value policy, and an analytic, high-precision, SVD,
  or physics oracle independent of the candidate provider.
- [ ] Scale sweeps and tolerance-boundary cases prove mathematically equivalent
  rescalings do not silently change semantic acceptance.
- [ ] Rank-deficient fixtures distinguish residual-zero solutions from the
  required minimum-norm solution; QR is not used as rank-revealing evidence.
- [ ] FP64 CPU and strict FP64 GPU results pass problem-specific tolerances.
- [ ] Mixed precision either certifies or escalates to FP64 according to policy.
- [ ] Device receipts prove no silent CPU fallback.
- [ ] Strict-device receipts include pre-import environment, topology, transfer
  policy, leaf-wise placement/dtype, child exit status, and cache identity.

### Integration and release gate

- [ ] Full supported test suite passes on the exact supported dependency stack.
- [ ] Latest-compatible dependency canary result is recorded separately.
- [ ] Native CPU, JAX CPU, and supported strict JAX GPU application lanes pass.
- [ ] Native-default-scale validation is run or remains explicitly `not_run`.
- [ ] Source distributions and wheels expose the intended extras and import
  behavior.
- [ ] Documentation states which providers are available, qualified, default,
  reference-only, or unsupported.
- [ ] Hash-verified locks install successfully in clean CPU and CUDA12
  environments for each supported provider subset.
- [ ] Dependency maintenance/license/security/removal reviews and maintainer
  acknowledgement are retained.

### Performance and concurrency gate

- [ ] On matched production inputs, record compile time, first solve,
  steady-state solve, total objective/residual evaluations, host
  synchronizations, peak RSS, and peak VRAM separately.
- [ ] Do not infer speed or memory improvement from numerical parity receipts.
- [ ] Define the permitted regression budget before changing a default and
  block promotion when it is exceeded.
- [ ] Run concurrent independent solves and prove resolver state, counters,
  callbacks, caches, and result provenance are not shared or cross-contaminated.
- [ ] Keep provider resolution immutable after import; no process-global
  mutable registry or callback/counter state is permitted.

### Reversibility gate

- [ ] `B0`, `N1`–`Nk`, `C1`, `C2`, and `D0` are provenance-bound and green.
- [ ] `PL+QL`, `PO+QO`, and `PL+PX+QX` are independently installable and green.
- [ ] Every `M` commit preserves its pre-migration provider behavior.
- [ ] Each `S_*` commit can be reverted alone.
- [ ] Each `M` commit can be reverted without changing provider availability.
- [ ] Each `X_*` deletion is downstream of its complete `M`/`Q`/`S` cone and
  can be omitted without affecting policy commits.
- [ ] Provider-subset cherry-pick rehearsals pass in disposable worktrees.

## Risks and Mitigations

- Risk: Reconstructing from the coupled port silently drops required behavior.
  Mitigation: Create a hunk inventory, map every solver hunk to one commit, use
  `git range-diff`, and run the full port regression suite.

- Risk: A provider-specific option leaks back into the public API.
  Mitigation: Keep provider option types inside adapters and add API/type tests
  that reject third-party objects in semantic contracts.

- Risk: Coexistence becomes permanent duplicated maintenance.
  Mitigation: Assign one owner per layer, mark reference-only implementations,
  and use evidence-gated, one-predecessor deletion decisions.

- Risk: A default-selection revert also removes implementation code.
  Mitigation: Prohibit provider code in `S` commits and rehearse isolated
  policy reverts.

- Risk: Similar algorithms produce materially different scientific outcomes.
  Mitigation: Qualify application certificates and failure regimes, retain
  reference lanes, and fail closed outside qualified regimes.

- Risk: Dependency drift changes numerical behavior or breaks installation.
  Mitigation: Enforce hash-verified CPU/CUDA locks in authoritative lanes and
  run a separate latest-compatible canary.

- Risk: Iterative Lineax solves regress ill-conditioned LM/root problems.
  Mitigation: Use QR only for declared full-rank problems, SVD for
  rank-deficient/minimum-norm contracts, and LSMR only in qualified
  conditioning regimes; retain refinement, FP64 escalation, and CPU/reference
  fallback.

- Risk: Unsupported bounded optimization is mistaken for unconstrained
  L-BFGS.
  Mitigation: Encode bounds in the semantic request and reject providers
  without true bound-aware/KKT capability.

- Risk: Reconstruction contaminates existing user work.
  Mitigation: Use a new worktree and never clean, reset, or repurpose the
  current dirty tree.

## Completion Criteria

- [ ] The reconstruction branch contains the selected canonical upstream SHA as
  an ancestor.
- [ ] The solver stack is a linear, independently green sequence matching the
  commit architecture above.
- [ ] Public problem and selection-policy contracts are provider-neutral;
  results retain immutable executed-provider provenance.
- [ ] Custom/SciPy references and each third-party provider can be tested
  independently.
- [ ] Provider availability and default selection are separate commits.
- [ ] Bounds and general constraints remain explicitly supported on SciPy CPU
  and explicitly unsupported on unqualified JAX routes.
- [ ] Qualification receipts cover numerical, failure, counter, precision, and
  placement contracts.
- [ ] Policy-only reverts and provider-subset cherry-picks are demonstrated.
- [ ] Domain algorithms and SIMSOPT scientific certification remain
  SIMSOPT-owned.
- [ ] Any predecessor deletion is separately certified, reviewed, and
  revertible.
- [ ] The solver matrix and user-facing documentation reflect actual state
  without claiming bitwise parity or unmeasured performance.
- [ ] No unrelated files from the original dirty worktree are modified.
- [ ] The unshipped donor `Driver` API and temporary reconstruction shims are
  absent from the final upstream series.
- [ ] API-evolution, dependency-review, rollback, and matched-performance
  artifacts are complete.

## Open Questions

- Which donor hunks are the minimal non-solver prerequisites in `N1`–`Nk`?
- Which supported Python/platform lanes require distinct
  `requirements/locks/` outputs?
- Which default switches (`S_*`) belong in the first upstream PR versus a
  follow-up qualification PR?
- How long should custom generic implementations remain as private reference
  lanes after all call sites qualify?
- Is any production call site a demonstrated requirement for GPU-resident box
  bounds?
- What problem-specific tolerances and certificates are authoritative for each
  domain call site?
- For Optimistix LM, which declared regimes select Lineax QR, SVD, or LSMR?
- What matched compile/runtime/memory regression budgets block each `S_*`
  promotion?
