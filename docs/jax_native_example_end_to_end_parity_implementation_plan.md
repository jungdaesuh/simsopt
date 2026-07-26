# Native SIMSOPT vs JAX Example End-to-End Parity Implementation Plan

**Status:** Draft
**Last updated:** 2026-07-26

## Purpose

Establish reproducible, fail-closed evidence that matched native SIMSOPT CPU
workflows and JAX workflows evaluate the same FP64 problem and reach
scientifically equivalent outcomes. The plan covers initial-state value and
derivative parity, final solve parity, strict JAX CPU/GPU execution, and
hash-bound structured artifacts.

This closes a gap in the JAX-first example suite: the candidate worktree has
ten ready JAX examples intended for CPU-smoke and strict-GPU execution, but
most compare against analytic, finite-difference, or reduced host oracles
rather than an identical native workflow from construction through final
state. Phase 0 makes the exact committed prerequisite baseline explicit before
any parity claim is authoritative.

## Goals

- Run each applicable native CPU/JAX pair from one canonical input bundle with
  identical initial parameters, quadrature/resolution, weights, constraints,
  random seed, FP64 policy, and comparable solve budgets.
- Compare objective, residual vector, gradient, residual Jacobian, and
  constraint vector at the identical initial state, with explicit
  `not_applicable` fields for problem types that do not define an observable.
- Compare optimized parameters, final objective/residual, feasibility,
  convergence category, raw status, and work counters without requiring
  different algorithms to have identical trajectories.
- Run the identical JAX case through `jax_cpu_parity` and `jax_gpu_parity`; the
  GPU lane must use FP64, set both the SIMSOPT policy
  `SIMSOPT_JAX_TRANSFER_GUARD=disallow` and JAX's effective
  `JAX_TRANSFER_GUARD=disallow`, and permit no CPU fallback.
- Preserve authentic RED -> GREEN -> REFACTOR receipts for each parity case and
  publish source/input/environment-bound JSON plus array sidecars.
- Classify every ready JAX example-to-native-source relationship as `full`,
  `reduced`, or `unsupported`, record `bounded` versus `native_default` scale
  independently, and prevent reduced or bounded evidence from being reported
  as native-default end-to-end parity.

## Non-Goals

- Require identical optimizer iterates, line-search decisions, iteration
  counts, or evaluation counts when native SciPy and SIMSOPT-owned JAX drivers
  implement different algorithms.
- Treat matching exit codes, final scalar objectives alone, or analytic-only
  checks as proof of native/JAX end-to-end parity.
- Port VMEC, SPEC, QSC, BOOZXFORM, MPI orchestration, plotting, or file-writing
  workflows solely to promote a parity classification.
- Import native example scripts whose top-level code executes solvers or writes
  artifacts. Matched cases reconstruct the declared workflow through public
  SIMSOPT and JAX APIs.
- Add benchmark or speedup claims. Runtime and memory may be recorded as
  diagnostics but are not correctness gates in this plan.
- Promote the currently planned single-stage vacuum example while its required
  outer derivative and accepted-state certificate remain unavailable.

## Current Context

### Baseline authority

- The committed baseline at review time is branch `pr/jax-port-squashed`,
  commit `6547da3a4`. At that commit, nine ready examples are CPU-only in
  `examples/jax/manifest.json`, and the serial JAX solver still imports
  Optimistix and uses a host callback. Therefore that commit does **not** yet
  satisfy the backend-neutral ten-example CPU/GPU prerequisite assumed by this
  plan.
- The review worktree contains an uncommitted predecessor slice that changes
  the manifest, examples, runner tests, and JAX serial-solver behavior. Those
  changes are useful candidate evidence, but are not shipped behavior and must
  not be cited as authoritative until they are committed, tested, and identified
  by revision.
- Phase 0 is a hard dependency: parity implementation starts from a clean,
  committed prerequisite revision whose manifest and solver contracts pass the
  named CPU/GPU tests. Exploratory dirty-tree runs must be labeled
  non-authoritative and may not promote a classification.

- `examples/jax/manifest.json` is the source of truth for 51 native example
  dispositions, 10 ready JAX lessons, one planned lesson, `inspired_by`
  lineage, host boundaries, and CPU/GPU lane membership.
- `examples/jax/run_examples.py` owns the isolated `cpu-smoke` and
  `gpu-strict` process environments and validates each child's backend,
  platform, precision, and status. It does not run or compare a native lane.
- `tests/integration/test_jax_examples.py` currently proves scientific smoke
  checks. Its oracles include analytic formulas, finite differences, host
  linear algebra, and selected native operations; it does not provide one
  normalized native/JAX result schema for complete paired workflows.
- `tests/solve/test_serial_jax.py` compares native SIMSOPT/SciPy and JAX solvers
  on deterministic toy quadratics. This is the strongest current whole-solve
  comparison, but it is not representative coverage of all ready examples.
- `benchmarks/validation_ladder_contract.py` already owns canonical tolerance
  buckets such as `direct_kernel`, `derivative_heavy`, and `gpu_runtime`.
  The first two require a direct-C++ oracle, so they do not automatically apply
  to native Python/SciPy workflows. Every parity case must route through this
  owner with compatible oracle semantics rather than embedding new tolerances
  in scripts or manifests.
- `src/simsopt_jax/solve/contracts.py` owns the typed `OptimizerResult`, driver,
  raw status, success flag, work counters, residual, Jacobian, and result
  fingerprint contracts.
- The older `non_banana_example_cpp_jax_cpu_parity` harness referenced by a
  sibling-checkout plan is absent from this checkout. This plan must not assume
  those modules, fixture IDs, manifests, or artifacts exist.
- The worktree contains unrelated modified and untracked files. Execution must
  preserve them and review only the parity-plan slice.
- The expanded non-GPU parity suite was run during this documentation review:
  51 tests passed and the surface-geometry final-parameter comparison failed
  (`max_abs_diff=2.45719087e-06` versus the current whole-solve route). This is
  implementation evidence, not a reason to weaken the central tolerance.
  Surface geometry remains reduced and non-authoritative until its matched
  workflow or equivalence contract passes through the normal RED -> GREEN
  cycle.

The lane semantics in this plan follow the official JAX configuration
contracts: [`JAX_PLATFORMS`](https://docs.jax.dev/en/latest/config_options.html#platforms)
is fixed before initialization and fails if a requested platform cannot be
initialized; [64-bit mode](https://docs.jax.dev/en/latest/default_dtypes.html)
must be enabled explicitly; transfer-guard `disallow` blocks implicit transfers
while explicit transfers remain available for result publication; and disabling
[GPU preallocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
changes allocation behavior and can increase fragmentation.

### Provisional parity classification

The classification below reflects the current uncommitted predecessor slice,
not shipped behavior. Phase 0 must first land a clean prerequisite revision;
Phase 1 then validates every relationship before any row or artifact becomes
authoritative. The table shows one primary relationship per JAX lesson only;
`parity_manifest.json` must still classify all 28 ready `inspired_by`
relationships individually.

| Ready JAX example | First matched native workflow | Provisional level | Required initial/final evidence |
|---|---|---:|---|
| `traceable-least-squares` | `1_Simple/just_a_quadratic.py` | full, bounded | residual, Jacobian, objective, gradient, parameters, status, counts |
| `curve-length-optimization` | `1_Simple/minimize_curve_length.py` | full, bounded | length, gradient, parameters, objective, status, counts |
| `surface-geometry-optimization` | `1_Simple/surf_vol_area.py` | reduced, bounded | area/volume residuals and Jacobian, parameters, status, counts |
| `coil-flux-optimization` | `1_Simple/stage_two_optimization_minimal.py` | reduced, bounded | flux/penalties, gradient, current/curve parameters, status, counts |
| `qfm-surface-optimization` | `1_Simple/qfm.py` | reduced, bounded | penalty/constraints, gradient/Jacobian, surface parameters, feasibility |
| `permanent-magnet-optimization` | `1_Simple/permanent_magnet_simple.py` | reduced, bounded | initial residual, selected moments, final residual, step/count metadata |
| `fieldline-and-particle-tracing` | `1_Simple/tracing_fieldlines_NCSX.py` | reduced, bounded | initial field/RHS, event/status, final state, invariants; optimizer fields N/A |
| `boozer-surface-optimization` | `2_Intermediate/QH_fixed_resolution_boozer.py` | reduced, bounded | residual/Jacobian, iota/G/surface state, feasibility, status, counts |
| `wireframe-optimization` | `2_Intermediate/wireframe_gsco_modular.py` | reduced, bounded | matrices/constraints, solution, objective, feasibility, step/count metadata |
| `coil-force-and-finite-build` | `2_Intermediate/strain_optimization.py` | reduced, bounded | force objective/gradient/frame; optimizer outcome fields N/A |

Every additional native source named by a combined JAX lesson must receive a
separate `full`, `reduced`, or `unsupported` relationship row. One successful
representative case must not silently cover all `inspired_by` sources.

## Rationale

The comparison unit must be a **typed parity case**, not an original script
process. Native scripts are teaching programs with different defaults,
side-effects, external solvers, and sometimes several workflows in one file.
Executing them beside a redesigned JAX lesson would compare different problems
while appearing authoritative.

A parity case instead owns one immutable, serialized input bundle and exposes
two implementations: native SIMSOPT CPU and JAX. The parent runner creates the
input once, hashes it, and passes the same bundle to isolated native CPU, JAX
CPU, and JAX GPU children. A separate arbiter compares structured outputs using
the central tolerance ladder. This keeps input identity, lane execution, and
scientific comparison as three distinct responsibilities.

### Design-it-twice

**Option A - parity-specific typed manifest, case registry, isolated runner,
and arbiter (selected).**

- Add `examples/jax/parity_manifest.json` for relationship classification and
  parity policy only. It references existing JAX example IDs and native source
  paths and is validated against `manifest.json` to prevent duplicated lineage.
- Add a small `examples/jax/parity/` package that owns immutable case inputs,
  normalized lane results, case implementations, subprocess commands, and the
  comparison arbiter.
- Add `examples/jax/run_parity.py` as the bounded CLI that creates an input
  bundle once and executes native CPU, JAX CPU, and optionally strict JAX GPU
  children in fresh processes.
- Advantages: exact input hashes, fail-closed lane isolation, reusable artifacts,
  and no changes to the public `src/` API.
- Cost: one new internal schema and runner, justified because paired comparison
  is a different abstraction from single-example smoke execution.

**Option B - extend each example script with `--native` and compare its JSON
output (rejected).**

- This would scatter native/JAX pairing, tolerance selection, fingerprints,
  status normalization, and artifact writing across ten scripts.
- Combined examples would still lack a truthful one-to-one native source
  mapping, and changing the result schema would require coordinated edits.

**Option C - import and execute original native scripts (rejected).**

- Several scripts run expensive solvers, MPI/external codes, plotting, or file
  output at import/top level. Their defaults also differ from bounded JAX smoke
  cases, so success would not establish identical-input parity.

### Information-hiding test

- `manifest.json` continues to own example readiness, lineage, and runtime
  lanes.
- `parity_manifest.json` owns relationship classification, scale tier, cost
  tier, and the exact comparison routing keyed by phase, observable, and lane
  pair. Each route names applicability, comparator, and central tolerance
  bucket. It also owns the ordered scientific workflow stages covered by the
  relationship and any scientific stages omitted by a reduced case; teaching
  I/O and plotting exclusions are recorded separately. A `full` row may not
  omit a scientific stage. Native paths must already be present in the
  referenced example's `inspired_by` list. Case implementations emit values
  and completed-stage receipts for those declared names but do not redefine
  policy.
- The case registry owns construction and execution of matched scientific
  problems.
- The runner owns process isolation and artifact provenance.
- The arbiter owns comparison and verdict semantics.
- `benchmarks/validation_ladder_contract.py` remains the sole tolerance owner.

Changing a tolerance, lane environment, or source relationship must therefore
require one owner change rather than edits across every parity case.

### Design tier and API evolution

This is Tier 3 because it introduces a repository CLI and versioned artifact
schema that CI and reviewers may consume.

- Observable behavior delta: a new opt-in parity runner and artifacts are
  added. Existing native/JAX example commands and `run_examples.py` behavior do
  not change.
- Caller inventory: developers running local parity checks, CPU CI, strict-GPU
  CI, integration tests, and reviewers reading published artifacts.
- Migration: none for existing commands. Artifact consumers must require an
  exact `schema_version`; incompatible future changes create a new version.
- Compatibility tests: exact CLI argv, schema fields, deterministic ordering,
  lane identity, input fingerprints, normalized statuses, and fail-closed
  malformed/mismatched artifact cases.
- Deprecation: not applicable to the additive v1 interface. A future v2 must
  retain a documented v1 reader or provide a conversion command before v1 is
  removed.
- Rollback: remove the parity manifest/package/runner, CI steps, tests, and docs
  together. Existing examples and public solver APIs remain independent.

## Assumptions

- “Native/C++” means the native SIMSOPT CPU workflow using its normal Python
  orchestration, SciPy controls where applicable, and `simsoptpp` kernels where
  the native surface uses them. Each artifact records whether its oracle is
  `simsoptpp`, native Python/SciPy, external, or analytic.
- `full` describes workflow-stage coverage, not production scale. A bounded
  case may count as `full` only when both lanes execute every declared stage
  and differ solely in the native/JAX implementation; its independent
  `scale_tier` remains `bounded`. Only a passing `full` + `native_default`
  artifact may be described as native-default full-example parity. A reduced
  subproblem is always labeled `reduced`, even if its numbers match.
- Same initial parameters and arrays means byte-identical dtype, shape, order,
  and payload hashes after loading, not merely numerically close construction.
- Comparable stopping criteria means a shared evaluation/iteration budget plus
  shared scientific terminal gates. Raw solver statuses and counts are retained
  but normalized convergence categories are compared across algorithms.
- Full-scale external-solver workflows remain manual/scheduled or unsupported;
  bounded deterministic cases own required presubmit evidence.
- GPU evidence is authoritative only from a real CUDA device and a clean
  current-checkout import path. CPU fallback, skips, wrong precision, or stale
  editable-source resolution fail the run.

## Implementation Plan

### TDD execution contract

Every implementation slice below is executed in strict **RED -> GREEN ->
REFACTOR** order. Acceptance criteria and test design may be written before a
RED, but no production implementation for that slice may be added first.

- An authentic RED is run against the immediate pre-GREEN revision and fails
  for the intended missing behavior or injected defect, not for test
  collection, an unavailable optional environment, or an unrelated failure.
- The receipt records the pre-GREEN commit/tree identity, exact command,
  expected failure, actual diagnostic, and exit status before implementation.
- GREEN makes the smallest production change that passes that RED and the
  affected regression suite. REFACTOR follows only while all GREEN tests stay
  passing.
- A test written after its production behavior already exists is regression
  coverage, not an authentic RED, and cannot satisfy the TDD receipt gate.
- Each case is promoted independently. A wave label controls dependency order;
  it never authorizes one representative RED or GREEN for the whole wave.

0. Establish the clean committed prerequisite baseline.
   - [ ] **RED:** On the committed baseline, require manifest tests to reject a
     ready example missing either required lane and solver tests to reject a
     strict-GPU default path that selects or executes Optimistix, Optax, SciPy,
     or a host callback in the numerical solve region. Optional solver contract
     imports and explicit opt-in drivers are not themselves failures.
   - [ ] **GREEN:** Land or identify the predecessor revision that owns the
     intended ten ready CPU/GPU examples, backend-neutral default serial
     solver, lane manifest, and focused tests. Do not include unrelated
     dirty-tree changes. Pass the manifest, example-runner, serial-solver,
     CPU-smoke, and real strict-GPU tests from a clean checkout; record the
     commit, resolved module paths, device, commands, and zero-skip results.
   - [ ] **REFACTOR:** Make the committed revision the sole baseline named by
     parity receipts. Dirty-tree exploratory evidence remains explicitly
     non-authoritative.

1. Define and RED-test the parity scope and classifications.
   - [ ] **RED:** Extend `tests/test_jax_examples_manifest.py` or add
     `tests/test_jax_example_parity_manifest.py` to require every relationship
     implied by ready `manifest.json` records to have exactly one parity row.
   - [ ] **RED:** Reject unknown example IDs, native paths absent from the
     referenced `inspired_by`, duplicate relationships/case IDs, hard-coded
     numeric tolerances, missing test owners, invalid scale tiers, or `full`
     rows with incomplete required observables or scientific workflow stages.
   - [ ] **GREEN:** Add `examples/jax/parity_manifest.json` with one row per
     ready JAX example/native `inspired_by` relationship and fields:
     `case_id`, `jax_example_id`, `native_source`, `classification`,
     `classification_reason`, `scale_tier`, `oracle_kind`, `cost_tier`,
     `workflow_stages`, `omitted_scientific_stages`,
     `excluded_teaching_stages`, `comparison_routes`, and
     `correctness_tests`. Each comparison route is an exact `(phase,
     observable, lane_pair) -> (applicability, comparator, tolerance_bucket)`
     mapping. Use only `full`, `reduced`, and `unsupported` classifications.
     `full`/`reduced` require a case ID; `unsupported` requires a concrete
     blocker and must not name executable lanes. Implement the immutable typed
     parser in `examples/jax/parity/_manifest.py`; do not expand the public
     `src/` API. Review the provisional table against live public native and
     JAX APIs and downgrade infeasible rows rather than creating a fake full
     case.
   - [ ] **REFACTOR:** Keep cross-manifest relationship validation in one
     function and preserve deterministic manifest order.

2. Specify the canonical input bundle and result schema test-first.
   - [ ] **RED:** Add schema tests for missing fields, non-finite arrays, dtype
     drift, inconsistent shapes, unknown schema versions, invalid normalized
     status, sidecar hash mismatch, traversal/symlink escape, and a dirty source
     mislabeled authoritative.
   - [ ] Define artifact schema v1 with required provenance: clean repository
     commit, Python/JAX/SIMSOPT versions, resolved source paths, device metadata,
     lane environment policy, case ID, native source, JAX example ID, random
     seed, input fingerprint, and configuration fingerprint. Authoritative runs
     require a clean tree. Exploratory dirty runs record a canonical tracked
     diff hash plus untracked-file inventory and are labeled non-authoritative.
   - [ ] Bind every result to a sorted executed-source manifest containing the
     canonical path and Git blob ID or SHA-256 for the runner, case module,
     manifests/configuration, and transitive native/JAX project modules used by
     the case. Each child snapshots the in-repository Python modules present in
     `sys.modules` after its lane completes, resolves and contains their source
     paths beneath the checkout, and hashes their executed bytes; the runner
     and declarative manifests are added explicitly because they need not be
     imported by every child. Validate the content hashes before comparison; a
     revision plus resolved paths alone is insufficient.
   - [ ] When `simsoptpp` is loaded, record the resolved extension path,
     SHA-256 of the loaded shared object, package/build version metadata, and a
     compatibility receipt tying that binary to the committed checkout. Reject
     a stale or unverifiable extension before treating it as a native/C++
     oracle.
   - [ ] Store bounded scalar/status metadata in canonical JSON. Store each
     parameter, residual, gradient, Jacobian, or constraint array in its own
     deterministic `.npy` sidecar, referenced by a canonical relative path,
     dtype, shape, memory order, and SHA-256. Canonicalize numeric byte order and
     C-contiguity, forbid object arrays/pickle, and pin the NPY format version so
     equal arrays have equal bytes across supported environments. Reject
     absolute paths, `..`, symlinks, containment escapes, missing files, or
     mismatched metadata.
   - [ ] Represent non-applicable quantities with an explicit applicability
     map and `null`, not omitted keys or empty arrays.
   - [ ] Normalize status to `converged`, `budget_exhausted`, `failed`, or
     `not_applicable` while preserving driver name, raw success/status/message,
     `nit`, `nfev`, and `njev`.
   - [ ] Each `LaneResult` records its ordered completed scientific stages; the
     arbiter requires exact agreement with the relationship's declared stage
     contract before comparing numerical observables.
   - [ ] **GREEN:** Add frozen typed contracts in
     `examples/jax/parity/contracts.py` for `ParityInputMetadata`, `LaneResult`,
     `InitialStateResult`, `FinalStateResult`, `ComparisonResult`, and
     `RunManifest`. Implement serialization and validation without a new direct
     dependency; use the standard library, NumPy, and existing project types.
   - [ ] **REFACTOR:** Centralize canonical JSON encoding, array fingerprinting,
     and source/environment provenance; no case may implement its own writer.

3. Build byte-identical input generation and lane isolation.
   - [ ] **RED:** Mutate one parameter, quadrature point, seed, weight,
     constraint, dtype, or stopping option and require the runner to fail before
     scientific comparison with a field-specific fingerprint diagnostic.
     Route each mutation through the real native and JAX case builders and
     prove that the corresponding effective-construction receipt changes; this
     catches ignored fields and default-option leakage.
   - [ ] Require all stochastic cases to use an explicit NumPy generator and
     named seed. Persist generated samples; native and JAX children must load
     them rather than regenerate them independently.
   - [ ] Record quadrature grids, resolution/order, free-DOF indices, weights,
     targets, bounds, constraints, solve budgets, and terminal thresholds in
     the bundle or configuration fingerprint.
   - [ ] Each child must emit an effective-construction receipt reconstructed
     from the instantiated problem and solver: applied parameter/DOF order,
     grids, weights, bounds, constraints, and effective stopping options. The
     arbiter compares this receipt with the canonical bundle; hashing the
     parent input alone is not proof that a lane consumed it.
   - [ ] **GREEN:** Add `examples/jax/parity/input_bundle.py` to create each
     canonical bundle once from NumPy arrays/scalars and record dtype, shape,
     memory order, and SHA-256 per leaf. Add `native-cpu`, `jax-cpu`, and
     `jax-gpu` subprocess environments in `examples/jax/parity/runtime.py`.
     Reuse the established runtime-policy values from `run_examples.py` through
     one shared owner rather than copying environment dictionaries.
   - [ ] Native CPU must resolve the current checkout and native SIMSOPT stack;
     JAX CPU/GPU must report `jax_cpu_parity`/`jax_gpu_parity`, FP64, and the
     expected platform. Strict GPU must disallow transfers and CPU fallback.
   - [ ] **REFACTOR:** Extract a shared lane-environment module only after
     regression tests prove the existing example-runner argv/environment and
     behavior remain unchanged.

4. Implement the paired runner and fail-closed arbiter.
   - [ ] **RED:** Add injected-child tests for missing lanes, nonzero children,
     malformed results, input/config/effective-construction fingerprint drift,
     wrong source checkout, wrong platform/precision/backend, non-finite
     required observables, unsupported masquerading as pass, direct pairwise
     tolerance failure, stable manifest ordering, and exact bounded argv.
   - [ ] **RED:** Add concurrent-writer, interrupted-publication, existing-run,
     and partial-artifact rejection tests.
   - [ ] **GREEN:** Implement deterministic aggregation and one summary JSON
     whose verdict is `pass` only when all required pairwise comparisons pass.
     Add `examples/jax/run_parity.py` with only `--case`, `--lanes`,
     `--artifact-root`, and `--smoke`; do not add arbitrary command passthrough
     or per-case tolerance flags. Add a static typed registry in
     `examples/jax/parity/cases/__init__.py`; do not use dynamic imports.
     Execute every lane in a fresh child process after its environment is fixed
     and before JAX-heavy imports, capturing exact argv, stdout, stderr, return
     code, elapsed time, and result paths. The arbiter must fail on missing
     lanes, nonzero children, malformed results, input/config fingerprint
     mismatch, wrong source checkout, wrong platform/precision/backend,
     non-finite required observables, unsupported masquerading as pass, or any
     tolerance failure. Require direct `native-cpu` versus `jax-cpu`, direct
     `native-cpu` versus `jax-gpu`, and `jax-cpu` versus `jax-gpu` comparisons
     for every applicable initial and final observable; adjacent comparisons
     never substitute for the direct native/GPU gate.
     Create each run with exclusive ownership beneath a unique
     `<UTC>-<nonce>.partial` directory, fsync required files, and atomically
     rename it to the final run ID only after validation, then fsync the parent
     directory. Never overwrite an existing run; retain a failed partial
     directory with an explicit failure marker for diagnosis.
   - [ ] **REFACTOR:** Keep subprocess mechanics independent from scientific
     comparison; the runner collects lane results and the arbiter compares them.

5. Establish initial-state parity before any solve comparison.
   - [ ] For each case, evaluate native CPU and JAX at the exact serialized
     initial parameters before invoking a solver or integrator.
   - [ ] Least-squares cases must record residual vector `r`, public objective
     `objective_sum_squares = r.T @ r`, residual Jacobian `J`, and public
     objective gradient `objective_gradient = 2 * J.T @ r`. Record a solver's
     half-squared `solver_cost = 0.5 * r.T @ r` separately when available; never
     compare it directly with the public objective. Scalar cases must record
     objective and gradient. Constrained cases additionally record constraint
     values and constraint Jacobian.
   - [ ] Tracing/fixed-state cases must name their analogous initial-state
     quantities (field/RHS/invariants or objective/gradient) and mark solver-only
     quantities not applicable.
   - [ ] Select tolerances through the central ladder only after validating that
     the bucket's oracle-kind requirements match the case. Use existing buckets
     such as `direct_kernel`, `derivative_heavy`, or `gpu_runtime` only when
     their direct-C++/derivative/GPU semantics actually apply. For native
     Python/SciPy workflow comparisons such as `just_a_quadratic.py`, add a
     centrally owned native-workflow bucket with a dedicated contract test,
     adversarial threshold test, and rationale rather than falsely labeling the
     Python oracle as direct C++.
     Analytic and external oracles likewise require compatible centrally owned
     semantics or remain non-promotable; no case-local numeric threshold is
     permitted.
   - [ ] **RED:** Write one public-behavior test per case that first fails on a
     genuine missing comparison or injected perturbation; preserve the exact
     failing command and diagnostic in a TDD receipt.
   - [ ] **RED:** Add a least-squares contract test whose nonzero residual and
     Jacobian catch a factor-of-two swap between the public objective gradient
     and the half-squared solver-cost gradient.
   - [ ] **GREEN:** Make all required native CPU vs JAX CPU, native CPU vs JAX
     GPU, and JAX CPU vs JAX GPU initial-state comparisons pass before enabling
     final solves for that case.
   - [ ] **REFACTOR:** Share mathematical recomposition helpers only where two
     completed cases use the same objective convention; keep tests DAMP.

6. Add final-outcome parity in dependency order.
   - [ ] For each case, first add and execute a case-owned RED that fails on the
     missing real native/JAX comparison or a scientifically meaningful injected
     perturbation. Then implement that case's smallest GREEN, run its affected
     regressions, and REFACTOR while green before starting the next case.
   - [ ] Wave A: traceable least-squares, curve length, and surface geometry.
     These establish scalar/least-squares result contracts and work-counter
     reporting with inexpensive deterministic fixtures.
   - [ ] Wave B: coil flux, permanent magnets, wireframe, and fixed-state
     coil-force/finite-build. These establish adapter publication, discrete
     decisions, constraints, and explicit N/A optimizer fields.
   - [ ] Wave C: QFM and Boozer. Require branch-stable initial states,
     feasibility/KKT gates, accepted-state publication, and original-residual
     checks; matching augmented or preconditioned residuals alone is invalid.
   - [ ] Wave D: field-line and particle tracing. Match physical initial state,
     integration interval, event semantics, tolerances, and final invariants;
     compare endpoints/events rather than demanding adaptive-step identity.
   - [ ] For every optimizing case, record final parameters, objective,
     residuals, gradient/Jacobian where defined, constraints/feasibility,
     normalized and raw status, driver, and work counters.
   - [ ] Apply the manifest's typed final comparison routes to every applicable
     lane pair. Each route specifies vector/scalar/constraint comparator and
     central tolerance bucket. For a case with non-unique minimizers, compare a
     declared equivalence invariant or quotient representation rather than raw
     parameters, and justify that policy in the case contract; individual
     feasibility or objective gates do not replace pairwise final parity.
   - [ ] Define common terminal scientific gates independently of driver status.
     A solver-reported success with failed residual/gradient/feasibility is a
     parity failure.
   - [ ] Compare `nit`/`nfev`/`njev` exactly only for the same algorithm and
     implementation contract. Otherwise require finite nonnegative counts and
     report absolute/relative deltas as diagnostics with an explicitly approved
     bound if one is scientifically necessary.
   - [ ] Record each case's RED -> GREEN -> REFACTOR evidence independently; do
     not batch-promote a wave based on one representative example.

7. Add real CPU/GPU evidence and memory/transfer gates.
   - [ ] **RED:** Prove strict GPU rejects a fabricated CPU result, a hidden
     host optimizer, an optional optimizer selected as the parity default, and
     a receipt that sets only the SIMSOPT policy without JAX's effective
     transfer guard before accepting a CUDA receipt.
   - [ ] Run every full/reduced JAX case on CPU and a real CUDA device with the
     identical input bundle and configuration fingerprint.
   - [ ] Require FP64 arrays for parameters, residuals, objectives, gradients,
     Jacobians, and solver state at every authoritative comparison boundary.
   - [ ] Require strict transfer guarding during the numerical region and fail
     on host callbacks, hidden SciPy/native solvers, CPU fallback, skipped GPU
     execution, or wrong device metadata.
   - [ ] Record peak host RSS and device memory as diagnostics for cases whose
     compiled solver state may scale quadratically. Name the measurement owner
     and method in the receipt: parent-observed child RSS and, only where the
     CUDA backend exposes a validated per-process counter, device allocation.
     Record device memory as `unavailable` rather than inventing a value when no
     supported counter exists. Synchronize JAX work with
     `jax.block_until_ready` at declared measurement boundaries, distinguish
     compile/warmup from steady-state execution, and add shape-scaling tests for
     any case that materializes dense Jacobian/Hessian state. These diagnostics
     do not support speed claims.
   - [ ] Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` for measurement receipts;
     document production preallocation policy separately so memory diagnostics
     are not confused with the supported runtime default.
   - [ ] **GREEN:** Publish real device metadata and zero-fallback receipts for
     all applicable cases.

8. Integrate artifacts, documentation, and CI.
   - [ ] **RED:** Add static workflow tests proving changes under parity
     manifests/cases reach the named CPU and strict-GPU jobs and that CI invokes
     the parity CLI rather than duplicating a case list. Add a narrow ignore
     test that initially fails for a probe below
     `.artifacts/jax-example-parity/`.
   - [ ] **GREEN:** Write local artifacts beneath
     `.artifacts/jax-example-parity/<UTC-run-id>/` and keep generated data out of
     Git. Upload selected JSON/NPY artifacts through CI with a retention policy.
   - [ ] Add only `.artifacts/jax-example-parity/` to `.gitignore`; do not
     broaden ignore or cleanup rules to unrelated artifacts or canonical input
     directories.
   - [ ] Add `docs/jax_native_example_parity_results.md` only after the first
     complete run; generate its tables from the aggregate JSON rather than
     copying values manually.
   - [ ] Extend `examples/jax/README.md` with the parity command, classification
     semantics, artifact layout, and the distinction between native
     Python/SciPy, `simsoptpp`, analytic, and external oracles.
   - [ ] Add CPU Wave A to the `jax-public-integration` job in
     `.github/workflows/jax_smoke.yml`. Add strict-GPU Wave A to that workflow's
     `jax-gpu-strict-purity` job only after bounded runtime is measured. Ensure
     the workflow is exercised by its pull-request targets or an explicit
     manual trigger before treating it as evidence.
   - [ ] Route expensive later waves only to a named scheduled/manual CUDA job
     that uses FP64, both effective transfer-guard settings at `disallow`,
     zero-skip/no-fallback checks, and artifact upload. The current
     `.github/workflows/jax_gpu_parity.yml` log-guard job is diagnostic and may
     not produce authoritative strict-parity evidence unless upgraded to this
     contract.
   - [ ] Preserve RED/GREEN/REFACTOR commands, source hashes, artifact hashes,
     device identity, and exact pass/fail counts in
     `docs/jax_examples_tdd_receipts.md` or a dedicated parity receipt file.

9. Complete the API, provenance, and final review gates.
   - [ ] Confirm the new CLI has no untyped pass-through options and the schema
     exposes only stable consumer data.
   - [ ] Confirm no `src/` module imports from `examples/jax/parity`, no native
     example is imported for top-level execution, and no new direct dependency
     was added.
   - [ ] Verify each `full` classification executes all claimed workflow stages;
     downgrade any fixed-state or reduced case whose artifact cannot prove the
     final accepted state.
   - [ ] Verify artifact paths are relative, source/input hashes resolve, array
     sidecars match, and secrets/environment-variable values are excluded.
   - [ ] Review the final path-scoped diff for duplicated tolerances, duplicated
     lane policy, hidden host transfers/solvers, stale lineage, weakened
     scientific thresholds, and unrelated dirty-tree changes.

## Validation Plan

- [ ] Manifest and schema RED/GREEN tests:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/test_jax_example_parity_manifest.py \
    tests/integration/test_jax_example_parity_artifacts.py \
    tests/integration/test_jax_example_parity_inputs.py \
    tests/integration/test_jax_example_parity_publication.py \
    tests/integration/test_jax_example_parity_runner.py \
    tests/integration/test_jax_example_parity_runtime.py
  ```

- [ ] Existing manifest/runner compatibility:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/test_jax_examples_manifest.py \
    tests/integration/test_jax_examples.py
  ```

- [ ] Wave A native/JAX CPU artifact:

  ```console
  PYTHONPATH="$PWD/src" ../.venv-simsopt-linux-x86/bin/python \
    examples/jax/run_parity.py \
    --case traceable-least-squares \
    --case curve-length-optimization \
    --case surface-geometry-optimization \
    --lanes native-cpu,jax-cpu \
    --smoke \
    --artifact-root .artifacts/jax-example-parity
  ```

- [ ] Wave A real strict-GPU artifact on the designated CUDA environment:

  ```console
  conda run --no-capture-output -n "${BENCHMARK_ENV_NAME:?set CUDA environment}" \
    python examples/jax/run_parity.py \
    --case traceable-least-squares \
    --case curve-length-optimization \
    --case surface-geometry-optimization \
    --lanes native-cpu,jax-cpu,jax-gpu \
    --smoke \
    --artifact-root .artifacts/jax-example-parity
  ```

- [ ] Full applicable registry run after every wave is implemented:

  ```console
  conda run --no-capture-output -n "${BENCHMARK_ENV_NAME:?set CUDA environment}" \
    python examples/jax/run_parity.py \
    --case all-applicable \
    --lanes native-cpu,jax-cpu,jax-gpu \
    --artifact-root .artifacts/jax-example-parity
  ```

- [ ] Artifact audit verifies every required lane, exact input/config
  and effective-construction fingerprint agreement, completed scientific
  stages, source path/revision and executed-source hashes, any loaded
  `simsoptpp` binary hash/compatibility receipt, FP64/device and effective
  transfer-guard metadata, sidecar SHA-256, path containment,
  authoritative-source status, observable applicability, tolerance result, and
  final verdict.
- [ ] Run focused native/JAX subsystem tests named by each case, including
  curve/surface objectives, Biot-Savart/flux, QFM, permanent magnets, tracing,
  Boozer, wireframe, force, and finite-build coverage.
- [ ] Run strict-GPU focused tests in a real CUDA process with zero skips and no
  CPU fallback.
- [ ] Static/type/format checks:

  ```console
  python -m ruff check examples/jax/parity examples/jax/run_parity.py \
    tests/test_jax_example_parity_manifest.py \
    tests/integration/test_jax_example_parity_*.py
  python -m ruff format --check examples/jax/parity examples/jax/run_parity.py \
    tests/test_jax_example_parity_manifest.py \
    tests/integration/test_jax_example_parity_*.py
  python -m compileall -q examples/jax/parity examples/jax/run_parity.py
  ```

  Run these from the project-pinned development environment. Do not introduce
  an unpinned ad hoc type checker as release evidence; add and configure a
  project-pinned checker in a separately reviewed change if static type-check
  enforcement becomes a repository gate.

- [ ] Import/source-boundary checks:

  ```console
  ! rg -n 'examples\.jax\.parity|from examples|import examples' \
    src/simsopt src/simsopt_jax src/simsopt_jax_adapters
  git check-ignore -q --no-index \
    .artifacts/jax-example-parity/probe/summary.json
  ```

  Native case functions may import SciPy. A lane-aware behavioral test must
  instead prove that `jax-cpu` and `jax-gpu` cannot select or execute SciPy,
  native callbacks, Optimistix, or Optax as the parity-default numerical path;
  a blanket source search over mixed native/JAX case modules is invalid.

- [ ] Stage only the reviewed approved-path set, then run both
  `git diff --check` and `git diff --cached --check`; inspect
  `git diff --cached --name-status` plus final `git status --short`. This binds
  new files as well as tracked working-tree and index changes while proving
  unrelated existing modifications remain intact.

## Risks and Mitigations

- Risk: A redesigned JAX lesson is compared with a native script that solves a
  different problem.
  Mitigation: One serialized input/config fingerprint is consumed by both lane
  implementations, and the parity manifest classifies each exact relationship.

- Risk: A reduced deterministic subproblem is presented as full native-example
  equivalence.
  Mitigation: classification and `scale_tier` are schema-enforced; result
  documents group bounded, native-default, reduced, and unsupported evidence
  separately and never derive native-default coverage from a bounded or reduced
  row.

- Risk: A parity receipt is generated from uncommitted prerequisite changes or
  omits untracked source inputs.
  Mitigation: authoritative runs require a clean named commit and resolved
  import paths; dirty runs are exploratory, inventory tracked and untracked
  state, and cannot promote a classification.

- Risk: Array publication is nondeterministic, partially written, or escapes
  the artifact root.
  Mitigation: canonical JSON plus one hash-bound `.npy` file per array is
  published through an exclusive partial directory and atomic rename; readers
  reject symlinks, traversal, containment escapes, and incomplete runs.

- Risk: Native and JAX solvers reach different valid minimizers in a nonconvex
  problem.
  Mitigation: Require initial-state value/derivative parity first, use
  branch-stable fixtures, compare feasibility and scientific invariants, and
  downgrade branch-divergent cases instead of relaxing thresholds.

- Risk: Different solver status/count semantics cause false failures or hide
  excess work.
  Mitigation: Preserve raw fields, compare normalized convergence categories,
  gate the final scientific state independently, and treat counter deltas as
  diagnostics unless a same-algorithm contract justifies an exact/bounded gate.

- Risk: Native example imports trigger MPI, external solvers, plotting, or
  writes.
  Mitigation: Reconstruct matched cases from public APIs and serialized inputs;
  do not import top-level example scripts.

- Risk: GPU results come from a stale editable checkout or CPU fallback.
  Mitigation: Record resolved module paths and source hashes in every child,
  require CUDA platform metadata and strict backend mode, and fail source or
  platform mismatches before numerical comparison.

- Risk: Full Jacobian/Hessian artifacts or on-device optimizer state exhaust
  host/device memory.
  Mitigation: Keep smoke fixtures bounded, store arrays in sidecars, use
  matrix-free solvers where intended, record memory diagnostics, and require a
  separate scale gate before increasing resolution.

- Risk: Tolerances are weakened until difficult examples pass.
  Mitigation: Route comparisons through the existing tolerance ladder; any new
  bucket requires an independent rationale, adversarial perturbation test, and
  review before the failing case is rerun.

- Risk: A second runner duplicates example lane policy.
  Mitigation: Extract one shared environment owner after compatibility RED tests
  exist; keep scientific pairing out of the ordinary smoke runner.

## Completion Criteria

- [ ] Every ready JAX example/native `inspired_by` relationship has exactly one
  validated `full`, `reduced`, or `unsupported` parity classification.
- [ ] Every `full` case proves byte-identical inputs and configuration across
  native CPU, JAX CPU, and applicable JAX GPU lanes.
- [ ] Every `full` relationship has a complete declared scientific-stage list,
  no omitted scientific stage, and matching completed-stage receipts from all
  required lanes; reduced and teaching-only exclusions remain explicit.
- [ ] Every applicable case passes required initial objective/residual,
  gradient/Jacobian, and constraint comparisons through centrally owned
  tolerances.
- [ ] Every optimizing full case passes final parameters, objective/residual,
  feasibility, normalized convergence, raw-status recording, and work-counter
  reporting gates.
- [ ] Every tracing or fixed-state full case passes its explicitly applicable
  endpoint/invariant or objective/gradient contract with optimizer fields
  explicitly marked not applicable.
- [ ] All ready applicable JAX cases pass on a real CUDA device in FP64 with
  both effective transfer-guard settings at `disallow` and no CPU fallback or
  hidden host solver.
- [ ] Aggregate JSON and NPY sidecars are source/input/environment hash-bound,
  schema-valid, and sufficient for an independent reviewer to reproduce every
  verdict.
- [ ] RED -> GREEN -> REFACTOR receipts exist for every promoted case; focused
  tests, existing example regressions, Ruff, format, compileall, import
  boundaries, and `git diff --check` pass.
- [ ] Documentation reports full, reduced, unsupported, CPU-only, and
  strict-GPU evidence separately; no analytic-only or reduced result is called
  native/C++ end-to-end parity.
- [ ] Unrelated dirty/untracked work remains unchanged.

## Open Questions

- Should “full example parity” require the original native script's default
  full-resolution run, or is a bounded matched workflow with all stages and
  identical inputs sufficient for a `full` workflow-coverage classification in
  presubmit while defaults run in scheduled CI? This plan assumes the latter,
  records the bounded scale independently, and reserves native-default parity
  claims for passing scheduled/manual evidence.
- What maximum wall time, host RSS, and device memory should determine whether
  Waves B-D run in presubmit versus scheduled CI?
- For algorithm-different solvers, should work-counter regressions remain
  reporting-only, or should each case define a reviewed maximum ratio after a
  stable baseline exists?
- How long should CI retain JSON/NPY parity artifacts, and which artifact store
  is authoritative for release claims?
