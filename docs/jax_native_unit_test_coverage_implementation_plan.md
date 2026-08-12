# JAX Coverage of Native SIMSOPT Unit-Test Capabilities Implementation Plan

**Status:** Draft  
**Last updated:** 2026-07-29

## Purpose

Close the gap between the behavior exercised by the native SIMSOPT test suite
and the behavior certified for the JAX implementation. This plan establishes a
complete, reviewable mapping from native test definitions to JAX-equivalent
tests, explicit host/hybrid boundaries, or justified non-JAX dispositions, then
implements the missing portable numerical capabilities using RED → GREEN →
REFACTOR test-driven development.

This is broader than the completed one-to-one example campaign. Example parity
proves selected end-to-end scientific workflows; this plan addresses the
lower-level values, derivatives, shapes, failure semantics, symmetry rules,
solver contracts, and device behavior exercised throughout the native unit
suite.

## Goals

- Trace every native test definition at the approved upstream baseline to one
  or more capability records with an explicit disposition and owner.
- Provide JAX CPU and strict JAX GPU parity for every portable numerical
  capability exposed by the supported JAX API.
- Implement missing JAX-native capabilities that are approved as in scope,
  without hidden native execution, host callbacks, or silent CPU fallback.
- Test values, derivatives, shapes, dtypes, batching/symmetry behavior, failure
  semantics, and state contracts wherever the native suite treats them as
  observable behavior.
- Preserve explicit host boundaries for VMEC, SPEC, MPI orchestration, file I/O,
  plotting, and other behavior that should not be reimplemented as an
  on-device numerical kernel.
- Fail CI when upstream adds or changes a native test without a corresponding
  coverage decision.
- Produce an auditable coverage report that distinguishes complete parity,
  partial coverage, missing JAX capability, shared backend-independent
  behavior, hybrid coverage, and justified native-only behavior.
- Retain FP64 scientific authority and add lean native-C++-CPU versus JAX-GPU
  performance and memory checks only for representative performance-critical
  kernels.

## Non-Goals

- Duplicate every native test file line-for-line or require identical test
  names.
- Claim coverage based on raw native/JAX test counts.
- Port plotting, serialization formats, logging, filesystem operations, MPI
  process management, or external solver implementations onto the GPU.
- Replace VMEC, SPEC, or other external scientific executables with new JAX
  implementations under this plan.
- Require bitwise-identical floating-point reductions across CPU and GPU.
- Require identical optimizer trajectories, iteration counts, or final
  parameters when multiple scientifically equivalent minima exist.
- Benchmark every unit test or create a large performance matrix.
- Reconstruct historical RED evidence that was not preserved.

## Current Context

- The inspected local upstream baseline is `upstream_check/master` at
  `377cf6651`. It contains 70 Python test files and 738 test definitions.
- Current remote upstream `master` is
  `4ad6fd99189b99d9722ad33aaeb5d30adc81680f`; it contains 71 Python test files
  and 753 test definitions. The remote delta demonstrates that the authority
  baseline must be refreshed and pinned before classification.
- The current remote definitions are distributed across:
  - `tests/configs`: 4 files
  - `tests/core`: 9 files
  - `tests/field`: 16 files
  - `tests/geo`: 20 files
  - `tests/mhd`: 9 files
  - `tests/objectives`: 4 files
  - `tests/solve`: 5 files
  - `tests/util`: 4 files
- The remote suite uses `unittest` and contains 91 `self.subTest(...)` sites
  plus 73 conditional skip sites. Test-method count is therefore a source
  traceability denominator, not the number of behavioral cases executed in
  every environment.
- No separate upstream C++ test-file suite was found under conventional test
  paths. The native tests are primarily Python tests that exercise SIMSOPT and
  the compiled `simsoptpp` bindings.
- The current checkout contains extensive JAX coverage under `tests/jax`,
  JAX-named tests in the native domain directories, and integration tests.
  However, there is no single source of truth mapping all native tests to
  equivalent JAX behavior.
- `src/simsopt_jax` contains the JAX-native numerical implementation;
  `src/simsopt_jax_adapters` owns explicit boundaries to native SIMSOPT and
  external systems.
- The example parity manifest currently records 26 full bounded relationships
  and one unsupported VMEC-hybrid certification relationship. That manifest is
  evidence for example workflows, not complete native unit-test coverage.
- Existing execution infrastructure already provides:
  - FP64 CPU and GPU policy;
  - transfer-guard enforcement;
  - native CPU, JAX CPU, and strict JAX GPU lanes;
  - centralized parity tolerances;
  - Pyright, Ruff, formatting, and compile checks;
  - example-level authority artifacts and provenance.
- The working tree contains unrelated modified and untracked user files. Any
  implementation must stage and commit only files belonging to this plan.
- The pre-execution blindspot decisions and rewritten execution prompt are
  recorded in `docs/jax_native_unit_test_coverage_unknowns.md`.

## Rationale

Filename matching is insufficient: one native test can cover several
capabilities, several native tests can exercise one kernel, and a JAX test may
add device-specific contracts that have no native counterpart. Coverage must
therefore be measured at two linked levels:

1. A complete native-test ledger proves that no upstream test definition was
   silently ignored.
2. A capability manifest groups those test definitions by observable behavior
   and owns the required JAX tests, lanes, tolerances, and disposition.

This keeps the coverage decision in one place while allowing clear,
behavior-oriented tests. It also prevents host-only utilities from being
misrepresented as missing GPU kernels and prevents a large number of JAX
regression tests from being mistaken for complete native coverage.

### Design considered twice

**Alternative A: filename and test-name pairing.** Generate a native test list
and require a similarly named JAX file or function. This is simple but creates
false positives, rewards duplicated tests, and cannot represent shared,
hybrid, or intentionally native-only behavior.

**Alternative B: native-test ledger plus capability manifest.** Require every
native test definition to map to a typed capability record, then require each
portable capability to own behavioral parity tests and device lanes. This adds
one small metadata layer but provides complete traceability without coupling
test organization to implementation organization.

**Decision:** use Alternative B. One coverage contract will own the schema,
allowed dispositions, and completeness rules so changes do not require
coordinated policy edits across scripts, tests, and documentation.

## Assumptions

- Before implementation begins, the upstream authority baseline will be
  refreshed and pinned to an immutable commit rather than implicitly following
  a moving branch.
- “Native test” means a test definition in the approved upstream `tests/`
  tree, including tests that exercise compiled `simsoptpp` behavior through
  Python.
- Every source test method is part of the traceability denominator. Material
  loop and `subTest` regimes are recorded as capability parameters rather than
  incorrectly treated as separate pytest node IDs.
- Coverage equivalence means equivalent observable scientific and API
  behavior, not identical internal algorithms.
- FP64 remains the scientific authority mode for native/JAX comparisons.
- A JAX-native numerical capability must execute on the selected JAX device.
  Explicit input staging and final reporting transfers are allowed; hidden
  callbacks and computational fallback are not.
- Host/hybrid and native-only dispositions require a concrete reason and an
  independent review. They are not escape hatches for difficult ports.
- Existing production APIs should be extended only when the capability review
  establishes a real public need; private native implementation details do not
  automatically justify new JAX public APIs.

## Coverage Contract

### Required artifacts

- `tests/fixtures/jax_native_unit_coverage_manifest.json`
  - Immutable baseline commit and source-tree hash.
  - Every native test definition and its capability IDs.
  - Every capability’s domain, disposition, observables, required lanes,
    JAX API, test IDs, tolerance owner, and blocker when applicable.
- `scripts/jax_native_unit_coverage.py`
  - Deterministic inventory, validation, report generation, and drift checking.
  - Read-only by default; explicit `--write` required to update generated
    artifacts.
- `tests/jax/test_native_unit_coverage_manifest.py`
  - Schema, completeness, path identity, source hash, disposition, and
    orphan-reference tests.
- `docs/jax_native_unit_test_coverage.md`
  - Generated human-readable coverage index and gap summary.
- `tests/jax/native_unit_parity/`
  - New behavioral parity tests when no suitable existing test owns the
    capability.

### Allowed dispositions

- `jax_equivalent`: all required observable behavior has native CPU, JAX CPU,
  and strict JAX GPU evidence.
- `jax_partial`: some required behavior is covered, but at least one observable,
  parameter regime, derivative order, failure mode, or device lane is missing.
- `jax_missing`: portable numerical behavior has no supported JAX
  implementation.
- `hybrid_boundary`: host/external computation is intentionally retained, and
  the boundary plus JAX-owned slice is tested.
- `shared_python`: backend-independent Python behavior is exercised once and
  does not need a duplicate JAX implementation.
- `native_only`: behavior is intentionally outside the JAX product boundary,
  with a concrete technical rationale and reviewer approval.

`jax_partial` and `jax_missing` are valid planning states but fail final
completion. Empty reasons, generic “unsupported” labels, missing tests, stale
paths, and unknown native tests fail closed.

### Required parity dimensions

Each capability record must explicitly select the applicable dimensions:

- value or residual;
- gradient, Jacobian, vector-Jacobian product, and Hessian behavior;
- shape, dtype, scalar/array convention, and PyTree structure;
- symmetry, periodicity, orientation, and coordinate convention;
- batching, broadcasting, chunking, and empty/singular edge cases;
- deterministic seeded behavior;
- mutation, cache invalidation, and state-token behavior when public;
- exception/status behavior for invalid input and non-convergence;
- CPU/GPU device placement and transfer behavior;
- JIT, `vmap`, and autodiff compatibility when promised by the JAX API.

Topology, indices, masks, shapes, status values, and discrete decisions require
exact equality. Floating-point tolerances must use the centralized tolerance
owner in `src/simsopt_jax/parity_tolerances.py`, be justified by conditioning
and scale, and must not be weakened merely to make a new port pass.

## Implementation Plan

1. Pin the authority baseline and build a complete native-test inventory.
   - [ ] Fetch or otherwise verify the intended upstream authority ref and
     record its immutable commit, merge-base relationship, and clean source
     tree hash.
   - [ ] Inventory every upstream `test_` function and test method with Python
     AST parsing so optional dependency failures cannot hide source tests.
   - [ ] In a full native authority environment, run `pytest --collect-only`
     and bind runtime-collected node IDs to their owning source definitions.
     Preserve material `subTest` loop regimes in capability metadata because
     pytest collection does not enumerate them as separate node IDs.
   - [ ] Record native test file/function hashes so renames and semantic edits
     are distinguishable from unchanged coverage.
   - [ ] Add RED tests proving the manifest validator rejects an omitted native
     test, duplicate mapping, stale hash, missing baseline commit, malformed
     disposition, orphan JAX test, and unreviewed `native_only` record.
   - [ ] Implement the deterministic inventory and validation CLI until those
     tests pass.
   - [ ] Generate the initial coverage document without claiming that
     unclassified records are covered.

2. Classify every native test by observable capability.
   - [ ] Review `tests/core` and map derivative plumbing, DOF/state semantics,
     serialization boundaries, and backend-independent framework behavior.
   - [ ] Review `tests/geo` and map curve/surface geometry, frames, distances,
     linking, finite-build, strain, QFM, and derivative orders.
   - [ ] Review `tests/field` and map Biot–Savart, coils, magnetic-field
     composition, interpolation, tracing, particles, sampling, self-field
     forces, wireframes, and I/O-only behavior.
   - [ ] Review `tests/objectives` and map values, residuals, gradients,
     Jacobians, constraints, and degenerate cases.
   - [ ] Review `tests/solve` and map least-squares, scalar minimization,
     constrained solving, MPI distribution, status, counters, and failure
     semantics.
   - [ ] Review `tests/mhd` and separate portable Boozer/geometry calculations
     from VMEC, SPEC, virtual-casing, and Fortran/external runtime boundaries.
   - [ ] Review `tests/configs` and `tests/util` and separate numerical
     constructors from fixtures, downloads, formatting, and I/O.
   - [ ] Require each native definition to have at least one capability ID and
     each capability to reference at least one native definition.
   - [ ] Require a concrete owner and rationale for every `hybrid_boundary`,
     `shared_python`, and `native_only` decision.
   - [ ] Publish the initial counts by domain and disposition. Do not convert
     `jax_partial` or `jax_missing` into a percentage “covered” claim.

3. Reuse and harden existing JAX tests before adding new ones.
   - [ ] Match current JAX tests to capability records by observable behavior,
     not filename similarity.
   - [ ] Reject source-text-only, import-only, and mock-only tests as scientific
     parity evidence unless the capability itself is a source/import contract.
   - [ ] Identify capabilities that already have correct native CPU/JAX
     CPU/JAX GPU coverage and bind their existing test IDs into the manifest.
   - [ ] Add missing assertions to an existing coherent test when that preserves
     one observable responsibility; otherwise create a focused parity test
     under `tests/jax/native_unit_parity/`.
   - [ ] Keep DAMP test fixtures when duplication makes the native/JAX
     scientific input visibly identical; extract only genuine shared scientific
     specifications.
   - [ ] Extend centralized tolerances only for genuinely new observable
     classes and add mutation tests proving the validator rejects weakened or
     unowned tolerances.

4. Execute RED → GREEN → REFACTOR for every portable gap.
   - [ ] **RED:** write a public-surface behavioral test using identical,
     immutable native/JAX inputs. Confirm it fails for the missing behavior,
     wrong derivative, wrong shape, wrong failure contract, or wrong device
     placement.
   - [ ] Preserve the failing command, test ID, checkout, and concise failure
     signature in a structured receipt. Do not synthesize historical failures.
   - [ ] **GREEN:** implement the smallest complete JAX-native behavior in the
     appropriate `src/simsopt_jax` module, or the explicit host boundary in
     `src/simsopt_jax_adapters`.
   - [ ] Prohibit `pure_callback`, `io_callback`, implicit NumPy conversion,
     hidden `device_get`, and native computational fallback inside JAX-native
     kernels.
   - [ ] Match public shapes, dtypes, coordinate conventions, normalization,
     derivative conventions, and failure semantics before optimizing.
   - [ ] Validate derivatives against both the native oracle and an independent
     directional finite-difference or Taylor test where numerically meaningful.
   - [ ] **REFACTOR:** remove duplicated scientific constants, preserve a
     single owner for policy/specification, and rerun RED and GREEN revisions
     to prove the test discriminates the behavior.
   - [ ] Update the manifest only after the required lanes pass; generated
     metadata must never be the mechanism that turns a failing test green.

5. Close gaps in capability-priority waves.
   - [ ] **Wave A — Existing implementation, missing evidence:** close
     `jax_partial` records where the JAX API already exists. This is the
     highest-priority and lowest-design-risk wave.
   - [ ] **Wave B — Core geometry and representations:** close portable curve,
     surface, frame, distance, finite-build, strain, QFM, symmetry, and
     derivative gaps.
   - [ ] **Wave C — Fields and objectives:** close Biot–Savart, coil,
     composition, interpolation, tracing, particle, force, permanent-magnet,
     wireframe, residual, and objective gaps.
   - [ ] **Wave D — Solvers and optimization contracts:** close supported
     least-squares and scalar minimization behavior, including status, budgets,
     counters, non-finite behavior, and equivalent-minimum comparisons.
   - [ ] Treat native constrained optimization as a separate architecture gate:
     either approve and implement a SIMSOPT-owned backend-neutral contract or
     retain an explicit `jax_missing` blocker. Do not disguise projection or a
     third-party optimizer as native solver equivalence.
   - [ ] **Wave E — MHD and hybrid boundaries:** certify JAX-owned Boozer and
     geometry slices; add boundary tests for VMEC/SPEC/virtual-casing host
     execution, mixed derivatives, dtype/shape transfer, MPI ownership, and
     failure propagation.
   - [ ] **Wave F — Configuration and utility behavior:** port only numerical
     constructors that belong in the JAX product. Classify plotting, logging,
     file formats, downloads, and backend-independent utilities explicitly.
   - [ ] After each wave, regenerate the coverage index and require zero new
     unclassified native tests.

6. Establish CPU, strict-GPU, and hybrid CI lanes.
   - [ ] Add a `jax_native_unit_parity` pytest marker and require each
     `jax_equivalent` capability to be collected by the appropriate lane.
   - [ ] Run public pure-JAX tests without `simsoptpp` where the API promises
     that independence.
   - [ ] Run native-oracle/JAX CPU parity in the existing JAX public integration
     workflow with FP64 enabled.
   - [ ] Run strict JAX GPU parity with the actual GPU backend, FP64, device
     identity checks, and transfer guard set to `disallow`.
   - [ ] Run hybrid boundary tests separately so intentional host computation
     is not mislabeled as strict GPU execution.
   - [ ] Keep slow/external tests scheduled, but require their manifest records
     to name the workflow, environment, and latest authority evidence.
   - [ ] Add a fail-closed drift job that inventories the approved upstream
     baseline and rejects unclassified additions or changed native tests.

7. Validate precision, performance, and memory without over-benchmarking.
   - [ ] Compare native CPU, JAX CPU, and strict JAX GPU values at identical
     inputs before comparing optimized or iterative final states.
   - [ ] Record absolute and relative errors for values, residuals, gradients,
     Jacobians, and Hessian-vector products as applicable.
   - [ ] For iterative algorithms, compare scientific success, feasibility,
     final objective/residual, and documented equivalent minima; report
     iterations and evaluations without assuming they must match.
   - [ ] Select one or two representative sizes only for each
     performance-critical kernel family.
   - [ ] Compare native C++/SIMSOPT CPU against JAX GPU for any GPU-speed claim;
     JAX CPU versus JAX GPU alone is insufficient.
   - [ ] Separate compilation/warmup from steady-state timing and synchronize
     devices before measurement.
   - [ ] Record wall time plus host RSS and GPU peak allocated memory for the
     representative cases.
   - [ ] Add scaling/memory regression checks for kernels whose intermediates
     can grow quadratically or whose transforms can materialize large
     Jacobians. Prefer structural or generous regression bounds over fragile
     machine-specific speed thresholds.
   - [ ] Publish measurements as diagnostic evidence unless hardware,
     checkout, inputs, repetitions, synchronization, and acceptance thresholds
     are all matched and provenance-bound.

8. Complete documentation and independent review.
   - [ ] Generate `docs/jax_native_unit_test_coverage.md` from the validated
     manifest; do not hand-edit generated coverage counts.
   - [ ] Document how maintainers classify a new upstream native test and add
     the corresponding JAX evidence.
   - [ ] Document the difference between unit-capability parity, example
     workflow parity, hybrid coverage, and full native API replacement.
   - [ ] Record every approved `native_only` and `hybrid_boundary` decision with
     its technical reason and reviewer.
   - [ ] Run an independent requirements, numerical correctness, device
     placement, test-quality, and documentation audit.
   - [ ] Materialize the exact staged candidate in a clean checkout and rerun
     all completion gates before the final scoped commit.

## Validation Plan

- [ ] Inventory and manifest are deterministic:

  ```bash
  python scripts/jax_native_unit_coverage.py \
    --check \
    --upstream-ref <APPROVED_UPSTREAM_SHA>
  python -m pytest -q tests/jax/test_native_unit_coverage_manifest.py
  ```

- [ ] Every approved upstream test definition is mapped exactly once at the
  ledger level, with no orphan capability or JAX test references.
- [ ] Native baseline collection succeeds in the authority environment:

  ```bash
  python -m pytest --collect-only -q \
    tests/core tests/field tests/geo tests/mhd \
    tests/objectives tests/solve tests/configs tests/util
  ```

- [ ] JAX CPU parity passes in FP64:

  ```bash
  JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
    python -m pytest -q -m jax_native_unit_parity
  ```

- [ ] Strict JAX GPU parity passes on a real GPU with no implicit transfers:

  ```bash
  JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda \
  SIMSOPT_JAX_TRANSFER_GUARD=disallow \
  JAX_TRANSFER_GUARD=disallow \
    python -m pytest -q -m "jax_native_unit_parity and jax_gpu_pure"
  ```

- [ ] Hybrid tests pass in their declared CPU/MPI/external-solver environment
  and never count as strict-GPU evidence.
- [ ] Existing example authority remains green:

  ```bash
  example_run_dir="$(python examples/jax/run_parity.py \
    --case all-applicable \
    --scale bounded \
    --lanes native-cpu,jax-cpu \
    --artifact-root .artifacts/jax-native-unit-regression)"
  python -m examples.jax.parity.audit \
    --run "${example_run_dir}" \
    --require-authoritative
  ```

  Strict-GPU example replay remains scheduled on the approved GPU runner.

- [ ] Representative native-CPU/JAX-GPU measurements include checkout, device,
  FP64, matched inputs, warmup policy, repetitions, synchronization, wall time,
  RSS, and VRAM provenance.
- [ ] The broad native and JAX suites pass without weakened tolerances or new
  unexplained skips.
- [ ] Pyright, Ruff, formatting, compilation, and diff integrity pass:

  ```bash
  python -m pyright
  python -m ruff check \
    src/simsopt_jax src/simsopt_jax_adapters \
    tests/jax scripts/jax_native_unit_coverage.py
  python -m ruff format --check \
    src/simsopt_jax src/simsopt_jax_adapters \
    tests/jax scripts/jax_native_unit_coverage.py
  python -m compileall -q \
    src/simsopt_jax src/simsopt_jax_adapters \
    tests/jax scripts/jax_native_unit_coverage.py
  git diff --check
  ```

- [ ] A clean-checkout independent audit verifies the manifest, source hashes,
  test results, lane identity, device placement, precision, and published
  coverage totals.

## Risks and Mitigations

- **Risk: Raw test counts create a false completeness claim.**  
  **Mitigation:** require complete native-test traceability plus
  behavior-oriented capability records; counts alone never establish parity.

- **Risk: One native test contains both portable numerical behavior and
  host-only behavior.**  
  **Mitigation:** map the test to multiple capability records and certify each
  observable at its correct JAX-native or hybrid boundary.

- **Risk: Optional dependencies prevent collection and hide native tests.**  
  **Mitigation:** use AST inventory as the completeness owner and full
  authority-environment collection to bind runnable methods and runtime
  requirements; preserve material `subTest` regimes through source review.

- **Risk: `native_only` becomes a convenient way to avoid difficult work.**  
  **Mitigation:** require a concrete technical rationale, named reviewer,
  immutable decision record, and fail-closed validation.

- **Risk: JAX tests call native code or NumPy and appear to pass on GPU.**  
  **Mitigation:** require strict transfer guards, device assertions, import
  boundaries, callback bans, and separate hybrid labeling.

- **Risk: Native and JAX tests accidentally use different conventions or
  fixtures.**  
  **Mitigation:** bind immutable scientific inputs and conventions once,
  compare initial-state observables first, and include the input hash in
  evidence.

- **Risk: Tolerances are weakened until mismatched implementations pass.**  
  **Mitigation:** centralize tolerances, mutation-test the validator, require
  conditioning-based justification, and independently review any change.

- **Risk: Porting every native test detail produces a second, shallow copy of
  SIMSOPT.**  
  **Mitigation:** port public numerical behavior, reuse shared Python behavior,
  and keep external/host responsibilities in explicit adapters.

- **Risk: Full Jacobians, Hessians, or on-device solver loops exhaust memory.**  
  **Mitigation:** test matrix-free products where they are the public contract,
  add representative scaling checks, inspect compiler output for unintended
  materialization, and avoid whole-outer-loop JIT when it creates unacceptable
  memory pressure.

- **Risk: Upstream changes invalidate the classification during execution.**  
  **Mitigation:** pin the authority commit per campaign, add a drift report for
  the moving upstream branch, and rebase the ledger only through an explicit
  reviewed update.

- **Risk: The campaign expands indefinitely into non-JAX product areas.**  
  **Mitigation:** execute in domain waves, require explicit scope decisions,
  and close shared/hybrid/native-only records without inventing inappropriate
  device implementations.

## Completion Criteria

- [ ] The approved upstream commit and source-tree identity are immutable and
  recorded.
- [ ] One hundred percent of native test definitions are present in the ledger
  with valid capability mappings.
- [ ] No portable in-scope capability remains `jax_partial` or `jax_missing`.
- [ ] Every `jax_equivalent` capability has passing native CPU, JAX CPU, and
  strict JAX GPU evidence for all declared observables.
- [ ] Every `hybrid_boundary` capability has passing host-boundary and JAX-slice
  tests and is excluded from strict-GPU claims.
- [ ] Every `shared_python` and `native_only` record has a concrete, reviewed
  rationale.
- [ ] Values, derivatives, shapes, dtypes, symmetry conventions, edge cases,
  and failure semantics satisfy their predefined contracts without weakened
  tolerances.
- [ ] Representative performance-critical kernels have matched native-CPU and
  JAX-GPU wall-time and memory evidence; no unsupported speed claim is made.
- [ ] The broad regression suite, Pyright, Ruff, formatting, compile checks,
  generated-report check, and diff integrity all pass.
- [ ] RED → GREEN → REFACTOR receipts are replayable for work performed under
  this plan; missing historical RED evidence remains labeled honestly.
- [ ] The generated coverage report is provenance-bound, independently audited,
  and consistent with the machine-readable manifest.
- [ ] Existing example parity and GPU purity contracts remain green.
- [ ] Only plan-owned implementation files are included in scoped commits;
  unrelated working-tree content remains untouched.

## Open Questions

- Should the first authority baseline be current remote upstream commit
  `4ad6fd99189b99d9722ad33aaeb5d30adc81680f`? The recommendation is yes, after
  one final remote identity check immediately before inventory generation.
- Who approves `native_only` and `hybrid_boundary` dispositions? The
  recommendation is one JAX maintainer plus one native-domain maintainer.
- Should constrained optimization become a supported backend-neutral JAX
  product capability? Until maintainers approve its semantics and ownership,
  it remains an explicit architecture blocker rather than an implied port.
- Which GPU runner is the durable authority for scheduled strict-GPU coverage?
  The runner must expose immutable checkout, environment, CUDA/JAX versions,
  device identity, FP64 state, and retained artifacts.
