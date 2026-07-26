# JAX Examples Directory Implementation Plan

**Status:** Draft
**Last updated:** 2026-07-26

## Purpose

Create a discoverable, JAX-first examples collection under `examples/jax/`.
The existing native examples provide a coverage catalog and scientific
reference, but JAX examples may combine, split, rename, or reorganize those
workflows to teach immutable state, explicit transformations, batching, and
device execution clearly.

This is a selective, evidence-backed collection. It must not imply that every
native example has a JAX counterpart or that a workflow is fully device-native
when it still depends on mutable host objects, VMEC, SPEC, MPI, visualization,
or another unported boundary.

## Goals

- Give users a coherent JAX-first teaching sequence under the familiar
  simple/intermediate/advanced tiers without requiring source-example filenames
  or code structure to match.
- Account for every tracked Python example as either a JAX candidate or a
  deliberately deferred concept in one machine-readable inventory, and derive
  whether each candidate is planned or covered from its JAX-example links.
- Demonstrate the public JAX API at the correct abstraction level: pure
  `simsopt_jax` kernels for traceable workflows and `simsopt_jax_adapters` for
  legacy `Optimizable` workflows.
- Prove each implemented JAX example with a deterministic CPU smoke lane and
  independent correctness checks for the scientific observables it claims.
- Add strict GPU execution only for examples whose manifest contract requires
  GPU residency; GPU jobs must fail rather than silently skip or fall back.
- Reuse canonical example inputs instead of copying data files into the JAX
  tree.

## Non-Goals

- Blindly copy all 51 native Python scripts and change only their imports.
- Preserve native script layout, defaults, outputs, or iteration structure when
  a clearer JAX-first design calls for a different presentation.
- Reimplement VMEC, SPEC, QSC, MPI, plotting, or serialization solely to make
  the directory appear complete.
- Make `examples/` a dependency of `src/simsopt`, `src/simsopt_jax`, or
  `src/simsopt_jax_adapters`.
- Replace the existing native example runners or change native CPU behavior.
- Turn examples into benchmarks or use them to make speedup claims.
- Duplicate unit-level kernel parity tests already present under `tests/`.
- Create placeholder JAX scripts that raise `NotImplementedError`; deferred
  coverage belongs in the inventory until a runnable workflow exists.

## Current Context

- Live HEAD `4bd849023` has 51 tracked Python examples: 10 in `1_Simple`, 26
  in `2_Intermediate`, 6 in `3_Advanced`, and 9 in
  `stellarator_benchmarks`.
- No tracked `examples/jax/` directory or JAX-specific example runner exists.
- Existing CI invokes `examples/run_serial_examples`,
  `examples/run_parallel_examples`, and `examples/run_vmec_examples`; those
  runners execute the native examples and remain unchanged by default.
- The public port is split between pure code in `src/simsopt_jax` and host
  compatibility code in `src/simsopt_jax_adapters`.
- Existing tested application surfaces include traceable serial/MPI solve
  contracts, Biot-Savart and squared-flux adapters, curve objectives, QFM,
  Boozer surfaces/objectives, tracing, permanent-magnet solves, wireframe
  solves, VMEC frozen diagnostics, and single-stage helper paths.
- Full VMEC and SPEC execution remain host/external-code workflows. Their
  presence in a source example is not proof of an end-to-end JAX port.
- JAX runtime selection is process-wide and must occur before importing
  JAX-heavy modules, so the example runner must isolate examples in fresh
  subprocesses.
- The checkout already contains unrelated modified and untracked work. The
  implementation must preserve it and stage only the example-plan slice.

## JAX-First Example Contract

The selected design is *inspired by*, not structurally mirrored from, the
native examples:

1. Every JAX example lives beneath `examples/jax/` in the most appropriate
   simple, intermediate, advanced, or stellarator-benchmark tier.
2. A JAX example may combine several native examples, split one native example
   into focused lessons, or use a descriptive JAX-specific filename.
3. Every JAX example uses a public `simsopt_jax` or
   `simsopt_jax_adapters` implementation directly. A wrapper that merely
   launches a native script with a backend environment variable is invalid.
4. Every host/device boundary is named in the script and the manifest.
5. Native implementations remain independent oracles for shared scientific
   observables where applicable, but example-level output and control-flow
   identity are not requirements.

Each source example receives exactly one authored disposition:

- `candidate`: the source concept is suitable inspiration for one or more JAX
  lessons now or in a later wave.
- `deferred`: no JAX lesson is currently justified. The manifest records the
  exact missing public surface, external limitation, or lack of a meaningful
  JAX teaching objective and the evidence needed to reconsider it.

Coverage is derived, not authored: every candidate must appear in at least one
JAX record's `inspired_by`; it is `planned` when all such records are planned
and `covered` when at least one is ready. A deferred source has no JAX link.
This keeps delivery status separate from execution architecture.

Every JAX example independently declares an execution kind:

- `pure`: the demonstrated workflow has no host boundary, so
  `host_boundaries` is empty.
- `adapter`: supported host construction, snapshot, or reporting seams remain,
  and each seam is named in non-empty `host_boundaries`.
- `hybrid`: native or external computation/control remains in the demonstrated
  workflow, and each boundary is named in non-empty `host_boundaries`.

## Rationale

### Design-it-twice

**Option A — `examples/jax/<existing-tier>/<jax-topic>.py` (selected).**

- Keeps all executable demonstrations under the existing top-level
  `examples/` directory.
- Preserves the familiar simple/intermediate/advanced taxonomy.
- Allows the current native runners to remain untouched while a separate JAX
  runner owns process-level runtime selection.
- Allows filenames and lesson boundaries to follow JAX concepts rather than
  native implementation history.

**Option B — add `*_jax.py` beside every native script (rejected).**

- Makes side-by-side comparison immediate, but mixes runtime dependencies in
  existing CI discovery, clutters the established teaching sequence, and
  makes output/input ownership ambiguous.

**Option C — create a new top-level `jax/examples/` tree (rejected).**

- Gives JAX a strong top-level identity, but introduces a new repository
  namespace, separates examples from existing documentation and inputs, and
  increases path-resolution and packaging complexity without adding a useful
  abstraction.

### Teaching-shape alternatives

**JAX-first redesigns inspired by native examples (selected).** Examples teach
public JAX concepts in their natural form and declare which native workflows
inspired them. This can change filenames, split or combine lessons, and omit
host-only details.

**Scientific mirrors with implementation freedom (rejected).** This would
preserve each native script's scientific workflow and outputs while allowing
JAX-specific restructuring. It provides tighter comparison but constrains the
teaching sequence around legacy workflow boundaries.

**Source-shaped mirrors (rejected).** This would preserve filenames and code
organization as closely as possible. It makes diffs easy to read but risks
teaching host-oriented patterns as if they were idiomatic JAX.

### Information-hiding test

The manifest owns example classification, dependency requirements, lane
membership, and smoke arguments. The runner owns one typed mapping from those
lanes to existing backend-runtime selections and child commands. Backend mode,
precision, transfer-guard, and platform semantics remain owned by the existing
runtime-policy modules; neither manifest nor runner redefines them. Individual
scripts own only their scientific workflow and structured result. Changing CI
lane composition must therefore require editing the manifest/runner contract,
not coordinated edits across every script.

The native input directories remain the single source of truth for VMEC/SPEC
and fixture data. Changing an input must not require updating a copied JAX
asset.

### Design tier and API-evolution gate

The implementation is Tier 3: it adds modules across examples, tests, docs, and
CI and exposes a user-facing runner CLI plus a machine-readable child-result
contract. The design is therefore fixed before test-first implementation:

- Observable delta: a new opt-in JAX example tree and runner are added; native
  example commands, ordering, outputs, and runtime defaults do not change.
- Caller inventory: direct repository users, the JAX CPU and strict-GPU CI
  jobs, manifest tests, and integration tests.
- Migration path: none for existing callers. New users enter through the README
  and choose a lane; example authors add one manifest record and implement the
  documented child contract.
- Compatibility proof: tests own exact runner ordering/argv, manifest schema,
  structured-result fields, failure propagation, and lane runtime assertions.
- Deprecation: not applicable to the initial additive interface. A future
  incompatible schema or CLI change requires an explicit version/migration
  plan.
- Rollback: revert the JAX workflow steps, documentation links, and
  `examples/jax/` slice together; native runners and inputs remain independent.

## Proposed Layout

```text
examples/
  jax/
    README.md
    manifest.json
    run_examples.py
    1_Simple/
      <JAX-first examples>
    2_Intermediate/
      <JAX-first examples>
    3_Advanced/
      <JAX-first examples>
    stellarator_benchmarks/
      <JAX-first examples, only when runnable>
tests/
  test_jax_examples_manifest.py
  integration/
    test_jax_examples.py
```

Do not add `inputs/` copies beneath `examples/jax/`. Scripts resolve repository
owned inputs from their canonical source-example path. An example that writes
artifacts must accept a caller-owned `--output-dir` or use a temporary
directory; examples with no file output do not need that option.

## Manifest Contract

`examples/jax/manifest.json` is the single source of truth for source coverage
and JAX example execution. It has two typed collections.

Each of the 51 `source_catalog` records has:

- `source`: native path relative to `examples/`.
- `disposition`: `candidate` or `deferred`.
- `deferred_reason`: an exact missing capability, external limitation, or
  absent teaching objective plus a reconsideration criterion; required only
  for `deferred` records.

Each `jax_examples` record has:

- `id`: stable descriptive identifier independent of a filename.
- `path`: executable path relative to `examples/jax/`.
- `status`: `planned` or `ready`; `ready` requires the file and all declared
  validation to exist and pass.
- `tier`: `1_Simple`, `2_Intermediate`, `3_Advanced`, or
  `stellarator_benchmarks`.
- `inspired_by`: one or more paths from `source_catalog`.
- `execution_kind`: `pure`, `adapter`, or `hybrid`.
- `jax_surfaces`: public `simsopt_jax` or `simsopt_jax_adapters` symbols the
  example teaches.
- `host_boundaries`: explicit remaining native/external operations.
- `extras`: required project extras such as `JAX`, `JAX_GPU`, `MPI`, `SPEC`,
  `VIS`, or `ALGS`.
- `smoke_args`: deterministic bounded arguments used by the runner.
- `correctness_tests`: owning scientific-oracle and integration tests.
- `lanes`: `cpu-smoke` and, only where required, `gpu-strict`.

`inspired_by` is the only stored source-to-JAX relation. The validator derives
the inverse mapping and source coverage so two relationship ledgers cannot
drift. It also enforces the `execution_kind`/`host_boundaries` invariants and
requires runnable files, tests, and lanes only for `ready` records. This permits
the complete source catalog and planned queue to merge before the first
executable example without overstating coverage.

The runner invokes each ready record exactly as
`[sys.executable, path, "--smoke", "--json", *smoke_args]`. A successful child
emits one machine-readable result with at least `example_id`, `backend_mode`,
`platform`, `precision`, `status`, and `observables`. The JSON object is the
final stdout line; human-oriented output may precede it. This final record is
the runner's acceptance interface.

The README explains these fields but does not reproduce the 51-row status
table. This avoids two status ledgers drifting apart.

## Confirmed Decisions

- Directory: `examples/jax/`, outside `src/` and beneath the existing
  top-level `examples/` directory.
- Coverage: create files only for meaningful, runnable JAX examples; catalog
  unsupported native concepts as deferred without placeholder scripts.
- Teaching model: JAX-first redesigns inspired by native examples, not
  source-shaped or output-identical mirrors.
- Delivery method: each implementation slice follows test-first
  RED → GREEN → REFACTOR, with the failing and passing commands recorded.
- Current deliverable: finalize this plan only; do not create the manifest,
  examples, runner, tests, or CI changes yet.

## Assumptions

- Native implementations remain independent correctness references for shared
  scientific quantities, but are not the required teaching structure.
- The initial delivery can be incremental; the complete source catalog and
  planned queue are required before the first ready JAX example merges, but
  runnable coverage of all 51 examples is not.
- JAX examples may construct native host objects when the manifest labels them
  as `adapter` or `hybrid`, but compiled numerical regions may not read mutable
  host state or perform implicit NumPy conversion.
- Every JAX example is directly executable from the repository checkout
  and supports a bounded smoke mode without changing its normal pedagogical
  defaults.
- Correctness tests own tolerances. A RED test may not be made GREEN by
  weakening a scientific tolerance without separate evidence and review.

## Initial Readiness Assessment

This is an implementation queue, not a claim of current end-to-end closure.
Phase 1 must confirm every row against public imports and focused tests.

### Wave 1 — smallest representative vertical slices

- `1_Simple/traceable_least_squares.py`, inspired by
  `1_Simple/just_a_quadratic.py`: immutable traceable problem state and JAX
  serial solve.
- `1_Simple/curve_length_optimization.py`, inspired by
  `1_Simple/minimize_curve_length.py`: curve snapshot plus `CurveLengthJAX`.
- `1_Simple/surface_geometry_optimization.py`, inspired by
  `1_Simple/surf_vol_area.py`: pure surface geometry/objective evaluation and
  traceable optimization.
- `1_Simple/coil_flux_optimization.py`, inspired by
  `1_Simple/stage_two_optimization_minimal.py`: `BiotSavartJAX`,
  `SquaredFluxJAX`, JAX curve objectives, and explicit state boundaries.
- `1_Simple/qfm_surface_optimization.py`, inspired by `1_Simple/qfm.py`:
  `BiotSavartJAX` and `QfmSurfaceJAX`/QFM solve path.
- `1_Simple/permanent_magnet_optimization.py`, inspired by
  `1_Simple/permanent_magnet_simple.py`: `PermanentMagnetGridJAX` and the public
  JAX permanent-magnet solver.
- `1_Simple/fieldline_tracing.py`, inspired by the QA and NCSX tracing scripts:
  a JAX field class plus the tracing adapter, with plotting outside the smoke
  result.

`1_Simple/logger_example.py` is expected to remain deferred unless it gains a
JAX-specific teaching purpose; logging alone is not a ported numerical surface.

### Wave 2 — composed field, Boozer, wireframe, and force workflows

- `2_Intermediate/boozer.py` and `boozerQA.py`.
- `2_Intermediate/stage_two_optimization.py` and
  `stage_two_optimization_planar_coils.py`.
- `2_Intermediate/tracing_boozer.py`.
- `2_Intermediate/wireframe_rcls_basic.py` and
  `wireframe_gsco_modular.py`.
- `3_Advanced/coil_forces.py`,
  `stage_two_optimization_finitebuild.py`, and
  `wireframe_gsco_multistep.py`.

### Wave 3 — large or mixed-boundary workflows

- NCSX field-line and particle tracing.
- Advanced permanent-magnet MUSE, PM4Stell, and QA workflows.
- Stochastic Stage-2 optimization.
- Single-stage vacuum optimization.

These examples require bounded fixtures and explicit transfer/compilation
receipts so normal pull-request CI does not become an hours-long campaign.

### Wave 4 — external-code and MPI inventory

- VMEC-, Boozer-VMEC-, SPEC-, QSC-, finite-beta-, and MPI-driven examples.
- All nine `stellarator_benchmarks` scripts.

Promote one of these only after identifying a meaningful JAX-owned region and
documenting the host/external boundary. Frozen VMEC diagnostic kernels do not
by themselves make a VMEC optimization example JAX-native.

## TDD Execution Contract

After the manifest, runner, and result interfaces above are reviewed, every
bounded implementation slice follows the same evidence-producing cycle. TDD
validates the approved design; it is not used to discover the module boundary:

1. **RED:** Add the smallest public, behavioral test that states the missing
   contract. Run its exact focused command and record the expected failure and
   actual failure reason. The test must fail because the behavior is absent or
   wrong, not because of a syntax error, unavailable fixture, unconditional
   skip, or `xfail`.
2. **GREEN:** Implement only enough production behavior to satisfy that test.
   Re-run the identical focused command, then the focused regressions for the
   public surfaces touched, and record both passing receipts.
3. **REFACTOR:** Only while GREEN, remove duplication and improve ownership or
   naming without changing the observable contract. Re-run the focused and
   broader slice commands and keep the final GREEN receipt.

Tests assert observable outputs, errors, process status, and device/runtime
facts. They do not mirror private implementation structure or prove behavior
only through mocks. When a new file does not yet exist, RED still exercises the
intended public entry point or validator behavior so the failure is meaningful.
No implementation task starts before its RED test is reviewed.

## Implementation Plan

1. Freeze the source catalog and manifest invariants.
   - [ ] **RED:** In `tests/test_jax_examples_manifest.py`, first assert the
     exact one-to-one set match between tracked native Python examples and the
     51 `source_catalog` paths. Add failing cases for duplicate/missing paths,
     invalid dispositions, invalid inspiration paths, stored inverse links,
     unlinked candidates, linked deferred sources, and incorrect
     deferred-reason presence.
   - [ ] **RED:** Add table-driven failures for the cross-field contract:
     `pure` requires no host boundary; `adapter` and `hybrid` require at least
     one named boundary; a ready record requires an executable file,
     correctness owners, and `cpu-smoke`; a deferred source has no link.
   - [ ] **GREEN:** Add the typed manifest parser and complete 51-row catalog.
     Record planned JAX lessons, public JAX symbols, owning tests, lane
     membership, and boundaries only where live public imports support them;
     symbol-name coincidence is not evidence.
   - [ ] **GREEN:** Derive the source-to-JAX inverse and `planned`/`covered`
     coverage in the validator. Do not store either derived value.
   - [ ] **REFACTOR:** Centralize enum and relationship validation, rerun the
     manifest tests, and manually review all 51 rows without changing the
     observed schema.

2. Build the subprocess runner test-first.
   - [ ] **RED:** In `tests/integration/test_jax_examples.py`, assert stable
     manifest order and the exact child argv
     `[sys.executable, path, "--smoke", "--json", *smoke_args]`.
   - [ ] **RED:** Inject a child that exits nonzero and require the runner to
     exit nonzero while reporting its ID, inspiration paths, exact command,
     stdout, and stderr.
   - [ ] **RED:** Add child-result cases for malformed JSON, a skip/unsupported
     sentinel, CPU fallback, wrong precision, and wrong backend. The strict GPU
     lane must reject every one rather than count it as a pass.
   - [ ] **GREEN:** Implement typed manifest loading, deterministic lane
     selection, one fresh subprocess per ready example, structured-result
     parsing, and fail-closed status propagation in
     `examples/jax/run_examples.py`.
   - [ ] **GREEN:** Map `cpu-smoke` to the existing `jax_cpu_parity` runtime and
     `gpu-strict` to the existing `jax_gpu_parity` runtime. Validate the child's
     resolved result instead of reimplementing runtime-policy semantics.
   - [ ] **GREEN:** For `gpu-strict`, set the existing CI contract before child
     startup: `SIMSOPT_BACKEND_MODE=jax_gpu_parity`,
     `SIMSOPT_BACKEND_STRICT=1`,
     `SIMSOPT_JAX_TRANSFER_GUARD=disallow`,
     `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`,
     `JAX_PLATFORMS=cuda`, `JAX_ENABLE_X64=1`, and
     `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
   - [ ] **REFACTOR:** Keep the lane-to-runtime mapping in one typed location,
     remove test-only duplication, and rerun manifest plus runner tests.
   - [ ] Document the author contract, install extras, runtime selection,
     result schema, and runner commands in `examples/jax/README.md`.

3. Deliver Wave 1 as two test-first vertical slices, then repeat the pattern.
   - [ ] **RED:** For `traceable_least_squares.py`, assert analytic optimum,
     objective, solver status, structured smoke result, and `cpu-smoke`
     execution before adding the script.
   - [ ] **GREEN:** Implement the smallest immutable, traceable example using
     the public JAX serial solver; make the focused correctness and runner tests
     pass.
   - [ ] **REFACTOR:** Improve the teaching flow while preserving observables,
     then rerun manifest, focused solver, correctness, and runner tests.
   - [ ] **RED:** For `coil_flux_optimization.py`, add independent value and
     gradient oracles, explicit adapter-boundary assertions, CPU result checks,
     and strict-GPU result checks before adding the script.
   - [ ] **GREEN:** Implement the public `BiotSavartJAX`, `SquaredFluxJAX`, and
     curve-objective workflow with immutable snapshots and no implicit transfer
     inside the compiled objective.
   - [ ] **REFACTOR:** Extract a helper only when the two completed examples
     demonstrate the same stable policy, then rerun both vertical slices.
   - [ ] Repeat RED → GREEN → REFACTOR for each remaining Wave 1 lesson.
     Each RED test names its independent observable oracle and documents any
     intentional pedagogical-default difference from the inspiration source.

4. Add later waves one observable contract at a time.
   - [ ] For Boozer lessons, RED-test value, derivatives, solver certificate,
     and final surface state before implementing `BoozerSurfaceJAX` workflows.
   - [ ] For Stage-2 lessons, RED-test explicit curve/current/surface snapshot
     ownership and absence of transfers inside the compiled objective.
   - [ ] For wireframe lessons, RED-test both current solution and field/error
     observables before implementing the public JAX solve-adapter workflow.
   - [ ] For tracing, RED-test deterministic reduced populations using
     termination status, event counts, and final states rather than complete
     adaptive trajectories.
   - [ ] For permanent magnets and single-stage workflows, RED-test objective,
     feasibility, solver status, and representative derivative paths against
     independent oracles before implementation.
   - [ ] Apply the full GREEN and REFACTOR gates from the TDD Execution Contract
     to every lesson; never batch several untested examples into one step.
   - [ ] Add force, finite-build, VMEC, SPEC, QSC, or MPI lessons only after
     every taught objective has a public JAX surface and the named external
     boundary has a failing behavioral test.

5. Make optional performance diagnostics technically valid.
   - [ ] RED-test the timing harness with an asynchronous stand-in whose result
     is incomplete until an explicit readiness barrier runs.
   - [ ] GREEN-separate first-call compile time from warmed repeated execution
     time and apply `block_until_ready` (or an equivalent JAX-tree barrier)
     before stopping each timer.
   - [ ] REFACTOR shared timing code only after two examples require it.
     Compile time, steady-state time, transfer count, and peak device memory
     remain diagnostics, never correctness or speedup claims without a separate
     approved performance contract.

6. Integrate documentation and CI through tested reachability.
   - [ ] **RED:** In `tests/test_jax_examples_manifest.py`, add a static workflow
     test that fails while both `push` and `pull_request` path filters in
     `.github/workflows/jax_smoke.yml` omit `examples/jax/**`, and that fails
     while the expected CPU/GPU runner steps are absent.
   - [ ] **GREEN:** Add `examples/jax/**` to both path filters, add `cpu-smoke`
     to the existing `jax-public-integration` job that installs `.[JAX]`, and
     add `gpu-strict` to the existing strict GPU job. Do not add JAX examples to
     the generic native-example workflow.
   - [ ] **GREEN:** Link `examples/jax/README.md` from `examples/README.md` and
     `docs/source/jax_migration.rst`; keep expensive external/MPI/full-size
     workflows in explicit scheduled or manual lanes.
   - [ ] **REFACTOR:** Keep lane membership in the manifest and runtime mapping
     in the runner so CI YAML contains commands, not a second script list.
   - [ ] Extend cleanup/ignore behavior only for JAX-owned artifact directories;
     never broaden deletion targets to canonical inputs.

7. Complete the API, design, and scope gates while GREEN.
   - [ ] Inventory every new CLI option and observable output contract. Require
     `--output-dir` only for examples that write artifacts.
   - [ ] Confirm no `src/` module imports from `examples` or the manifest and no
     example imports/forwards to a native example module.
   - [ ] Confirm no new direct dependency was introduced; if one is necessary,
     run the dependency-review gate and obtain user acknowledgement.
   - [ ] Re-read every opening comment against its actual pure, adapter, or
     hybrid behavior and named host boundaries.
   - [ ] Scan the path-scoped final diff for copied inputs, forwarding wrappers,
     implicit fallbacks, mutable state captured by JIT, stale source links, and
     unrelated dirty-tree files.

## Validation Plan

- [ ] For every implementation slice, preserve the RED receipt showing the
  intended behavioral failure, followed by the identical GREEN command and the
  post-REFACTOR focused/broader passing receipts.
- [ ] `python -m compileall -q examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py`
- [ ] `ruff check examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py`
- [ ] `ruff format --check examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py`
- [ ] `python -m pytest -q tests/test_jax_examples_manifest.py`
- [ ] `python -m pytest -q tests/integration/test_jax_examples.py -m 'not slow'`
- [ ] `python examples/jax/run_examples.py --lane cpu-smoke`
- [ ] Run each JAX example in a fresh process under
  `SIMSOPT_BACKEND_MODE=jax_cpu_parity` with FP64 enabled.
- [ ] On the designated CUDA environment, run
  `python examples/jax/run_examples.py --lane gpu-strict` and require every
  selected child to report `backend_mode="jax_gpu_parity"`, `platform="gpu"`,
  `precision="fp64"`, and `status="ok"`, with no skip, unsupported, or fallback
  outcome.
- [ ] For each JAX example, verify its declared scientific observables and
  derivatives against an independent analytic, finite-difference, or native CPU
  oracle using tolerances owned by the correctness test, not hard-coded
  independently in the script.
- [ ] Run existing focused tests for every public JAX surface used by the
  examples, including serial/MPI solve, flux, curve objectives, QFM, Boozer,
  tracing, permanent magnets, wireframe, finite-build, force, and single-stage
  paths as applicable.
- [ ] Re-run the unchanged native example runner(s) touched by shared-input or
  documentation changes to prove native behavior remains intact.
- [ ] `! rg -n '(from|import) examples|examples\.' src/simsopt src/simsopt_jax src/simsopt_jax_adapters`
- [ ] `git diff --check -- examples/jax examples/README.md docs/source/jax_migration.rst .github/workflows/jax_smoke.yml tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py`
- [ ] `git diff --cached --check` after staging exactly the intended slice.
- [ ] Review `git status --short` and stage only the intended JAX-examples slice.

## Risks and Mitigations

- Risk: A JAX-first redesign loses the scientific lineage that motivated it.
  Mitigation: Keep explicit `inspired_by` links and correctness contracts in the
  manifest while allowing code structure and user-facing outputs to differ.
- Risk: “JAX example” is mistaken for “fully device-native.”
  Mitigation: Derive delivery coverage separately from execution kind and name
  every host boundary in both the script and manifest.
- Risk: Backend configuration occurs after JAX initialization.
  Mitigation: Execute every example in a fresh subprocess whose environment is
  set by the runner before import.
- Risk: CI cost grows with compilation and external solvers.
  Mitigation: Maintain bounded smoke inputs, tier lanes by cost, and keep full
  runs scheduled/manual.
- Risk: Input or output files are duplicated or overwritten.
  Mitigation: Reuse canonical native inputs read-only and write to isolated
  caller-owned output directories.
- Risk: A GPU job passes through CPU fallback or skip behavior.
  Mitigation: Validate the structured child result against backend, platform,
  precision, and status; make fallback, skip, unsupported, malformed output,
  and nonzero exit lane failures.
- Risk: Asynchronous JAX dispatch makes timing diagnostics falsely optimistic.
  Mitigation: Separate cold compile from warmed execution and wait for every
  measured result before stopping the timer.
- Risk: Tests are added after implementation and merely confirm its structure.
  Mitigation: Require reviewed behavioral RED evidence before each production
  slice, then retain identical-command GREEN and post-REFACTOR receipts.
- Risk: The manifest and README become competing ledgers.
  Mitigation: Keep per-example state only in the manifest; README documents the
  schema and workflow.
- Risk: Existing dirty work is accidentally included.
  Mitigation: Use path-scoped status/diff checks and stage only files created or
  intentionally edited for this plan.

## Completion Criteria

- [ ] All 51 tracked native Python examples have exactly one valid manifest
  record.
- [ ] Every candidate source derives as `planned` or `covered`; each covered
  source links through `inspired_by` to at least one ready, executable JAX
  example with public JAX surfaces, correctness owners, and required lanes.
- [ ] Every deferred source names a concrete missing boundary, external
  limitation, or absent teaching objective plus a reconsideration criterion;
  it has no JAX link and no placeholder script.
- [ ] Every implementation slice has an authentic behavioral RED receipt,
  identical-command GREEN receipt, and post-REFACTOR focused/broader receipt.
- [ ] Wave 1 JAX examples pass CPU smoke and declared correctness tests.
- [ ] GPU-required Wave 1 examples pass strict CUDA execution with no fallback or
  skips.
- [ ] Native example scripts, runners, and inputs retain their prior behavior.
- [ ] Documentation tells users when to choose pure, adapter, or hybrid
  examples and how to run them.
- [ ] No source package imports example code or manifest data.
- [ ] Both JAX workflow path filters reach `examples/jax/**`; the existing JAX
  CPU and strict-GPU jobs invoke the corresponding runner lanes.
- [ ] The path-scoped final diff passes formatting, static, focused test,
  manifest, and dirty-tree scope checks.

## Open Questions

- None for plan finalization. Implementation remains intentionally unauthorized
  by the current plan-only decision.
