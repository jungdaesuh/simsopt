# One-to-One JAX Example Mirrors and Native Parity Implementation Plan

**Status:** Bounded implementation complete; scheduled native-default authority
remains `not_run`
**Last updated:** 2026-07-29

## Purpose

Replace the current many-to-one, `inspired_by` JAX-example coverage model with
an executable one-to-one mirror contract. Every native example whose scientific
workflow is supported by the existing JAX implementation must have a
recognizable `examples/jax/<tier>/<same-name>.py` counterpart that teaches the
public JAX API and is checked against the corresponding native
SIMSOPT/`simsoptpp` workflow.

This plan also defines two scientifically distinct single-stage examples:

- a VMEC-free, implicit-Boozer vacuum workflow that can run through the JAX
  CPU/GPU path; and
- a VMEC-hybrid workflow in which VMEC remains an explicit CPU/MPI host solve
  while the JAX-owned coil calculations run on the selected JAX device.

The plan extends, but does not retroactively broaden, the completed bounded
evidence in `docs/jax_native_example_end_to_end_parity_implementation_plan.md`.
Existing receipts remain evidence for their exact recorded cases only.

## Bounded implementation closure

The production implementation is complete for the bounded delivery at
`11340c829690fdc0652e47588f5da549829c056a`:

- manifest schema v3 contains 36 ready examples and a 52-row native-source
  catalog; parity schema v2 contains 26 full relationships and one explicitly
  unsupported relationship;
- all 25 external-solver-free candidate sources have exact-path JAX mirrors,
  and the VMEC-free Boozer/vacuum single-stage workflow is the 26th full parity
  case;
- ordinary JAX execution defaults to fast intent, while parity intent remains
  explicit and FP64;
- the exact-source bounded authority run
  `20260729T005942Z-5ade9aee` passed its independent fail-closed audit with 26
  cases, 78 native-CPU/JAX-CPU/strict-JAX-GPU lane receipts, and 1,248
  comparisons;
- the full authority run completed in 29 minutes 21 seconds (run identifier
  timestamp to completion marker) with a largest parent-observed child peak
  RSS (VmHWM, excluding descendants) of 2,292,080,640 bytes and a largest
  child-reported (`ru_maxrss`) host peak RSS of 2,219,270,144 bytes. The largest measured GPU allocation was
  737,743,616 bytes in particle tracing; no lane exhausted host or device
  memory;
- the separate VMEC-hybrid example passed locally with VMEC on CPU and the JAX
  slice on both CPU and strict GPU. Both lanes completed four VMEC evaluations
  without a VMEC failure; their final objectives differed by approximately
  `4.4e-11`. This is local hybrid evidence recorded from the example's JSON
  stdout without a persisted machine-readable receipt, and it is not a claim
  that VMEC ran on GPU;
- all ready examples passed the isolated strict-transfer execution suite.
  Pyright 1.1.411 reported zero errors and warnings, Ruff lint and format
  checks passed, and compile and diff checks passed.

Performance evidence is deliberately lean. The authority run retains per-lane
wall time, per-process peak RSS (parent-observed child VmHWM and
child-reported `ru_maxrss`), and GPU peak allocation where the backend
exposes it. Representative maxima and the single-stage hybrid measurements are
enough to establish bounded completion without adding a benchmark framework or
claiming a speedup.

Two limitations remain visible rather than being converted into claims:

1. Native-default-scale scheduled authority remains `not_run`; the bounded
   results do not certify native-default workloads.
2. The machine-readable log contains 32 authentic, structurally valid TDD
   behaviors. Historical fixes without preserved failing revisions remain
   post-hoc and are not relabeled as RED -> GREEN -> REFACTOR receipts.

## Goals

- Give every eligible native example exactly one discoverable JAX mirror at the
  same tier and filename.
- Make each mirror an executable lesson in the public `simsopt_jax` or
  `simsopt_jax_adapters` API rather than a parity-harness-only fixture.
- Default ordinary JAX CPU/GPU example execution to fast intent while retaining
  explicit FP64 parity intent.
- Compare identical-input native CPU, JAX CPU, and strict JAX GPU outcomes at
  the initial state and after optimization or integration.
- Measure numerical precision for every mirror, and retain bounded wall-time
  and peak-memory sanity evidence without turning each example into a bespoke
  benchmark.
- Maintain bounded deterministic parity in normal CI and native-default-scale
  parity in scheduled/manual authority runs.
- Preserve an explicit host/device contract for adapter examples and the
  VMEC-hybrid workflow.
- Require authentic per-mirror RED -> GREEN -> REFACTOR evidence before a
  mirror can be marked ready.
- Keep tutorials useful without allowing one tutorial to count as several
  native ports.

## Non-Goals

- Port VMEC, SPEC, QSC, or MPI itself to JAX in this delivery.
- Claim that native VMEC executes on a GPU in the VMEC-hybrid example.
- Differentiate through native VMEC with `jax.grad` or hide VMEC behind a JAX
  host callback.
- Treat frozen VMEC output diagnostics as a boundary-to-equilibrium JAX solve.
- Restore obsolete historical `*_jax.py` scripts whose APIs no longer match the
  supported public port.
- Require iteration counts or evaluation counts from different solvers to be
  numerically identical.
- Make fast-mode results certification evidence, require a speedup to call a
  mirror complete, or make an unmeasured performance claim.
- Count logging-only, external-package-owned, or scientifically different
  workflows as JAX mirrors.

## Current Context

### Historical repository context

- `examples/jax/manifest.json` currently contains 10 ready JAX lessons and one
  planned single-stage lesson.
- The 10 ready lessons have passed the existing CPU/GPU x fast/parity execution
  matrix.
- `examples/jax/parity_manifest.json` currently classifies 28 ready-example
  relationships as 2 `full`, 6 `reduced`, and 20 `unsupported`.
- The current bounded baseline is local-only evidence from clean executable
  revision `3b401b54bf4b12d1a35b70dd9621080ca9620ff6`, run ID
  `20260727T141144Z-cb97f4d1`, under
  `.artifacts/jax-example-parity/`. Its required-authority audit passed 8 cases,
  24 lane receipts, and 252 comparisons across native CPU, JAX CPU, and strict
  RTX 5090 JAX GPU in FP64. The summary SHA-256 is
  `f317f5c69fcb52a83ebb598df897273160537ecae421f6a15c1db1c2574b0819`
  and the isolated generated-report SHA-256 is
  `297d13deaa6f1500c966aed2795bca8725d76c1fa8235378189e5d885875d43f`.
  This does not certify later mirror implementations; Phase 1 must verify these
  identities before reusing the baseline.
- Current combined lessons cite several native sources through `inspired_by`.
  For example, `coil_flux_optimization.py` cites four stage-two sources but has
  only reduced parity with `stage_two_optimization_minimal.py`; the other three
  relationships are explicitly unsupported.
- The current port provides JAX evaluation of frozen VMEC spline state,
  geometry, and fieldline diagnostics. It does not provide a JAX VMEC
  equilibrium solve or a differentiable boundary-to-equilibrium map.
- Native `examples/3_Advanced/single_stage_optimization.py` invokes VMEC inside
  the joint boundary/coil optimization. The autoresearch banana single-stage
  workflow instead performs an implicit `BoozerSurface` solve from the coil
  field and uses a precomputed `wout` only as seed/reference data.
- The worktree already contains unrelated modified and untracked files. Every
  implementation commit must remain scoped and preserve them.
- Execution through `8b8965fce23175724560ab67391b11bcc7ab60c2` has implemented
  all eight Wave-A source mirrors with replayable per-source TDD receipts.
  The focused Wave-A CPU matrix and strict RTX 5090 strain lane pass, but
  schema activation, the complete ready-mirror CPU/GPU matrix, and lean
  representative timing/RSS/VRAM checks remain later gates.
  Validation timings have not been promoted into performance claims.
- Standard `2_Intermediate/stage_two_optimization.py` is the first Wave-B
  source to complete source-owned exact CPU parity, a batched public workflow
  boundary, explicit strict-transfer-safe device parameter placement, and
  immutable RED -> GREEN -> REFACTOR receipt replay. Strict RTX 5090 FP64
  execution passed with `JAX_TRANSFER_GUARD=disallow`. Reusing one compiled
  parametric problem across the two source stages reduced the bounded cold CPU
  run from 19.23 s and 1,030,840 KiB peak RSS to 16.90 s and 1,019,196 KiB.
  These are focused development measurements, not a speedup claim.
- Planar `2_Intermediate/stage_two_optimization_planar_coils.py` now has
  source-owned exact native/JAX parity, a reusable two-stage device workflow,
  bounded L-BFGS history, and strict-GPU topology diagnostics whose arrays are
  explicit JIT operands. Native CPU, JAX CPU, and strict RTX 5090 GPU exact
  lanes pass. Its four immutable RED -> GREEN -> REFACTOR receipts validate
  structurally; manifest promotion remains part of the paired schema
  activation.

### Target classifications

| Classification | Meaning | Device claim |
|---|---|---|
| `mirror` | One native source and one JAX example execute the same scientific stages with matched inputs | Full JAX CPU/GPU claim when all numerical stages are device-compatible |
| `adapter` | Native SIMSOPT objects are constructed on the host, then immutable state enters a JAX-owned numerical path | Full claim covers the declared JAX region; host setup is explicit |
| `hybrid` | An external/native solver remains part of the scientific loop and JAX owns a separate numerical slice | GPU claim applies only to the JAX slice |
| `tutorial` | A useful combined or introductory JAX lesson with no one-to-one native-source claim | Execution only; never mirror coverage |
| `blocked` | A faithful mirror cannot be built with the supported surfaces yet | No parity or device claim |
| `not_applicable` | The native source has no meaningful JAX numerical lesson | No port expected |

### Initial one-to-one candidate inventory

The first delivery investigates and targets 25 existing native sources whose
normal execution appears not to require an external scientific solver. Phase 1
must prove each candidate's dependency and public-JAX-surface eligibility before
it is promoted to `mirror` or `adapter`; the table is not itself proof that all
25 workflows are already implementable. A newly demonstrated capability gap
must remain visible as `blocked`, with evidence and user review, rather than
silently shrinking the inventory. The two currently full and six currently
reduced cases still need exact-path mirror identities; passing a renamed
combined lesson does not satisfy the new discoverability contract.

| Wave | Native source | Current evidence | Target |
|---|---|---|---|
| A | `1_Simple/just_a_quadratic.py` | full, bounded | exact-name mirror |
| A | `1_Simple/minimize_curve_length.py` | full, bounded | exact-name mirror |
| A | `1_Simple/surf_vol_area.py` | reduced, bounded | full mirror |
| A | `1_Simple/stage_two_optimization_minimal.py` | reduced, bounded | full mirror |
| A | `1_Simple/qfm.py` | reduced, bounded | full mirror |
| A | `1_Simple/permanent_magnet_simple.py` | reduced, bounded | full mirror |
| A | `2_Intermediate/wireframe_rcls_basic.py` | reduced, bounded | full mirror |
| A | `2_Intermediate/strain_optimization.py` | reduced, bounded | full mirror |
| B | `2_Intermediate/stage_two_optimization.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/stage_two_optimization_planar_coils.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/stage_two_optimization_stochastic.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/permanent_magnet_MUSE.py` | unsupported relationship | full mirror; VMEC post-check remains off |
| B | `2_Intermediate/permanent_magnet_PM4Stell.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/permanent_magnet_QA.py` | unsupported relationship | full mirror; VMEC post-check remains off |
| B | `1_Simple/tracing_fieldlines_NCSX.py` | scientifically different combined lesson | full mirror |
| B | `1_Simple/tracing_fieldlines_QA.py` | unsupported relationship | full mirror |
| B | `1_Simple/tracing_particle.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/boozer.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/boozerQA.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/wireframe_gsco_modular.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/wireframe_gsco_sector_saddle.py` | unsupported relationship | full mirror |
| B | `2_Intermediate/wireframe_rcls_with_ports.py` | unsupported relationship | full mirror |
| B | `3_Advanced/wireframe_gsco_multistep.py` | unsupported relationship | full mirror |
| B | `3_Advanced/coil_forces.py` | unsupported relationship | full mirror |
| B | `3_Advanced/stage_two_optimization_finitebuild.py` | unsupported relationship | full mirror |

The following remain outside the 25-source external-solver-free target:

- `QH_fixed_resolution_boozer.py`, `tracing_boozer.py`, and
  `resolution_increase_boozer.py` invoke VMEC in their current workflows.
- `QSC.py` is owned primarily by the external `qsc` package.
- `boozerQA_ls_mpi.py` requires a separate MPI execution and result contract.
- `logger_example.py` has no JAX-specific numerical teaching objective.
- VMEC/SPEC equilibrium examples remain hybrid or blocked unless a bounded,
  truthful JAX-owned region is identified.

## Rationale

The current combined-lesson design is effective for teaching broad JAX
capabilities, but it cannot answer whether a particular native workflow has
been ported. A one-to-one identity makes discoverability, scientific lineage,
test ownership, and parity status obvious from the path alone.

The implementation must keep three concerns separate:

1. The native-source catalog decides whether a source is eligible, hybrid,
   blocked, or not applicable.
2. The example registry decides what user-facing JAX programs can run and on
   which declared devices.
3. The parity registry decides which matched comparisons certify a particular
   native/JAX pair.

These concerns may live in the existing manifest files, but each fact must have
one owner. Native-source eligibility must not be copied into each runner, and
device/profile policy must continue to come from the central runtime resolver.

The versioned ownership boundary is:

- `examples/jax/manifest.json` schema v3 migrates all 51 current native catalog
  rows and executable JAX records, then grows to 52 rows when the new
  Boozer/vacuum native example is added. A native row owns its source path,
  disposition, reason/blocker, reconsideration condition, dependencies, and at
  most one stable mirror example ID. An executable record alone owns its
  physical JAX path, classification, public surfaces, host/device scope, and
  readiness.
  Exact same-tier/same-filename identity is derived and validated from the
  native source path plus that referenced executable path; it is never stored
  as a second catalog path.
- `examples/jax/parity_manifest.json` schema v2 owns scientific applicability,
  workflow stage coverage, observable routes, scale, oracle kind, and tolerance
  references. Each coverage relationship references one native catalog source
  and one executable example ID; it does not own either path or disposition.
- New parity receipts, summaries, completion markers, and reports use artifact
  schema v2 and bind both manifest schema versions and hashes.
  Historical artifact-v1 readers remain read-only; new writers emit only the
  new artifact version, and the auditor rejects cross-version mixtures.

### Design-it-twice decision

Two designs were considered:

1. Keep `inspired_by` as the coverage contract and add more relationship rows.
   This avoids a successor schema but continues to let one tutorial appear to
   cover several workflows and leaves readiness unrelated to one-to-one parity.
2. Promote the native-source catalog to the authoritative mirror inventory,
   classify executable records as mirrors/adapters/hybrids/tutorials, and make
   parity cases reference exactly one eligible source and one executable mirror.

Use design 2. It makes mirror identity explicit, keeps tutorials honest, and
allows blocked sources to remain visible without masquerading as executable
coverage.

### Information-hiding test

- Changing fast/parity device policy must require edits only in the central
  runtime resolver and its tests, not in every example.
- Changing native-source eligibility must require editing only the catalog row
  and its validation evidence, not example runners and CI matrices.
- Changing parity tolerances must require editing only the existing central
  tolerance owner, not individual examples.
- Changing a scientific input must change one typed input bundle whose hash is
  consumed by every lane.

### Design tier

This program is Tier 4 because it includes an integrity-sensitive manifest
schema migration and changes public example identity and evidence semantics.
Individual mirror implementation slices may be reviewed as Tier 3 only after
the Tier-4 schema and rollback contract is approved. Observable changes include
new paths, removal of multi-source coverage claims, new execution
classifications, and scoped GPU semantics for hybrid examples.

## Assumptions

- Native SIMSOPT/`simsoptpp` host construction is allowed when a JAX adapter
  snapshots immutable state and the boundary is documented.
- Precomputed input data such as FOCUS files or a `wout` seed is allowed when
  no external solver is executed by the example; provenance and checksums must
  remain visible.
- `permanent_magnet_MUSE.py` and `permanent_magnet_QA.py` are eligible with
  their optional `vmec_flag` post-check disabled, matching their current
  default execution.
- Fast remains the ordinary JAX CPU/GPU default and is non-certifying. Parity
  requires explicit intent.
- FP64 remains mandatory for scientific parity on both JAX CPU and GPU.
- Native-default-scale authority may require a scheduled runner and data not
  suitable for per-PR CI.
- A VMEC-hybrid example is valuable even though its entire workflow is not
  on-device, provided the host solve and JAX device slice are reported
  separately.
- No new third-party optimizer or equilibrium-solver dependency is required by
  this plan.

## Implementation Plan

Every behavior-changing slice below follows RED -> GREEN -> REFACTOR. Before
production code for a mirror is added, retain the failing command, exit status,
failure excerpt, pre-GREEN revision, and the exact behavior being proved. Each
mirror owns its own RED; a wave-level failure cannot stand in for 25 source
contracts. A test written after the behavior exists is regression coverage, not
an authentic RED. Performance measurements are taken only after GREEN
scientific correctness; a slow RED is never treated as a functional RED.
Each RED must fail at the intended missing or incorrect scientific behavior,
not because of an unavailable dependency, broken test fixture, or unrelated
exception; GREEN must run the unchanged RED test.

1. Freeze the baseline and publish the one-to-one inventory.
   - [ ] Record HEAD, worktree status, Python/JAX versions, CPU devices, CUDA
     device/runtime, `simsoptpp` identity, and current manifest hashes without
     modifying unrelated files.
   - [ ] Re-run the current manifest/parity schema tests and record the existing
     10-ready/1-planned and 2-full/6-reduced/20-unsupported baseline.
   - [ ] Add a checked-in inventory fixture containing every native source,
     current disposition, runtime dependencies, current public JAX surface
     coverage, a minimal executable capability probe, and recommended target
     classification.
   - [ ] Require an explicit reason and evidence pointer for every `blocked` or
     `not_applicable` source.
   - [ ] Store new TDD evidence in
     a versioned `docs/jax_examples_one_to_one_tdd_receipts.json` and generate
     `docs/jax_examples_one_to_one_tdd_receipts.md` from it; do not rewrite the
     older receipts as if they were one-to-one RED evidence.
   - [ ] Add a receipt validator that requires, per mirror and behavior slice,
     native source ID, mirror ID, RED revision/command/exit/failure excerpt,
     GREEN revision/command/exit, REFACTOR revision/command/exit, test and source
     hashes, timestamps, and monotonic revision ancestry. Readiness validation
     must consume the JSON receipt; prose alone cannot promote a mirror.
   - [ ] Make the receipt auditor materialize each bound revision in an isolated
     clean checkout and rerun every exact phase command. Require identical RED
     test bytes at RED, GREEN, and REFACTOR; verify recorded source/test hashes;
     match the RED exit and intended-failure discriminator; and require the
     unchanged test to exit zero at GREEN and REFACTOR. Schema-valid but stale,
     fabricated, unreplayable, or post-hoc receipts must fail readiness.

2. Migrate the manifest contracts to example schema v3 and parity schema v2
   without losing read compatibility.
   - [ ] **RED:** Add schema tests that reject a ready mirror with zero or more
     than one native source, duplicate mirror ownership, a tutorial counted as
     coverage, an eligible source without a mirror, and a hybrid record whose
     GPU scope is undeclared.
   - [ ] **RED:** Add migration tests proving the exact current example-schema-v2
     and parity-schema-v1 bytes map deterministically to the target schemas;
     reject mixed old/new manifests, unknown versions, partial conversion, and
     rollback that restores only one contract file.
   - [ ] Define typed classifications for `mirror`, `adapter`, `hybrid`,
     `tutorial`, `blocked`, and `not_applicable`; do not pass arbitrary strings
     through runner code.
   - [ ] Make the native-source catalog the sole owner of source eligibility,
     blocker/reconsideration condition, stable mirror example ID, dependencies,
     and port status; do not store the executable path there.
   - [ ] Make executable example records own path, teaching kind, public JAX
     surfaces, required extras, host boundaries, supported device scopes, and
     readiness.
   - [ ] Make parity records reference exactly one native catalog row and one
     mirror record; remove duplicate `inspired_by` coverage authority.
   - [ ] Check in an explicit 51-row v2-to-v3 disposition mapping before code
     migration. Map the 25 candidate rows individually; map
     `QH_fixed_resolution_boozer.py`, `resolution_increase_boozer.py`, and
     `tracing_boozer.py` to VMEC-blocked; map
     `single_stage_optimization.py` to hybrid; and preserve a reason for every
     remaining deferred/not-applicable source.
   - [ ] Retain read-only adapters for example schema v2, parity schema v1, and
     their historical artifact version for one documented deprecation interval.
     All writers emit only the new canonical versions.
   - [ ] Inventory every existing JAX path and `example_id`, parity reference,
     command, document link, and artifact consumer. Keep an old path/ID as a
     non-covering tutorial or explicit compatibility alias for the interval,
     with warning and removal tests; do not silently reuse an ID for a
     scientifically different exact-name mirror.
   - [ ] Generate a no-write migration candidate, publish its SHA-256 and
     semantic diff, and obtain explicit approval before activating it.
   - [ ] **GREEN:** Activate only the approved byte-identical candidates and
     preserve the prior manifests as tested rollback inputs.
   - [ ] Test one atomic rollback procedure, such as reverting the activation
     commit, that restores both manifests, activation-specific readers/tests,
     artifact observability, and compatibility behavior together;
     `git checkout -- <current file>` is not a post-activation rollback.

3. Establish the user-facing mirror module contract.
   - [ ] **RED:** Require every mirror path to equal
     `examples/jax/<native tier>/<native filename>` and reject filename aliases
     as mirror coverage.
   - [ ] Require every mirror to expose a small typed input/result boundary,
     deterministic JSON reporting, a `main()` entrypoint, and nonzero exit on
     scientific failure.
   - [ ] Require examples to consume public `simsopt_jax` or
     `simsopt_jax_adapters` entrypoints; private APIs require a separately
     approved public-surface task before the mirror can be ready.
   - [ ] For adapter mirrors, document host construction, immutable snapshot,
     JAX numerical region, accepted-state publication, and reporting boundary
     in the module docstring and manifest.
   - [ ] Prohibit imports from `tests/` and `benchmarks/` in user-facing
     examples. Move genuinely shared production contracts to an owned source
     module rather than importing validation code.
   - [ ] Keep native and JAX implementations independent enough to avoid a
     shared-bug oracle; share only typed inputs, immutable data, and comparison
     contracts.
   - [ ] Ensure imports do not select a backend or initialize JAX before the
     runner establishes the requested device profile.

4. Convert the eight existing applicable cases into exact-name mirrors.
   - [ ] For each Wave-A source, write and execute a source-owned RED proving
     the exact mirror is missing or fails at least one required scientific
     observable.
   - [ ] Migrate the existing implementation where it already performs the
     correct workflow; avoid pass-through wrapper files whose only purpose is
     satisfying a path assertion.
   - [ ] Expand each of the six reduced cases until all declared scientific
     stages of its bounded native workflow are executed.
   - [ ] Preserve useful combined lessons as `tutorial` records only when they
     teach a materially different composition; otherwise remove the duplicate
     during the same migration.
   - [ ] Add native CPU/JAX CPU/JAX GPU parity routes for every applicable
     initial and final observable.
   - [ ] **GREEN:** Require all eight exact-name mirrors to pass bounded CPU and
     strict-GPU parity before marking them ready.
   - [ ] **REFACTOR:** Remove obsolete aliases and duplicated input builders
     after compatibility tests prove no supported CLI path regresses.

5. Implement the external-solver-free Wave-B mirrors.
   - [ ] Implement separate stage-two mirrors for standard, planar-coil, and
     stochastic workflows; preserve their distinct parameterizations, random
     seed semantics, and optimizer contracts.
   - [ ] For the stochastic workflow, materialize the ordered FP64 in-sample
     and out-of-sample perturbation tensors with the native PCG64DXSM process
     into the canonical input bundle. Bind shape, dtype, byte order, seed,
     generator identity, and SHA-256, and make every lane consume those exact
     arrays; a shared scalar seed across NumPy and JAX PRNGs is insufficient.
   - [ ] Implement separate MUSE, PM4Stell, and QA permanent-magnet mirrors;
     checksum input data and keep optional VMEC postprocessing disabled and
     explicitly out of the parity claim.
   - [ ] Implement separate NCSX/QA fieldline and particle-tracing mirrors;
     compare the correct physical workflow rather than a generic Reiman-field
     lesson.
   - [ ] Implement separate `boozer.py` and `boozerQA.py` mirrors using the
     supported `BoozerSurfaceJAX` and residual surfaces without invoking VMEC.
   - [ ] Implement separate GSCO modular, GSCO sector-saddle, RCLS-with-ports,
     and GSCO-multistep wireframe mirrors; do not use one bounded RCLS case as
     evidence for GSCO.
   - [ ] Implement separate coil-force and finite-build mirrors with the same
     current, curve, frame, quadrature, and force/strain definitions as native.
   - [ ] For every source above, retain its own RED, smallest GREEN, and
     green-preserving REFACTOR receipt before changing its status to ready.
   - [ ] Reject wave-level readiness if even one source is still represented
     only by a combined tutorial.

6. Add the VMEC-free Boozer/vacuum single-stage pair.
   - [ ] Define a bounded native SIMSOPT reference workflow derived from the
     autoresearch implicit-Boozer formulation: coil DOFs -> `BiotSavart` ->
     `BoozerSurface` solve -> iota/volume/QS/residual objectives -> implicit
     derivative to coil DOFs.
   - [ ] Add the native reference at
     `examples/3_Advanced/single_stage_boozer_vacuum_optimization.py`; do not
     cite the VMEC-based upstream example as its lineage.
   - [ ] Add
     `examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py` using
     public JAX Boozer/session/solver APIs.
   - [ ] Add the new native source as catalog row 52 with its stable mirror ID,
     add its executable record, and add its parity-schema-v2 relationship before
     either example can be marked ready.
   - [ ] **RED:** Prove disagreement is detected independently for inner-solve
     status, iota, volume, QS/residual objective, coil gradient, and final coil
     parameters.
   - [ ] Run fast by default on selected CPU/GPU and explicit FP64 parity on
     both devices.
   - [ ] Verify the JAX path has no VMEC import, subprocess, host callback, or
     hidden SciPy optimizer.
   - [ ] Measure host RSS, device memory, cold compile/setup time, and warmed
     solve time against the matched native reference so the inner implicit
     solve is not accidentally differentiated by unrolling its entire iteration
     history.

7. Add the VMEC-hybrid single-stage mirror.
   - [ ] Add
     `examples/jax/3_Advanced/single_stage_optimization.py` as the exact mirror
     of native `examples/3_Advanced/single_stage_optimization.py`.
   - [ ] Keep VMEC execution on CPU/MPI with explicit working-directory,
     process, input, output, failure, and cleanup ownership.
   - [ ] Run coil field, quadratic flux, coil regularization, and supported
     mixed surface/coil derivatives through the JAX CPU/GPU adapter path.
   - [ ] Combine VMEC-owned and JAX-owned objective/gradient blocks in a
     host-controlled outer loop; do not wrap VMEC in `jax.pure_callback` or JIT
     the complete outer loop.
   - [ ] **RED:** Inject a stale VMEC output, failed VMEC solve, mismatched
     boundary hash, missing MPI result, CPU fallback inside the JAX GPU slice,
     and incorrect mixed derivative; require each to fail closed.
   - [ ] Record separate timings, devices, precision, evaluations, and transfer
     receipts for VMEC and JAX regions.
   - [ ] Bind VMEC finite-difference method and step sizes, failure threshold,
     MPI layout, and host outer-solver stopping configuration into the canonical
     input/configuration fingerprint before claiming controlled derivative or
     final-state parity.
   - [ ] Label GPU results `jax_slice_gpu`, never full-workflow GPU or on-device
     VMEC.
   - [ ] Compare the hybrid result against the original native workflow using
     identical boundary/coil inputs and VMEC configuration.
   - [ ] Keep a pure-JAX VMEC equilibrium solve as a separately scoped future
     project requiring its own design and scientific validation plan.

8. Extend fast/parity execution without duplicating policy.
   - [ ] Keep `run_examples.py --device cpu|gpu` mapped by the central resolver
     to fast intent when `--intent` is absent.
   - [ ] Keep explicit `--intent parity` mapped to the existing CPU/GPU parity
     profiles with FP64 and deterministic parity policy.
   - [ ] Add a typed execution-scope field so runners distinguish full-device,
     adapter-host-setup, and hybrid-JAX-slice execution.
   - [ ] Require strict GPU mirrors to report the selected CUDA device and fail
     on CPU fallback within their declared JAX region.
   - [ ] Allow declared host setup and VMEC regions only at named boundaries;
     never relax the transfer guard globally to make a hybrid pass.
   - [ ] Keep fast receipts non-certifying. Allow only statements directly
     supported by the matched measurement artifact; do not turn measurement
     into a default-promotion or universal-speedup claim.

9. Expand matched precision parity and artifact publication.
   - [ ] Give every mirror a typed canonical input bundle and record byte/hash
     identity across all required lanes.
   - [ ] Execute the native source-owned callable or approved source-adjacent
     native adapter; a reimplemented oracle that merely cites the source path is
     insufficient.
   - [ ] Compare applicable initial objective, residual, Jacobian, gradient,
     constraints, geometry/field invariants, and tracing initial state.
   - [ ] Compare applicable final parameters, objective/residual, feasibility,
     normalized convergence, trace endpoints/invariants, and raw status.
   - [ ] For every scalar, vector, and matrix observable, retain shape/dtype,
     maximum absolute error, maximum relative error with an explicit near-zero
     rule, and a scale-aware norm discrepancy; use the existing checked-in
     tolerance owner and never weaken tolerances to admit a new mirror.
   - [ ] Require both JAX parity profiles to pass the certification comparison.
     Apply the same scientific-success and discrepancy checks to fast profiles,
     but publish those results as non-certifying diagnostics.
   - [ ] Record iteration and evaluation counts as diagnostics with documented
     solver semantics; do not require equality when algorithms differ.
   - [ ] Add pairwise native CPU:JAX CPU, native CPU:JAX GPU, and JAX CPU:JAX
     GPU routes for every applicable observable.
   - [ ] Extend the existing atomic, no-replace, hash-bound artifact publication
     and independent audit machinery rather than introducing a second format.
   - [ ] Generate a results table that separates full, reduced, blocked,
     bounded, native-default, full-GPU, adapter, and hybrid evidence.

10. Keep performance and memory validation proportional to example risk.
    - [ ] Record one bounded native-CPU smoke and one bounded JAX CPU/strict-GPU
      smoke for each mirror, including wall time and peak host RSS. This is a
      regression sanity check, not a speedup study.
    - [ ] Synchronize JAX results before timing and retain the canonical input,
      scale, revision, backend mode, platform, FP64 policy, and solver counts.
    - [ ] For GPU smokes, retain peak process-attributed VRAM when the existing
      checked monitor can provide it. Never substitute host RSS for VRAM.
    - [ ] Run detailed cold/warm timing and allocation-sensitive VRAM checks
      once per representative workload class: optimization, tracing, Boozer,
      permanent magnets, wireframes, finite-build/forces, and hybrid execution.
      Re-run an individual mirror only when its smoke reveals a regression.
    - [ ] Keep native-default measurements scheduled/manual. Record an OOM as a
      failure and investigate its cause; do not build a large benchmark
      framework solely to repeat the same check across every example.
    - [ ] Report JAX CPU and native CPU only under matched host/resource policy.
      Label GPU-versus-CPU results as cross-device observations and make no
      speedup or commensurate RSS/VRAM claim.

11. Add bounded CI and native-default scheduled authority.
    - [ ] Materialize the exact candidate revision in a fresh checkout, build
      `simsopt`/`simsoptpp` from that checkout, require clean tracked and
      untracked inventories before and after execution, and bind the installed
      package/native-extension hashes to every authority receipt.
    - [ ] Run every ready mirror on JAX CPU fast and parity in normal CI with
      deterministic bounded inputs.
    - [ ] Make `.github/workflows/jax_smoke.yml` own required bounded CPU-fast,
      CPU-parity, GPU-fast, and risk-ranked strict-GPU-parity PR commands. Make
      `.github/workflows/jax_gpu_parity.yml` own the full strict-GPU and
      native-default scheduled/manual authority matrix. Static workflow tests
      must prove these trigger/command routes, explicit parity intent, zero-skip
      GPU behavior, and artifact eligibility.
    - [ ] Add `bounded` and `native_default` scale as typed manifest/parity
      properties; do not infer scale from command names.
    - [ ] Add and test the explicit `--scale bounded|native_default` selector to
      `run_examples.py`, `run_parity.py`, and the measurement collector, with
      selector conflicts and unsupported per-source scales rejected.
    - [ ] Preserve `bounded` as the no-`--scale` compatibility default, but make
      every CI and authority command explicit. Replace the current unconditional
      child `--smoke` emission with typed scale propagation through runner,
      child argv, canonical input builder, native adapter, result JSON, parity
      receipt, and measurement artifact; only bounded scale may map to smoke.
    - [ ] Add mutation tests proving the emitted child argv and effective input,
      receipt, and artifact scales agree for every runner/collector, and that a
      parser-only selector or a native-default request executing bounded inputs
      fails closed.
    - [ ] Run native-default-scale authority for each scientifically practical
      mirror on a scheduled/manual lane with immutable inputs and environment
      receipts.
    - [ ] Calibrate per-source runtime and storage from non-promotional pilot
      receipts, then require the self-hosted runner owner's recorded budget
      approval before enabling the recurring native-default schedule.
    - [ ] Add a workflow-dispatch-only
      `.github/workflows/jax_vmec_hybrid_authority.yml` for the VMEC-hybrid lane.
      It cannot promote the hybrid until an approved runner proves immutable
      VMEC/MPI build identity, CPU/GPU availability, runtime budget, and output
      provenance; report the JAX device slice separately.
    - [ ] Make missing native-default evidence visible as `not_run`, never a
      bounded pass promoted by omission.

12. Update documentation, compatibility, and developer gates.
    - [ ] Rewrite `examples/jax/README.md` around exact native/JAX pairs and show
      CPU fast, GPU fast, CPU parity, GPU parity, Boozer/vacuum single-stage,
      and VMEC-hybrid commands.
    - [x] Add a generated native-to-JAX index showing mirror path,
      classification, runtime dependencies, device scope, scale, and latest
      evidence status.
    - [ ] Mark combined examples as tutorials in help text and remove wording
      that presents `inspired_by` as port coverage.
    - [ ] Document the example-schema-v2/parity-schema-v1 deprecation interval,
      migration command, rollback, and removal gate.
    - [ ] Expand the pinned Pyright include set monotonically as touched example,
      manifest, and parity modules become clean; do not suppress diagnostic
      categories or add blanket ignores.
    - [ ] Run Ruff, formatting, Pyright, compileall, manifest tests, parity
      contract tests, import-boundary tests, and `git diff --check` for every
      implementation slice.

## Validation Plan

- [x] Manifest and mirror-identity tests:

  ```bash
  python -m pytest -q \
    tests/test_jax_examples_manifest.py \
    tests/test_jax_example_parity_manifest.py
  ```

- [x] Example runner and bounded CPU matrix:

  ```bash
  python -m pytest -q tests/integration/test_jax_examples.py
  python examples/jax/run_examples.py --device cpu --scale bounded
  python examples/jax/run_examples.py \
    --device cpu --intent parity --scale bounded
  ```

- [x] Parity input, runner, publication, artifact, and runtime contracts:

  ```bash
  python -m pytest -q \
    tests/integration/test_jax_example_parity_inputs.py \
    tests/integration/test_jax_example_parity_runner.py \
    tests/integration/test_jax_example_parity_publication.py \
    tests/integration/test_jax_example_parity_artifacts.py \
    tests/integration/test_jax_example_parity_runtime.py
  ```

- [x] Real strict-GPU example matrix on the designated CUDA runner:

  ```bash
  python examples/jax/run_examples.py --device gpu --scale bounded
  python examples/jax/run_examples.py \
    --device gpu --intent parity --scale bounded
  ```

- [x] Matched native/JAX bounded authority and independent audit:

  ```bash
  matched_parity_run_dir="$(python examples/jax/run_parity.py \
    --case all-applicable \
    --scale bounded \
    --lanes native-cpu,jax-cpu,jax-gpu \
    --artifact-root .artifacts/jax-example-parity)"
  python -m examples.jax.parity.audit \
    --run "${matched_parity_run_dir}" \
    --require-authoritative
  ```

- [ ] After the typed `--scale` selector is implemented, run native-default
  authority from its clean scheduled checkout and audit the emitted directory:

  ```bash
  native_default_run_dir="$(python examples/jax/run_parity.py \
    --case all-applicable \
    --scale native_default \
    --lanes native-cpu,jax-cpu,jax-gpu \
    --artifact-root .artifacts/jax-example-parity)"
  python -m examples.jax.parity.audit \
    --run "${native_default_run_dir}" \
    --require-authoritative
  ```

- [x] Representative bounded performance and peak-memory sanity runs:

  ```bash
  /usr/bin/time -v python examples/jax/run_examples.py \
    --device cpu --intent fast --scale bounded
  /usr/bin/time -v python examples/jax/run_examples.py \
    --device gpu --intent parity --scale bounded
  ```

- [x] Machine-validated authentic TDD receipts:

  ```bash
  python -m pytest -q tests/test_jax_examples_one_to_one_tdd_receipts.py
  python examples/jax/validate_tdd_receipts.py \
    --replay \
    docs/jax_examples_one_to_one_tdd_receipts.json
  ```
- [x] Verify the GPU lane reports CUDA, FP64, no CPU fallback in each declared
  JAX region, and no undeclared host numerical solve.
- [x] Verify VMEC-hybrid receipts identify VMEC CPU/MPI separately and never
  label the entire workflow GPU/on-device.
- [x] Verify native/JAX inputs, executed-source hashes, data checksums,
  `simsoptpp` identity, and effective construction fingerprints match.
- [ ] Verify every promoted mirror has an authentic source-owned RED, GREEN, and
  REFACTOR receipt tied to immutable revisions.
- [x] Static and typing checks:

  ```bash
  ruff check examples/jax tests/test_jax_examples_manifest.py \
    tests/test_jax_example_parity_manifest.py
  ruff format --check examples/jax tests/test_jax_examples_manifest.py \
    tests/test_jax_example_parity_manifest.py
  pyright --warnings
  python -m compileall -q examples/jax src/simsopt_jax src/simsopt_jax_adapters
  git diff --check
  ```

- [x] Confirm unrelated modified/untracked files are byte-identical to the
  preflight inventory and that each commit contains only its intended slice.

## Risks and Mitigations

- Risk: Exact-name mirrors become shallow wrappers around combined tutorials.
  Mitigation: Require the exact mirror to own and execute the native source's
  scientific stages; reject pass-through-only files in review and parity tests.

- Risk: Sharing implementation code makes native and JAX lanes agree on the
  same bug.
  Mitigation: Share typed inputs and immutable data only; keep numerical
  implementations independent and recompute comparisons in the auditor.

- Risk: A source is called full parity after only a reduced smoke workflow.
  Mitigation: Record workflow stages and scale independently; require all
  declared stages for `full` and label bounded versus native-default evidence.

- Risk: Optional VMEC code in permanent-magnet examples silently enters the
  external-solver-free lane.
  Mitigation: Pin and test `vmec_flag=False`; reject VMEC imports/execution in
  those JAX lane receipts.

- Risk: A VMEC-hybrid run is advertised as GPU VMEC or fully on-device.
  Mitigation: Use a typed `jax_slice_gpu` scope, separate receipts, and report
  generation tests that reject broader wording.

- Risk: Wrapping VMEC in a callback makes the objective appear traceable while
  breaking differentiation, transfer safety, or reproducibility.
  Mitigation: Keep VMEC in an explicit host-controlled outer loop and prohibit
  VMEC callbacks in static import/AST tests.

- Risk: Differentiating through an unrolled Boozer/optimizer loop causes device
  memory growth.
  Mitigation: Use the existing implicit/custom derivative surface, measure peak
  memory, and test that iteration history is not retained as the reverse-mode
  tape.

- Risk: Example proliferation duplicates device, input, and tolerance policy.
  Mitigation: Keep those policies in the existing central runtime, input-bundle,
  and comparator owners; examples contain workflow-specific science only.

- Risk: Native-default-scale jobs are too expensive or unavailable in CI.
  Mitigation: Keep bounded deterministic CI mandatory and schedule immutable
  native-default authority separately, with missing evidence reported as
  `not_run`.

- Risk: Manifest migration invalidates prior artifacts or erases historical
  meaning.
  Mitigation: Retain example-schema-v2/parity-schema-v1 read compatibility,
  bind artifacts to both schema and source hashes, dry-run both migrations,
  obtain approval, and test atomic rollback.

- Risk: Timing or memory numbers compare different work, cache states, or
  resource classes.
  Mitigation: Fail closed on input/provenance mismatch, retain cold and warm
  samples separately, synchronize JAX, use common host-RSS ownership, and label
  GPU-versus-native results as cross-device observations.

- Risk: The dirty worktree causes unrelated user files to enter commits.
  Mitigation: Record preflight status, stage explicit paths, inspect the staged
  diff, and verify unrelated file hashes after every slice.

## Completion Criteria

- [x] All 25 candidate external-solver-free native sources have passed the
  executable capability audit and have exact-path JAX mirrors; any contrary
  capability finding requires an explicit user-approved plan amendment rather
  than silent removal. No ready mirror owns more than one native source.
- [x] Every ready mirror teaches public supported JAX APIs and passes its
  declared CPU/GPU execution contract with fast as ordinary default.
- [x] Every ready mirror has matched bounded native CPU/JAX CPU/strict JAX GPU
  parity for all applicable initial and final observables, with retained
  absolute, relative, and scale-aware precision discrepancies.
- [x] Every ready mirror has bounded timing and peak-host-memory sanity evidence
  for native/JAX CPU and strict-GPU execution. Representative workload classes
  retain detailed cold/warm and VRAM evidence, and every required execution
  finishes without OOM.
- [ ] Native-default-scale evidence is separately recorded for every practical
  mirror; absent evidence remains explicitly `not_run`.
- [x] The VMEC-free Boozer/vacuum single-stage pair passes native/JAX CPU/GPU
  parity without invoking VMEC and is represented by the 52nd native catalog
  row, one exact-name executable record, and one parity relationship.
- [x] The VMEC-hybrid single-stage mirror matches the original native workflow,
  with VMEC CPU/MPI and JAX CPU/GPU-slice evidence reported separately.
- [x] Combined lessons that remain are classified and documented as tutorials
  and contribute zero one-to-one mirror coverage.
- [x] Every blocked/not-applicable source has a concrete, validated reason and a
  reconsideration condition.
- [ ] Every promoted mirror has authentic per-source RED -> GREEN -> REFACTOR
  evidence accepted by the machine-readable receipt validator; no historical
  test is relabeled retroactively.
- [x] Example schema v3, parity schema v2, compatibility, atomic rollback,
  artifact audit, documentation, Ruff, format, Pyright, compileall, and
  `git diff --check` gates pass.
- [x] Generated reports distinguish full/reduced, bounded/native-default,
  full-device/adapter/hybrid, CPU/GPU-slice, and certification/non-certification
  claims.
- [x] Unrelated worktree content remains untouched and implementation commits
  are reviewable independently.

## Open Questions

- What per-source runtime budget will the self-hosted GPU-runner owner approve
  for native-default scheduled authority? Phase 11 must calibrate bounded and
  native-default durations first; until approval, these entries remain
  `not_run` and cannot complete the corresponding authority gate.
- Which VMEC/MPI build will be installed on the workflow-dispatch hybrid
  authority runner? Its immutable identity and a successful preflight receipt
  are required before the hybrid can be promoted.
- After the example-schema-v2/parity-schema-v1 compatibility interval, should
  `inspired_by` be removed
  entirely or retained as non-authoritative tutorial provenance? The
  recommended choice is to retain it only as tutorial provenance and reject it
  anywhere coverage is computed.
