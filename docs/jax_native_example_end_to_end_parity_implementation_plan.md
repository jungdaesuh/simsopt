# JAX Fast Execution and Native Example End-to-End Parity Implementation Plan

**Status:** Blocked
**Last updated:** 2026-07-26

**Outcome:** Fast-default and explicit-parity support planned; candidate
certification still has scientific, safety, TDD, and delivery blockers

## Implementation outcome

Candidate commit `799c656e186642bdb7e296e46ad1c6cd61277839`, a detached direct
child of reviewed branch baseline
`b6775bf23030bafbd602ce3131dea948e7b8bb4b`, implements the planned harness.
It is not contained by a local branch or tag and therefore is not delivered by
the current branch history. Its clean, non-smoke authority run is
`.artifacts/jax-example-parity/20260726T225943Z-09dfdc3e`, bound to detached
source snapshot `799c656e186642bdb7e296e46ad1c6cd61277839`. A clean-checkout
audit replay recomputed all 228 **declared** numerical comparisons and reported
8 cases, 24 authoritative lane receipts, and an aggregate `pass`. The JAX GPU
receipts identify an NVIDIA GeForce RTX 5090, FP64, and all three effective JAX
transfer directions at `disallow`. This is historical, local candidate
evidence, not a fresh GPU rerun or evidence that the implementation is
scientifically complete or shipped.

The candidate classified all 28 ready `inspired_by` relationships. Eight
are executable bounded cases: two `full` and six `reduced`; the other 20 name
concrete unsupported blockers. The candidate results table at
`docs/jax_native_example_parity_results.md` is worktree-only, and the parity
section of `docs/jax_examples_tdd_receipts.md` is not present on the current
branch. Neither is shipped evidence until delivered with the implementation.

Final review found scientific-contract gaps: QFM passes despite violating the
central constraint threshold and reports failed JAX solver statuses as
normalized `converged`; the full traceable least-squares case omits three
final-Jacobian lane routes; and the auditor does not revalidate the retained
input bundle. It also found two unresolved artifact-safety defects: the publish
check plus POSIX rename does not provide no-replace semantics under a
publish-time race, and sidecar path validation is vulnerable to a symlink
time-of-check/time-of-use replacement. The retained numerical artifact is
still useful evidence that the declared routes agree, but it is not authority
for the requested end-to-end contract. The plan cannot be `Done` until Phase
11 repairs these defects and the complete reviewed slice becomes reachable.

### Delivery blockers

- [ ] Implement and RED-test the fast-default/explicit-parity contract in Phase
  10 without weakening parity certification.
- [ ] Repair and RED-test the scientific completeness, no-replace publication,
  and descriptor-relative no-follow sidecar requirements in Phase 11.
- [ ] Create a new clean authority run and independent audit after those
  repairs; the retained `799c656e1` artifact may not close the repaired safety
  contract.
- [ ] Deliver the repaired implementation, generated results table, parity TDD
  receipt, and this status document in a reachable reviewed branch history.
- [ ] Put the authority artifact in a durable shared store with retention and
  an immutable identifier, or explicitly scope the final claim to local-only
  evidence. The current ignored `.artifacts/` path exists only on this host.

Unless an item is explicitly reopened below, checked implementation items
describe behavior and evidence in candidate `799c656e1`; they do not mean that
the current branch has delivered the item.

## Purpose

Make performance-oriented JAX execution the normal CPU/GPU user path while
preserving an explicit, reproducible, fail-closed parity path proving that
matched native SIMSOPT CPU and JAX workflows evaluate the same FP64 problem and
reach scientifically equivalent outcomes. The plan covers fast-mode defaults,
initial-state value and derivative parity, final solve parity, strict JAX
CPU/GPU certification, and hash-bound structured artifacts.

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
- Support JAX CPU and GPU in both `fast` and `parity` execution intents. `fast`
  is the default whenever a caller selects JAX without an explicit intent;
  `parity` is an explicit opt-in and is the only intent that may produce
  certification evidence.
- Keep device selection orthogonal to execution intent: CPU/GPU chooses
  placement, while fast/parity chooses numerical and verification policy. Both
  intents use the same public objective, solver family, accepted-state
  publication, and scientific-success contract.
- Preserve authentic RED -> GREEN -> REFACTOR receipts for each parity case and
  publish source/input/environment-bound JSON plus array sidecars.
- Classify every ready JAX example-to-native-source relationship as `full`,
  `reduced`, or `unsupported`, record `bounded` versus `native_default` scale
  independently, and prevent reduced or bounded evidence from being reported
  as native-default end-to-end parity.

## Requested Matched-Comparison Contract

The five requested requirements map to one fail-closed execution contract.
Only `full` or `reduced` relationships with an implemented parity case are
executable; `unsupported` relationships remain explicit rather than producing
partial or synthetic parity evidence.

| Requirement | Single owner | TDD acceptance gate | Persisted evidence |
|---|---|---|---|
| Identical native/JAX inputs | `examples/jax/parity/input_bundle.py` owns canonical typed inputs, with FP64 required for real-valued scientific arrays; each case emits an effective-construction receipt | Mutating parameters, quadrature/resolution, weights, constraints, seed, dtype, or stopping options must produce an authentic RED before comparison | Input and configuration fingerprints plus per-lane construction receipts |
| Same-state objective/derivative/constraint comparison | Case implementations emit normalized initial-state observables; `examples/jax/parity/arbiter.py` owns pairwise verdicts | Missing, non-finite, factor-of-two-swapped, or tolerance-violating objective, residual, gradient, Jacobian, constraint, or constraint-Jacobian data must fail | Canonical JSON metadata and hash-bound NPY array sidecars |
| Final-outcome comparison | `examples/jax/parity_manifest.json` owns applicable final comparison routes and central tolerance buckets | Perturbed parameters, final objective/residual, feasibility, convergence category, raw status, or applicable required counters must fail; algorithm-different counts remain reported diagnostics unless a reviewed bound exists | Final state, status, driver, applicable `nit`, `nfev`, `njev`, and every direct comparison result |
| JAX CPU and strict GPU execution | Shared lane-environment policy and `examples/jax/run_parity.py` own isolated `native-cpu`, `jax-cpu`, and `jax-gpu` children | CPU fallback, wrong precision/backend, missing effective transfer guards, hidden host solver, skipped GPU execution, or source drift must fail | Device metadata, FP64 policy, transfer-guard state, exact argv, stdout/stderr, and lane result paths |
| RED -> GREEN parity tests and structured publication | The parity tests own public failure cases; publication code owns atomic run artifacts | Every promoted case requires a case-owned RED, smallest GREEN, and green-preserving REFACTOR receipt | Source/input/environment-bound aggregate JSON, NPY sidecars, TDD receipts, and generated results table |

The mandatory direct comparison graph for each applicable observable is:
`native-cpu` versus `jax-cpu`, `native-cpu` versus `jax-gpu`, and `jax-cpu`
versus `jax-gpu`. Passing two adjacent comparisons never substitutes for the
direct native/GPU comparison.

## Fast and Parity Execution Contract

The public decision is two-dimensional: `device = cpu | gpu` and
`intent = fast | parity`. `intent` defaults to `fast`; the device remains an
explicit caller or environment choice. The existing internal mode names remain
the runtime SSOT:

| Device | Default fast mode | Explicit parity mode |
|---|---|---|
| CPU | `jax_cpu_fast` | `jax_cpu_parity` |
| GPU | `jax_gpu_fast` | `jax_gpu_parity` |

With every backend selector unset, the repository continues to default to
`native_cpu`; this change does not make JAX or CUDA an implicit dependency. The
changed default applies only after JAX has been selected through the typed
public API or compatible legacy backend/platform environment without an
explicit `SIMSOPT_BACKEND_MODE`. Parity is never inferred from CI location,
strictness, or device availability, and an unavailable requested device must
fail rather than fall back to another device. The retained legacy lane aliases
are explicit parity selectors during their compatibility interval.

Selector precedence is defined per entrypoint rather than by one ambiguous
global slogan:

1. `set_backend()` arguments are explicit process configuration. A full mode
   is mutually exclusive with `device`/`intent`; the API rejects mixed forms.
2. `run_examples.py --device/--intent` or the legacy `--lane` alias is the
   operator's explicit child profile. The runner removes inherited backend,
   platform, precision, strictness, transfer, and JAX-platform selectors before
   emitting the selected child environment; it records the effective profile.
3. In environment-only resolution, `SIMSOPT_BACKEND_MODE` wins over the legacy
   backend/platform pair. When that pair explicitly selects JAX, its CPU/CUDA
   platform maps to the corresponding fast mode.
4. With no explicit API, CLI, or environment selector, `native_cpu` remains the
   default.

Fast and parity are both supported production paths, not correctness versus
incorrectness:

- **Fast** retains FP64 scientific arrays and the same terminal correctness
  gates, but may use performance-tuned chunking/sharding, default matmul
  precision, normal transfer logging, compilation caches, and backend reduction
  order. The `default` versus `highest` matmul setting is retained as provenance,
  not presumed evidence of an FP64 speed or accuracy difference; only matched
  measurements can justify the fast label. Fast receipts are diagnostic and
  must be labeled non-certifying.
- **Parity** uses the stable comparison policy, highest matmul precision,
  strict effective transfer guards where required, deterministic GPU settings,
  source/input binding, direct native/JAX lane comparisons, and authoritative
  artifacts. It is slower by design and must be requested explicitly.
- A fast result may be numerically compared for diagnostics, but it cannot be
  relabeled, promoted, or accepted as parity evidence. Conversely, parity mode
  must not silently select fast tuning to recover performance.

The dedicated `examples/jax/run_parity.py` command remains parity-only and does
not accept a fast-mode option. The ordinary example runner exposes the
orthogonal interface:

```console
python examples/jax/run_examples.py --device cpu
python examples/jax/run_examples.py --device gpu
python examples/jax/run_examples.py --device cpu --intent parity
python examples/jax/run_examples.py --device gpu --intent parity
```

The first two commands use fast mode by default. Existing `--lane cpu-smoke`
and `--lane gpu-strict` commands retain their current parity semantics during a
documented deprecation interval; they must never be silently reinterpreted as
fast aliases.

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
- Publish generalized speedup or memory-efficiency claims from the example
  measurements. Matched runtime and memory evidence is a gate for naming and
  defaulting the performance-oriented profiles, but it is separate from the
  scientific correctness and parity-certification verdicts and remains scoped
  to the measured workload, device, software stack, and artifact.
- Make fast-mode output eligible for native/JAX certification, weaken FP64 or
  scientific-success requirements in fast mode, or make parity the implicit
  default for JAX users.
- Promote the currently planned single-stage vacuum example while its required
  outer derivative and accepted-state certificate remain unavailable.

## Planning-time context

### Baseline authority at plan review

This section preserves the evidence that drove the implementation. It is
historical; the candidate outcome above supersedes its provisional worktree
state and classifications, subject to the unresolved delivery and safety gates.

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
- The live runtime already declares `jax_cpu_fast`, `jax_cpu_parity`,
  `jax_gpu_fast`, and `jax_gpu_parity` in
  `src/simsopt_jax/backend/_runtime_policy.py`. However, its legacy JAX resolver
  currently maps CPU/CUDA to parity when no explicit full mode is set, so fast
  is not yet the JAX default requested by this plan.
- The live example runner exposes only `cpu-smoke` and `gpu-strict`; the shared
  lane environment maps both to parity modes. Fast is therefore implemented at
  the low-level runtime policy but is not yet a first-class ordinary-example
  runner path.
- The expanded non-GPU parity suite was run during the original 2026-07-26
  plan review:
  51 tests passed and the surface-geometry final-parameter comparison failed
  (`max_abs_diff=2.45719087e-06` versus the current whole-solve route). This is
  implementation evidence, not a reason to weaken the central tolerance.
  At that stage, surface geometry remained reduced and non-authoritative. The
  later candidate authority artifact promoted a bounded reduced comparison; it
  did not promote full or native-default parity.

The lane semantics in this plan follow the official JAX configuration
contracts: [`JAX_PLATFORMS`](https://docs.jax.dev/en/latest/config_options.html#platforms)
is fixed before initialization and fails if a requested platform cannot be
initialized; [64-bit mode](https://docs.jax.dev/en/latest/default_dtypes.html)
must be enabled explicitly; transfer-guard `disallow` blocks implicit transfers
while explicit transfers remain available for result publication; and disabling
[GPU preallocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
changes allocation behavior and can increase fragmentation.

### Historical provisional parity classification

The classification below reflects the uncommitted predecessor slice that was
available when the plan was reviewed, not the final implementation. It is
retained to show which rows were downgraded after live API validation. The
candidate source of truth is `examples/jax/parity_manifest.json`, which
classifies all 28 ready `inspired_by` relationships individually.

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

For the user-facing fast/parity choice:

**Mode Option A - orthogonal device and execution intent with one resolver
(selected).**

- `device = cpu | gpu` selects placement and `intent = fast | parity` selects
  policy. The shared profile owner maps the pair to one existing internal mode.
- The intent defaults to fast. Explicit `SIMSOPT_BACKEND_MODE` remains the
  low-level escape hatch and provenance identity; the parity runner requests
  parity directly and cannot be downgraded by defaults.
- Advantages: one obvious default, no four-way user-facing cross product, and
  one policy owner for the runtime, runner, validation, documentation, and CI.

**Mode Option B - expose four independent lane names everywhere (rejected).**

- Requiring callers to choose among `jax_cpu_fast`, `jax_cpu_parity`,
  `jax_gpu_fast`, and `jax_gpu_parity` leaks the runtime representation and
  duplicates device/mode branching across CLIs and workflows.

**Mode Option C - infer parity from strictness, CI, or the command name
(rejected).**

- Hidden context would make identical commands change numerical policy and
  could silently turn fast results into certification evidence. Parity is an
  explicit intent, not an inference from transfer guards, process location, or
  CI environment.

For native/JAX matched comparison:

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

- `manifest.json` continues to own example readiness, lineage, and supported
  devices. Fast/parity support is a suite-wide execution contract, not a field
  copied into every ready example. A pure typed resolver in
  `src/simsopt_jax/backend/_runtime_policy.py` owns the
  `(device, intent) -> explicit mode` mapping and certification eligibility.
  `examples/jax/_lane_environment.py` separately owns process-isolation and
  guard overlays derived from that resolved profile. The runner validates
  child results against the resolver, and CI invokes the public CLI; neither
  keeps another four-mode mapping.
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

Changing a tolerance, execution profile, or source relationship must therefore
require one owner change rather than edits across every example or parity case.

### Design tier and API evolution

This is Tier 4 because it changes a public runtime API and migrates the
source-controlled example-manifest schema. Tier 3 API-evolution gates still
apply, plus a schema rollback, read-only dry-run, rollout observability, and
explicit user sign-off before the v2 manifest is written or activated.

- Observable behavior delta: JAX selected without an explicit full mode resolves
  to fast instead of parity; `run_examples.py --device <cpu|gpu>` defaults to
  fast, while `--intent parity` is explicit. Existing `--lane cpu-smoke` and
  `--lane gpu-strict` commands keep their current parity behavior during
  deprecation. The dedicated parity runner and its artifacts remain parity-only.
- Caller inventory: `simsopt_jax.config.set_backend`/`use_runtime` callers,
  direct `SIMSOPT_BACKEND_MODE` users, legacy backend/platform-environment
  users, example-runner users, parity-runner users, manifest readers, CPU CI,
  GPU CI, integration fixtures, benchmark launchers, documentation commands,
  and artifact/report consumers.
- Migration: ordinary callers may omit intent and receive fast mode. Callers
  requiring certification must set `intent="parity"`, pass `--intent parity`,
  set an explicit `jax_*_parity` mode, or use `run_parity.py`. Artifact consumers
  continue to require an exact `schema_version`.
- Compatibility tests: fully unset selection remains `native_cpu`; explicit JAX
  legacy selection defaults to fast; explicit-mode precedence; all four
  device/intent profiles; retained `jax_cpu_float32_smoke`; `use_runtime`
  forwarding; legacy lane aliases; inherited-environment scrubbing; exact CLI
  argv; schema migration; deterministic ordering; lane identity; input
  fingerprints; normalized statuses; and fail-closed malformed/mismatched
  artifact cases.
- Deprecation: retain legacy `--lane cpu-smoke` and `--lane gpu-strict` parity
  semantics with a warning for at least one documented release; remove them
  only after CI, docs, and downstream callers use `--device/--intent`. A future
  artifact schema v2 must retain a v1 reader or conversion command before v1 is
  removed.
- Manifest migration: absence of `schema_version` is exactly v1 and requires
  the current `lanes` field. Version 2 requires `schema_version: 2` and
  `devices`, forbids `lanes` and per-example `intents`, and rejects unknown
  versions or mixed field sets. The production manifest has one canonical v2
  writer; the v1 path is read-only compatibility.
- Dry-run and observability: before changing the canonical manifest, run the
  v1-to-v2 converter in read-only mode against the committed v1 file and all
  compatibility fixtures. It must print the candidate v2 bytes and a semantic
  diff proving identical example IDs, readiness, lineage, paths, ordering, and
  CPU/GPU capability, without writing the repository. Runner receipts and CI
  logs record `manifest_schema_version` and `used_legacy_manifest_adapter`; v1
  use emits one actionable deprecation warning. During the compatibility
  release, CI exercises both readers and fails if the canonical manifest or a
  generated artifact unexpectedly uses v1.
- Sign-off: after RED tests and the dry-run report pass, present the exact
  candidate manifest digest, semantic diff, compatibility duration,
  observability fields, and rollback command to the user. Do not write or
  activate canonical v2 until the user explicitly approves that migration.
- Rollback: restore the previous JAX selection resolver and example-runner CLI
  while leaving explicit low-level modes intact. For manifest rollback, restore
  the canonical v1 `lanes` file but retain the v2 reader until every emitted v2
  file and known downstream consumer has aged out; validate both readers before
  and after rollback. Remove parity manifest/package/runner, CI steps, tests,
  and docs together only if the whole parity feature is rolled back. Existing
  examples and public solver APIs remain independent.

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
- Fast means performance-oriented FP64 JAX policy, not reduced scientific
  correctness, FP32 smoke, CPU fallback, or permission to publish parity claims.
- Device choice remains explicit. This plan does not add automatic GPU
  discovery that changes placement across machines.

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
   - [x] **RED:** On the committed baseline, require manifest tests to reject a
     ready example missing either required lane and solver tests to reject a
     strict-GPU default path that selects or executes Optimistix, Optax, SciPy,
     or a host callback in the numerical solve region. Optional solver contract
     imports and explicit opt-in drivers are not themselves failures.
   - [x] **GREEN:** Land or identify the predecessor revision that owns the
     intended ten ready CPU/GPU examples, backend-neutral default serial
     solver, lane manifest, and focused tests. Do not include unrelated
     dirty-tree changes. Pass the manifest, example-runner, serial-solver,
     CPU-smoke, and real strict-GPU tests from a clean checkout; record the
     commit, resolved module paths, device, commands, and zero-skip results.
   - [x] **REFACTOR:** Make the committed revision the sole baseline named by
     parity receipts. Dirty-tree exploratory evidence remains explicitly
     non-authoritative.

1. Define and RED-test the parity scope and classifications.
   - [x] **RED:** Extend `tests/test_jax_examples_manifest.py` or add
     `tests/test_jax_example_parity_manifest.py` to require every relationship
     implied by ready `manifest.json` records to have exactly one parity row.
   - [x] **RED:** Reject unknown example IDs, native paths absent from the
     referenced `inspired_by`, duplicate relationships/case IDs, hard-coded
     numeric tolerances, missing test owners, invalid scale tiers, or `full`
     rows with incomplete required observables or scientific workflow stages.
   - [x] **GREEN:** Add `examples/jax/parity_manifest.json` with one row per
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
   - [x] **REFACTOR:** Keep cross-manifest relationship validation in one
     function and preserve deterministic manifest order.

2. Specify the canonical input bundle and result schema test-first.
   - [x] **RED:** Add schema tests for missing fields, non-finite arrays, dtype
     drift, inconsistent shapes, unknown schema versions, invalid normalized
     status, sidecar hash mismatch, traversal/symlink escape, and a dirty source
     mislabeled authoritative.
   - [x] Define artifact schema v1 with required provenance: clean repository
     commit, Python/JAX/SIMSOPT versions, resolved source paths, device metadata,
     lane environment policy, case ID, native source, JAX example ID, random
     seed, input fingerprint, and configuration fingerprint. Authoritative runs
     require a clean tree. Exploratory dirty runs record a canonical tracked
     diff hash plus untracked-file inventory and are labeled non-authoritative.
   - [x] Bind every result to a sorted executed-source manifest containing the
     canonical path and Git blob ID or SHA-256 for the runner, case module,
     manifests/configuration, and transitive native/JAX project modules used by
     the case. Each child snapshots the in-repository Python modules present in
     `sys.modules` after its lane completes, resolves and contains their source
     paths beneath the checkout, and hashes their executed bytes; the runner
     and declarative manifests are added explicitly because they need not be
     imported by every child. Validate the content hashes before comparison; a
     revision plus resolved paths alone is insufficient.
   - [x] When `simsoptpp` is loaded, record the resolved extension path,
     SHA-256 of the loaded shared object, package/build version metadata, and a
     compatibility receipt tying that binary to the committed checkout. Reject
     a stale or unverifiable extension before treating it as a native/C++
     oracle.
   - [x] Store bounded scalar/status metadata in canonical JSON. Store each
     parameter, residual, gradient, Jacobian, or constraint array in its own
     deterministic `.npy` sidecar, referenced by a canonical relative path,
     dtype, shape, memory order, and SHA-256. Canonicalize numeric byte order and
     C-contiguity, forbid object arrays/pickle, and pin the NPY format version so
     equal arrays have equal bytes across supported environments. Reject
     absolute paths, `..`, symlinks, containment escapes, missing files, or
     mismatched metadata.
   - [x] Represent non-applicable quantities with an explicit applicability
     map and `null`, not omitted keys or empty arrays.
   - [x] Normalize status to `converged`, `budget_exhausted`, `failed`, or
     `not_applicable` while preserving driver name, raw success/status/message,
     `nit`, `nfev`, and `njev`.
   - [x] Each `LaneResult` records its ordered completed scientific stages; the
     arbiter requires exact agreement with the relationship's declared stage
     contract before comparing numerical observables.
   - [x] **GREEN:** Add frozen typed contracts in
     `examples/jax/parity/contracts.py` for `ParityInputMetadata`, `LaneResult`,
     `InitialStateResult`, `FinalStateResult`, `ComparisonResult`, and
     `RunManifest`. Implement serialization and validation without a new direct
     dependency; use the standard library, NumPy, and existing project types.
   - [x] **REFACTOR:** Centralize canonical JSON encoding, array fingerprinting,
     and source/environment provenance; no case may implement its own writer.

3. Build byte-identical input generation and lane isolation.
   - [x] **RED:** Mutate one parameter, quadrature point, seed, weight,
     constraint, dtype, or stopping option and require the runner to fail before
     scientific comparison with a field-specific fingerprint diagnostic.
     Route each mutation through the real native and JAX case builders and
     prove that the corresponding effective-construction receipt changes; this
     catches ignored fields and default-option leakage.
   - [x] Require all stochastic cases to use an explicit NumPy generator and
     named seed. Persist generated samples; native and JAX children must load
     them rather than regenerate them independently.
   - [x] Record quadrature grids, resolution/order, free-DOF indices, weights,
     targets, bounds, constraints, solve budgets, and terminal thresholds in
     the bundle or configuration fingerprint.
   - [x] Each child must emit an effective-construction receipt reconstructed
     from the instantiated problem and solver: applied parameter/DOF order,
     grids, weights, bounds, constraints, and effective stopping options. The
     arbiter compares this receipt with the canonical bundle; hashing the
     parent input alone is not proof that a lane consumed it.
   - [x] **GREEN:** Add `examples/jax/parity/input_bundle.py` to create each
     canonical bundle once from NumPy arrays/scalars and record dtype, shape,
     memory order, and SHA-256 per leaf. Add `native-cpu`, `jax-cpu`, and
     `jax-gpu` subprocess environments in `examples/jax/parity/runtime.py`.
     Reuse the established runtime-policy values from `run_examples.py` through
     one shared owner rather than copying environment dictionaries.
   - [x] Native CPU must resolve the current checkout and native SIMSOPT stack;
     JAX CPU/GPU must report `jax_cpu_parity`/`jax_gpu_parity`, FP64, and the
     expected platform. Strict GPU must disallow transfers and CPU fallback.
   - [x] **REFACTOR:** Extract a shared lane-environment module only after
     regression tests prove the existing example-runner argv/environment and
     behavior remain unchanged.

4. Implement the paired runner and fail-closed arbiter.
   - [x] **RED:** Add injected-child tests for missing lanes, nonzero children,
     malformed results, input/config/effective-construction fingerprint drift,
     wrong source checkout, wrong platform/precision/backend, non-finite
     required observables, unsupported masquerading as pass, direct pairwise
     tolerance failure, stable manifest ordering, and exact bounded argv.
   - [x] **RED:** Add concurrent-writer, interrupted-publication, existing-run,
     and partial-artifact rejection tests. These candidate tests did not cover
     the publish-time destination race or path-component replacement race
     reopened in Phase 11.
   - [x] **GREEN:** Implement deterministic aggregation and one summary JSON
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
     directory. The candidate does not yet prove the required no-overwrite
     behavior under the publish-time race; Phase 11 owns that correction.
     Retain a failed partial directory with an explicit failure marker for
     diagnosis.
   - [x] **REFACTOR:** Keep subprocess mechanics independent from scientific
     comparison; the runner collects lane results and the arbiter compares them.

5. Establish initial-state parity before any solve comparison.
   - [x] For each case, evaluate native CPU and JAX at the exact serialized
     initial parameters before invoking a solver or integrator.
   - [ ] Least-squares cases must record residual vector `r`, public objective
     `objective_sum_squares = r.T @ r`, residual Jacobian `J`, and public
     objective gradient `objective_gradient = 2 * J.T @ r`. Record a solver's
     half-squared `solver_cost = 0.5 * r.T @ r` separately when available; never
     compare it directly with the public objective. The candidate preserves the
     public objective convention but omits `solver_cost` where available.
     Scalar cases must record objective and gradient. Constrained cases
     additionally record constraint values and constraint Jacobian.
   - [x] Tracing/fixed-state cases must name their analogous initial-state
     quantities (field/RHS/invariants or objective/gradient) and mark solver-only
     quantities not applicable.
   - [x] Select tolerances through the central ladder only after validating that
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
     failing command and diagnostic in a TDD receipt. Existing summaries do not
     authenticate this per-case failing-first history.
   - [x] **RED:** Add a least-squares contract test whose nonzero residual and
     Jacobian catch a factor-of-two swap between the public objective gradient
     and the half-squared solver-cost gradient.
   - [x] **GREEN:** Make all required native CPU vs JAX CPU, native CPU vs JAX
     GPU, and JAX CPU vs JAX GPU initial-state comparisons pass before enabling
     final solves for that case.
   - [x] **REFACTOR:** Share mathematical recomposition helpers only where two
     completed cases use the same objective convention; keep tests DAMP.

6. Add final-outcome parity in dependency order.
   - [ ] For each case, first add and execute a case-owned RED that fails on the
     missing real native/JAX comparison or a scientifically meaningful injected
     perturbation. Then implement that case's smallest GREEN, run its affected
     regressions, and REFACTOR while green before starting the next case. The
     retained receipt does not prove this ordering for every promoted case.
   - [x] Wave A: traceable least-squares, curve length, and surface geometry.
     These establish scalar/least-squares result contracts and work-counter
     reporting with inexpensive deterministic fixtures.
   - [x] Wave B: coil flux, permanent magnets, wireframe, and fixed-state
     coil-force/finite-build. These establish adapter publication, discrete
     decisions, constraints, and explicit N/A optimizer fields.
   - [ ] Wave C executable scope: implement bounded reduced QFM with
     branch-stable initial state, feasibility/KKT gates, accepted-state
     publication, and original-residual checks; matching augmented or
     preconditioned residuals alone is invalid. The candidate QFM artifact
     violates the central constraint tolerance and conflates raw solver failure
     with normalized convergence; Phase 11 reopens it. Classify every Boozer
     relationship as `unsupported` with its concrete blocker; no Boozer solve
     parity was executed.
   - [x] Wave D disposition: classify every field-line and particle-tracing
     relationship as `unsupported` with its concrete blocker. No tracing parity
     case was executed; a future promotion must match physical initial state,
     interval, event semantics, tolerances, endpoints, and final invariants
     without demanding adaptive-step identity.
   - [ ] For every optimizing case, record final parameters, objective,
     residuals, gradient/Jacobian where defined, constraints/feasibility,
     normalized and raw status, driver, and work counters. Candidate QFM omits
     `nfev` and `njev`; least-squares cases also omit the separately promised
     half-squared `solver_cost` where available.
   - [ ] Apply the manifest's typed final comparison routes to every applicable
     lane pair. Each route specifies vector/scalar/constraint comparator and
     central tolerance bucket. For a case with non-unique minimizers, compare a
     declared equivalence invariant or quotient representation rather than raw
     parameters, and justify that policy in the case contract; individual
     feasibility or objective gates do not replace pairwise final parity. The
     candidate traceable least-squares case marks final `residual_jacobian`
     applicable but defines no three-lane routes for it.
   - [ ] Define common terminal scientific gates independently of driver status.
     A solver-reported success with failed residual/gradient/feasibility is a
     parity failure. Scientific acceptance must not rewrite a failed raw driver
     status to normalized `converged`; both dimensions remain explicit.
   - [ ] Compare `nit`/`nfev`/`njev` exactly only for the same algorithm and
     implementation contract. Otherwise require finite nonnegative counts and
     report absolute/relative deltas as diagnostics with an explicitly approved
     bound if one is scientifically necessary. QFM does not yet satisfy this
     reporting contract.
   - [ ] Record each case's RED -> GREEN -> REFACTOR evidence independently; do
     not batch-promote a wave based on one representative example. Phase 11
     owns the unresolved evidence-recovery gate.

7. Add real CPU/GPU evidence and memory/transfer gates.
   - [x] **RED:** Prove strict GPU rejects a fabricated CPU result, a hidden
     host optimizer, an optional optimizer selected as the parity default, and
     a receipt that sets only the SIMSOPT policy without JAX's effective
     transfer guard before accepting a CUDA receipt.
   - [x] Run every full/reduced JAX case on CPU and a real CUDA device with the
     identical input bundle and configuration fingerprint.
   - [x] Require FP64 arrays for parameters, residuals, objectives, gradients,
     Jacobians, and solver state at every authoritative comparison boundary.
   - [x] Require strict transfer guarding during the numerical region and fail
     on host callbacks, hidden SciPy/native solvers, CPU fallback, skipped GPU
     execution, or wrong device metadata.
   - [x] Record peak host RSS and device memory as diagnostics for cases whose
     compiled solver state may scale quadratically. Name the measurement owner
     and method in the receipt: parent-observed child RSS and, only where the
     CUDA backend exposes a validated per-process counter, device allocation.
     Record device memory as `unavailable` rather than inventing a value when no
     supported counter exists. Synchronize JAX work with
     `jax.block_until_ready` at declared measurement boundaries, distinguish
     compile/warmup from steady-state execution, and add shape-scaling tests for
     any case that materializes dense Jacobian/Hessian state. These diagnostics
     do not support speed claims.
   - [x] Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` for measurement receipts;
     document production preallocation policy separately so memory diagnostics
     are not confused with the supported runtime default.
   - [x] **GREEN:** Publish real device metadata and zero-fallback receipts for
     all applicable cases.

8. Integrate artifacts, documentation, and CI.
   - [x] **RED:** Add static workflow tests proving changes under parity
     manifests/cases reach the named CPU and strict-GPU jobs and that CI invokes
     the parity CLI rather than duplicating a case list. Add a narrow ignore
     test that initially fails for a probe below
     `.artifacts/jax-example-parity/`.
   - [x] **GREEN:** Write local artifacts beneath
     `.artifacts/jax-example-parity/<UTC-run-id>/` and keep generated data out of
     Git. Upload selected JSON/NPY artifacts through CI with a retention policy.
   - [x] Add only `.artifacts/jax-example-parity/` to `.gitignore`; do not
     broaden ignore or cleanup rules to unrelated artifacts or canonical input
     directories.
   - [ ] Deliver `docs/jax_native_example_parity_results.md` after the repaired
     complete run; generate its tables from the aggregate JSON rather than
     copying values manually. The current worktree-only table regenerates
     byte-for-byte from the retained candidate aggregate but is not in current
     branch history or candidate `799c656e1`.
   - [x] Extend `examples/jax/README.md` with the parity command, classification
     semantics, artifact layout, and the distinction between native
     Python/SciPy, `simsoptpp`, analytic, and external oracles.
   - [x] Add CPU Wave A to the `jax-public-integration` job in
     `.github/workflows/jax_smoke.yml`. Add strict-GPU Wave A to that workflow's
     `jax-gpu-strict-purity` job only after bounded runtime is measured. Ensure
     the workflow is exercised by its pull-request targets or an explicit
     manual trigger before treating it as evidence.
   - [x] Route expensive later waves only to a named scheduled/manual CUDA job
     that uses FP64, both effective transfer-guard settings at `disallow`,
     zero-skip/no-fallback checks, and artifact upload. The current
     `.github/workflows/jax_gpu_parity.yml` log-guard job is diagnostic and may
     not produce authoritative strict-parity evidence unless upgraded to this
     contract.
   - [ ] Preserve authentic per-case RED/GREEN/REFACTOR commands and pre-GREEN
     identities, plus source hashes, artifact hashes, device identity, and exact
     pass/fail counts in `docs/jax_examples_tdd_receipts.md` or a dedicated
     parity receipt file. Current summaries do not prove each case's
     failing-first sequence, and the parity receipt update is not delivered.

9. Complete the API, provenance, and final review gates.
   - [x] Confirm the new CLI has no untyped pass-through options and the schema
     exposes only stable consumer data.
   - [x] Confirm no `src/` module imports from `examples/jax/parity`, no native
     example is imported for top-level execution, and no new direct dependency
     was added.
   - [x] Verify each `full` classification executes all claimed workflow stages;
     downgrade any fixed-state or reduced case whose artifact cannot prove the
     final accepted state.
   - [x] Verify artifact paths are relative, source/input hashes resolve, array
     sidecars match, and secrets/environment-variable values are excluded.
   - [ ] Review the final path-scoped diff for duplicated tolerances, duplicated
     lane policy, hidden host transfers/solvers, stale lineage, weakened
     scientific thresholds, artifact races, and unrelated dirty-tree changes.
     The 2026-07-26 adversarial review failed this gate on two artifact-safety
     races and the delivery/evidence gaps below.

10. Make fast the default while supporting explicit parity on CPU and GPU.
    - [x] **RED:** In `tests/test_backend_precision_policy.py` and subprocess
      import-smoke coverage, prove the current explicitly selected legacy JAX
      CPU/CUDA resolver selects `jax_cpu_parity`/`jax_gpu_parity` when no full
      mode is set. Add expectations that an explicitly selected JAX CPU/CUDA
      backend instead selects `jax_cpu_fast`/`jax_gpu_fast`, while a fully unset
      selector remains `native_cpu` and an explicit full mode wins over legacy
      environment values.
    - [x] **GREEN:** Change the single mode resolver in
      `src/simsopt_jax/backend/_runtime_policy.py` so a selected JAX platform
      defaults to its fast mode. Preserve the four fast/parity modes, plus
      `native_cpu` and `jax_cpu_float32_smoke`, and their current policy tables;
      do not implement the default with duplicated environment conditionals in
      callers.
    - [x] **RED:** Add public-API tests for `set_backend("jax", device="cpu")`
      and `set_backend("jax", device="gpu", intent="parity")`, parameterized
      across both devices and intents. Require `device`, default `intent` to
      `fast`, reject `set_backend("jax")`, invalid device/intent values, and a
      full mode combined with any device/intent keywords before JAX
      initialization. Preserve every canonical explicit mode including
      `jax_cpu_float32_smoke`. Mirror the typed convenience selector through
      `use_runtime` without duplicating its resolution logic.
    - [x] **GREEN:** Extend the existing `set_backend` facade with the typed JAX
      convenience selector and resolve it through the same owner as environment
      selection. Define and publicly re-export `BackendMode`, `JaxDevice`, and
      `ExecutionIntent` `Literal` aliases. Use overloads that require `device`
      for the `Literal["jax"]` convenience form and expose only `None` for
      device/intent on the canonical-`BackendMode` form; apply the same contract
      to `use_runtime`. Type `BackendConfig.mode`, `BackendPolicy.mode`, and the
      pure resolver with `BackendMode`. The runtime implementation still rejects
      invalid dynamically typed calls. Provenance always records one of all six
      canonical modes, never the ambiguous convenience token `jax`.
    - [x] **RED:** Add `run_examples.py` parser and child-result tests proving
      `--device cpu|gpu` without `--intent` selects the matching fast mode,
      `--intent parity` selects the matching parity mode, an unavailable GPU
      cannot fall back to CPU, and mixed legacy/new selectors fail with a clear
      migration diagnostic. Start from parent environments containing each
      backend/platform/full-mode selector and prove the child environment is
      scrubbed and pinned to the CLI-selected profile. Prove legacy
      `--lane cpu-smoke` and `--lane gpu-strict` retain their current parity
      modes and emit a deprecation warning.
    - [ ] **RED:** Add manifest migration tests for absent/v1 and explicit-v2
      versions, unknown versions, mixed `lanes`/`devices`, per-example
      `intents`, semantic drift, deterministic candidate bytes, and a dry run
      that must not modify its input or repository.
    - [ ] **GREEN:** Implement the dual reader and
      `examples/jax/migrate_manifest.py --dry-run` without changing the
      canonical manifest. The dry run prints candidate v2 bytes, SHA-256,
      normalized semantic diff, and observed schema/adapter metadata. Run it
      against the committed v1 manifest and every compatibility fixture, retain
      the report, and prove the repository hash is unchanged.
    - [ ] **USER SIGN-OFF:** Present the dry-run report, exact candidate digest,
      one-release compatibility interval, observability fields, and rollback
      command. Stop before writing or activating v2 until the user explicitly
      approves the schema migration.
    - [ ] **GREEN after sign-off:** Evolve `examples/jax/run_examples.py`,
      `examples/jax/_lane_environment.py`, and the canonical
      `examples/jax/manifest.json` to the versioned `devices` capability field.
      The v2 production writer contains no `lanes` or per-example `intents`
      copy; the parser retains its read-only v1 adapter for legacy `lanes`, with
      fixture tests, through the documented compatibility interval. Legacy CLI
      aliases resolve independently of manifest storage. The runtime resolver
      alone maps `(device, intent)` to backend mode and certification
      eligibility; the lane-environment owner derives process guards from that
      profile. Record v1-adapter use and v2 activation in runner/CI receipts.
    - [ ] **RED:** Through the new runner interface, parameterize every ready
      example over CPU/GPU crossed with fast/parity. The immediate pre-GREEN
      revision must fail because the public selector/result-validation path is
      absent or parity-only, not because a scientific threshold is weakened.
      Require the same public solver family, objective convention,
      accepted-state publication, FP64 scientific dtype, and independent
      terminal-success checks. Compare scientific observables through centrally
      owned fast-versus-parity diagnostic tolerances without requiring bitwise
      equality or identical work counters.
    - [ ] **GREEN:** Remove any example-local assumption that parity is the only
      valid backend metadata. Fast-mode success remains fail-closed on wrong
      device, non-finite output, failed scientific gates, or hidden host solver.
      Both ordinary-runner profiles set strict fallback rejection explicitly;
      add a behavioral fallback-injection test instead of trusting a receipt
      label. The generic low-level full-mode API retains its documented
      `strict` option for compatibility. Only performance/reduction policy and
      certification eligibility differ.
    - [ ] **RED:** Inject `jax_cpu_fast` and `jax_gpu_fast` into parity-runner
      children and receipts and require arbitration/audit rejection. Also inject
      parent fast-mode environment values and prove `run_parity.py` overrides
      them with explicit parity child profiles rather than inheriting them.
    - [ ] **GREEN:** Keep `examples/jax/run_parity.py` parity-only. It accepts
      neither `--intent fast` nor fast receipts, and it continues to require
      `native-cpu`, `jax-cpu`, and `jax-gpu` certification lanes as selected.
    - [ ] **REFACTOR:** Use one immutable runtime profile resolver for mode and
      eligibility. Derive the separately owned subprocess guard overlay and
      result expectations from its output, and make CI exercise the public CLI
      rather than copy profile values. Keep parity comparison/tolerance routes
      out of the fast runner and performance tuning out of the parity arbiter.
    - [ ] **RED:** Add
      `tests/benchmarks/test_jax_example_execution_mode_contract.py` for a new
      `benchmarks/jax_example_execution_mode_contract.py` owner. Reject missing
      profiles, unmatched inputs, failed scientific repetitions, fewer than one
      empty-cache child plus one untimed cache warmup and seven balanced
      warmed-cache child pairs, unsynchronized timings, missing host RSS,
      unavailable GPU-memory evidence for GPU promotion, threshold edits without
      a schema/version change, and fast receipts presented as parity evidence.
    - [ ] **GREEN:** Add the dependency-light contract and
      `benchmarks/run_jax_example_execution_mode_benchmark.py`. On the same host
      and device, use byte-identical inputs, alternate fast/parity order, retain
      every outcome, and measure the real isolated example-child path. Give each
      profile an empty cache for one labeled cold end-to-end child, run one
      untimed warmup child, then collect seven synchronized warmed-cache child
      pairs. Do not call the cold measurement "compile time" unless separately
      instrumented. Compute paired medians plus a deterministic one-sided 95%
      bootstrap lower bound. Before collecting promotion data, check in the
      initial device-specific rule: every repeat passes the same scientific
      gates; warm median speedup is at least `1.05x` with lower bound at least
      `1.00x`; cold end-to-end time is no more than `1.25x` parity; peak host RSS
      and, on GPU, process-attributed device memory are each no more than
      `1.25x` parity; no child may OOM or receive a resource-limit kill, and
      dense materialization must respect the resolved
      `max_dense_jacobian_bytes`. GPU device memory marked `unavailable` is
      reportable diagnostics but cannot promote GPU fast as the default. The
      artifact records all repeats, cache identities,
      synchronization points, environment/source hashes, device identity, and
      metric owners. Add only `.artifacts/jax-example-execution-modes/` to
      `.gitignore`, with a narrow ignore regression; do not broaden the existing
      parity-artifact rule.
    - [ ] Run the matched contract on the checked-in representative IDs
      `traceable-least-squares`, `curve-length-optimization`,
      `surface-geometry-optimization`, `coil-flux-optimization`, and
      `fieldline-and-particle-tracing`, covering least-squares, scalar
      optimization, dense derivatives, field kernels, and tracing. Compare each
      pair on the same physical CPU host or the same single CUDA device and
      retain the CPU model or GPU model/UUID, driver, JAX/JAXLIB/XLA versions,
      clock/power policy when observable, and concurrent-load preflight. The
      initial GPU reference device is the project's RTX 5090 authority runner;
      other GPU models remain separately labeled evidence. If either reference
      device misses its gate, keep that device's existing default and repair its
      fast tuning; never weaken scientific gates, discard repeats, or relabel
      parity. These measurements support only the scoped default decision, not
      a general speedup claim.
    - [x] Update `examples/jax/README.md`, `docs/source/jax_gpu_setup.rst`, and
      `docs/source/jax_migration.rst` with the default, the four fast/parity
      profiles, retained float32-smoke mode, public API/CLI examples,
      legacy-lane deprecation, evidence eligibility, and rollback instructions.
      Record the behavior delta, caller inventory, dry-run, observability,
      explicit sign-off, and rollback evidence required by the Tier-4 gate.
    - [x] Update `.github/workflows/jax_smoke.yml` and its static reachability
      assertions so `jax-public-integration` runs
      `run_examples.py --device cpu` and the bounded explicit
      `--device cpu --intent parity` matrix, while `jax-gpu-strict-purity` runs
      `run_examples.py --device gpu` and the bounded explicit
      `--device gpu --intent parity` matrix with zero skips/fallbacks. Keep the
      complete three-lane certification matrix in the named scheduled/manual
      `.github/workflows/jax_gpu_parity.yml` authority job via `run_parity.py`.
      CI commands must spell `--intent parity` or use `run_parity.py`; job
      location alone never implies parity. The workflow tests must assert PR
      path triggers, exact commands, child modes, zero-skip GPU behavior, and
      that only parity jobs upload certification-eligible artifacts.

    Phase 10's all-ready scientific matrix becomes GREEN only after the known
    QFM feasibility/status repair in Phase 11. Resolver, API, CLI, schema, and
    parity-isolation slices may turn GREEN first, but they may not mark the
    four-profile example gate complete or weaken QFM to do so.

11. Repair the scientific, artifact-safety, and TDD-evidence blockers.
    - [ ] **RED:** Add a QFM authority regression using the retained terminal
      state: `abs(constraint_value)` is about `1.23158e-8` and must fail the
      centrally owned `terminal_constraint_norm_atol=1e-10` gate. Add separate
      tests proving a raw failed/budget-exhausted driver status cannot be
      serialized as normalized `converged`, even if another scientific
      acceptance predicate passes.
    - [ ] **GREEN:** Make QFM terminal success apply every central feasibility,
      residual, and gradient threshold. Preserve driver termination and
      scientific acceptance as separate typed fields; either make every
      required lane satisfy the reviewed convergence contract or downgrade the
      relationship instead of rewriting status. Publish finite nonnegative
      `nit`, `nfev`, and `njev` for the optimizing case, or revise applicability
      through a separately reviewed schema contract rather than emitting
      unreviewed nulls.
    - [ ] **RED:** Add manifest/arbiter/auditor tests requiring exactly the three
      direct lane routes for every observable marked applicable in a lane
      receipt. The retained traceable least-squares final
      `residual_jacobian` must expose the current three-route omission.
    - [ ] **GREEN:** Add the missing final-Jacobian routes and make route
      completeness schema-enforced before arbitration. Serialize the distinct
      half-squared `solver_cost` for least-squares lanes when the solver exposes
      it, without changing the public `r.T @ r` objective convention.
    - [ ] **RED:** Mutate the retained `input_bundle.json` and one input NPY
      sidecar without changing lane-receipt fingerprint strings; require the
      independent auditor to fail. This must expose the candidate auditor's
      summary/receipt string agreement without input-bundle reload.
    - [ ] **GREEN:** Make the auditor load the canonical bundle and every input
      sidecar, validate their hashes and metadata, recompute input/configuration
      fingerprints, and bind those recomputed values to every lane and the
      aggregate before numerical arbitration.
    - [ ] **REFACTOR:** Keep feasibility/status semantics, route completeness,
      and input-bundle validation in their existing single owners. Do not add
      case-local thresholds or a second audit-only comparison policy.
    - [ ] **RED:** Add a deterministic publish-time interleaving test that
      creates an empty final directory or symlink after the initial destination
      check but before publication. It must demonstrate that the candidate's
      check-then-`Path.rename()` can replace an existing empty destination.
    - [ ] **GREEN:** Replace check-then-rename with a logical-publication
      protocol that reserves the final run directory exclusively, never
      replaces it, and exposes the run to readers only after an atomically
      created completion marker. Fsync the completed tree, marker, final
      directory, and parent in durability order. A colliding writer must fail
      without modifying either writer's bytes.
    - [ ] **RED:** Add deterministic read and write tests that replace a checked
      path component or leaf with a symlink between validation and open. Prove
      the candidate pathname-based `_contained_path()` followed by `open()` or
      `np.load()` can access a substituted external file.
    - [ ] **GREEN:** Traverse artifact directories relative to trusted directory
      descriptors with no-follow semantics. Open each leaf once with the
      appropriate no-follow and exclusive-create flags, and hash, serialize, or
      load through that same descriptor. Do not validate one pathname and later
      reopen it by name. Fail explicitly on platforms that cannot enforce the
      contract rather than silently weakening it.
    - [ ] **REFACTOR:** Centralize the descriptor-safe filesystem operations and
      logical-publication protocol; keep artifact serialization, publication,
      and audit readers on the same primitive. Run concurrent stress tests only
      as supplementary evidence after the deterministic interleavings pass.
    - [ ] Recover immutable pre-GREEN revisions and exact commands, diagnostics,
      and exit codes for every promoted case's claimed authentic RED. The
      retained receipt currently provides table summaries and a generic
      placeholder, not the required per-case pre-GREEN identity and invocation.
      If that historical evidence does not exist, label those tests post-hoc
      regression coverage and leave this TDD plan blocked; never fabricate,
      retroactively relabel, or waive a RED within this plan.

12. Publish the repaired authority slice.
    - [ ] Integrate the repaired candidate into a reachable branch commit and
      confirm its parentage from the intended baseline. Do not cite detached
      `799c656e1` as shipped state.
    - [ ] From a clean checkout of the delivered commit, run the complete CPU
      suite, the real strict-GPU all-applicable run with zero skips/fallbacks,
      and the independent audit. Generate the results document from that new
      aggregate rather than copying the retained table.
    - [ ] Commit the implementation, manifests, workflows, tests, generated
      results, authentic receipt, and final status update as one reviewed
      delivery unit. Upload the immutable authority artifact to the selected
      durable store and record its retention and digest.

    Run the repaired authority from its clean delivered checkout with the
    project-pinned CUDA-capable interpreter. `PARITY_ARTIFACT_ROOT` must be the
    absolute staging path for the selected durable artifact store. The runner
    prints only the published run directory on success, so command substitution
    binds the subsequent audit and report to that exact run:

    ```console
    set -euo pipefail
    DELIVERED_SHA="$(git rev-parse HEAD)"
    PARITY_PYTHON="${PARITY_PYTHON:?set project-pinned CUDA interpreter}"
    PARITY_ARTIFACT_ROOT="${PARITY_ARTIFACT_ROOT:?set absolute artifact root}"
    test "${PARITY_ARTIFACT_ROOT#/}" != "$PARITY_ARTIFACT_ROOT"
    test -z "$(git status --porcelain=v1)"
    git merge-base --is-ancestor \
      b6775bf23030bafbd602ce3131dea948e7b8bb4b "$DELIVERED_SHA"
    RUN_DIRECTORY="$(
      SIMSOPT_PARITY_SIMSOPTPP_BUILD_COMMIT="$DELIVERED_SHA" \
      PYTHONPATH="$PWD/src:$PWD" "$PARITY_PYTHON" \
        examples/jax/run_parity.py \
        --case all-applicable \
        --lanes native-cpu,jax-cpu,jax-gpu \
        --artifact-root "$PARITY_ARTIFACT_ROOT"
    )"
    test -d "$RUN_DIRECTORY"
    PYTHONPATH="$PWD/src:$PWD" "$PARITY_PYTHON" \
      examples/jax/parity/audit.py \
      --run "$RUN_DIRECTORY" --repo-root "$PWD" --require-authoritative
    PYTHONPATH="$PWD/src:$PWD" "$PARITY_PYTHON" \
      examples/jax/parity/report.py \
      --summary "$RUN_DIRECTORY/summary.json" \
      --output docs/jax_native_example_parity_results.md
    sha256sum "$RUN_DIRECTORY/summary.json" \
      docs/jax_native_example_parity_results.md
    ```

    Record the two printed digests, exact interpreter/environment identity,
    durable artifact URI, and retention policy in the final receipt before
    changing this plan to `Done`.

## Validation Plan

- [ ] Fast-default and explicit-parity resolver/API matrix:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/test_backend_precision_policy.py \
    tests/test_jax_import_smoke.py \
    tests/integration/test_jax_execution_modes.py
  ```

  The test module must cover unset/default selectors, explicit full-mode
  precedence within each API/environment surface,
  `set_backend("jax", device=..., intent=...)`, `use_runtime` forwarding,
  `jax_cpu_float32_smoke` preservation, conflict rejection, cached
  configuration/provenance, and absence of JAX-import-order leakage. A fully
  unset environment must remain native CPU; only an explicitly selected JAX
  backend takes the new fast default.

- [ ] All ready examples on CPU fast and CPU parity:

  ```console
  python examples/jax/run_examples.py --device cpu
  python examples/jax/run_examples.py --device cpu --intent parity
  ```

- [ ] All ready examples on a real GPU in fast and parity modes, with zero
  skips and no CPU fallback:

  ```console
  python examples/jax/run_examples.py --device gpu
  python examples/jax/run_examples.py --device gpu --intent parity
  ```

  Fast must report `jax_gpu_fast`; parity must report `jax_gpu_parity` and the
  strict transfer/determinism contract. Both must report FP64 and pass the same
  independent scientific checks.

- [ ] Legacy runner compatibility and migration diagnostics:

  ```console
  python examples/jax/run_examples.py --lane cpu-smoke
  python examples/jax/run_examples.py --lane gpu-strict
  ```

  Both aliases retain their historical parity behavior during deprecation and
  emit the documented warning. Combining `--lane` with `--device` or
  `--intent` must fail before child execution. Parameterized tests seed every
  parent selector with conflicting values and prove the runner scrubs them
  before pinning the requested child profile.

- [ ] Tier-4 manifest migration dry-run and sign-off gate:

  ```console
  python examples/jax/migrate_manifest.py \
    --input examples/jax/manifest.json --dry-run
  ```

  Before and after hashes of the committed v1 input and repository diff must be
  identical. Retain the printed v2 candidate digest, semantic diff, schema
  version, legacy-adapter signal, compatibility duration, and rollback command.
  The canonical manifest remains v1 until the user explicitly signs off on
  those exact bytes; after activation, CI replays both v1 fixtures and canonical
  v2 and records reader-version observability.

- [ ] Parity isolation tests prove fast modes cannot enter authoritative
  artifacts:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/integration/test_jax_example_parity_runtime.py \
    tests/integration/test_jax_example_parity_runner.py \
    tests/integration/test_jax_execution_modes.py
  ```

- [ ] Matched fast-mode promotion contract:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/benchmarks/test_jax_example_execution_mode_contract.py
  python benchmarks/run_jax_example_execution_mode_benchmark.py \
    --device cpu --artifact-root .artifacts/jax-example-execution-modes
  python benchmarks/run_jax_example_execution_mode_benchmark.py \
    --device gpu --artifact-root .artifacts/jax-example-execution-modes
  ```

  The contract test owns the checked-in workload IDs, balanced seven-pair
  schedule, scientific admission checks, timing statistic, memory metric
  availability, provenance schema, and device-specific promotion thresholds.
  Both benchmark runs must retain every repeat and pass the checked-in rule
  before that device changes its default. Benchmark artifacts are explicitly
  non-certifying.

- [ ] Static workflow routing proves the PR-required CPU/GPU fast and bounded
  parity commands plus the scheduled/manual full parity authority command:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/test_jax_examples_manifest.py \
    tests/test_jax_example_parity_manifest.py
  ```

- [x] Manifest and schema RED/GREEN tests:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/test_jax_example_parity_manifest.py \
    tests/integration/test_jax_example_parity_artifacts.py \
    tests/integration/test_jax_example_parity_inputs.py \
    tests/integration/test_jax_example_parity_publication.py \
    tests/integration/test_jax_example_parity_runner.py \
    tests/integration/test_jax_example_parity_runtime.py
  ```

- [x] Existing manifest/runner compatibility:

  ```console
  MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
    tests/test_jax_examples_manifest.py \
    tests/integration/test_jax_examples.py
  ```

- [x] Wave A native/JAX CPU artifact:

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

- [x] Wave A real strict-GPU artifact on the designated CUDA environment:

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

- [x] Full applicable registry run after every wave is implemented:

  ```console
  conda run --no-capture-output -n "${BENCHMARK_ENV_NAME:?set CUDA environment}" \
    python examples/jax/run_parity.py \
    --case all-applicable \
    --lanes native-cpu,jax-cpu,jax-gpu \
    --artifact-root .artifacts/jax-example-parity
  ```

- [ ] Final artifact audit verifies every required lane, exact input/config
  and effective-construction fingerprint agreement, route completeness,
  completed scientific
  stages, source path/revision and executed-source hashes, any loaded
  `simsoptpp` binary hash/compatibility receipt, FP64/device and effective
  transfer-guard metadata, sidecar SHA-256, path containment,
  authoritative-source status, observable applicability, tolerance result, and
  final verdict. The candidate auditor recomputes all declared comparison
  routes and validates receipt fingerprint agreement, but it does not reload
  and re-hash the retained input bundle or prove applicable-observable route
  completeness. The local replay below therefore validates only that narrower
  candidate contract; it does not rerun the scientific lanes. It requires the generated,
  ignored `src/simsopt/_version.py` whose bytes were executed by the authority
  run. The following command reproduced the 8-case/24-receipt/228-comparison
  `pass` from this host's retained artifact:

  ```console
  (
    set -euo pipefail
    PARITY_AUDIT_ROOT="$(mktemp -d)"
    cleanup_parity_audit() {
      git worktree remove "$PARITY_AUDIT_ROOT/checkout" 2>/dev/null || true
      rmdir "$PARITY_AUDIT_ROOT" 2>/dev/null || true
    }
    trap cleanup_parity_audit EXIT
    git worktree add --detach "$PARITY_AUDIT_ROOT/checkout" \
      799c656e186642bdb7e296e46ad1c6cd61277839
    cd "$PARITY_AUDIT_ROOT/checkout"
    uvx --from setuptools-scm==10.2.1 setuptools-scm \
      --force-write-version-files
    test "$(sha256sum src/simsopt/_version.py | cut -d' ' -f1)" = \
      42b6fe4a2ed69ed6234ac4e32af34c849c97b18e96a4a3e33aeaab54dc7612ef
    PYTHONPATH="$PWD/src:$PWD" \
      /home/jungdaesuh/simsopt_mixed_artifacts/v0c_62a262b09c_20260715T2142Z/runtime-env/bin/python \
      examples/jax/parity/audit.py \
      --run /home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.artifacts/jax-example-parity/20260726T225943Z-09dfdc3e \
      --repo-root "$PWD" --require-authoritative
  )
  ```

  This recipe is intentionally local and historical: the interpreter and
  artifact paths are host-specific. Phase 12 must replace it with a durable
  environment and artifact reference for shipped evidence.
- [x] Run focused native/JAX subsystem tests named by each case, including
  curve/surface objectives, Biot-Savart/flux, QFM, permanent magnets, tracing,
  Boozer, wireframe, force, and finite-build coverage.
- [x] Run strict-GPU focused tests in a real CUDA process with zero skips and no
  CPU fallback.
- [x] Static/type/format checks:

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

- [x] Import/source-boundary checks:

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
  unrelated existing modifications remain intact. Repeat this for the repaired
  delivered slice; the detached candidate and current worktree are not that
  final delivery.

## Risks and Mitigations

- Risk: Making fast the default accidentally weakens scientific correctness.
  Mitigation: Both intents retain FP64, the same solver/objective family,
  fail-closed accepted-state publication, and terminal scientific gates. Only
  tuning, reproducibility policy, transfer enforcement, and evidence eligibility
  differ.

- Risk: A fast run is mislabeled or promoted as parity evidence.
  Mitigation: Persist the explicit low-level mode and certification-eligibility
  field, reject fast modes in the parity runner/arbiter/auditor, and require
  parity commands to opt in before child creation.

- Risk: Device and intent policy is duplicated across runtime, examples, tests,
  and CI, causing drift.
  Mitigation: One pure runtime resolver maps `(device, intent)` to mode and
  eligibility; the example environment owner derives only process-isolation
  overlays, result validation consumes the resolved profile, and CI calls the
  public CLI. Ratchet tests reject copied four-mode mappings.

- Risk: Presubmit runs only the fast default and parity silently regresses.
  Mitigation: Keep bounded explicit CPU parity and strict-GPU parity jobs in
  presubmit, plus the complete scheduled/manual authority run. CI names and
  environment location are never substitutes for an explicit parity selector.

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
  Target mitigation: canonical JSON plus one hash-bound `.npy` file per array,
  descriptor-relative no-follow I/O, exclusive logical publication, and an
  atomic completion marker. Phase 11 must make readers reject replacement
  races as well as pre-existing symlinks, traversal, containment escapes, and
  incomplete runs.

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

- [ ] Explicitly selecting JAX CPU or GPU without an intent resolves to
  `jax_cpu_fast` or `jax_gpu_fast`; a fully unset selector remains
  `native_cpu`, and explicit full modes retain their documented precedence
  within the API and environment surfaces.
- [ ] The public API and example CLI support CPU/GPU crossed with fast/parity,
  default intent to fast, reject ambiguous selectors, and retain documented
  legacy lane behavior for the deprecation interval. The existing explicit
  `jax_cpu_float32_smoke` mode remains supported outside this convenience
  cross-product.
- [ ] Manifest v1/v2 readers, no-write dry-run, semantic replay, observability,
  rollback, and explicit user sign-off satisfy the Tier-4 migration gate before
  canonical v2 activation; absence of `schema_version` has only the documented
  v1 meaning.
- [ ] Every ready example passes CPU fast, CPU parity, real-GPU fast, and
  real-GPU parity with FP64 and the same scientific-success contract; fast and
  parity may use their centrally owned tuning and diagnostic tolerances.
- [ ] Fast receipts are explicitly non-certifying, and injected fast modes are
  rejected by the parity runner, arbiter, auditor, report generator, and CI
  authority gates.
- [ ] Matched CPU and GPU artifacts demonstrate that the selected fast policy
  meets the checked-in paired timing and memory promotion rule without changing
  scientific outcomes; every repeat is retained, and no speed claim relies on
  a single cold run.
- [ ] Runtime, example runner, manifest validation, result validation, and CI
  consume the pure runtime profile resolver; process isolation remains owned by
  one example-environment module, and no duplicated four-mode mapping remains.
- [x] Every ready JAX example/native `inspired_by` relationship has exactly one
  validated `full`, `reduced`, or `unsupported` parity classification.
- [ ] Every `full` case proves byte-identical inputs and configuration across
  native CPU, JAX CPU, and applicable JAX GPU lanes. Candidate inputs were
  independently checked, but the authority auditor does not yet derive this
  proof from the retained bundle.
- [x] Every `full` relationship has a complete declared scientific-stage list,
  no omitted scientific stage, and matching completed-stage receipts from all
  required lanes; reduced and teaching-only exclusions remain explicit.
- [x] Every applicable case passes required initial objective/residual,
  gradient/Jacobian, and constraint comparisons through centrally owned
  tolerances.
- [ ] Every optimizing full case passes final parameters, objective/residual,
  feasibility, normalized convergence, raw-status recording, and work-counter
  reporting gates. The traceable least-squares full case lacks final-Jacobian
  route completeness and the promised solver-cost field.
- [x] Every tracing or fixed-state full case passes its explicitly applicable
  endpoint/invariant or objective/gradient contract with optimizer fields
  explicitly marked not applicable.
- [ ] All ready applicable JAX cases pass the scientific contract on a real
  CUDA device in FP64 with
  both effective transfer-guard settings at `disallow` and no CPU fallback or
  hidden host solver. All candidate lanes executed with the required GPU
  policy, but QFM did not satisfy feasibility/status requirements.
- [ ] Aggregate JSON and NPY sidecars are source/input/environment hash-bound,
  schema-valid, race-safe, durably retained, and sufficient for an independent
  reviewer to recompute every retained verdict. Candidate recomputation passes;
  race safety and durable retention remain open.
- [ ] Authentic RED -> GREEN -> REFACTOR receipts exist for every promoted
  case; focused tests, existing example regressions, Ruff, format, compileall,
  import boundaries, and `git diff --check` pass. Current case receipts do not
  prove failing-first order at the required per-case granularity.
- [ ] Delivered documentation reports full, reduced, unsupported, CPU-only, and
  strict-GPU evidence separately; no analytic-only or reduced result is called
  native/C++ end-to-end parity. Candidate wording is corrected here, but the
  generated results and receipt updates remain outside current branch history.
- [x] Unrelated dirty/untracked work remains unchanged.

## Open Questions

- Should “full example parity” require the original native script's default
  full-resolution run, or is a bounded matched workflow with all stages and
  identical inputs sufficient for a `full` workflow-coverage classification in
  presubmit while defaults run in scheduled CI? This plan assumes the latter,
  records the bounded scale independently, and reserves native-default parity
  claims for passing scheduled/manual evidence.
- What absolute wall time, host RSS, and device-memory budgets should determine
  whether parity Waves B-D run in presubmit versus scheduled CI? This scheduling
  decision is separate from Phase 10's checked-in paired fast-mode promotion
  ratios and cannot weaken them.
- For algorithm-different solvers, should work-counter regressions remain
  reporting-only, or should each case define a reviewed maximum ratio after a
  stable baseline exists?
- How long should CI retain JSON/NPY parity artifacts, and which artifact store
  is authoritative for release claims?
