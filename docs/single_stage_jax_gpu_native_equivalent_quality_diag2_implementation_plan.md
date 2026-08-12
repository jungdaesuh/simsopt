# Single-Stage JAX GPU Native-Equivalent DIAG2 Plan

**Status:** Pre-execution design and qualification
**Last updated:** 2026-08-11

## Purpose

This file is the single source of truth for `NEQ-GNTR1-DIAG2`, a
crash-safe, resource-isolated successor to the failed `NEQ-GNTR1-DIAG1`
evidence route. DIAG2 must retain the unchanged `NEQ-GNTR1` numerical route
while ensuring that the long-lived supervisor owns no allocation on the RTX
5090 and that every supervised terminal outcome produces an immutable,
independently reloadable artifact.

`NEQ-GNTR1-DIAG2` names an evidence producer and receipt schema only. It is
not a new optimizer, physical model, objective, quality predicate, or
performance route.

## Prior Failure and Successor Boundary

The single DIAG1 cold authorization was consumed at
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag1-prospective`. Its
preflight passed, but its cold child crashed after repeated CUDA allocation
failures. The exact child retained a peak of `4,984,930,304 B`
(`4.642578125 GiB`). Contemporaneous non-artifact observation found the
long-lived supervisor holding approximately `24.6 GiB` on the same GPU.

The executed runner called the JAX-based fullspace bootstrap and scaling
construction in the persistent parent before launching either child. This
initialized the CUDA backend and retained the parent allocation. The child
then failed with `CUDA_ERROR_OUT_OF_MEMORY` followed by
`CUDA_ERROR_LAUNCH_FAILED`. This is an infrastructure/resource-isolation
failure, not evidence about GNTR convergence or terminal physics.

DIAG1 also represented a non-producing crashed child as canonical `{}` while
asserting the producer schema on its `ArtifactRef`. The incomplete-receipt
builder resolved that false typed reference before deriving `COLD_CRASH`, so
the tree never received `diagnostic.json`, an exact manifest, or its seal.

DIAG1 bytes and its writable incomplete tree are historical failure evidence.
They must not be mutated, chmod'ed, backfilled, promoted, or rerun. DIAG2 uses
a new plan, route/schema identity, source snapshot, and absent output root.

## Goals

- Keep the persistent supervisor GPU-zero through both authorization gates.
- Derive the exact parent policy authority with a pure-NumPy reconstruction
  from the independently validated native-reference evidence.
- Prove by raw parent-PID/UUID observations that the supervisor owns no GPU
  process allocation immediately before preflight and immediately before cold.
- Represent missing child producer bytes as an absent optional reference,
  never as an empty object carrying an asserted producer schema.
- Seal and independently reload every compile failure/OOM, timeout, crash,
  protocol failure, monitor failure, source failure, and numerical-evidence
  failure.
- Preserve the complete DIAG1 numerical telemetry contract when the cold child
  completes: all 300 history slots, unconditional terminal evidence, seven
  phase scopes, FP64, 716 DOFs, 255 equalities, native-reference binding, and
  zero guarded hot transfers/callbacks.
- Execute exactly one lower/compile-only RTX 5090 preflight and, only after a
  strict raw-evidence gate passes, exactly one pristine cold diagnostic.

## Non-Goals

- Changing any optimizer expression, GN model, HVP, projection, correction,
  acceptance predicate, trust-radius update, termination order, tolerance,
  scaling definition, quality predicate, physical equation, objective term,
  DOF, equality, or continuation branch.
- Warm execution, tuning, replacement, optimizer/cold rerun, a second cold, A100 execution,
  endpoint promotion, formal speed comparison, or scientific convergence
  claims.
- Inferring an algorithmic next route from a crash, incomplete trace, missing
  producer, or otherwise incomplete numerical evidence.
- Repairing or canonically relabeling the failed DIAG1 tree.

## Frozen Identities and Numerical Contract

Pre-DIAG2 live identities:

- repository HEAD: `52dea17ddf3012cf923fc92da78c0d73a17f4625`;
- executed DIAG1 source manifest:
  `d33001f37fadd3b06d04a1fa3ac6f51075afe9da9c400efe2c3558c9c2ba6cfd`;
- DIAG1 outcome-ledger document:
  `f86897888b1c92baab791bf1d411e97fc177adda248e0ade902bc30a71215133`;
- runner:
  `bde4238914353c5bfdf68be0fac1b93c987099f76060b958dd39720d2b0d876c`;
- DIAG1 receipt:
  `54f0b2134e9cd138b80b47e550822726313ae7421285e38611d2f74cd200c53c`;
- snapshot authority:
  `985ddf4b61ab4ecdf1ab6c0e130a1ed98d7e807f233f76ba27fb2bf74f247374`;
- trace annotations:
  `9d50e5fca9dddc8b933f5039beb0ed5f25339dea78e2c5a12bacf67489881ea7`;
- projected GNTR core:
  `c28a598a56eae109b3e61f846ae58c34b97a2cdc5fe92fdb15af0a668eb380de`;
- NEQ adapter:
  `abf9726e487eb4bda9f82c6092415e988e5a346383c89cec732fe7185b6e6fac`;
- fullspace GNTR bridge:
  `62b7dec2194f7c381d676abeed852ff1c4acba9e1a5f8d764a845abcd040f436`;
- fullspace scaling:
  `475cb63ddc183e343c1ae40faf7e0abf8bad5e6c288eabe38d31ce416e18cde4`;
- fullspace objective:
  `ca3a09f57fcabe4e448b9c50256bf28cc3750005cf52199ace2061d3e55f19fd`;
- fullspace bootstrap adapter:
  `910b59131cc9137fee65a8d14222eeccbc0cf3d61d300a63250a95469c413e4e`;
- native endpoint adapter:
  `bad745833c598072e3b205599fd55eb4e35dec61e87fa6679552b8343d9d2934`;
- native-reference validator:
  `faf7614ad827e3603b1ba8e4a792394e50fb8be2146bff5bb34f002cb41d96e6`;
- input-bundle loader:
  `303439ea4dcf9b444ad3410c088fb17bc25cd4701dabc88bcf5106faf9a8e87b`;
- native Boozer-QA case:
  `3bf7c04ec64b340a7dbb8c08b0cd55cbc0bbd0cb41942976cd33451087894832`;
- native-reference manifest:
  `5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db`.

The core, adapter, trace scopes, input bundle, native reference, and all
physics/objective source reachable by the numerical child are frozen. DIAG2
may change only runner/supervisor, snapshot environment authority, diagnostic
receipt/schema, their tests, and this SSOT unless a new reviewed plan revision
explicitly authorizes otherwise.

Qualification and production both construct
`frozen-numerical-subset.json`, schema
`single-stage-neq-gntr1-frozen-numerical-subset-v1`, as a canonical sorted
array of exact `{relative_path, sha256}` entries for only these paths:

- `src/simsopt_jax/runtime/trace_annotations.py`;
- `src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py`;
- `src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py`;
- `src/simsopt_jax/solve/fullspace_native_equivalent_quality.py`;
- `src/simsopt_jax/solve/fullspace.py`;
- `src/simsopt_jax/objectives/single_stage_fullspace.py`;
- `src/simsopt_jax_adapters/geo/single_stage_fullspace.py`;
- `src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py`;
- `benchmarks/single_stage_native_equivalent_reference.py`;
- `examples/jax/parity/input_bundle.py`;
- `examples/jax/parity/cases/native_boozerqa.py`.

The hashes are the corresponding values listed above; runner, diagnostic
receipt, snapshot authority, tests, and SSOT are explicitly excluded from this
frozen subset. The receipt resolves every entry against the immutable source
snapshot and rejects any delta. It also compares the new source
snapshot with the executed DIAG1 manifest under an explicit delta allowlist
containing only the runner, diagnostic receipt, snapshot authority, their
direct tests, `benchmarks/process_gpu_monitor.py`,
`benchmarks/single_stage_fullspace_process_gpu_monitor.py`, their direct tests,
and this SSOT. Added or changed numerical-source paths outside
that allowlist are blocking even though the full new snapshot is separately
content-addressed.

The numerical contract remains:

- state size 716, equality size 255, residual size 2110;
- FP64/x64 authoritative state, residual, Jacobian, gradient, multiplier, and
  equality evidence;
- objective maximum `4.4822246533126125e-08`;
- scaled feasibility maximum `1e-10`;
- componentwise raw equality bound
  `abs(q_gpu[i]) <= abs(q_native[i]) + 1e-12 + 1e-10*abs(q_native[i])`;
- `maximum_accepted_steps=256`, `maximum_attempts=300`, and every other frozen
  GNTR option unchanged;
- fatal, first quality hit, 256 accepts, then 300 attempts termination order;
- no callback or host transfer inside the synchronized loop; exactly one
  declared post-timing terminal transfer for complete numerical evidence.

## Design-It-Twice Decision

### Parent policy authority

**Alternative A: pure-NumPy reconstruction from the validated reference.
Selected.** The frozen equality order is 254 Boozer components followed by
`signed_volume - volume_target`. The validated reference retains both its
signed-volume observable and its raw equality vector, so the parent derives

```text
volume_target = reference_volume - native_raw_equalities[254]
D[0:254] = 1 / sqrt(254)
D[254] = 1 / abs(volume_target)
```

in explicit little-endian FP64 NumPy. For the frozen reference this must give
`volume_target=-0x1.296a9ce4a271dp-2`,
`D[0]=0x1.0101828467ee9p-4`, `D[254]=0x1.b8b3b0469c959p+1`,
constraint-scale raw SHA-256
`ee71932a5d6a0dfb0ca4dc9d852bf1f32e669dbc81ced26903db956027e1155e`,
and policy SHA-256
`6face7116d36d2eae954bb5b3bde465f37c990ceb9b319ba2c20bb62ad1a6f99`.
The receipt independently repeats the derivation from validated reference
bytes. Any mismatch blocks before preflight. This duplicates only the frozen
three-line scaling identity, not the Boozer solve or physical model.

**Alternative B: one short CPU-only JAX policy-authority subprocess.** This
would reuse the bootstrap implementation, but adds a third process, internal
mode, runtime schema, CPU compilation cost, and extra sealing/failure surface.
It is retained only as a separately reviewed fallback if the exact NumPy
identity cannot reproduce the frozen bytes or an import-only canary allocates
the GPU.

### Parent GPU-zero proof

**Alternative A: infer zero use from environment variables or lack of a JAX
device call.** Rejected. Environment intent is not allocation evidence.

**Alternative B: exact parent-PID/physical-UUID observations. Selected.**
The supervisor queries `nvidia-smi` without initializing JAX and retains the
raw query bytes plus parsed rows. A gate passes only when the exact supervisor
PID is absent from compute applications on the frozen GPU UUID. Observations
are taken after pure-NumPy policy reconstruction completes and immediately before
preflight, then again after preflight exits and immediately before cold. The
receipt independently parses and binds PID, UUID, timestamp, argv/query,
stdout/stderr hashes, and zero matching rows. Any missing, ambiguous, failed,
or nonzero observation prevents the next GPU child.

### Missing producer evidence

**Alternative A: always write `producer.json`, using `{}` on failure.**
Rejected because physical existence is not schema validity and caused DIAG1's
unsealed failure.

**Alternative B: optional producer reference. Selected.**
`SupervisedSample.producer` is `Mapping | None` internally. A producer file and typed
reference are published only after exit code zero, canonical JSON parsing,
exact object type, and expected schema validation. Timeout, crash, launch
failure, malformed/empty stdout, and monitor failure retain raw stdout/stderr,
process, child terminal, and memory evidence when available. Their producer
slot is `ABSENT` except that an exit-zero monitor-finalization failure may
retain a valid parsed producer. A post-launch source-post failure likewise
preserves any independently valid producer. Preflight success requires a
`PRESENT` valid producer;
failure receipt construction never resolves an absent producer.

### Failure publication

**Alternative A: build the complete receipt first and fall back on any
exception.** Rejected because malformed optional evidence can fail before the
failure taxonomy and leave the output writable.

**Alternative B: derive terminal class first from mandatory raw supervision
evidence, then validate only evidence applicable to that class. Selected.** A
single `finally`-equivalent publication boundary is not introduced with a
broad catch; instead every explicit supervisor outcome returns a typed raw
result and flows through one total receipt builder. The incomplete builder's
required inputs are process/terminal and source/resource authorities that
exist for that stage. Optional producer/numerical/trace slots carry exact
`ABSENT` reasons.
Manifest construction role-closes every physically retained regular file,
including partial traces and opaque source bytes under their exact failure
roles. Publication writes the receipt and manifest, chmods every file 0444 and
directory 0555, and deep-loads the artifact before returning.

## DIAG2 Route and Schema

- evidence route: `NEQ-GNTR1-DIAG2`;
- numerical route: `NEQ-GNTR1`;
- receipt schema: `single-stage-neq-gntr1-no-hit-diagnostic-v2`;
- manifest schema:
  `single-stage-neq-gntr1-no-hit-diagnostic-artifact-manifest-v2`;
- policy-authority schema: `single-stage-neq-gntr1-policy-authority-v2`;
- supervisor-resource schema:
  `single-stage-neq-gntr1-supervisor-gpu-zero-v1`.

DIAG2 loaders must exact-parse the version they claim. DIAG1 constants and
historical bytes remain distinguishable; no DIAG2 loader may silently accept a
v1 object as v2. Because DIAG1 never produced a canonical artifact, no
migration or rewrite is authorized.

### Exact v2 evidence slots and paths

The receipt has exactly these evidence slots and canonical paths:

| Slot | Canonical path |
| --- | --- |
| `source_manifest` | `source-snapshot/source-manifest.json` |
| `frozen_numerical_subset` | `frozen-numerical-subset.json` |
| `native_reference` | `native-reference/reference.json` |
| `policy_authority` | `policy-authority.json` |
| `supervisor_before_preflight` | `supervisor/before-preflight.json` |
| `preflight_producer` | `preflight/producer.json` |
| `preflight_terminal` | `preflight/terminal.json` |
| `preflight_process` | `preflight/process.json` |
| `preflight_memory` | `preflight/gpu-memory.json` |
| `preflight_memory_samples` | `preflight/gpu-memory-samples.json` |
| `preflight_runtime` | `preflight/runtime-evidence.json` |
| `preflight_policy` | `preflight/policy.json` |
| `supervisor_before_cold` | `supervisor/before-cold.json` |
| `cold_producer` | `cold/producer.json` |
| `cold_terminal` | `cold/terminal.json` |
| `cold_process` | `cold/process.json` |
| `cold_memory` | `cold/gpu-memory.json` |
| `cold_memory_samples` | `cold/gpu-memory-samples.json` |
| `cold_runtime` | `cold/runtime-evidence.json` |
| `cold_policy` | `cold/policy.json` |
| `cold_history` | `cold/history.json` |
| `cold_terminal_numerical` | `cold/terminal-numerical.json` |
| `cold_raw_trace` | the sole Chrome file below `cold/raw-trace/plugins/profile/<run>/` |
| `cold_trace_intervals` | `cold/trace-intervals.json` |
| `execution` | `execution.json` |
| `supervisor_terminal` | `supervisor-terminal.json` |

Every slot serializes as the `PRESENT`/`ABSENT` union below. The supervisor
terminal is always `PRESENT` for every handled run after staging creation.
Complete numerical evidence requires every slot `PRESENT`. An outcome-specific
manifest contains exactly the paths reachable from its `PRESENT` slots plus
their nested raw streams, terminal arrays, immutable source snapshot, copied
native reference, and `diagnostic.json`. Except for the closed retained-failure
table below, `ABSENT` slots contribute no path or role. The raw-trace role additionally requires exactly one matching
Chrome/XPlane basename pair; a handled partial-trace failure role-closes the
single retained sibling and keeps `cold_raw_trace` `ABSENT`.

Fixed top-level roles are the slot names above except that the
`cold_raw_trace` Chrome artifact has role `raw_trace_chrome`; its mandatory
XPlane sibling has role `raw_trace_xplane`. Additional fixed role
`diagnostic.json -> diagnostic_receipt` applies. `artifact-manifest.json` is the
separately validated manifest root and is excluded from its own entries and
from the observed regular-file comparison. Nested roles are
`preflight_stdout`, `preflight_stderr`, `cold_stdout`, `cold_stderr`,
`before_preflight_gpu_inventory_stdout`,
`before_preflight_gpu_inventory_stderr`,
`before_preflight_compute_apps_stdout`,
`before_preflight_compute_apps_stderr`, `before_cold_gpu_inventory_stdout`,
`before_cold_gpu_inventory_stderr`, `before_cold_compute_apps_stdout`,
`before_cold_compute_apps_stderr`, `terminal_array`,
`source_snapshot`, `source_snapshot_opaque_failure`, `native_reference`,
`native_reference_opaque_failure`,
`raw_trace_chrome`, `raw_trace_xplane`, `invalid_setup_authority_failure`, and
`untyped_evidence_failure`.
Role/path equality is exact; role
aliases and unknown roles fail.

### Exact new payloads

`frozen-numerical-subset.json` has exactly `schema_version`, `plan_sha256`, and
`entries`; entries are the sorted exact `{relative_path, sha256}` rows defined
above. It is published immediately after the source snapshot. Failure to
construct or validate it is `SOURCE_PUBLICATION_FAILURE` with reason
`FROZEN_SUBSET_INVALID`.

`policy-authority.json` has exactly: `schema_version`, `route`, `plan_sha256`,
`derivation_kind` (literal `VALIDATED_REFERENCE_NUMPY_FP64`),
`native_reference`, `reference_volume`, `volume_target`,
`native_raw_equalities`, `native_raw_equalities_sha256`,
`constraint_inverse_scale`, `constraint_inverse_scale_sha256`,
`objective_target`, `state_size`, `equality_size`,
`objective_residual_size`, `component_absolute_tolerance`,
`component_relative_tolerance`, `scaled_feasibility_tolerance`,
`residual_value_defect_tolerance`, `residual_gradient_defect_tolerance`,
`transpose_defect_tolerance`, `gntr_options`, and `policy_sha256`.

Each supervisor-zero document has exactly: `schema_version`, `route`,
`plan_sha256`, `stage` (`BEFORE_PREFLIGHT` or `BEFORE_COLD`),
`captured_at_monotonic_ns`, `captured_at_unix_ns`, `supervisor_pid`,
`supervisor_start_ticks`, `gpu_uuid`, `visible_device`, `gpu_inventory_query`,
`compute_apps_query`, and `matching_rows`. Each query object has exactly
`argv`, `query_executable_sha256`, `launched`, `timed_out`, nullable
`returncode`, `stdout`, and `stderr`. `stdout`/`stderr` are exact ArtifactRefs.
The literal argvs are:

```text
nvidia-smi --query-gpu=uuid,memory.total --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits
```

The inventory query must identify the frozen UUID exactly once and supplies
its physical memory independently of the possibly empty compute-app query.
The compute-app query supplies zero or more rows, each with exactly `pid`,
`gpu_uuid`, and `used_memory_mib`. Launch failure is
`launched=false,timed_out=false,returncode=null`; timeout is
`launched=true,timed_out=true,returncode=null`; success requires
`launched=true,timed_out=false,returncode=0`; command failure is
`launched=true,timed_out=false,returncode!=0`. Every unsuccessful or malformed
query requires `matching_rows=[]` because its output is non-authoritative.
Successful inventory requires exactly one frozen-UUID row. Successful compute
parsing retains every row and derives `matching_rows` as exactly those rows
whose PID and UUID equal the supervisor identity and frozen UUID; the gate
requires that derived tuple empty.

Raw query paths and schemas are exact:

- `supervisor/before-preflight-gpu-inventory.stdout.bin` and `.stderr.bin`;
- `supervisor/before-preflight-compute-apps.stdout.bin` and `.stderr.bin`;
- `supervisor/before-cold-gpu-inventory.stdout.bin` and `.stderr.bin`;
- `supervisor/before-cold-compute-apps.stdout.bin` and `.stderr.bin`.

Their roles are the stage/query/stream names and their schemas are respectively
`raw-supervisor-gpu-inventory-stdout-v1`,
`raw-supervisor-gpu-inventory-stderr-v1`,
`raw-supervisor-compute-apps-stdout-v1`, and
`raw-supervisor-compute-apps-stderr-v1`.

`supervisor-terminal.json` has exactly: `schema_version`, `route`,
`plan_sha256`, `disposition` (`COMPLETE` or `INCOMPLETE`), nullable
`failure_stage`, nullable structured `failure_reason`, `launched_children`,
`policy_authority_produced`, `preflight_authorized`, `cold_authorized`,
`publication`,
`engineering_campaign_receipt_produced` (literal false),
`promotion_authorized` (literal false), `formal_comparison` (literal
`NOT_PRODUCED`), and `algorithm_route_selection` (literal `NOT_PRODUCED` for
incomplete evidence). A failure reason has exactly `code` and
`detail_sha256`.

`publication` has exactly `staging_root`, `final_root`, and `nonce`. Both roots
share the same resolved, nonsymlink parent; `nonce` is 32 lowercase hex digits;
the staging basename is exactly `<final-basename>.partial-<nonce>`. Every
recorded artifact-local absolute path is historical execution evidence. Child
cwd, snapshot import/module origins including the copied `simsoptpp` native
extension, snapshot runner argv operand, and the path operand following
`--reference` must be rooted at that exact staging string. Serialized
ArtifactRef paths remain canonical relative strings: they resolve beneath
staging during prepublication validation and beneath final after rename with
identical relative path/hash/size/schema. External absolute authorities are not
relocated: argv[0]/runtime Python executable, `--input-root`, source worktree
repository root, and `nvidia-smi` executable must instead equal their separately
frozen or independently validated identities. The original native-extension
source-copy path is a pre-output input only; the final artifact certifies the
copied extension through its immutable source-manifest path and hash and makes
no unretained original-path claim.
The final loader requires its own resolved path to equal `final_root` and
rechecks both the artifact-local relocation relation and external identities.

`launched_children` is exactly `[]`, `["preflight"]`, or
`["preflight","cold"]` in that order. `COMPLETE` requires null failure fields,
both authorization booleans true, both children, policy authority true, and
`algorithm_route_selection` in exactly `RETRY_MODEL_REUSE`,
`RADIUS_RETRACTION`, or `CONDITIONING_MODEL_CHANGE`. `INCOMPLETE` requires
non-null stage/reason, `algorithm_route_selection=NOT_PRODUCED`, and booleans
that exactly match the reached schedule prefix.

`diagnostic.json` has exactly `schema_version`, `route`, `numerical_route`,
`plan_sha256`, `evidence_slots`, `verdict`, `historical_relation`, `quality`,
`phase_attribution`, `next_route`, `failure`,
`engineering_campaign_receipt_produced`, `promotion_authorized`, and
`formal_comparison`. Complete evidence requires `failure=null`, non-null
quality/phase attribution, and the same three-value next-route domain.
Incomplete evidence requires non-null failure, null quality/phase attribution,
`next_route=NOT_PRODUCED`, both campaign/promotion booleans false, and
`formal_comparison=NOT_PRODUCED`.

`verdict` is exactly `DIAGNOSTIC_COMPLETE_NO_HIT`,
`DIAGNOSTIC_COMPLETE_QUALITY_HIT`, or `DIAGNOSTIC_INCOMPLETE`.
`historical_relation` is exactly `MATCHES_RETAINED_AGGREGATES`,
`DIVERGES_FROM_RETAINED_AGGREGATES`, or `NOT_COMPARABLE_INCOMPLETE`.
`failure`, when non-null, has exactly `stage` and `reason`, where `reason` is
the exact `{code,detail_sha256}` object. Incomplete receipts require
`NOT_COMPARABLE_INCOMPLETE`; complete receipts derive the matching/diverging
value from the frozen `300/203/no-hit` DIAG1 aggregates. Quality-hit/no-hit is
recomputed from accepted-state replay, never producer summary booleans.
The verdict biconditionals are exact: `failure != null` iff
`verdict=DIAGNOSTIC_INCOMPLETE`; with `failure=null`, recomputed replay hit iff
`verdict=DIAGNOSTIC_COMPLETE_QUALITY_HIT`, and recomputed replay no-hit iff
`verdict=DIAGNOSTIC_COMPLETE_NO_HIT`.
Receipt and supervisor terminal must exactly agree on route/plan, failure
stage/reason, launched-child sequence, campaign/promotion/formal literals, and
next-route/algorithm-route value. Any disagreement fails parsing.

## Exact Schedule and State Machine

1. Apply cheap path/interpreter/reference/UUID checks before creating output.
2. Keep the final output absent; create only its resolved sibling staging root,
   then publish the immutable source snapshot and validated copied reference
   inside staging.
3. Reconstruct and publish the policy authority with pure NumPy from the
   validated copied reference. No JAX array/backend call and no child exists.
4. Capture and publish `supervisor_before_preflight`; require exact
   parent-PID absence on the frozen RTX UUID.
5. Launch exactly one annotated lower/compile-only GPU preflight child. It may
   compile the exact loop/finalizer/map/replay/terminal executables but must not
   dispatch the solver, finalizer, replay, or endpoint audit.
6. Independently validate the preflight producer, terminal, process, memory,
   runtime, callback scan, and supervisor-zero authority, then revalidate all
   four setup authorities in the frozen order: source snapshot, frozen subset,
   native reference, and policy authority.
7. On any failure, publish/seal/deep-load one incomplete artifact and stop.
8. Capture and publish `supervisor_before_cold`; require exact parent-PID
   absence again.
9. Launch exactly one new cold GPU child. No persistent compilation cache is
   shared with preflight or policy authority.
10. Publish the cold supervision evidence, then revalidate all four setup
    authorities in the frozen order before classifying any cold child failure.
    A setup-integrity failure wins by the frozen stage precedence and retains
    the truthful subordinate cold vector. Otherwise, on non-complete terminal
    or missing producer, publish/seal/deep-load one incomplete artifact and
    stop before publishing numerical evidence.
11. On complete cold with valid setup authorities, publish existing history,
    terminal, replay, raw trace,
    process, memory, runtime, source, policy, and phase evidence; independently
    build, seal, and deep-load the complete receipt.

There is no loop, retry option, sample-count option, or replacement path in
DIAG2. The exact child sequence is one GPU preflight and zero or one GPU cold.
Warm paths and `campaign.json` are forbidden artifact members.

## Supervisor GPU-Zero Evidence

Each observation retains:

- schema, stage, monotonic and wall-clock timestamps;
- supervisor PID and `/proc/<pid>/stat` start ticks;
- frozen physical GPU UUID;
- exact `nvidia-smi` argv, executable identity, return code, raw stdout/stderr
  artifact references and hashes;
- parsed matching compute-process rows and their used-memory bytes.

The independent gate requires successful command execution, canonical parsing,
one frozen GPU identity, the exact current supervisor PID/start ticks, and zero
matching compute-process rows. A reported numeric zero from a matching row is
not sufficient; the parent PID must be absent. The two observations are
distinct and ordered around preflight. Preflight and cold child PIDs must differ
from each other and from the supervisor.

The exact supervisor environment forces `JAX_PLATFORMS=cpu`, removes legacy
`JAX_PLATFORM_NAME`, disables the compilation cache, and disables JAX GPU
preallocation. `CUDA_VISIBLE_DEVICES` remains the frozen UUID solely for the
management query and GPU-child construction. The import-path canary uses the
literal prospective supervisor argv/environment and proves parent-PID absence.
The supervisor must not call `jax.devices`, `device_get`, create a JAX array,
build the fullspace bootstrap, or invoke any backend-initializing helper.
`build_child_invocation` replaces the parent platform with
`JAX_PLATFORMS=cuda`, retains x64, keeps compilation cache disabled, and sets
GPU preallocation true for preflight and cold.

## Optional Producer and Failure Contract

The raw child terminal is mandatory for every launched child. Producer evidence
is optional and present only for a canonical schema-valid stdout document.
Every v2 evidence slot is an exact discriminated union, never a bare nullable
reference:

```json
{"state":"PRESENT","artifact":{"relative_path":"...","sha256":"...","size_bytes":1,"schema_version":"..."}}
```

or

```json
{"state":"ABSENT","reason":"CHILD_EXIT_NONZERO"}
```

`FailureReasonCode` is exactly: `SOURCE_PRE`, `SOURCE_POST`,
`FROZEN_SUBSET_INVALID`, `REFERENCE_INVALID`, `POLICY_DERIVATION_INVALID`,
`GPU_QUERY_FAILED`, `GPU_PARENT_PID_PRESENT`, `CHILD_LAUNCH_FAILED`,
`CHILD_TIMEOUT`, `CHILD_COMPILE_FAILED`, `CHILD_COMPILE_OOM`,
`CHILD_EXIT_NONZERO`, `PRODUCER_DECODE_FAILED`, `PRODUCER_SCHEMA_INVALID`,
`RUNTIME_SCHEMA_INVALID`, `POLICY_SCHEMA_INVALID`,
`NUMERICAL_SCHEMA_INVALID`,
`MONITOR_BINDING_FAILED`, `MONITOR_FINALIZATION_FAILED`,
`MEMORY_LIMIT_EXCEEDED`, `TRACE_NORMALIZATION_FAILED`, and
`SEMANTIC_VALIDATION_FAILED`. `AbsenceReason` is exactly `NOT_REACHED` plus
every `FailureReasonCode` value; spellings are identical. An absent slot
requires its canonical path not to exist except for a path admitted by the
closed retained-failure table below. A present JSON artifact requires its
in-document schema to match the reference.

`FailureStage` precedence is exactly this ordered tuple:

1. `SOURCE_PUBLICATION_FAILURE`;
2. `NATIVE_REFERENCE_FAILURE`;
3. `POLICY_AUTHORITY_FAILURE`;
4. `GPU_ZERO_BEFORE_PREFLIGHT_FAILURE`;
5. `PREFLIGHT_SOURCE_FAILURE`;
6. `PREFLIGHT_SUPERVISOR_FAILURE`;
7. `PREFLIGHT_TIMEOUT`;
8. `PREFLIGHT_MONITOR_FAILURE`;
9. `PREFLIGHT_PROTOCOL_FAILURE`;
10. `PREFLIGHT_COMPILE_FAILURE`;
11. `PREFLIGHT_CRASH`;
12. `PREFLIGHT_RESOURCE_FAILURE`;
13. `GPU_ZERO_BEFORE_COLD_FAILURE`;
14. `COLD_SOURCE_FAILURE`;
15. `COLD_SUPERVISOR_FAILURE`;
16. `COLD_TIMEOUT`;
17. `COLD_MONITOR_FAILURE`;
18. `COLD_PROTOCOL_FAILURE`;
19. `COLD_COMPILE_FAILURE`;
20. `COLD_CRASH`;
21. `COLD_RESOURCE_FAILURE`;
22. `NUMERICAL_EVIDENCE_INCOMPLETE`.

The receipt owns one pure failure selector used whenever two or more
independently observed candidate failures coexist. Raw observations are first
collapsed to at most one legal reason per stage using the stage-local
offending-path/first-authority/subordinate precedence rules frozen below. The
selector input is that normalized nonempty sequence of exact, legal
stage/reason pairs; it rejects an empty sequence, illegal pairs, and duplicate
stages that prove the caller skipped normalization. It returns the candidate
whose stage occurs first in the frozen tuple above, independent of input order.
Runner code must call this selector rather than reproduce the global
precedence locally wherever coexisting candidates are available. Qualification
crosses same-stage competing observations against each local precedence,
crosses every adjacent global-stage pair in both input orders, and rejects an
unnormalized duplicate-stage input.

The receipt derives the first applicable stage in that tuple. A valid producer
may remain
`PRESENT` under `MONITOR_FINALIZATION_FAILED` only when exit code is zero and
its bytes passed the exact mode-specific parser; monitor binding, timeout,
crash, source-pre, and prelaunch failures require the producer slot `ABSENT`
with the matching reason. Source-post preserves any independently valid
producer. Protocol failure caused by producer decode/schema also
requires it absent; protocol failure caused by runtime/policy schema may retain
an independently valid producer present.

Within each preflight/cold protocol stage the offending-path precedence is
producer stdout decode, producer schema, runtime schema, then policy schema.
Within numerical evidence it is history, terminal numerical, raw trace,
trace intervals, then execution. Only the first failing path supplies the
direct reason; all earlier minimum-typed artifacts remain `PRESENT`, the
failing untyped path is retained only under its failure role, and later
unproduced slots are `ABSENT/NOT_REACHED`.

For exact slot-state derivation, define ordered authority groups:

- `SETUP_SOURCE` = `source_manifest`, `frozen_numerical_subset`;
- `SETUP_REFERENCE` = `native_reference`;
- `SETUP_POLICY` = `policy_authority`;
- `ZERO_PREFLIGHT` = `supervisor_before_preflight`;
- `PREFLIGHT` = every `preflight_*` slot;
- `ZERO_COLD` = `supervisor_before_cold`;
- `COLD_SUPERVISION` = `cold_producer`, `cold_terminal`, `cold_process`,
  `cold_memory`, `cold_memory_samples`, `cold_runtime`, `cold_policy`;
- `COLD_NUMERICAL` = `cold_history`, `cold_terminal_numerical`,
  `cold_raw_trace`, `cold_trace_intervals`, `execution`;
- `TERMINAL` = `supervisor_terminal`.

The mechanical stage table is:

| Failure stage | Allowed direct reason code(s) | Own authority group |
| --- | --- | --- |
| `SOURCE_PUBLICATION_FAILURE` | `SOURCE_PRE`, `SOURCE_POST`, `FROZEN_SUBSET_INVALID` | `SETUP_SOURCE` |
| `NATIVE_REFERENCE_FAILURE` | `REFERENCE_INVALID` | `SETUP_REFERENCE` |
| `POLICY_AUTHORITY_FAILURE` | `POLICY_DERIVATION_INVALID` | `SETUP_POLICY` |
| `GPU_ZERO_BEFORE_PREFLIGHT_FAILURE` | `GPU_QUERY_FAILED`, `GPU_PARENT_PID_PRESENT` | `ZERO_PREFLIGHT` |
| `PREFLIGHT_SOURCE_FAILURE` | `SOURCE_POST`, `FROZEN_SUBSET_INVALID`, `REFERENCE_INVALID`, `POLICY_DERIVATION_INVALID` | `PREFLIGHT` |
| `PREFLIGHT_SUPERVISOR_FAILURE` | `CHILD_LAUNCH_FAILED` | `PREFLIGHT` |
| `PREFLIGHT_TIMEOUT` | `CHILD_TIMEOUT` | `PREFLIGHT` |
| `PREFLIGHT_MONITOR_FAILURE` | `MONITOR_BINDING_FAILED`, `MONITOR_FINALIZATION_FAILED` | `PREFLIGHT` |
| `PREFLIGHT_PROTOCOL_FAILURE` | `PRODUCER_DECODE_FAILED`, `PRODUCER_SCHEMA_INVALID`, `RUNTIME_SCHEMA_INVALID`, `POLICY_SCHEMA_INVALID` | `PREFLIGHT` |
| `PREFLIGHT_COMPILE_FAILURE` | `CHILD_COMPILE_FAILED`, `CHILD_COMPILE_OOM` | `PREFLIGHT` |
| `PREFLIGHT_CRASH` | `CHILD_EXIT_NONZERO` | `PREFLIGHT` |
| `PREFLIGHT_RESOURCE_FAILURE` | `MEMORY_LIMIT_EXCEEDED` | `PREFLIGHT` |
| `GPU_ZERO_BEFORE_COLD_FAILURE` | `GPU_QUERY_FAILED`, `GPU_PARENT_PID_PRESENT` | `ZERO_COLD` |
| `COLD_SOURCE_FAILURE` | `SOURCE_POST`, `FROZEN_SUBSET_INVALID`, `REFERENCE_INVALID`, `POLICY_DERIVATION_INVALID` | `COLD_SUPERVISION` |
| `COLD_SUPERVISOR_FAILURE` | `CHILD_LAUNCH_FAILED` | `COLD_SUPERVISION` |
| `COLD_TIMEOUT` | `CHILD_TIMEOUT` | `COLD_SUPERVISION` |
| `COLD_MONITOR_FAILURE` | `MONITOR_BINDING_FAILED`, `MONITOR_FINALIZATION_FAILED` | `COLD_SUPERVISION` |
| `COLD_PROTOCOL_FAILURE` | `PRODUCER_DECODE_FAILED`, `PRODUCER_SCHEMA_INVALID`, `RUNTIME_SCHEMA_INVALID`, `POLICY_SCHEMA_INVALID` | `COLD_SUPERVISION` |
| `COLD_COMPILE_FAILURE` | `CHILD_COMPILE_FAILED`, `CHILD_COMPILE_OOM` | `COLD_SUPERVISION` |
| `COLD_CRASH` | `CHILD_EXIT_NONZERO` | `COLD_SUPERVISION` |
| `COLD_RESOURCE_FAILURE` | `MEMORY_LIMIT_EXCEEDED` | `COLD_SUPERVISION` |
| `NUMERICAL_EVIDENCE_INCOMPLETE` | `NUMERICAL_SCHEMA_INVALID`, `TRACE_NORMALIZATION_FAILED`, `SEMANTIC_VALIDATION_FAILED` | `COLD_NUMERICAL` |

For a row, all earlier authority groups are present. The own group follows its
group-specific rules below. Every later group is absent with `NOT_REACHED`.
The terminal group is present. Setup source/reference/policy slots are present
iff their minimum typed authority was produced. For an initial setup failure,
the first semantically invalid but minimum-typed slot remains `PRESENT`; a slot
that fails minimum typing is `ABSENT` with the direct reason; and every
never-attempted later setup slot is `ABSENT/NOT_REACHED`.
GPU-zero authority documents are always present because their typed query
union represents launch/timeout/return/parse failure. This table, not enum
declaration order or exception text, is the sole stage/reason mapping.

For each launched-child group, abbreviate producer `P`, terminal `T`, process
`X`, memory `M`, memory samples `S`, runtime `R`, and policy `Q`. The exact
own-group vector rule is:

- `CHILD_LAUNCH_FAILED`: all seven slots are absent with the direct reason; no
  child artifact path may exist. Initial `SOURCE_PRE` belongs only to
  `SOURCE_PUBLICATION_FAILURE`; all downstream child slots are
  `ABSENT/NOT_REACHED` under the ordinary prefix rule.
- after successful launch, `T` and `X` are always present because the
  supervisor constructs them from raw process state;
- `M` and `S` are both present after successful monitor finalization and both
  absent with `MONITOR_BINDING_FAILED` or `MONITOR_FINALIZATION_FAILED`
  otherwise;
- `P` is present iff exit code is zero, canonical mode-specific producer
  parsing passes, and the selected stage permits retention as stated above; it is
  absent with `PRODUCER_DECODE_FAILED`, `PRODUCER_SCHEMA_INVALID`,
  `CHILD_TIMEOUT`, or `CHILD_EXIT_NONZERO` according to the reached terminal;
  a schema-valid compile-failure/OOM producer remains present;
- `R` and `Q` are present iff their minimum typed boundary passes. If never
  produced they are absent with the selected direct terminal reason; if raw
  files exist but fail typing they are absent with `RUNTIME_SCHEMA_INVALID` or
  `POLICY_SCHEMA_INVALID` and those bytes use the untyped role;
- `SOURCE_POST`, `FROZEN_SUBSET_INVALID`, `REFERENCE_INVALID`, or
  `POLICY_DERIVATION_INVALID` discovered by the mandatory post-launch setup
  revalidation applies the same produced-evidence rules after launch, while
  the outer stage remains the higher-precedence source failure.

Post-launch setup-integrity failure is the sole exception to the ordinary
earlier-group prefix rule. It is required because a child may have already
produced valid immutable evidence before the supervisor discovers that a setup
authority changed. The receipt derives exactly one first failing setup
authority in this order: `source_manifest`, `frozen_numerical_subset`,
`native_reference`, then `policy_authority`, with direct reason respectively
`SOURCE_POST`, `FROZEN_SUBSET_INVALID`, `REFERENCE_INVALID`, or
`POLICY_DERIVATION_INVALID`. Its slot and every other setup slot remain
`PRESENT` when their own canonical minimum-typed ArtifactRef boundary passes;
semantic invalidity is not rewritten as absence. If the first authority itself
fails that minimum boundary, only its slot is `ABSENT` with the direct reason
and its physical bytes use the closed invalid/opaque role below. Validation
stops after the first semantic failure, so later setup slots are checked only
at their minimum-typed boundary and are not assigned additional semantic
failures. Any later setup slot that fails its minimum-typed boundary is
`ABSENT` with that slot's canonical setup reason and its bytes use the matching
closed failure role; the outer stage/reason remains the first semantic failure.
A later slot whose minimum boundary passes remains `PRESENT`. A preflight
discovery retains the complete truthful preflight child-evidence vector
derived from the raw child outcome, including timeout, monitor, protocol,
compile/OOM, crash, or resource failure. The setup-integrity stage/reason wins
by `FailureStage` precedence whenever post-launch revalidation is reached. It
makes `ZERO_COLD`, both cold groups, and their paths `ABSENT/NOT_REACHED`. A
post-cold discovery retains both GPU-zero documents, the preflight group, and
the complete truthful cold child-evidence vector derived from the raw child
outcome under the same precedence rule;
the numerical group is `ABSENT/NOT_REACHED` because the mandatory setup
revalidation precedes numerical publication. In both cases the supervisor
terminal is present
and joins the same stage/reason. No invalid minimum-typed setup ArtifactRef is
represented as `PRESENT`, and no valid child evidence is discarded to restore
a fictitious authority prefix.

The subordinate launched-child vector under an outer post-launch setup failure
is derived independently; it is not serialized as a second outer failure.
`T` and `X` are always `PRESENT`. `M/S` are both `PRESENT` iff monitor
finalization produced their minimum-typed documents, otherwise both are
`ABSENT` with `MONITOR_BINDING_FAILED` or `MONITOR_FINALIZATION_FAILED`.
Both `T` and `X` contain the same required `monitor_failure_kind`, exactly one
of `NONE`, `BINDING`, or `FINALIZATION`; the validator rejects disagreement.
For `BINDING`, terminal status is `MONITOR_FAILURE`, child PID is positive,
child start ticks are the literal zero sentinel, the parent-monotonic process
interval is positive, memory/sample slots are absent, and no bound child-start
identity join is attempted. Binding is attempted synchronously immediately
after `Popen`; on failure the supervisor kills and reaps the child without
entering the timed communicate/monitor-finalization path. Therefore `BINDING`
is mutually exclusive with timeout, `FINALIZATION`, producer decode/schema,
compile, and memory-limit outcomes; any evidence claiming such a combination
is invalid. The killed child may have emitted arbitrary partial stdout/stderr;
those raw bytes remain retained and hash-bound but are never parsed or
promoted to a producer under `BINDING`. Mutual exclusion applies to a typed
producer/outcome claim, not to physical raw-stream nonemptiness; any
`PRESENT` producer slot under `BINDING` is invalid. For `FINALIZATION`, child PID and start ticks are
positive, bound argv identity exists, the parent-monotonic interval is
positive, memory/sample slots are absent, and the terminal preserves the
independently derived process/producer outcome while the monitor failure
participates in precedence: timeout remains terminal `TIMEOUT`, nonzero exit
remains terminal `CRASH`, and exit-zero success or compile failure/OOM uses
terminal `COMPLETE` with the exact producer status.
It does not rewrite an exit-zero child to terminal `MONITOR_FAILURE`; the typed
`FINALIZATION` discriminator is the monitor-failure authority. Because
finalization produced no typed memory/sample documents, `FINALIZATION` is
mutually exclusive with `MEMORY_LIMIT_EXCEEDED`; that resource outcome requires
typed `M/S` and therefore `monitor_failure_kind=NONE`. For `NONE`, child PID/start ticks and the
bound identity are positive and memory/sample slots are paired and present.
GPU-zero ordering uses the retained parent-monotonic process interval for all
launched children, including `BINDING`; PID/start-tick equality joins apply
only when start ticks are positive. PID distinctness applies whenever the PID
is positive.
`R/Q` are `PRESENT` iff their minimum-typed documents exist; an existing
minimum-untyped file is `ABSENT` with `RUNTIME_SCHEMA_INVALID` or
`POLICY_SCHEMA_INVALID` and the untyped role. For mode success,
compile/OOM, memory-limit failure, or exit-zero monitor-finalization with a
valid producer, a genuinely missing `R/Q` file is also `ABSENT` with
`RUNTIME_SCHEMA_INVALID` or `POLICY_SCHEMA_INVALID`; those outcomes require
both documents. For timeout, monitor binding, any monitor-finalization
failure not covered by the preceding exit-zero-valid-producer case, crash, or producer
decode/schema failure, a genuinely unproduced `R/Q` file is `ABSENT` with the
subordinate terminal reason. `P` follows this exact table:

| Subordinate raw child outcome | `P` state | Required evidence |
| --- | --- | --- |
| mode success (`SUCCESS` preflight, `COMPLETE` cold) | `PRESENT` | exit zero and exact mode-specific producer |
| timeout | `ABSENT/CHILD_TIMEOUT` | child terminal `TIMEOUT` |
| monitor binding failure | `ABSENT/MONITOR_BINDING_FAILED` | no bound child start identity |
| monitor finalization failure | `PRESENT` only for exit-zero exact producer; otherwise `ABSENT/MONITOR_FINALIZATION_FAILED` | child terminal/process plus monitor absence |
| crash | `ABSENT/CHILD_EXIT_NONZERO` | child terminal `CRASH` and nonzero process exit |
| producer decode failure | `ABSENT/PRODUCER_DECODE_FAILED` | raw stdout fails canonical JSON decoding |
| producer schema failure | `ABSENT/PRODUCER_SCHEMA_INVALID` | raw stdout decodes but fails the exact mode schema |
| compile failure or compile OOM | `PRESENT` | exit-zero exact producer with `COMPILE_FAILURE` or `COMPILE_OOM` |
| memory-limit failure | `PRESENT` | exact success producer and typed memory fraction at least `0.8` |

When more than one non-binding subordinate condition is observable, its
slot-state reason uses this precedence: timeout; monitor finalization; crash;
producer decode; producer schema; compile OOM; compile failure; memory limit;
success. The synchronous binding branch is selected before this list and is
mutually exclusive with every listed condition. Runtime/policy minimum typing is then applied independently as stated
above. The validator recomputes this class from child terminal, process exit
and identity, the typed monitor discriminator, monitor documents, raw stdout,
and producer bytes. `SOURCE_PRE` cannot coexist with a launched child; as an
initial setup failure its downstream child slots are `ABSENT/NOT_REACHED`.
For post-preflight setup failure, `launched_children=["preflight"]`,
`preflight_authorized=false`, and `cold_authorized=false`. For post-cold setup
failure, `launched_children=["preflight","cold"]`,
`preflight_authorized=true`, and `cold_authorized=true`. The supervisor
terminal contains only the outer setup failure and these flags; the retained
child evidence proves the subordinate outcome.

These seven decisions are independently recomputed from terminal/process,
monitor, producer stdout, and physical typed/untyped evidence. There is no
unnamed direct-failure slot.

Every handled artifact has `TERMINAL` present. Except for the explicitly
enumerated post-launch setup-integrity case above, a stage requires every group
strictly before its named stage present. Its own attempted authority document
is present when one was canonically produced (including a failed GPU-zero
document); otherwise that slot is absent with the stage's reason code. Every
strictly downstream group is absent with `NOT_REACHED`. Launched-child
terminal/process/raw stdout/stderr are present even on child failure. Producer
is present only under the rule above. Memory and memory-samples are present
only after successful monitor finalization; otherwise absent with the matching
monitor reason. Runtime/policy are present only when the child published and
their canonical syntax, exact keys/types/schema, and ArtifactRef
hash/size/path checks pass. Such a typed artifact remains `PRESENT` even when
its cross-artifact semantics fail; semantic evidence is never discarded to
manufacture absence. A physically retained artifact that cannot meet that
minimum typed boundary is not assigned a typed slot and is handled only by the
closed untyped-failure roles below. A genuinely unproduced numerical slot is
absent with `TRACE_NORMALIZATION_FAILED` or `SEMANTIC_VALIDATION_FAILED`; later
slots carry `NOT_REACHED`. This prefix rule, the one closed post-launch
setup-integrity exception, and the ordered tuple form the total stage-to-slot
cross-table; any other stage/reason/state combination fails.

The only retained files not reachable from a `PRESENT` slot, plus the exact
nested semantic-failure role overrides for a minimum-typed `PRESENT` source or
reference authority, are this closed failure table:

| Applicable stage | Canonical pattern | Role | Cardinality |
| --- | --- | --- | --- |
| any setup or post-launch setup failure with `source_manifest=ABSENT` | retained `source-snapshot/**` regular files | `source_snapshot_opaque_failure` | zero or more |
| any setup or post-launch setup failure with `source_manifest=PRESENT` and semantic source failure | `source-snapshot/**` regular files except `source-snapshot/source-manifest.json` | `source_snapshot_opaque_failure` | one or more |
| any setup or post-launch setup failure with `native_reference=ABSENT` | retained `native-reference/**` regular files | `native_reference_opaque_failure` | zero or more |
| any setup or post-launch setup failure with `native_reference=PRESENT` and semantic reference failure | `native-reference/**` regular files except `native-reference/reference.json` | `native_reference_opaque_failure` | one or more |
| `SOURCE_PUBLICATION_FAILURE`, or any post-launch setup failure, with `frozen_numerical_subset=ABSENT` | retained `frozen-numerical-subset.json` failing the minimum typed boundary | `invalid_setup_authority_failure` | zero or one |
| `POLICY_AUTHORITY_FAILURE`, or any post-launch setup failure, with `policy_authority=ABSENT` | retained `policy-authority.json` failing the minimum typed boundary | `invalid_setup_authority_failure` | zero or one |
| any `COLD_*` or `NUMERICAL_EVIDENCE_INCOMPLETE` after profiler start | `cold/raw-trace/plugins/profile/<run>/<base>.trace.json.gz` | `raw_trace_chrome` | zero or one |
| any `COLD_*` or `NUMERICAL_EVIDENCE_INCOMPLETE` after profiler start | `cold/raw-trace/plugins/profile/<run>/<base>.xplane.pb` | `raw_trace_xplane` | zero or one |
| any `PREFLIGHT_*` after successful launch | `preflight/runtime-evidence.json` or `preflight/policy.json` failing the minimum typed boundary | `untyped_evidence_failure` | zero or one per path |
| any `COLD_*` after successful launch | `cold/runtime-evidence.json`, `cold/policy.json`, `cold/history.json`, `cold/terminal-numerical.json`, `cold/trace-intervals.json`, or `execution.json` failing the minimum typed boundary | `untyped_evidence_failure` | zero or one per path |
| `NUMERICAL_EVIDENCE_INCOMPLETE` | `cold/history.json`, `cold/terminal-numerical.json`, `cold/trace-intervals.json`, or `execution.json` failing the minimum typed boundary | `untyped_evidence_failure` | zero or one per path |

If both trace siblings exist they must share directory and basename. No other
opaque, partial, temporary, producer, policy, reference, or source file is
permitted in a final artifact. In particular, malformed producer bytes remain
the already-bound raw stdout stream; `producer.json` is written only after the
minimum typed boundary passes.

An OOM must not be inferred merely from a failed allocation substring and used
as a scientific verdict. The outer result remains `DIAGNOSTIC_INCOMPLETE`; the
raw stderr/process evidence is retained and hash-bound for a nonpromoting
resource diagnosis. No derived `allocator_failure_observed` field is serialized
in the v2 receipt, supervisor terminal, or process payload.

Every incomplete receipt fixes:

- `engineering_campaign_receipt_produced=false`;
- `promotion_authorized=false`;
- `formal_comparison=NOT_PRODUCED`;
- `algorithm_route_selection=NOT_PRODUCED`.

Parent/supervisor failures are their own terminal class and must not be
misrepresented as child `CRASH`. Text substrings never select the failure
stage. The incomplete builder must not require or resolve absent producer, history,
terminal numerical, replay, or trace references. It must nevertheless hash and
role-close all physically retained stdout, stderr, process, memory, partial
trace, source, runtime, reference, and supervisor-resource bytes applicable to
the failure.

## Complete Numerical Evidence

On a complete cold producer, DIAG2 preserves the DIAG1 evidence contract
unchanged:

- all 300 fixed history slots and exact active-prefix/outcome applicability;
- accepted optimizer/physical ledgers and exact prefix mask;
- unconditional returned endpoint, objective ledger, all 255 raw/scaled
  equalities, gradients, Jacobian, multipliers, raw stationarity and KKT;
- accepted-state quality replay and independently recomputed first hit;
- signed objective/component/scaled-feasibility margins;
- residual reconstruction, transpose, projection, solve, and correction
  certificates;
- seven exact profiler scopes, interval ownership/unions/overlap, at least 0.90
  attribution coverage, and a loop-only trace envelope;
- exact process/runtime/source/reference/policy identity;
- exact child PID/start/UUID memory samples and peak fraction below 0.8;
- zero guarded hot H2D, zero guarded hot D2H, zero Python callbacks, and one
  post-timing final D2H.

Only a complete, independently reloadable numerical receipt may select an
algorithm-development route under the existing deterministic DIAG1 formulas.
Any incomplete evidence fixes selection to `NOT_PRODUCED`.

## Artifact and Seal Contract

- The final output parent is resolved strictly before work, every existing path
  component is rejected if it is a symlink, and the resolved parent is outside
  the resolved repository. Final and staging identities use that resolved
  parent; the final leaf must be absent and not a dangling symlink.
- Work occurs in a sibling `<output>.partial-<nonce>` staging directory. Raw
  evidence is written first and each child terminal last. Receipt and manifest
  validation run while staging remains writable; then all files/directories are
  sealed, the sealed staging tree is deep-loaded, and the staging directory is
  atomically published with Linux `renameat2(RENAME_NOREPLACE)` to the
  never-existing final output. If `renameat2`/`RENAME_NOREPLACE` is unavailable
  or fails, publication fails closed; an exists-check plus ordinary
  `rename`/`replace` fallback is forbidden.
  The public loader then independently deep-loads the final path a second time.
  A supervisor crash
  may leave only a visibly partial directory, never an apparently canonical
  final root.
- Canonical JSON forbids NaN/Infinity, duplicate keys, unknown keys, booleans
  as numbers, and noncanonical encodings.
- Binary arrays and raw streams use content-addressed refs with exact path,
  role, schema, size, and SHA-256.
- The role-bearing manifest must equal both the observed regular-file set and
  the receipt-derived expected set. Missing and extra bytes fail.
- Symlinks, hardlink ambiguity, path traversal, writable files, writable
  directories, and special files fail.
- Publication order is raw evidence, supervisor terminal, receipt, role
  manifest, writable semantic/exact-set validation, recursive file chmod 0444,
  recursive directory/root chmod 0555, sealed-staging deep load, no-replace
  atomic rename, destination-parent directory `fsync`, then independent
  final-path deep load. Before rename, every regular file is fsynced without
  following symlinks and every staging directory is fsynced bottom-up after
  chmod; any fsync failure is fail-closed.
- If the public deep load fails, the terminal result is not successful; no
  promotion or route claim may escape.

## Qualification and Mutation Matrix

All tests run with explicit `JAX_PLATFORMS=cpu`. No qualification test may
create a CUDA context, allocation, compilation, or kernel on the RTX 5090
before the reviewed production preflight. The import-path canary may use only
the read-only NVIDIA management query whose required result is parent-PID
absence.

Required qualification:

- frozen-file hashes and exact optimizer/core/adapter/trace byte audit;
- old public caller inventory and compatibility tests;
- pure-NumPy policy derivation boundary/bit-pattern/hash and canonical payload
  tests, including coherent reference-volume/equality mutations;
- fresh subprocess proof that importing/running the supervisor parent path does
  not load a CUDA backend or acquire a GPU PID;
- exact GPU-zero parser tests for absent PID, wrong UUID, duplicate PID,
  nonzero PID, malformed/empty output, failed command, PID reuse/start mismatch,
  and two-gate ordering;
- dual-query mutations for launched/timed-out/return-code state, inventory versus
  compute argv, stdout/stderr ref and stage swaps, malformed-success rows,
  failed-query nonempty matching rows, and first/second-stage interchange;
- successful mocked integrated sequence proves policy/preflight/cold exactly
  once and no warm/campaign/retry;
- preflight failure proves no cold launch;
- absent producer cases for timeout, crash, malformed stdout, empty stdout,
  monitor failure, and launch failure;
- every producer PRESENT/ABSENT reason crossed with terminal status, including
  complete-plus-absent, failure-plus-present, absent-plus-file, and
  present-minus-file rejections;
- slot-parser mutations for an unknown/missing state or key, invalid absence
  reason, `PRESENT+reason`, `ABSENT+artifact`, and wrong path/schema/hash/size;
- schema-valid producer required for COMPLETE and preflight authorization;
- sealed public deep-load tests for compile failure, compile OOM, timeout,
  crash, allocator-OOM stderr, protocol failure, monitor failure, source-pre,
  source-post, supervisor-resource failure, semantic numerical failure, and
  partial Chrome/XPlane trace variants;
- coherent post-preflight and post-cold mutations of each setup authority:
  source snapshot bytes, frozen-subset bytes, native-reference bytes, and
  policy-authority bytes. Each case must preserve already-valid child evidence
  and derive only the exact first semantic reason. A minimum-typed offending
  slot remains `PRESENT`; a minimum-typing failure is `ABSENT` with that reason.
  Nested or untyped physical bytes receive only the exact closed failure role.
  Every case must seal, publish, and independently deep-load as the matching
  nonpromoting source failure;
- initial setup failures crossed with the same minimum-typed semantic-invalid
  versus minimum-typing-invalid state rule, including coherent frozen-subset,
  native-reference, and policy-authority semantic mutations;
- combined setup-integrity mutations in which the first authority fails
  semantically and one or more later authorities fail minimum typing. The outer
  stage/reason must remain the first semantic failure, while each later invalid
  slot is `ABSENT` with its own canonical reason and exact closed role;
- every post-launch setup reason crossed with representative timeout, monitor,
  protocol, compile/OOM, crash, and resource child outcomes. The source/setup
  stage must win while child slots preserve the exact subordinate raw outcome;
- monitor-binding crossed with timeout/finalization/producer/compile/resource
  claims must be rejected as impossible state combinations;
- monitor-finalization crossed with memory-limit failure must be rejected
  because no typed memory/sample authority exists;
- distinct sealed cases for every preflight supervisor/timeout/compile/crash/
  protocol/monitor/resource/source stage, setup reference/policy/source stages,
  and the first versus second GPU-zero gate;
- explicit cold compile-failure and cold compile-OOM terminal/seal cases;
- pairwise competing-failure tests for every adjacent precedence boundary and
  exhaustive `FailureStage` x direct `FailureReasonCode` x authority-group
  prefix x slot state/absence reason x retained-opaque-path validation;
- diagnostic-to-supervisor-terminal mutations for route/plan, disposition,
  stage, reason code/detail, child order, authorization flags, campaign/
  promotion/formal literals, and next-route equality;
- verdict/failure/replay biconditional mutations for all three verdicts;
- coherent mutation tests that rewrite raw bytes and all immediate hashes for
  producer presence, terminal taxonomy, supervisor PID/UUID/rows, policy
  authority, process/runtime/source/memory joins, manifest roles/modes, and
  every complete numerical evidence family;
- fault injection after raw writes, terminal publication, receipt, manifest,
  each seal stage, deep load, and atomic rename; injections before rename must
  leave the final path absent, while an injected post-rename return failure must
  leave a sealed, independently reloadable final artifact;
- manifest mutations for missing/extra/duplicate/out-of-order entries, digest,
  size, role, nested stream, Chrome/XPlane sibling, mode, symlink, hardlink,
  special file, and path traversal;
- v1/v2 route/schema/plan cross-rejection;
- Ruff check/format, compileall, diff-check, isolated CPU test files, and a
  fresh independent source/receipt/runner/integrated launch audit.

No production GPU command is authorized until all qualification commands,
counts, hashes, and independent GO verdicts are appended to this file without
changing the pre-execution contract.

## Caller Inventory, Compatibility, and Migration

The affected production surfaces are the diagnostic path in
`benchmarks/run_single_stage_native_equivalent_quality_campaign.py`, the
diagnostic receipt module, snapshot runtime/environment authority, and their
direct tests. Engineering `run_campaign`, legacy `run_preflight`, and the
public optimizer/adapter APIs remain unchanged.

`SupervisedSample.producer` changes internally from a required mapping to an
optional mapping. All diagnostic callers must branch on presence. Engineering
campaign callers either retain the existing required-producer contract through
a separate type or explicitly reject `None`; DIAG2 must not weaken engineering
promotion gates.

DIAG2 adds discriminated evidence slots, raw supervisor-resource and
policy-authority fields, structured failure reasons, and v2 receipt keys. V1
parsers/constants remain explicit where retained; no silent defaulting or alias
fallback is permitted. The CLI remains mutually exclusive:
`--diagnostic-only` cannot combine with engineering campaign, preflight-only,
or internal child flags.

## Rollback and Recovery

Before production execution, rollback removes only DIAG2 runner/receipt/schema
surfaces and restores their direct callers; frozen numerical files and DIAG1
artifacts remain untouched. After final publication, never delete or mutate the
DIAG2 root. Every handled typed child or supervisor outcome must seal its
truthful artifact. An unrecoverable death of the supervisor process itself may
leave only a noncanonical `.partial-*` staging directory and no final path; it
cannot be adjudicated or promoted. A new attempt after
any launched DIAG2 cold requires a new SSOT/schema/output root; the current
plan cannot authorize replacement.

## Production Authorization

After qualification and independent GO, use one literal command with:

- `CUDA_VISIBLE_DEVICES=GPU-7951f78e-c05d-e01c-303f-d644f4341fe1` for the
  supervisor's physical identity check and GPU children;
- compilation cache absent/disabled, x64 true, preallocation true only in GPU
  children, and the frozen profiler capacity;
- the exact `.venv-qn-gpu/bin/python`, validated native reference, frozen input
  root, and a new absent DIAG2 output root.

The runner must use only NumPy/reference bytes for parent policy derivation and
must not rely on caller environment alone for resource isolation.

## Completion Criteria

- [ ] Final pre-execution SSOT hash and every frozen/change-authorized file hash
      are recorded; independent reviewers issue GO.
- [ ] Explicit CPU/static/mutation qualification passes in isolated processes.
- [ ] Pure-NumPy policy reconstruction reproduces the frozen FP64/policy bytes,
      and both retained supervisor GPU observations independently prove
      parent-PID absence.
- [ ] Exactly one passing lower-only RTX 5090 preflight and at most one cold are
      present; no warm, tuning, replacement, optimizer/cold rerun, or second
      cold exists. The post-timing accepted-state diagnostic replay is retained.
- [ ] Either the complete numerical diagnostic or a truthful failure receipt is
      canonical, exact-manifested, 0444/0555 sealed, and independently reloaded
      from the final public path.
- [ ] No algorithmic route is claimed unless complete raw trajectory, terminal,
      phase, source/runtime/resource, and policy evidence all pass.
- [ ] The executed outcome and exact artifact hashes are appended without
      rewriting the pre-execution contract.

## Open Questions

- None. Any change to the schedule, failure precedence, GPU-zero definition,
  policy authority, numerical route, evidence schema, or output root requires a
  new reviewed SSOT revision before execution.
## Qualification Record (append-only; excluded from `DIAG2_PLAN_SHA256`)

The pre-execution contract is the exact byte prefix ending immediately before
this heading. Its SHA-256 is
`38bf768c8c851347e9178596f6dcec8f3fb43ff88030a3dd953066999df97f78`,
which is also the frozen `DIAG2_PLAN_SHA256`. This append-only record does not
revise that contract.

Qualification completed on 2026-08-11 under repository HEAD
`52dea17ddf3012cf923fc92da78c0d73a17f4625` without a production GPU launch.
The controlling explicit-CPU command was:

```text
env JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
.venv-qn-cpu/bin/python -m pytest -q \
tests/benchmarks/test_process_gpu_monitor.py \
tests/benchmarks/test_single_stage_fullspace_snapshot.py \
tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py \
tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py \
tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py
```

Result: `801 passed in 328.31s (0:05:28)`.

The following post-test static commands all passed:

```text
.venv-qn-cpu/bin/python -m ruff check <10 qualified production/test files>
.venv-qn-cpu/bin/python -m ruff format --check <10 qualified production/test files>
.venv-qn-cpu/bin/python -m compileall -q <10 qualified production/test files>
git diff --check
```

## Effective Fresh-Attempt Ledger (append-only controlling clarification)

This EOF entry is the sole controlling byte/review ledger for the fresh R1
authorization. Every earlier hash list and GO list above remains historical
evidence for the initial qualification or Attempt 1 and is expressly
superseded for R1 authorization. The unusual textual ordering above resulted
from appending Attempt 1 and root-fix records before the remainder of the
initial qualification record; it does not alter the frozen pre-execution
prefix or make the stale Attempt-1 hashes effective for R1.

The effective frozen contract prefix is
`38bf768c8c851347e9178596f6dcec8f3fb43ff88030a3dd953066999df97f78`.
The effective R1 qualification result is the unchanged controlling command
with `807 passed in 332.80s (0:05:32)`, followed by the exact 11-file static
commands recorded above, all passing. The sole effective 11-file SHA-256
manifest for R1 is:

- `benchmarks/process_gpu_monitor.py`:
  `6f2e6d3c144a5e31b45533ef288e0bd1dba84066780f0bdffd443f89c3316f39`;
- `benchmarks/single_stage_fullspace_process_gpu_monitor.py`:
  `57918ef564eb66705dd8789cc30f18bc095d3a66b4224888d84daa583f9549e3`;
- `benchmarks/run_single_stage_native_equivalent_quality_campaign.py`:
  `d67df375bde3a6cf542536837b127c480476fbc670b38177c2ed7e3dd73e749c`;
- `benchmarks/single_stage_fullspace_snapshot.py`:
  `985ddf4b61ab4ecdf1ab6c0e130a1ed98d7e807f233f76ba27fb2bf74f247374`;
- `benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py`:
  `bb7104974dbded616cc1f673eb5987b2bc857dc93144a359518e334b04cf17db`;
- `tests/benchmarks/_diag2_fixture.py`:
  `bad70f7ac5d6021ad0bf3c99efcc1e8d7f9023360265bc67b24c4e43b67c275d`;
- `tests/benchmarks/test_process_gpu_monitor.py`:
  `f776670184493012293638e32711a49340dccdc83a77e008cbd590a1dbad16e1`;
- `tests/benchmarks/test_single_stage_fullspace_snapshot.py`:
  `ca0f4ed6dbd4bab6fb6b85c4c5901192282bce042b652646913955829d0bdf95`;
- `tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py`:
  `01b894c2e461209f72165dae9dc7b3c596dd6789c38a1a1dbec185d696fb79b3`;
- `tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py`:
  `f111b94f4d039517cdfb66e09a342b6e34119d2fcfef48c97211b33c558b3275`;
- `tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py`:
  `e82179c675b9ca372433a096d03e4c2b861e22472fc81c4ffada8005f85dbd11`.

Pre-clarification fresh audits independently found the implementation,
receipt/schema, atomic-publication, and qualification-coverage surfaces correct
on this exact manifest; they withheld overall authorization solely because the
effective manifest and review chronology were ambiguous. This controlling EOF
entry closes that record defect. R1 launch remains conditional on their fresh
post-clarification GO verdicts, which must be appended after they are issued.

The only candidate final root remains
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-r1-20260811T123501Z`.
It and every matching `.partial-*` sibling must be absent immediately before
launch. Attempt 1's existing partial remains immutable and out of scope.

## Attempt 1 Outcome (append-only)

The authorized command targeted
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-20260811T120706Z`
and exited `1` after 5.6 seconds. The canonical final root is absent. The sole
retained staging tree is
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-20260811T120706Z.partial-aeafe206e452e429fa3ba94558531edb`.
It contains 645 regular files, 58 descendant directories, no symlinks, and is
not sealed (`0755` staging root; `0775` preflight and supervisor directories).
It must remain an unmodified, noncanonical forensic partial and cannot be
completed, relabeled, promoted, or used for a scientific or performance claim.

The first supervisor GPU-zero observation passed. Exactly one preflight child
launched (PID `1470196`, start ticks `47646434`), used five retained GPU-memory
samples with a 526 MiB peak, and exited `1` after 2.029777491 seconds. The child
failed before prepare/lower/compile because `preflight/` did not yet exist when
it attempted to publish `preflight/runtime-evidence.json`. No solver loop was
dispatched, no cold directory or before-cold observation exists, and no cold
authorization was consumed.

The parent then retained the raw preflight evidence but failed before
`finish()` because `_publish_diag2_supervision` labeled v2 terminal/process
files with v1 ArtifactRef schemas. Consequently no supervisor terminal,
diagnostic receipt, artifact manifest, seal, atomic rename, or final-path deep
load exists.

Key retained forensic SHA-256 identities are:

- preflight terminal: `ff4909da618710df35ab1ee35c6a4f012c1186334c69076029965244895e80ba`;
- preflight process: `5bbc5d0c01da9c05626c5023a0f1202e92629bf19cf5b86947b86ee08dd5685b`;
- empty preflight stdout: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- preflight stderr: `83e73a95caf58bef8ddac98bdfb8e8a67a28f630f94b555a49fa9061346265cc`;
- preflight memory summary: `c4c3d48ade70df480ef1afce1fdc33e8eb4266b8620250a3d8d02a96f84280c5`;
- preflight memory samples: `1d25a421985d72e06c10cf9e31725cf3fe08707efc82ef8fd9f9b0ecda193579`;
- before-preflight supervisor observation: `b6c8711e7e6e9033226a290a917f306d691265ad52e47fc148e1573390126147`;
- source manifest: `e9b2eec39d6c9a04c751830c4909994a88e73a19408a323b223c871cc3256e5a`;
- frozen numerical subset: `b4132a4064fb6ff25f2005f367c66e1afaeb62e379902dfed6b76a9bdcde4e17`;
- policy authority: `f30441a3d4de8d4cf7dc657abf4d692236114323c48c8bf955ded35d9ef107c6`.

Attempt 1 is spent and may not be repeated or repaired. Because no cold
launched, a single fresh attempt under the unchanged pre-execution contract may
be separately authorized only after both orchestration defects are fixed at
their root, the changed implementation is fully requalified and rehashed,
independent reviewers issue GO, and an explicit fresh authorization with a new
absent output root is appended below.

## Root-Fix Requalification and Fresh Authorization (append-only)

The runner now creates each exclusive `preflight/` or `cold/` directory before
launching its child, requires the directory to exist when supervision evidence
is published, and tags the process and terminal ArtifactRefs with
`DIAG2_PROCESS_SCHEMA_VERSION` and `DIAG2_CHILD_TERMINAL_SCHEMA_VERSION`.
Receipt production and all frozen numerical files are unchanged. Regression
tests use the real receipt resolver/classifier and independently reject v2
documents wrapped in legacy v1 references for both child modes.

The exact controlling explicit-CPU command recorded above was rerun unchanged
on these bytes. Result: `807 passed in 332.80s (0:05:32)`. The exact 11-file
Ruff check, Ruff format check, compileall, and `git diff --check` commands
recorded above all passed; Ruff reported `11 files already formatted`.

Updated exact SHA-256 identities are:

- runner: `d67df375bde3a6cf542536837b127c480476fbc670b38177c2ed7e3dd73e749c`;
- runner tests: `01b894c2e461209f72165dae9dc7b3c596dd6789c38a1a1dbec185d696fb79b3`;
- v2 contract matrix tests: `e82179c675b9ca372433a096d03e4c2b861e22472fc81c4ffada8005f85dbd11`.

All other change-authorized hashes remain exactly as recorded in the initial
qualification ledger. The eleven frozen numerical/source hashes and sealed
native-reference manifest remain unchanged.

Because Attempt 1 launched no cold child and produced no canonical final
artifact, the post-cold replacement prohibition was not triggered. The prior
literal launch authorization is superseded. Subject to fresh independent GO on
the exact hashes above, exactly one fresh DIAG2 command is authorized at the
new absent final root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-r1-20260811T123501Z`:
one lower-only preflight child and, only after the strict retained preflight and
second supervisor-zero gates pass, at most one cold child.

This authorization does not permit mutation or reuse of Attempt 1's partial,
warm execution, tuning, cache reuse, replacement cold execution, or a second
cold. Any handled failure must publish the truthful sealed nonpromoting final
artifact. Any launched cold consumes this authorization regardless of outcome.

The exact change-authorized qualification bytes were:

- `benchmarks/process_gpu_monitor.py`:
  `6f2e6d3c144a5e31b45533ef288e0bd1dba84066780f0bdffd443f89c3316f39`;
- `benchmarks/single_stage_fullspace_process_gpu_monitor.py`:
  `57918ef564eb66705dd8789cc30f18bc095d3a66b4224888d84daa583f9549e3`;
- `benchmarks/run_single_stage_native_equivalent_quality_campaign.py`:
  `1602991ee0fff64eada0dec605dccf9d921c0cda76b868864910ad706ef5cdb9`;
- `benchmarks/single_stage_fullspace_snapshot.py`:
  `985ddf4b61ab4ecdf1ab6c0e130a1ed98d7e807f233f76ba27fb2bf74f247374`;
- `benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py`:
  `bb7104974dbded616cc1f673eb5987b2bc857dc93144a359518e334b04cf17db`;
- `tests/benchmarks/_diag2_fixture.py`:
  `bad70f7ac5d6021ad0bf3c99efcc1e8d7f9023360265bc67b24c4e43b67c275d`;
- `tests/benchmarks/test_process_gpu_monitor.py`:
  `f776670184493012293638e32711a49340dccdc83a77e008cbd590a1dbad16e1`;
- `tests/benchmarks/test_single_stage_fullspace_snapshot.py`:
  `ca0f4ed6dbd4bab6fb6b85c4c5901192282bce042b652646913955829d0bdf95`;
- `tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py`:
  `b1b01da3581556a789b672083b085191452c3b79a3279854aa0219aadfc1a699`;
- `tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py`:
  `f111b94f4d039517cdfb66e09a342b6e34119d2fcfef48c97211b33c558b3275`;
- `tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py`:
  `d489ccf2e1800a17f7f0a3f5040ed6578e32e7d31a93ca14fbcc45613b671cff`.

The eleven frozen numerical/source paths were rehashed after qualification and
still exactly matched the identities frozen above: trace annotations
`9d50e5fc...`, projected GNTR core `c28a598a...`, fullspace GNTR bridge
`62b7dec2...`, NEQ adapter `abf9726e...`, fullspace scaling `475cb63d...`,
fullspace objective `ca3a09f5...`, bootstrap adapter `910b5913...`, native
endpoint adapter `bad74583...`, native-reference validator `faf7614a...`,
input-bundle loader `303439ea...`, and native Boozer-QA case `3bf7c04e...`.

Independent read-only reviews issued GO on these exact production bytes:

- `/root/diag_runner_map`: implementation and integrated launch-readiness GO;
- `/root/diag_runner_map/ssot_atomic_review`: schema and atomic-publication GO;
- `/root/diag_runner_map/v2_test_matrix`: qualification-coverage GO.

The native reference remained sealed and validator-usable with manifest
SHA-256 `5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db`.
No optimizer numerical file changed, no production artifact was created, and no
GPU computation was performed during qualification.

### Static-command clarification

The placeholder-form static commands above are superseded by this exact
append-only argv record. All four commands passed after the controlling test
run; Ruff format reported `11 files already formatted`.

```text
.venv-qn-cpu/bin/python -m ruff check benchmarks/process_gpu_monitor.py benchmarks/single_stage_fullspace_process_gpu_monitor.py benchmarks/run_single_stage_native_equivalent_quality_campaign.py benchmarks/single_stage_fullspace_snapshot.py benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/_diag2_fixture.py tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py
.venv-qn-cpu/bin/python -m ruff format --check benchmarks/process_gpu_monitor.py benchmarks/single_stage_fullspace_process_gpu_monitor.py benchmarks/run_single_stage_native_equivalent_quality_campaign.py benchmarks/single_stage_fullspace_snapshot.py benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/_diag2_fixture.py tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py
.venv-qn-cpu/bin/python -m compileall -q benchmarks/process_gpu_monitor.py benchmarks/single_stage_fullspace_process_gpu_monitor.py benchmarks/run_single_stage_native_equivalent_quality_campaign.py benchmarks/single_stage_fullspace_snapshot.py benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/_diag2_fixture.py tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py
git diff --check
```

## Final EOF R1 Ledger (append-only and controlling)

This final entry supersedes every earlier effective-byte and review claim for
R1. Earlier sections remain immutable historical records of initial
qualification, Attempt 1, or pre-clarification audits. The frozen contract
prefix remains
`38bf768c8c851347e9178596f6dcec8f3fb43ff88030a3dd953066999df97f78`.

The controlling CPU result is `807 passed in 332.80s (0:05:32)`. The literal
11-file Ruff check, Ruff format check, compileall, and diff-check commands
immediately above all passed. The sole effective R1 SHA-256 manifest is:

- monitor `6f2e6d3c144a5e31b45533ef288e0bd1dba84066780f0bdffd443f89c3316f39`;
- process identity helper `57918ef564eb66705dd8789cc30f18bc095d3a66b4224888d84daa583f9549e3`;
- runner `d67df375bde3a6cf542536837b127c480476fbc670b38177c2ed7e3dd73e749c`;
- snapshot `985ddf4b61ab4ecdf1ab6c0e130a1ed98d7e807f233f76ba27fb2bf74f247374`;
- receipt `bb7104974dbded616cc1f673eb5987b2bc857dc93144a359518e334b04cf17db`;
- shared fixture `bad70f7ac5d6021ad0bf3c99efcc1e8d7f9023360265bc67b24c4e43b67c275d`;
- monitor tests `f776670184493012293638e32711a49340dccdc83a77e008cbd590a1dbad16e1`;
- snapshot tests `ca0f4ed6dbd4bab6fb6b85c4c5901192282bce042b652646913955829d0bdf95`;
- runner tests `01b894c2e461209f72165dae9dc7b3c596dd6789c38a1a1dbec185d696fb79b3`;
- receipt tests `f111b94f4d039517cdfb66e09a342b6e34119d2fcfef48c97211b33c558b3275`;
- v2 matrix tests `e82179c675b9ca372433a096d03e4c2b861e22472fc81c4ffada8005f85dbd11`.

All eleven frozen numerical/source hashes and native-reference manifest
`5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db`
remain unchanged. Fresh pre-EOF reviewers found the implementation,
receipt/schema, atomic-publication, and qualification-coverage surfaces GO on
this manifest and withheld launch only for ledger ambiguity. R1 remains
conditional on their post-EOF confirmation.

Subject to that confirmation, the sole fresh target is the currently absent
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-r1-20260811T123501Z`.
Attempt 1's partial is immutable. Exactly one R1 preflight and, only after all
strict gates pass, at most one cold are permitted. No warm, tuning, partial
reuse, replacement cold, or second cold is authorized.

## R1 Executed Outcome (append-only final adjudication)

The R1 command completed with exit code zero from the supervisor and atomically
published the canonical final root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-r1-20260811T123501Z`.
The public CPU-only final-path loader independently returned
`DIAGNOSTIC_INCOMPLETE`, failure `COLD_CRASH/CHILD_EXIT_NONZERO`, historical
relation `NOT_COMPARABLE_INCOMPLETE`, and next route `NOT_PRODUCED`.
Engineering campaign receipt production and promotion authorization are false;
formal comparison is `NOT_PRODUCED`.

Exactly `preflight` then `cold` launched. Preflight completed successfully in
compile-only mode with zero callbacks and no solver dispatch. The cold child
ran 321.842099928 seconds, dispatched the compiled solver loop, and failed at
`jax.block_until_ready(loop_result)` while retrieving `branch_index` with
`CUDA_ERROR_LAUNCH_FAILED`. Its stdout is empty and no cold producer exists.
Exact-PID monitoring retained 2,678 samples with peak 26,023,559,168 bytes
(24,818 MiB, 76.11249118287484% of device memory). The retained stderr contains
no `OUT_OF_MEMORY` or `RESOURCE_EXHAUSTED` marker, so the artifact does not
adjudicate an allocator OOM.

No cold history, typed trace interval, terminal numerical evidence, or
execution evidence was produced. The raw Chrome and XPlane failure traces are
retained and manifested, but they cannot support an accepted-state trajectory,
quality/no-hit, convergence, speed, or algorithmic next-route claim.

Final artifact integrity is independently GO: 665 regular files all `0444`, 63
descendant directories plus the root all `0555`, zero symlinks or hardlinks,
and 664 sorted unique manifest entries exactly covering every regular file
except the manifest itself. No staging sibling, warm path, or `campaign.json`
exists.

Key final SHA-256 identities are:

- diagnostic receipt: `b671fea9294991ad3749fc115dab24ce3abe60c23b699da9dbe8904641bcdae7`;
- artifact manifest: `c6d244c9e82d31edc04036fff0722b75b59334d36aec599746c9e8d302500029`;
- supervisor terminal: `6de766bb34376deed92743ecb364c368a4cf268e0c7b8844da208a5c624ef3ef`;
- cold terminal: `4b84a1719f3da974c72df2fffd819242539cb15a0788329ad7301ac9e9f7ea21`;
- cold process: `efd55cb365b9a13308f9c748c4a31f58bf3ea097de2d6e0a20806c835cd089c5`;
- cold stderr: `debed91bf17f9eea2d0a56bd53df59ed4dfc1872065fee6a67d3cd7e1eb0e26a`;
- raw Chrome trace: `4db22ea34a8092e3fc55e366a4ffe83055951bc2425322ed5f874c2bfe9b8c4e`;
- raw XPlane trace: `9983075bf5c00f2c1fc98be9492e08111f8f436081b6abf0e6c4f263d8f57fad`.

This authorization is consumed. The artifact is the truthful final outcome of
the goal: immutable and independently reloadable, but scientifically
incomplete and nonpromoting. Any successor cold requires a new SSOT, schema,
output root, and explicit authorization.
