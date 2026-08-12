# NEQ-GNTR3 DIAG4 Step-Bound-Safeguard Successor Contract

Status: unconsumed implementation contract only. This document does not
authorize a GPU launch, no DIAG4 authority JSON exists, and no result described
below is a parity, GPU, or speed result.

## Objective and claim boundary

Produce one canonical, deeply reloadable, trace-free artifact for an
identity-distinct numerical route that answers the scientific question before
the performance question:

1. does NEQ-GNTR3 satisfy the frozen native-equivalent mathematical and physical
   quality policy; and
2. only if it does, what synchronized solve time was observed relative to the
   historical native C++ engineering threshold?

No timing observation may promote a route whose numerical or physics gate did
not pass. CPU qualification, a successful preflight, finite values, a zero exit,
or a sealed artifact is not numerical parity.

## Consumed evidence and successor diagnosis

The consumed DIAG3 attempt is the immutable partial root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag3-cb0-20260811T150010Z.partial-56a1ec6d730cc005db84f99e9965b868`.
It is not a scientific receipt and must never be mutated or promoted.

DIAG3 established two independent failures:

- numerical: `ATTEMPT_LIMIT`, 300 logical attempts, 203 accepted, 94
  `RETRY_FEASIBILITY`, two `RETRY_STEP_BOUNDS`, one `RETRY_OBJECTIVE`, and no
  native-quality latch; and
- evidence finalization: profiler XSpace exceeded 2 GiB and Chrome normalization
  found no in-envelope device intervals after valid history and terminal arrays
  had already been written.

NEQ-GNTR2, the max-two nonlinear-retraction route specified by the prior form of
this unconsumed plan, removes the dominant feasibility rejection but leaves a
structural step-bound failure mode. This revision does not relabel NEQ-GNTR2.
It defines the additive NEQ-GNTR3 numerical identity and retains DIAG4 as the
evidence-loader generation.

The sole GNTR3-versus-GNTR2 option delta is:

`enable_step_bound_safeguard: false -> true`.

Increasing only the logical-attempt budget, weakening any frozen bound, retrying
DIAG3, relabeling a canary, or restoring profiling is prohibited.

## Route, loader, and schema identity

The successor identities are:

- numerical route: `NEQ-GNTR3`;
- evidence route: `NEQ-GNTR3-DIAG4`;
- loader generation: `v4`, with no `v5` loader or v5 alias;
- numerical result: `single-stage-fullspace-neq-gntr3-result-v1`;
- scientific receipt: `single-stage-neq-gntr3-trace-free-diagnostic-v1`;
- artifact manifest:
  `single-stage-neq-gntr3-trace-free-artifact-manifest-v1`;
- execution-source authority:
  `single-stage-neq-gntr3-execution-source-authority-v1`;
- cold producer: `single-stage-neq-gntr3-trace-free-cold-result-v1`;
- committed numerical bundle:
  `single-stage-neq-gntr3-trace-free-numerical-bundle-v1`;
- solve timing: `single-stage-neq-gntr3-solve-timing-v1`;
- safeguard telemetry:
  `single-stage-neq-gntr3-step-bound-safeguard-telemetry-v1`;
- successor authority: `single-stage-neq-gntr3-diag4-authorization-v1`;
- authority consumption:
  `single-stage-neq-gntr3-diag4-authority-consumption-v1`; and
- qualification record: `single-stage-neq-gntr3-diag4-qualification-v1`.

The in-place v4 implementation must replace the unconsumed GNTR2-DIAG4 route
constants and fixtures. No GNTR2-DIAG4 authority or result is grandfathered.
Legacy v1, v2, and v3 receipt loaders remain byte-for-byte behaviorally
unchanged. They must not dispatch to v4. The v4 loader accepts only its exact
GNTR3 route/schema family and rejects v1/v2/v3 objects, the obsolete unconsumed
GNTR2-DIAG4 schema family, and any invented v5 label. Legacy loaders reject all
v4 objects.

The v4 receipt evidence vector has exactly 26 slots in this order:

1. `source_manifest`;
2. `frozen_numerical_subset`;
3. `native_reference`;
4. `policy_authority`;
5. `supervisor_before_preflight`;
6. `preflight_producer`;
7. `preflight_terminal`;
8. `preflight_process`;
9. `preflight_memory`;
10. `preflight_memory_samples`;
11. `preflight_runtime`;
12. `preflight_policy`;
13. `supervisor_before_cold`;
14. `cold_producer`;
15. `cold_terminal`;
16. `cold_process`;
17. `cold_memory`;
18. `cold_memory_samples`;
19. `cold_runtime`;
20. `cold_policy`;
21. `cold_history`;
22. `cold_terminal_numerical`;
23. `cold_solve_timing`;
24. `cold_safeguard_telemetry`;
25. `execution`; and
26. `supervisor_terminal`.

Each slot is exactly either `PRESENT` with one typed artifact reference or
`ABSENT` with one closed reason. The group prefixes are setup slots 1--4,
preflight slots 5--12, cold slots 13--24, and terminal slots 25--26. A later
execution group among setup, preflight, and cold cannot contain a `PRESENT` slot
unless the gate closing every earlier execution group passed. Terminal slots are
the explicit exception: after staging exists, they close a handled earlier-stage
failure and may be `PRESENT` while later execution groups are absent. Within a
launched preflight or cold group, raw supervision slots may be `PRESENT` even
when its producer is absent. Slots 21--24 are one atomic scientific subgroup:
they are all `PRESENT` only after the parent commits the numerical bundle, and
otherwise all four are `ABSENT`. Slot order, group prefix, and subgroup
cardinality are schema, not presentation choices.

The v4 vector has no `cold_raw_trace` or `cold_trace_intervals` slot. The v4
loader and manifest reject those legacy slot names; every path under
`cold/raw-trace`, `plugins/profile`, or any trace-interval location; and roles
`raw_trace_chrome`, `raw_trace_xplane`, or `trace_intervals`. Unknown trace
aliases fail identically.

The frozen contract-prefix SHA-256 is computed over every byte preceding the
`## Qualification Record` heading. The qualification record binds that prefix,
every qualified and frozen source/test byte, the native reference, the consumed
DIAG3 evidence manifest, the two historical CPU20 files, the complete
current-byte CPU qualification artifact, and the exact absent GPU output root.
It must not claim a hash of the file containing itself. Only after the canonical
record is appended may a separate authority bind the completed plan's full-file
SHA-256. This file remains unconsumed while the record is blank.

## Numerical-policy inheritance and options identity

All physics, objective, layout, scaling, derivative, endpoint, quality, and
trust-region values remain equal to NEQ-GNTR2. In particular:

- 716 physical/optimizer coordinates, 255 equalities, and 2110 objective
  residuals;
- native raw-equality vector and constraint inverse scale;
- native objective target `4.4822246533126125e-08`;
- component absolute/relative bounds and scaled-feasibility tolerance `1e-10`;
- maximum 256 accepted steps and 300 logical attempts;
- initial/minimum/maximum trust radii;
- Steihaug iteration and certificate tolerances;
- `maximum_nonlinear_corrections = 2`;
- correction-step ratio and corrected-radius bounds;
- accepted-state quality predicate; and
- terminal replay, derivative transpose, KKT, and endpoint checks.

The legacy NEQ-GNTR1 base-policy SHA-256 remains
`6face7116d36d2eae954bb5b3bde465f37c990ceb9b319ba2c20bb62ad1a6f99`.

Options identity is versioned independently from result schemas:

- `projected-gauss-newton-options-v1` has the exact historical ordered field set
  ending in `maximum_nonlinear_corrections`. It excludes
  `enable_step_bound_safeguard`. Existing NEQ-GNTR1 and NEQ-GNTR2 v1 hashes must
  remain unchanged when the new dataclass field exists; serialization must use
  an explicit v1 field tuple and must not iterate all current dataclass fields.
- `projected-gauss-newton-options-v2` has the same ordered v1 prefix followed by
  exactly `enable_step_bound_safeguard`. NEQ-GNTR3 uses v2 with the literal
  value `true`. No legacy route may silently acquire a v2 identity.
- The v1 and v2 tags, field order, field names, Python scalar types, and exact
  numeric values are hash input. A bool-as-int substitution, missing field,
  reordered field, v1 hash drift, or v2 hash collision fails qualification.

For the frozen option values, the required exact hashes are NEQ-GNTR1 v1
`dcd481184681563551a0631d1da93b8ec0f12aacc81d4dd2e4a1e55cdc9787f7`,
NEQ-GNTR2 v1
`ed187997094f68888c2c36009551a10dec510ac6661766306c2832f621d74460`,
and NEQ-GNTR3 v2
`d9c2545f79815e057e55ae03648a0d23ab0518af51b8a6fb36246bfff796917f`.
Qualification independently recomputes all three from explicit field tuples.

The NEQ-GNTR3 canary/result identity binds the complete `FullSpaceProblem`, base
policy, v2 options, scaling, bootstrap state, initial physical state, numerical
result schema, and route. It must differ from NEQ-GNTR2 even when every state and
policy byte is shared.

## Fixed safeguard state machine

One logical attempt has fixed conditional capacity three: the initial subtrial
at the incoming radius followed by at most two complete re-solves. Each re-solve
uses the clipped quarter radius returned by the immediately preceding executed
subtrial. Thus, before minimum-radius clipping, the radii are exactly `r`,
`0.25*r`, and `0.0625*r`. Capacity is compile-time policy; it is not a tuning
parameter.

The exact continuation trigger is true only when every clause below is true for
the immediately preceding subtrial:

1. its outcome is exactly `RETRY_STEP_BOUNDS`;
2. its predicted reduction is finite and strictly greater than zero;
3. its actual reduction is finite and strictly greater than zero;
4. its returned next radius is strictly less than its incoming radius; and
5. fewer than three subtrials have executed in this logical attempt.

The next radius is exactly the existing structural-failure update
`clip(0.25 * incoming_radius, minimum_trust_radius, maximum_trust_radius)`.
The next subtrial recomputes the entire current linearization, Steihaug solve,
nonlinear retraction, candidate, certificates, and acceptance logic at that
radius. It may not reuse a rejected tangent, correction, candidate, derivative,
or work count.

The safeguard stops at the first false trigger or after the third subtrial. It
never activates for feasibility, objective, nonfinite, correction-certificate,
Steihaug, curvature, or other fatal/retry outcomes. A step-bound row with a
nonpositive/nonfinite reduction or without strict radius decrease also stops.
There is no hidden fourth subtrial.

All subtrials in one safeguard sequence read the same logical-attempt incumbent:
coordinates, objective, constraints, accepted-step count, accepted ledger,
quality latch, first-quality indices, mechanism latch, fatal/status state, and
logical attempt number. Intermediate rejected subtrials may update only the
private selected-result candidate and the subtrial evidence ledger. They never
mutate the incumbent or accepted ledger.

The last executed subtrial is the selected subtrial. Only its ordinary attempt
result is committed to the loop state and legacy one-row history. Therefore:

- `attempts` increments exactly once per logical attempt, not per subtrial;
- `retryable_rejections` increments once only if the selected result rejects;
- `accepted_steps`, the accepted physical/optimizer ledger, quality latch,
  first-quality indices, and terminal state change only if the selected result
  accepts;
- the next logical attempt receives only the selected result's radius and state;
  and
- a fatal selected result terminates the loop exactly as the legacy attempt
  would.

The outer loop remains fixed at 300 logical attempts. With three conditional
subtrials per attempt, the maximum is exactly 900 complete subtrials. The 900
ceiling is a work/evidence bound, not permission for 900 logical rejections or a
larger accepted-step budget.

## Safeguard telemetry and work accounting

The numerical result and committed safeguard-telemetry document contain exactly
24 typed envelopes: four per-attempt outer-history fields, two per-attempt
safeguard scalar fields, and 18 per-subtrial safeguard matrix fields.

The four outer-history fields have shape `(300,)` in this order:

1. `nonlinear_corrections`, dtype signed 32-bit integer;
2. `maximum_individual_correction_step_ratio`, dtype FP64;
3. `correction_path_step_ratio`, dtype FP64;
4. `steihaug_solve_calls`, dtype signed 32-bit integer.

The two scalar fields have shape `(300,)`, dtype signed 32-bit integer:

1. `subtrial_count`;
2. `selected_subtrial_index`.

The 18 matrix fields have shape `(300, 3)` in this order:

1. `subtrial_trust_radius`;
2. `subtrial_outcome`;
3. `subtrial_actual_reduction`;
4. `subtrial_predicted_reduction`;
5. `subtrial_maximum_individual_correction_step_ratio`;
6. `subtrial_correction_path_step_ratio`;
7. `subtrial_corrected_radius_ratio`;
8. `subtrial_steihaug_iterations`;
9. `subtrial_steihaug_hvp_evaluations`;
10. `subtrial_steihaug_solve_calls`;
11. `subtrial_total_hvp_evaluations`;
12. `subtrial_nonlinear_corrections`;
13. `subtrial_joint_evaluations`;
14. `subtrial_joint_linearizations`;
15. `subtrial_joint_value_evaluations`;
16. `subtrial_objective_residual_linearizations`;
17. `subtrial_gram_factorizations`;
18. `subtrial_gram_solves`.

Matrix fields 1 and 3--7 are FP64. Matrix field 2 and matrix fields 8--18 are
signed 32-bit integer. The exact 24-envelope set admits no additional envelope
or alternate spelling without a new result schema.

For each active logical-attempt row, `subtrial_count` is in `[1, 3]`,
`selected_subtrial_index == subtrial_count - 1`, and executed columns are the
exact prefix `[0, subtrial_count)`. For inactive outer-loop rows,
`subtrial_count == 0` and `selected_subtrial_index == -1`.

Every one of the 24 fields is serialized as its own typed envelope with exactly
the keys `{dtype,shape,values,sha256}`. The two integer outer-history envelopes,
count, index, outcome, and work envelopes use `dtype="<i4"`; the two FP64
outer-history envelopes and float-matrix envelopes use `dtype="<f8"`. Shape is
exactly `[300]` for the six vectors or `[300,3]` for the 18 matrices. Envelope
`sha256` is computed over canonical JSON of exactly `{dtype,shape,values}`, with
the hash field excluded. There is no global dtype, shape, values, or checksum
metadata that can override an envelope. The runner rejects raw dtype/shape drift
before conversion; the receipt independently checks exact envelope keys, values,
and hash.

Unused raw FP64 columns are canonical quiet NaN values and serialize only as JSON
`null`; a JSON NaN string/token is forbidden. Unused integer work columns are
zero. Unused `subtrial_outcome` columns are the int32 code for the exact
`INACTIVE` enum value. Any non-prefix use, alternate null/NaN/padding convention,
nonzero inactive work, wrong dtype/shape, envelope/hash drift, or selected index
outside the executed prefix is invalid.

Executed radii are finite, positive, within the frozen trust-radius interval,
and match the exact initial/quarter/quarter recurrence including clipping.
Every executed outcome is a valid non-`INACTIVE` enum. Executed floating
telemetry is finite exactly when that outcome reached the corresponding value;
otherwise it has the canonical unavailable NaN. The validator applies the same
outcome-conditioned availability rules as ordinary history and rejects a NaN
used to conceal a value that was computed.

For every executed column, work values are nonnegative and exactly obey:

- `trial_evaluated = (subtrial_nonlinear_corrections > 0)`;
- `subtrial_total_hvp_evaluations = 3 +
  subtrial_steihaug_hvp_evaluations + int(trial_evaluated)`;
- `subtrial_joint_evaluations = 1 + (2 +
  subtrial_nonlinear_corrections if trial_evaluated else 0)`;
- `subtrial_joint_linearizations = 1 +
  subtrial_nonlinear_corrections`;
- `subtrial_joint_value_evaluations = 2 if trial_evaluated else 0`;
- `subtrial_objective_residual_linearizations = 1`;
- `subtrial_gram_factorizations = 1 +
  subtrial_nonlinear_corrections`; and
- `subtrial_gram_solves = 2 + subtrial_nonlinear_corrections +
  subtrial_steihaug_solve_calls * (3 +
  subtrial_steihaug_hvp_evaluations)`.

The fixed `2` in the work equations is
`maximum_nonlinear_corrections`, not an independently configurable telemetry
constant. Steihaug iterations, HVP evaluations, and solve calls must also obey
the frozen solver bounds and ordinary-history certificate relations.

The selected matrix column must exactly equal the corresponding ordinary
history values for trust radius, outcome, reductions, step-bound ratios,
Steihaug counts, and nonlinear-correction count. In particular, the four outer
envelopes independently join to the selected columns of
`subtrial_nonlinear_corrections`,
`subtrial_maximum_individual_correction_step_ratio`,
`subtrial_correction_path_step_ratio`, and `subtrial_steihaug_solve_calls`,
respectively. These envelopes preserve four independently hashable
selected-column joins; they project existing ordinary-history leaves and do not
add or alter a legacy-history leaf or parser. Ordinary history contains only the
selected subtrial. Nonselected executed columns remain immutable evidence and
contribute to work totals but never to accepted-state counts.

The summary schema has exactly these 19 ordered keys:

1. `total_subtrials`;
2. `total_shadow_subtrials`;
3. `maximum_subtrial_count`;
4. `logical_attempts_with_1_subtrial`;
5. `logical_attempts_with_2_subtrials`;
6. `logical_attempts_with_3_subtrials`;
7. `recovered_step_bound_attempts`;
8. `exhausted_step_bound_attempts`;
9. `total_steihaug_iterations`;
10. `total_steihaug_hvp_evaluations`;
11. `total_steihaug_solve_calls`;
12. `total_hvp_evaluations`;
13. `total_nonlinear_corrections`;
14. `total_joint_evaluations`;
15. `total_joint_linearizations`;
16. `total_joint_value_evaluations`;
17. `total_objective_residual_linearizations`;
18. `total_gram_factorizations`;
19. `total_gram_solves`.

It has no selected-outcome-count key. Recovered/exhausted counts and any broader
selected-outcome census are derived from the selected-row and ordinary-history
join. For active row `i`, let `selected_outcome[i]` be
`subtrial_outcome[i, selected_subtrial_index[i]]`. The two step-bound counts are
frozen exactly as:

- `recovered_step_bound_attempts = sum_i int(subtrial_count[i] > 1 and
  selected_outcome[i] == ACCEPTED)`; and
- `exhausted_step_bound_attempts = sum_i int(subtrial_count[i] == 3 and
  selected_outcome[i] == RETRY_STEP_BOUNDS)`.

Every summary is independently recomputed only from the fixed two
safeguard scalar arrays and 18 subtrial matrices; the four outer-history
envelopes are independent join evidence, never additional summary inputs or
keys. The summary is joined to the producer, history, terminal result, route,
result schema, plan, problem, base policy, v2 options, scaling, bootstrap, and
initial-state hashes. Total executed subtrials is at most 900.

Mutations of any outer-history field, either safeguard scalar field, any matrix
element, shape, dtype, padding, trigger, radius recurrence, selected-column
join, work equation, summary, identity, or 900 ceiling fail at
`NUMERICAL_COMMIT/SAFEGUARD_TELEMETRY_INVALID`. The legacy history schema and all
v1/v2/v3 parsers remain unchanged.

## Historical nonpromoting CPU20 route-selection evidence

The completed CPU20 safeguard replay is historical route-selection evidence
only. It selected the step-bound safeguard for implementation; it is not
current-byte qualification and cannot authorize GPU. Its exact files are:

- result `/tmp/diag4_safeguard_cpu20_result.json`, SHA-256
  `7f08eadfe17e3a18f4c8f480d482945ddff3df4ed5a6092f4252ef1d054846fe`;
- harness `/tmp/diag4_safeguard_cpu20.py`, SHA-256
  `bb339b9968885f642af2d247c92ef01a26f9dc8a0e639f539a29cc7060897525`.

The literal recorded command is:

```text
/usr/bin/env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src /home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-cpu/bin/python /tmp/diag4_safeguard_cpu20.py
```

It ran at Git HEAD `52dea17ddf3012cf923fc92da78c0d73a17f4625` with problem SHA-256
`e6df89b64cf2bf4d5d6dce48b62380a0bc975cba0365eb0762dc78e75f1de6df`,
scaling SHA-256
`6888ae5ab558d466fa6fa931577b5df8cf9981884d8bf3ae7d7a9d9e8c379ff2`,
initial physical/bootstrap-state SHA-256
`9d3dd46e70fbc46c2c50cb8d347d08754843e7d5e4d6b63864a848b0e5b0cdcc`,
and base-policy SHA-256
`6face7116d36d2eae954bb5b3bde465f37c990ceb9b319ba2c20bb62ad1a6f99`.
The harness reported three production-NEQ-GNTR2-to-safeguarded-CPU20 option
differences: `maximum_accepted_steps: [256, 20]`,
`maximum_attempts: [300, 20]`, and
`enable_step_bound_safeguard: [false, true]`. The first two are bounded-harness
budget reductions only. CPU20-base and safeguarded CPU20 both use the 20/20
budget, so their sole option difference is
`enable_step_bound_safeguard: [false, true]`. The production route delta from
NEQ-GNTR2 to NEQ-GNTR3 remains only that boolean; its 256/300 production budget
is unchanged.

The historical harness also reported CPU20-base options SHA-256
`73699aed67e93c696c57c4d124fdee508fc81bbfbc1648a58c8d82e5b05ba794`,
production-NEQ-GNTR2 options SHA-256
`790344781f1105d00498985e10a0f72fafb9bcd7d628a611356819dfeb5a3c4d`,
safeguarded options SHA-256
`d490b943babf8d7c10bd01b8d76187901dbc563b752e96dda6ac07df09114663`,
and canary identity SHA-256
`257d6c73b2159da16c44883c59d566b39ac80df358a75eec1cb943e1d3c0968f`.
Those values used the historical canary serializer and are preserved as evidence
only; they are not the explicit production v1/v2 hashes above.

Its embedded historical source/test map is exact:

- `src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py`:
  `00076da8c28301f0fd02d57cf602f3c118dbd6e5dc53dc35d4cd72fe99b44138`;
- `src/simsopt_jax/solve/fullspace_native_equivalent_quality.py`:
  `3e33a91375befe17ec87d56d507d0ce8aaa5144a2d88cd384ab19275b7f1ec25`;
- `src/simsopt_jax_adapters/geo/single_stage_fullspace.py`:
  `910b59131cc9137fee65a8d14222eeccbc0cf3d61d300a63250a95469c413e4e`;
- `tests/geo/test_fullspace_native_equivalent_quality.py`:
  `6441ec8b600636a08862f931d7c1c253baffb399f9effbacd8a35ffac650b8d3`;
- `tests/geo/test_projected_gauss_newton_trust_region.py`:
  `1d7542ce8e19d6e268f9f5d51d8f24966e81d27ce4d5b0aabd528eef4f114b66`.

The fullspace source subsequently changed to add the GNTR3 production API; at
this reconciliation it is
`4f9526e32ff6b5d490cdc100714cd589dd2e2f582ef4311db897b02479e17fd0`,
not the historical `3e33a913...` byte identity. Qualification must record the
actual live hash again after implementation freezes; the reconciliation hash is
diagnostic, not a frozen future value.

The exact bounded result was 20 attempts, 20 accepted steps, zero retryable
rejections, all 20 accepted-attempt indices `1..20`, terminal objective
`4.6469179685276666e-05`, terminal scaled feasibility
`1.6147031125045906e-15`, and terminal scaled stationarity
`3.0372115655007777e-04`. All recorded accepted states and final-certificate
values were finite, the local frozen mechanism gates passed, and the subtrial
structure/trigger/work gates passed. Synchronized loop time was
`497.86480433499673 s`, but timing is qualification diagnostics only.

The result explicitly records `promotion_eligible=false` and
`endpoint_local_audit_passed=false`. Its objective is far above the native target
`4.4822246533126125e-08`. It therefore establishes no native parity, no terminal
quality hit, no GPU behavior, and no speed claim. It cannot populate a DIAG4
scientific slot or replace the authoritative bounded cold.

The qualification record binds both historical file hashes, their paths,
command, reported identities, all frozen gate values, all gate booleans, exact
result summary, and the five embedded historical source/test hashes. The harness
records `one_shot_no_retry=true`: it must never be rerun or regenerated. If
either `/tmp` file is absent or differs when qualification is sealed,
qualification stops and a new SSOT is required; prose is not a substitute.

## Decisive complete current-byte CPU qualification

Only a separate complete current-byte FP64 CPU trajectory plus its independently
compiled endpoint audit can authorize creation of the GPU authority. It must use
the exact final frozen numerical map, qualified runner/receipt bytes, GNTR3 v2
options, full 256-accepted-step/300-logical-attempt policy, native reference,
input bundle, bootstrap, scaling, and initial physical state later bound by the
GPU authority.

The dedicated producer is
`benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py`, with
contract tests in
`tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py`.
It is a CPU-only qualification harness, not a DIAG4 GPU child and not an
authority consumer. It compiles and dispatches the exact production
`NEQ_GNTR3_OPTIONS` loop, fully materializes the accepted ledger, executes the
independently compiled terminal endpoint graph, reconstructs every scientific
gate, writes all four outer-history, two safeguard-scalar, and 18 safeguard-matrix
typed envelopes, and reuses the public history, safeguard-telemetry, terminal,
and endpoint validators used by the production receipt path.

The current-byte CPU qualification schema is
`single-stage-neq-gntr3-cpu-trajectory-qualification-v1`. Its exact final root is
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-cpu-qualification-20260811T214932Z`,
which must be absent before the one authorized qualification run. Its source
manifest covers the exact canonical execution-source authority described below,
not a rediscovered directory glob. The qualification document and artifact
manifest both contain exact fields `execution_source_manifest_sha256` and
`execution_source_entries_sha256`; they must equal the qualified manifest-file
bytes and its internally recomputed aggregate, respectively. Its artifact
manifest also closes the command/environment, interpreter,
route/result/options identities, raw history and arrays, accepted ledger,
endpoint audit, safeguard telemetry, scientific reconstruction, and every file
mode/size/digest.

The harness creates exactly one same-parent staging root, binds its inode, writes
only beneath it, seals and independently deep-loads it, then publishes
staging-to-final with no replacement and fsyncs the parent before a final
independent deep load. Collision, partial publication, rename/fsync/deep-load
failure, noncanonical bytes, or any existing final root is terminal. The run is
one-shot with no retry or replacement under this SSOT.

Authorization requires this CPU artifact to deep-load from its final path and
yield `SCIENTIFIC/QUALITY_HIT`: the loop-returned state, accepted-state first-hit
replay, objective target, all 255 component gates, scaled feasibility,
derivatives, KKT evidence, finite physics observables, and independent endpoint
evaluation must all pass. `INCOMPLETE`, `NO_HIT`, attempt exhaustion, endpoint
audit failure, identity drift, a partial root, or noncanonical evidence blocks
GPU authority. CPU duration is diagnostic and creates no CPU/GPU or speed claim.

The decisive artifact is generated once only after all implementation/test edits
and four pre-run byte reviews are complete. Any later qualified/frozen byte
change invalidates this SSOT and requires a successor plan, root, and
qualification; it does not permit replacement under this one-shot contract. The
historical CPU20 harness likewise remains non-rerunnable.

## Canonical execution-source authority

The sole repository SSOT for every byte copied into an execution-source tree is
`benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json`.
It is strict canonical JSON with schema
`single-stage-neq-gntr3-execution-source-authority-v1` and exactly the top-level
keys `{schema_version,entries,entries_sha256}`. `entries` is an object mapping a
canonical repository-relative POSIX path to an object with exactly
`{sha256,size_bytes}`. Paths are nonempty, normalized, unique after
normalization, contain no `.` or `..` component, and name regular nonsymlink
files. SHA-256 values are lowercase 64-digit hex over exact raw file bytes;
`size_bytes` is the exact nonnegative integer byte length.

Live source mode is intentionally not an entry identity field. The bootstrap
requires a readable regular nonsymlink source through its held descriptor, then
publishes every copied source with the canonical sealed mode `0444`; source-side
permission bits neither select bytes nor survive into the execution tree. File
type, path, size, and bytes remain mandatory, while readability and destination
mode are independently validated lifecycle conditions.

`canonical_json_bytes(value)` is frozen here as UTF-8 encoding of
`json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"),
sort_keys=True) + "\n"`. The exact aggregate is
`entries_sha256 = sha256(canonical_json_bytes(entries)).hexdigest()`. The
manifest file itself is serialized with the same canonical algorithm. Its raw
file-byte SHA-256 is `execution_source_manifest_sha256`; the aggregate is
`execution_source_entries_sha256`. Neither hash may be substituted for the
other.

The exact deduplicated entry-key set is:

1. every filesystem-visible regular nonsymlink `*.py` beneath `benchmarks/`,
   `src/`, and `examples/`, recursively, without applying Git, ignore-file,
   `rg`, or `fd` filtering;
2. the exact 23-path `qualified_files` set below minus the manifest path itself;
3. the exact 11-path `frozen_numerical_entries` set below.

At this freeze the first class contains exactly 591 paths: 113 below
`benchmarks/`, 322 below `src/`, and 156 below `examples/`. It includes the
Git-ignored `src/simsopt/_version.py`. After deduplication with the 12 qualified
test paths, `entries` contains exactly 603 paths. The manifest is excluded from
its own `entries` to avoid a self-hash cycle; it is the 23rd qualified file and
is copied separately as immutable control metadata. This plan is not an entry.
Any
missing entry, additional current `*.py`, ignored-file omission, symlink or
special-file substitution, duplicate/alias path, size drift, byte drift, count
drift, noncanonical JSON, extra key, or aggregate mismatch invalidates the
execution-source authority before execution.

The qualifier's stdlib-only pre-import bootstrap opens and validates the
canonical manifest, recomputes both hashes and the exact no-ignore membership,
then opens each listed source with no-follow semantics and verifies its type,
size, and digest from the held descriptor. It copies exactly the 603 entry keys
from those descriptors plus the separately held manifest control file; broad
directory traversal is validation only and can never select a copied file. It
seals the copy, proves exact set equality to `entries` plus the manifest, and
executes the worker only from that sealed tree. The worker and final validator
join every imported production-source binding and the entrypoint to its listed
entry; an import outside the sealed tree or an unlisted copied byte is invalid.

The CPU source snapshot has exactly four disjoint membership classes: the 603
repository-relative execution entries; the manifest at its own repository path
with role `execution_source_manifest`; the blank-ledger plan copied to the
distinct snapshot path `control/prequalification-plan.md` with role
`prequalification_plan`; and exactly one native-extension entry with role
`native_extension`. The control plan path is not a repository entry alias and
cannot collide with the manifest or an execution entry. The CPU qualification
field `prequalification_plan_control` has schema
`single-stage-neq-gntr3-prequalification-plan-control-v1` and exactly the keys
`{schema_version,snapshot_relative_path,source_relative_path,sha256,size_bytes,plan_prefix_sha256}`.
Its snapshot path is exactly `control/prequalification-plan.md`; its source path
is exactly
`docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md`;
`sha256` and `size_bytes` bind the complete blank-ledger plan bytes, while
`plan_prefix_sha256` independently equals the frozen plan prefix. The CPU
qualification joins this typed control, its recorded manifest bytes, both
execution-source hashes, exact source-snapshot classes, and imported-source
bindings.

The later GPU source snapshot has exactly three membership classes: the same
603 execution entries and manifest bytes from the sealed CPU-qualified snapshot,
not from a changed worktree, plus exactly one separately typed native-extension
entry. It does not carry the blank-ledger control plan. After the Qualification
Record is appended, the completed live plan is bound separately by plan-prefix
SHA-256 and completed-plan SHA-256 and remains outside both the execution
manifest and GPU source snapshot. The authority, native reference, inputs, and
consumed DIAG3 evidence likewise remain separately typed controls and cannot
overwrite a snapshot entry.

The native extension has an independent three-way identity. The authority binds
exact fields `native_extension_path`, `native_extension_sha256`, and
`native_extension_size_bytes`. `native_extension_path` is the exact resolved
absolute loaded live `simsoptpp.__file__` path and must equal the absolute loaded
path reported by both CPU and GPU runtime identities. Each CPU/GPU snapshot
`native_extension` entry instead has the deterministic repository-independent
`relative_path == "native/" + Path(native_extension_path).name`; no additional
snapshot-path authority field exists. Both snapshot entries' raw-byte SHA-256
and size equal `native_extension_sha256` and `native_extension_size_bytes` and
the held live descriptor.

The CPU qualification scientific-evidence top level contains
`execution_source_manifest_sha256`, `execution_source_entries_sha256`,
`prequalification_plan_control`, `native_extension_path`,
`native_extension_sha256`, and `native_extension_size_bytes`. Its CPU runtime
identity repeats the native absolute path/hash/size triple, and the summary,
runtime, CPU snapshot entry, and live descriptor must join exactly. The GPU
runtime and GPU snapshot entry later join the same authority triple.

The authority supervisor opens the current live native binary with no-follow
semantics, proves this identity, retains its descriptor through the complete
authority lifetime, and revalidates the descriptor/path inode, absolute path,
size, and bytes before each child, after final publication, and before unlock.
The GPU child imports and reports that held, current exact binary; a copied
stale extension, alternate installation, or different loaded path/hash/size is
invalid. Native live mode is omitted for the same reason as other live-source
mode: descriptor accessibility is a lifecycle gate, while path, type, size, and
bytes are the execution identity.

### DIAG4 sealed-supervisor bootstrap

The DIAG4 supervisor entry is frozen before the CPU one-shot as
`benchmarks/run_single_stage_native_equivalent_quality_campaign.py`, one of the
603 execution-manifest entries. No new public CLI option is introduced. The
existing exact `--diagnostic-successor-authority AUTHORITY_PATH` option is the
sole bootstrap selector; stdlib-only argument scanning reads the canonical
authority schema, and only
`single-stage-neq-gntr3-diag4-authorization-v1` activates this DIAG4 bootstrap.

The entry module performs this bootstrap before importing any repository module,
JAX, jaxlib, NumPy, simsopt, or native extension. Using only the standard
library, it validates the authority path and canonical bytes, exact DIAG4
schema/route, both execution-source hashes, decisive CPU qualification
reference, sealed CPU snapshot manifest, and the fixed entry's path, size, and
SHA-256 against the 603-entry map. The exact sealed entry is
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-cpu-qualification-20260811T214932Z/source-snapshot/benchmarks/run_single_stage_native_equivalent_quality_campaign.py`.
The root is the already frozen decisive CPU qualification root, so neither the
authority nor the runner contains a hash of future authority bytes and no
self-reference is introduced.

If the current resolved entry is not that sealed entry, the stdlib bootstrap
changes directory to the exact frozen CPU root's `source-snapshot` directory and
calls `os.execve` exactly once with the authority-bound interpreter, `-B`, the
sealed entry, and the unchanged original CLI arguments. It sets `PYTHONPATH` to
the sealed snapshot's `src` directory followed by the sealed snapshot root and
sets `PYTHONDONTWRITEBYTECODE=1`; all other environment values remain subject to
the frozen supervisor/child policy. The live launcher may not import
repository/JAX code, claim an output, consume authority, observe a GPU, create
staging, or launch a child. If the current entry is already the sealed path, the
same stdlib-only validations must pass before normal imports, preventing
recursion without a new environment marker. Any missing/extra source, path
alias, entry hash/size drift, wrong CPU root, noncanonical authority, or attempt
to skip or repeat re-exec fails before repository/JAX imports.

Only the sealed supervisor then performs the complete descriptor-bound authority
claim and schedule below. Its `_prepare_diag4_snapshot` copies the exact 603
execution entries, manifest bytes, and native entry from the sealed CPU-qualified
snapshot. It preserves the CPU roles, paths, sizes, and hashes, independently
proves the copied native entry equal to the held current live native descriptor,
proves exact GPU snapshot set equality, and never calls Git/worktree capture,
live repository enumeration, directory-glob selection, or a live-source snapshot
publisher. The GPU child still loads the held/current live native binary under
the three-way identity above. The completed plan, authority, native reference,
inputs, and consumed DIAG3 evidence remain separate held controls and are not
injected into the GPU source snapshot.

This bootstrap and `_prepare_diag4_snapshot` replacement are DIAG4-only. DIAG1,
DIAG2, and DIAG3 CLI interpretation, import policy, source enumeration, snapshot
membership, schemas, and behavior remain unchanged.

## Exact frozen and qualified map membership

`frozen_numerical_entries` contains exactly these 11 repository-relative paths,
with one SHA-256 each and no extras:

1. `benchmarks/single_stage_native_equivalent_reference.py`;
2. `examples/jax/parity/cases/native_boozerqa.py`;
3. `examples/jax/parity/input_bundle.py`;
4. `src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py`;
5. `src/simsopt_jax/objectives/single_stage_fullspace.py`;
6. `src/simsopt_jax/runtime/trace_annotations.py`;
7. `src/simsopt_jax/solve/fullspace.py`;
8. `src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py`;
9. `src/simsopt_jax/solve/fullspace_native_equivalent_quality.py`;
10. `src/simsopt_jax_adapters/geo/single_stage_fullspace.py`;
11. `src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py`.

`qualified_files` contains exactly these 23 repository-relative paths, with one
SHA-256 each and no extras. The serialized authority order is exactly the
lexicographic order shown here:

1. `benchmarks/process_gpu_monitor.py`;
2. `benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py`;
3. `benchmarks/run_single_stage_native_equivalent_quality_campaign.py`;
4. `benchmarks/single_stage_fullspace_process_gpu_monitor.py`;
5. `benchmarks/single_stage_fullspace_snapshot.py`;
6. `benchmarks/single_stage_native_equivalent_endpoint_audit.py`;
7. `benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py`;
8. `benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json`;
9. `benchmarks/single_stage_native_equivalent_quality_receipt.py`;
10. `benchmarks/single_stage_native_equivalent_quality_successor_authority.py`;
11. `benchmarks/single_stage_native_equivalent_reference.py`;
12. `tests/benchmarks/_diag2_fixture.py`;
13. `tests/benchmarks/test_process_gpu_monitor.py`;
14. `tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py`;
15. `tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py`;
16. `tests/benchmarks/test_single_stage_fullspace_snapshot.py`;
17. `tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py`;
18. `tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py`;
19. `tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py`;
20. `tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py`;
21. `tests/benchmarks/test_single_stage_native_equivalent_reference.py`;
22. `tests/geo/test_fullspace_native_equivalent_quality.py`;
23. `tests/geo/test_projected_gauss_newton_trust_region.py`.

This completed plan and the later authority are separately bound and are not
members of either map. The native-reference manifest/tree, exact interpreter,
input root, consumed-DIAG3 manifest/tree, historical CPU20 files, decisive
current-byte CPU qualification root/manifest, and absent GPU output claim are
separately typed authority fields. The source snapshot membership is exactly the
execution-source authority set defined above, not the smaller qualified/frozen
union. Missing members, additional members, duplicate paths, path aliases,
absolute paths in a repository map, symlinks, or conflicting hashes fail before
any child.

## Execution policy and trace-free schedule

The supervisor is CPU-only and establishes exact-PID GPU absence before each
authorized child boundary. Each child is a pristine CUDA/FP64 process with the
frozen command-buffer-disabled policy, compilation cache disabled, and
preallocation enabled.

The only permitted schedule is:

1. validate and exclusively claim the exact authority, completed plan, qualified
   map, frozen map, execution-source manifest bytes and entry aggregate, sealed
   CPU-qualified source snapshot, held live native-extension three-way identity,
   native reference, input, interpreter, consumed DIAG3 tree, exact output-root
   and authority-bound rollback-partial absence, and consumption-marker absence;
2. create and inode-bind exactly one staging sibling;
3. publish and deep-load the source snapshot and setup authorities;
4. prove supervisor GPU-zero;
5. durably consume authority;
6. call exactly one compile-only preflight `Popen` for the exact GNTR3 executable;
7. validate the complete preflight producer and setup authorities;
8. prove supervisor GPU-zero and revalidate every held identity;
9. launch at most one pristine trace-free cold;
10. validate or quarantine the child numerical bundle;
11. reconstruct the scientific receipt, seal, deep-load staging, atomically
    rename with `RENAME_NOREPLACE`, fsync the parent, and independently deep-load
    the final path while all claims remain held; and
12. revalidate and finalize every held authority identity before unlock.

Warm runs, retries, replacement runs, campaign loops, tuning, trace canaries, a
second preflight, and a second cold are forbidden under this authority.

The authoritative preflight and cold must not call `jax.profiler.start_trace`,
`jax.profiler.stop_trace`, Chrome normalization, XPlane parsing, or any equivalent
profiler API. No raw-trace or trace-interval path may exist in the artifact.

The cold child must:

1. compile the exact NEQ-GNTR3 route and prove zero Python callbacks;
2. block the initial device state ready;
3. record `solve_started_monotonic_ns` using `time.perf_counter_ns`;
4. dispatch the compiled bounded loop exactly once;
5. block the complete loop result ready;
6. record `solve_stopped_monotonic_ns`;
7. run the untimed finalizer, accepted-ledger replay, and independent endpoint
   audit exactly once; and
8. perform only the frozen final evidence transfers.

The solve-timing evidence contains exact integer timestamps, a finite positive
`synchronized_solve_seconds`, backend/UUID/process identity, callback and transfer
counts, route/result/options/problem/policy/source hashes, and these exact
ordered fields:

- `process_started_monotonic_ns`;
- `state_ready_monotonic_ns`;
- `solve_started_monotonic_ns`;
- `solve_stopped_monotonic_ns`;
- `finalizer_completed_monotonic_ns`;
- `endpoint_audit_completed_monotonic_ns`;
- `serialization_started_monotonic_ns`; and
- `process_stopped_monotonic_ns` from the parent process record.

It also contains exact integer counters `profiler_start_calls=0`,
`profiler_stop_calls=0`, `trace_normalization_calls=0`, and the literal claim
`profiler_enabled=false`. Counters join independently to the route-owned
call-audit object. Exact arithmetic is
`synchronized_solve_seconds == (solve_stopped_monotonic_ns -
solve_started_monotonic_ns) / 1e9`. The parent process interval strictly
contains all child timestamps in the order above.

Phase attribution is `NOT_PRODUCED`. No profiler or CPU20 timing may be spliced
into authoritative solve timing.

## Durable authority consumption

The supervisor takes exclusive nonblocking locks on the authority inode, the
completed-plan inode, the canonical execution-source manifest inode, every
manifest entry and CPU-snapshot leaf, the live native-extension descriptor,
every other discovered authority-bound leaf, the complete authority/root-parent
inode chains, and the exact output claim. The authority carries exact fields
`execution_source_manifest_sha256`
and `execution_source_entries_sha256`, equal to the decisive CPU artifact and
every reviewer record. It retains all claims from initial claim through final or
visible-partial fsync and independent deep load. Path, device, inode, byte,
manifest membership, both source hashes, and absence bindings are revalidated
before every child, after final publication, and before unlock. A mismatch is
identity failure; it cannot be repaired by rediscovery, live-worktree copying,
or hash replacement.

Authority consumption is a durable sibling marker named
`.OUTPUT_NAME.diag4-authority-consumed.json`, where `OUTPUT_NAME` is the exact
authority-bound output-root basename. The marker payload contains exactly the
consumption schema, evidence route, authority SHA-256, plan-prefix SHA-256,
completed-plan SHA-256, and absolute output root.

Immediately before the first preflight `Popen`, the held supervisor:

1. revalidates every lock and proves the final root and marker absent with only
   its one bound staging sibling present;
2. creates one PID-qualified pending marker using `O_EXCL|O_NOFOLLOW`;
3. writes canonical JSON completely, changes the mode to `0444`, fsyncs the
   file, and verifies its descriptor/path inode and bytes;
4. publishes the final marker without replacement by linking the held
   descriptor, then immediately treats authority state as
   `CONSUMPTION_UNCERTAIN`;
5. verifies final marker inode/bytes, removes only the bound pending name, fsyncs
   the output parent, and changes state to `CONSUMED`; and
6. revalidates the held final marker descriptor before calling `Popen`.

The authority JSON itself remains immutable; the marker is the one-shot
consumption evidence. A preexisting final/pending marker, competing partial/final
root, collision, replacement, unlink, inode drift, or byte drift fails closed.

Failure before final-marker publication is `AUTHORITY_CONSUMPTION_FAILED` only
when the implementation proves no final marker was published and safely removes
the bound pending file. Any failure at or after possible publication, or any
failure to prove/remove the pending state, is
`AUTHORITY_CONSUMPTION_UNCERTAIN`; it is treated as consumed forever. Neither
case launches a child. Once the final marker is durably verified, an exception
or OS-level failure from the immediately following first `Popen` is
`PREFLIGHT_LAUNCH_FAILED`, with authority consumed and no retry.

Pre-consumption setup/GPU-zero failures may seal exactly one zero-child failure
artifact and finalize an unconsumed prelaunch state. They do not create a
consumption marker. After consumption starts, no replacement preflight or cold
is permitted. Future claims observing the marker or uncertain state terminate
as `AUTHORITY_ALREADY_CONSUMED`.

## Atomic numerical-result publication

The child writes post-solve scientific output only beneath
`cold/.numerical-result.pending`. The closed pending tree contains history,
terminal numerical document and arrays, solve timing, and safeguard telemetry.
Runtime and policy remain earlier supervision authorities at
`cold/runtime-evidence.json` and `cold/policy.json`. Canonical stdout is retained
and published by the parent at `cold/producer.json`; its logical references name
future committed paths. The child never renames the bundle. The existing
terminal numerical document is the independent endpoint-audit artifact; no
second endpoint file or native host callback is invented.

After a zero exit and exact producer parse, the parent validates the complete
closed pending tree and atomically renames it without replacement to
`cold/numerical-result`, followed by parent-directory fsync. Only committed paths
populate typed scientific slots.

On timeout, nonzero exit, producer decode/schema failure, or invalid pending
bytes, a present pending tree is atomically renamed without replacement to
`cold/uncommitted-numerical-result` and the parent directory is fsynced. Those
bytes are opaque, nonpromoting, and never populate typed numerical slots. If no
pending tree exists, no quarantine path is created. A present invalid pending
tree whose no-replace quarantine rename, parent fsync, or closed-path validation
fails yields `QUARANTINE_FAILED`; successful quarantine yields
`PENDING_RESULT_INVALID`. These outcomes are mutually exclusive. Any rename,
fsync, symlink, hardlink, special-file, collision, or post-rename validation
failure leaves a visible overall partial and never a misleading final artifact.

No pending directory may exist when sealing. The final manifest closes every
regular byte, role, mode, size, and digest; all files are `0444`, directories
`0555`, link counts one, and no symlink or special file is permitted.

## Failure convergence and required vectors

Every terminal outcome has exactly one stage and one reason from this ordered
table. Stages are compared by row order. Reasons within one stage are compared
left to right; the first true candidate wins. Candidate derivation is pure and
order-independent before selection.

| Order | Stage | Same-stage reason precedence |
| ---: | --- | --- |
| 0 | `AUTHORITY` | `AUTHORITY_INVALID`, `OUTPUT_ROOT_NOT_ABSENT`, `LOCK_CLAIM_FAILED`, `IDENTITY_REVALIDATION_FAILED`, `AUTHORITY_ALREADY_CONSUMED` |
| 1 | `SETUP` | `SOURCE_PUBLICATION_FAILED`, `FROZEN_NUMERICAL_SUBSET_INVALID`, `NATIVE_REFERENCE_INVALID`, `POLICY_AUTHORITY_INVALID`, `SETUP_DEEP_LOAD_FAILED` |
| 2 | `BEFORE_PREFLIGHT` | `SUPERVISOR_GPU_OBSERVATION_INVALID`, `SUPERVISOR_GPU_NONZERO`, `AUTHORITY_CONSUMPTION_FAILED`, `AUTHORITY_CONSUMPTION_UNCERTAIN` |
| 3 | `PREFLIGHT` | `PREFLIGHT_LAUNCH_FAILED`, `PREFLIGHT_TIMEOUT`, `PREFLIGHT_MONITOR_FAILED`, `PREFLIGHT_EXIT_NONZERO`, `PREFLIGHT_PROTOCOL_INVALID`, `PREFLIGHT_PRODUCER_INVALID`, `PREFLIGHT_GATE_FAILED` |
| 4 | `BEFORE_COLD` | `SUPERVISOR_GPU_OBSERVATION_INVALID`, `SUPERVISOR_GPU_NONZERO`, `SOURCE_REVALIDATION_FAILED`, `IDENTITY_REVALIDATION_FAILED`, `CONSUMPTION_MARKER_INVALID` |
| 5 | `COLD` | `COLD_LAUNCH_FAILED`, `COLD_TIMEOUT`, `COLD_MONITOR_FAILED`, `COLD_EXIT_NONZERO`, `COLD_PROTOCOL_INVALID`, `COLD_PRODUCER_INVALID` |
| 6 | `NUMERICAL_COMMIT` | `PENDING_RESULT_ABSENT`, `TIMING_INVALID`, `SAFEGUARD_TELEMETRY_INVALID`, `NUMERICAL_IDENTITY_MISMATCH`, `QUARANTINE_FAILED`, `PENDING_RESULT_INVALID`, `COMMIT_COLLISION`, `COMMIT_RENAME_FAILED`, `COMMIT_FSYNC_FAILED`, `COMMITTED_DEEP_LOAD_FAILED` |
| 7 | `RECEIPT` | `EVIDENCE_VECTOR_INVALID`, `GROUP_PREFIX_INVALID`, `SCIENTIFIC_RECONSTRUCTION_FAILED`, `RECEIPT_SCHEMA_INVALID` |
| 8 | `PUBLICATION` | `MANIFEST_INVALID`, `MODE_OR_LINK_INVALID`, `STAGING_DEEP_LOAD_FAILED`, `FINAL_COLLISION`, `FINAL_RENAME_FAILED` |
| 9 | `SCIENTIFIC` | `INCOMPLETE`, `NO_HIT`, `QUALITY_HIT` |

`PREFLIGHT_LAUNCH_FAILED` includes first-`Popen` failure after durable authority
consumption. `PENDING_RESULT_ABSENT` applies only after a zero-exit cold required
to produce the bundle. An earlier timeout/nonzero exit remains a `COLD` reason
and may have no pending tree. `QUARANTINE_FAILED` applies only when a present
invalid pending tree cannot be closed under its opaque no-replace path;
`PENDING_RESULT_INVALID` applies only after that quarantine succeeds. They can
never both be candidates for one execution.

Failures after the sealed staging root has been renamed to the final path are
physical publication failures outside the schema-visible terminal stage/reason
table. Their exact typed `reason` enum is
`FINAL_FSYNC_FAILED`, `FINAL_DEEP_LOAD_FAILED`,
`POST_FINAL_AUTHORITY_REVALIDATION_FAILED`, or
`POST_FINAL_AUTHORITY_FINALIZATION_FAILED`. The supervisor never rewrites the
sealed terminal reason or any sealed artifact byte after one of these failures.
It attempts exactly one descriptor-bound, inode-checked, no-replace rename from
the final path to the exact authority-bound partial path, then verifies and
fsyncs that partial lifecycle while retaining all claims.

If that rollback succeeds, the final path is absent, the partial path is
visible, and the sealed bytes are unchanged; the partial is invalid,
unadjudicated, nonpromoting, and has `speed=NOT_PRODUCED`. If rollback fails or
is ambiguous, no retry, unlink, rewrite, or alternate path is permitted. The
typed rollback-hard wrapper retains exactly one original physical `reason` from
the four-value enum plus the rollback cause, state, and exact observed
final/partial path lifecycle; it does not invent a second original or terminal
reason. Any final path still visible is invalid, unadjudicated, nonpromoting,
and has `speed=NOT_PRODUCED`. Post-final authority revalidation and finalization
failures use this same physical-failure and single-rollback contract.

For every stage before `SCIENTIFIC`, promotion is false and speed is
`NOT_PRODUCED`. A complete committed result failing any parity clause is
`SCIENTIFIC/NO_HIT`; complete passing parity is `SCIENTIFIC/QUALITY_HIT`, which
only permits conditional engineering-timing context. Opaque quarantined bytes
never populate typed slots.

The mutation suite contains positive canonical controls and at least these
independent one-fact negatives, each with exact slot assertions and proof of no
later-stage side effect:

| Mutated fact | Required terminal stage/reason or rejection |
| --- | --- |
| plan prefix, completed-plan hash, authority schema/inode/root/launch counts | corresponding `AUTHORITY` reason; no child |
| final output, competing partial, final/pending consumption marker | `AUTHORITY/OUTPUT_ROOT_NOT_ABSENT` or `AUTHORITY_ALREADY_CONSUMED`; no child |
| qualified/frozen membership; execution-source manifest byte hash, entry aggregate, 603-member count, sealed-supervisor entry/re-exec/pre-import boundary, DIAG4-only `_prepare_diag4_snapshot` copied-set equality, prequalification-plan slot, CPU/GPU snapshot join, or native-extension path/hash/size three-way identity; source, native reference, or base policy | exact pre-import rejection or `SETUP` reason; no child |
| supervisor PID/start/UUID observation | `BEFORE_PREFLIGHT` or `BEFORE_COLD`; next child absent |
| consumption pending create/write/chmod/fsync before publication with proven cleanup | `BEFORE_PREFLIGHT/AUTHORITY_CONSUMPTION_FAILED`; no child |
| publication/binding/unlink/directory-fsync ambiguity | `BEFORE_PREFLIGHT/AUTHORITY_CONSUMPTION_UNCERTAIN`; marker treated consumed; no child |
| first preflight `Popen` failure after marker fsync | consumed authority and `PREFLIGHT_LAUNCH_FAILED`; no retry |
| marker replacement, inode/byte drift after consumption | `BEFORE_COLD/CONSUMPTION_MARKER_INVALID`; cold absent |
| GNTR3 route/result schema, sole option delta, v2 options, problem, scaling, bootstrap, initial state | `PREFLIGHT_GATE_FAILED` or `NUMERICAL_IDENTITY_MISMATCH` at first boundary |
| a legacy v1 options hash changed after adding the bool | immediate identity/qualification rejection; no child |
| callback/transfer count or command-buffer/runtime policy | producer or preflight gate invalid; cold absent |
| ordered timestamp, integer type, containment, or solve arithmetic | `NUMERICAL_COMMIT/TIMING_INVALID` |
| profiler call counter or `profiler_enabled` claim | `NUMERICAL_COMMIT/TIMING_INVALID` |
| trace slot, path, role, alias, or invented v5 label | immediate v4 schema/manifest rejection |
| safeguard trigger, quarter-radius sequence, logical-attempt count, immutable incumbent, or selected-result semantics | focused numerical test failure and first typed boundary rejection |
| any of four outer-history envelopes; either safeguard scalar; any envelope key/hash; matrix value/shape/dtype/null/padding; independent selected join; work equation; exact subtrial-derived 19-key summary; or 900 bound | `NUMERICAL_COMMIT/SAFEGUARD_TELEMETRY_INVALID` |
| one of slots 21--24 independently absent/present | `RECEIPT/EVIDENCE_VECTOR_INVALID` |
| slot order, later-group presence, or group-prefix reason | `RECEIPT/GROUP_PREFIX_INVALID` |
| absent pending tree after zero exit | `NUMERICAL_COMMIT/PENDING_RESULT_ABSENT` |
| present malformed pending tree and successful quarantine | `NUMERICAL_COMMIT/PENDING_RESULT_INVALID`; typed scientific slots absent |
| present malformed pending tree and quarantine failure | `NUMERICAL_COMMIT/QUARANTINE_FAILED`; exact opaque-path state retained; typed scientific slots absent |
| pending or final rename collision | `COMMIT_COLLISION` or `FINAL_COLLISION`; no overwrite |
| numerical-result rename/fsync/deep-load failure | matching `NUMERICAL_COMMIT` exact reason; typed scientific slots absent |
| staging validation or final rename failure before final visibility | matching schema-visible `PUBLICATION` reason; final absent |
| final fsync/deep-load or post-final authority revalidation/finalization failure | exact typed out-of-band physical reason; sealed terminal unchanged; exactly one descriptor-bound final-to-authority-partial rollback |
| symlink, hardlink, special file, mode, size, digest, unknown path/role | `PUBLICATION/MANIFEST_INVALID` or `MODE_OR_LINK_INVALID` |
| objective, equality component, scaled feasibility, first-hit, or endpoint reconstruction | `SCIENTIFIC/NO_HIT`, never a speed verdict |
| v1/v2/v3 object to v4, v4 object to legacy, obsolete GNTR2-DIAG4 object, or v5 alias | exact cross-generation rejection |

## Scientific parity gate and performance interpretation

Positive parity requires all of the following from independently reconstructed
committed evidence:

- accepted-state quality latch and first-hit replay agree;
- objective is no greater than the frozen native target;
- all 255 raw-equality component bounds pass;
- scaled feasibility is no greater than `1e-10`;
- terminal state is exactly the loop-returned state;
- objective residual reconstruction and gradient defects pass;
- derivative transpose certificate passes;
- all terminal arrays and physics observables are finite and source-bound; and
- independent endpoint evaluation agrees with the terminal record.

Failure of any clause yields `NO_HIT` or `INCOMPLETE`, never approximate parity.

For this frozen problem, the objective target, all 255 componentwise raw-equality
bounds, and scaled-feasibility bound are the gating physics proxy. The 254 Boozer
equations plus signed-volume equation constrain the physical solve; the unchanged
weighted objective constrains the non-QS, residual, iota, major-radius, and length
terms. DIAG4 invents no tolerance.

The endpoint source-binds and independently reevaluates raw objective terms
`non_qs`, `residual`, `iota`, `major_radius`, and `length`, and observables
`iota`, `G`, `volume`, `major_radius`, `total_length`, `non_qs_ratio`,
`boozer_residual_value`, and `boozer_residual_rms`. The serialized/receipt wire
key `boozer_residual_value` is derived from the raw source field
`observables.boozer_residual_scalar`. They must be finite and exactly agree
between committed terminal record and same-state evaluation.
Except where already represented in the objective/equalities, they are
diagnostic and cannot establish or weaken parity.

The sealed native trajectory duration `287.30421751597896 s` is an engineering
threshold, not a claim-compatible formal speed authority: it is a historical
native SIMSOPT/SciPy/C++ execution under a different optimizer budget.

If DIAG4 parity does not pass, speed is `NOT_PRODUCED` regardless of synchronized
duration. If parity passes, the receipt may report the observed ratio to the
historical threshold only as non-formal engineering context. A formal
GPU-faster-than-native-C++ verdict requires a separately frozen claim-compatible
native baseline or remeasurement with matched endpoint, scope, timing boundary,
source/runtime identity, and statistical policy.

## Required CPU qualification and literal commands

All qualification commands run with explicit CPU selection and no GPU child.
The exact CPU20 command and hashes above are a completed input to qualification;
they do not replace the suite below.

After the static/test suite passes and the four pre-run reviewers approve the
same bytes, the one-shot decisive current-byte CPU command is exactly:

```text
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv-qn-cpu/bin/python benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py --output-root /home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-cpu-qualification-20260811T214932Z
```

It is run once only after proving the exact final root and every same-prefix
staging root absent. A nonzero exit, timeout, partial, `NO_HIT`, or invalid
artifact is terminal for this SSOT; the command is not repeated.

The controlling pytest command is exactly:

```text
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv-qn-cpu/bin/python -m pytest -q --basetemp /home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-pytest-qualification-20260811T223700Z tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py tests/benchmarks/test_single_stage_native_equivalent_reference.py tests/geo/test_fullspace_native_equivalent_quality.py tests/geo/test_projected_gauss_newton_trust_region.py
```

The exact basetemp root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-pytest-qualification-20260811T223700Z`
must be absent before this command. After its pytest evidence is durably retained,
the basetemp root must be cleaned and its absence proved before the qualification
record is completed.

The Ruff check command is exactly:

```text
.venv-qn-cpu/bin/python -m ruff check benchmarks/process_gpu_monitor.py benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py benchmarks/run_single_stage_native_equivalent_quality_campaign.py benchmarks/single_stage_fullspace_process_gpu_monitor.py benchmarks/single_stage_fullspace_snapshot.py benchmarks/single_stage_native_equivalent_endpoint_audit.py benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py benchmarks/single_stage_native_equivalent_quality_receipt.py benchmarks/single_stage_native_equivalent_quality_successor_authority.py benchmarks/single_stage_native_equivalent_reference.py examples/jax/parity/cases/native_boozerqa.py examples/jax/parity/input_bundle.py src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py src/simsopt_jax/objectives/single_stage_fullspace.py src/simsopt_jax/runtime/trace_annotations.py src/simsopt_jax/solve/fullspace.py src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py src/simsopt_jax/solve/fullspace_native_equivalent_quality.py src/simsopt_jax_adapters/geo/single_stage_fullspace.py src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py tests/benchmarks/_diag2_fixture.py tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py tests/benchmarks/test_single_stage_native_equivalent_reference.py tests/geo/test_fullspace_native_equivalent_quality.py tests/geo/test_projected_gauss_newton_trust_region.py
```

The Ruff format command is exactly:

```text
.venv-qn-cpu/bin/python -m ruff format --check benchmarks/process_gpu_monitor.py benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py benchmarks/run_single_stage_native_equivalent_quality_campaign.py benchmarks/single_stage_fullspace_process_gpu_monitor.py benchmarks/single_stage_fullspace_snapshot.py benchmarks/single_stage_native_equivalent_endpoint_audit.py benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py benchmarks/single_stage_native_equivalent_quality_receipt.py benchmarks/single_stage_native_equivalent_quality_successor_authority.py benchmarks/single_stage_native_equivalent_reference.py examples/jax/parity/cases/native_boozerqa.py examples/jax/parity/input_bundle.py src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py src/simsopt_jax/objectives/single_stage_fullspace.py src/simsopt_jax/runtime/trace_annotations.py src/simsopt_jax/solve/fullspace.py src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py src/simsopt_jax/solve/fullspace_native_equivalent_quality.py src/simsopt_jax_adapters/geo/single_stage_fullspace.py src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py tests/benchmarks/_diag2_fixture.py tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py tests/benchmarks/test_single_stage_native_equivalent_reference.py tests/geo/test_fullspace_native_equivalent_quality.py tests/geo/test_projected_gauss_newton_trust_region.py
```

The compile command is exactly:

```text
.venv-qn-cpu/bin/python -m compileall -q benchmarks/process_gpu_monitor.py benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py benchmarks/run_single_stage_native_equivalent_quality_campaign.py benchmarks/single_stage_fullspace_process_gpu_monitor.py benchmarks/single_stage_fullspace_snapshot.py benchmarks/single_stage_native_equivalent_endpoint_audit.py benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py benchmarks/single_stage_native_equivalent_quality_receipt.py benchmarks/single_stage_native_equivalent_quality_successor_authority.py benchmarks/single_stage_native_equivalent_reference.py examples/jax/parity/cases/native_boozerqa.py examples/jax/parity/input_bundle.py src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py src/simsopt_jax/objectives/single_stage_fullspace.py src/simsopt_jax/runtime/trace_annotations.py src/simsopt_jax/solve/fullspace.py src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py src/simsopt_jax/solve/fullspace_native_equivalent_quality.py src/simsopt_jax_adapters/geo/single_stage_fullspace.py src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py tests/benchmarks/_diag2_fixture.py tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py tests/benchmarks/test_single_stage_native_equivalent_reference.py tests/geo/test_fullspace_native_equivalent_quality.py tests/geo/test_projected_gauss_newton_trust_region.py
```

The whitespace command is exactly:

```text
git diff --check -- docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md benchmarks/process_gpu_monitor.py benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py benchmarks/run_single_stage_native_equivalent_quality_campaign.py benchmarks/single_stage_fullspace_process_gpu_monitor.py benchmarks/single_stage_fullspace_snapshot.py benchmarks/single_stage_native_equivalent_endpoint_audit.py benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json benchmarks/single_stage_native_equivalent_quality_receipt.py benchmarks/single_stage_native_equivalent_quality_successor_authority.py benchmarks/single_stage_native_equivalent_reference.py examples/jax/parity/cases/native_boozerqa.py examples/jax/parity/input_bundle.py src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py src/simsopt_jax/objectives/single_stage_fullspace.py src/simsopt_jax/runtime/trace_annotations.py src/simsopt_jax/solve/fullspace.py src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py src/simsopt_jax/solve/fullspace_native_equivalent_quality.py src/simsopt_jax_adapters/geo/single_stage_fullspace.py src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py tests/benchmarks/_diag2_fixture.py tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py tests/benchmarks/test_single_stage_native_equivalent_reference.py tests/geo/test_fullspace_native_equivalent_quality.py tests/geo/test_projected_gauss_newton_trust_region.py
```

The suite must cover default/max-one compatibility, max-two iterative
retraction, safeguard disabled compatibility, exact trigger/stop/radius rules,
immutable incumbent, one-logical-attempt accounting, third-subtrial capacity,
up-to-900 bounds, all telemetry/work formulas and mutations, whole-solver JIT,
CPU20 byte/schema/gate validation, exact 603-entry no-ignore execution-source
membership, canonical-manifest and dual-hash validation, exact copied-set,
collision-free blank-plan control, DIAG4 sealed-supervisor re-exec before any
repository/JAX import, live-launcher side-effect prohibition, CPU-to-GPU exact
snapshot copying without Git/live enumeration, CPU/GPU snapshot joins,
native-extension three-way identity, full-physics prefix behavior, v1 hash
preservation, v2 identity, result identity, trace prohibition, timing, atomic
commit/quarantine, authority consumption crash boundaries, every failure stage,
scientific `QUALITY_HIT`/`NO_HIT`, and v1/v2/v3/v4 cross-rejection. The
fullspace-route fixtures must construct the real `(N, 3)` matrix leaves with
their exact FP64/int32 dtypes and `-1` selected-index padding; a scalar or FP64
mock for an integer matrix is prohibited even when focused optimizer tests cover
the live pytree. The dedicated CPU-harness tests cover final-root absence,
one-shot/no-retry behavior, source closure, exact production options, complete
trajectory and endpoint execution, public-validator reuse, quality-hit gating,
atomic no-replace publication, fsync/deep-load failures, and `speed=NOT_PRODUCED`.
The production mutation suite separately covers every schema-visible pre-rename
publication reason, all four typed out-of-band physical publication reasons,
and successful, failed, and ambiguous single rollback lifecycles.

Qualification is incomplete unless all four independent reviewer lanes issue GO
on the exact final qualified bytes:

1. `numerical-controller`: safeguard trigger, incumbent, logical-attempt,
   telemetry/work, options identity, and CPU20 interpretation;
2. `receipt-schema`: v4 schema/vector, legacy loader preservation, mutations,
   reconstruction, and claim boundaries;
3. `source-snapshot`: exact maps, dependency/source closure, native/DIAG3/CPU20
   provenance, relocation, and byte identities;
4. `atomic-lifecycle`: locks, authority creation/consumption, crash boundaries,
   pending/commit/quarantine, publication, and terminal convergence.

Each reviewer record contains exact fields
`reviewed_execution_source_manifest_sha256` and
`reviewed_execution_source_entries_sha256`, in addition to reviewer identity,
session, reviewed qualified-map hash, reviewed frozen-map hash, completed
command receipts, duration/counts, and verdict. Both execution-source hashes
must equal the CPU qualification and the later authority. One reviewer cannot
fill multiple lanes. Any manifest, entry, qualified, frozen, or other source
change after a GO invalidates all GOs and prevents authority creation.

## Authorization and rollback

The DIAG4 authority JSON is created only after the canonical qualification record
is appended at physical EOF, all literal commands pass, all hashes are frozen,
the exact output root is absent, and four independent GOs are recorded. Until
then, the authorization path must remain absent.

The authority binds exactly one preflight, at most one cold, zero warm,
`retry_allowed=false`, the exact GNTR3 numerical identity, v4 evidence route,
completed plan, maps, `execution_source_manifest_sha256`,
`execution_source_entries_sha256`, the exact sealed CPU-qualified source
snapshot, `native_extension_path`, `native_extension_sha256`,
`native_extension_size_bytes`, the held live native descriptor, interpreter,
input/reference/DIAG3 roots, GPU UUID, and one absent output root plus its one
exact absent rollback-partial sibling. It cannot authorize GNTR2, a canary, a
rebuilt or rediscovered source tree, another native binary, another output or
partial root, or another loader generation.

Rollback is source-level selection of the unchanged legacy route. It never
relabels DIAG4 artifacts as older schemas. Any consumed/uncertain authority or
launched child requires a new SSOT/schema/root/authority for a successor. Any
execution-source manifest/entry drift, newly discovered Python path, or dual-hash
join failure likewise requires a successor manifest and SSOT; it cannot mutate
or regenerate this authority in place.

## Qualification Record
