# NEQ-GNTR1 DIAG3 Command-Buffer Recovery SSOT

## Status and objective

This document authorizes one successor diagnostic after the sealed DIAG2 R1
artifact at
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-r1-20260811T123501Z`
ended `DIAGNOSTIC_INCOMPLETE / COLD_CRASH / CHILD_EXIT_NONZERO`. The R1
authorization is consumed and its artifact is immutable.

The successor objective is narrower than an engineering campaign:

1. reproduce the unchanged NEQ-GNTR1 numerical route with XLA GPU command-buffer
   execution disabled before JAX is imported in each GPU child;
2. run exactly one compile-only preflight and, only after the existing strict
   gate passes, at most one pristine cold solve;
3. seal and deep-load the existing independently validated DIAG2 scientific
   evidence schema; and
4. only if complete evidence passes, adjudicate numerical/physics parity and
   compare the synchronized cold timing with the sealed native C++ reference.

Incomplete evidence remains `NOT_PRODUCED`: GPU activity, utilization, trace
events, or a child process duration cannot establish parity or speed.

## Design tier and alternatives

This is a Tier 4 integrity change because it adds a launch-authorization schema
to a one-shot GPU evidence workflow. Rollback is to decline launch; no existing
artifact or historical reader is rewritten.

### Design A: duplicate or rename the DIAG2 receipt schema

Copying the receipt implementation to a DIAG3 module, or globally changing its
route constants, would duplicate thousands of lines and reinterpret sealed R1
bytes. It would create change amplification across the runner, receipt, manifest,
and mutation matrix. Rejected.

### Design B: additive successor launch authority, unchanged scientific schema

Add one exact canonical launch-authority schema. The parent validates it before
creating output and binds it to the new root, inputs, interpreter, GPU UUID,
command-buffer policy, current orchestration bytes, frozen numerical hashes,
native-reference manifest, and consumed R1 identities. The source snapshot
retains the authority and this SSOT. The scientific evidence continues to use
the already-qualified DIAG2 receipt schema without fallback or reinterpretation.

Selected. The changed knowledge is execution authorization, so its schema lives
at the orchestration boundary. Numerical evidence did not change and remains
owned by the DIAG2 receipt.

## Frozen identities and boundaries

- Successor route: `NEQ-GNTR1-DIAG3-CB0`.
- Authorization schema:
  `single-stage-neq-gntr1-command-buffer-recovery-authorization-v1`.
- Scientific evidence route/schema: unchanged `NEQ-GNTR1-DIAG2` /
  `single-stage-neq-gntr1-no-hit-diagnostic-v2`.
- GPU UUID: `GPU-7951f78e-c05d-e01c-303f-d644f4341fe1`.
- Native-reference manifest SHA-256:
  `5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db`.
- R1 diagnostic SHA-256:
  `b671fea9294991ad3749fc115dab24ce3abe60c23b699da9dbe8904641bcdae7`.
- R1 manifest SHA-256:
  `c6d244c9e82d31edc04036fff0722b75b59334d36aec599746c9e8d302500029`.
- R1 cold stderr SHA-256:
  `debed91bf17f9eea2d0a56bd53df59ed4dfc1872065fee6a67d3cd7e1eb0e26a`.
- R1 Chrome trace SHA-256:
  `4db22ea34a8092e3fc55e366a4ffe83055951bc2425322ed5f874c2bfe9b8c4e`.
- R1 XPlane trace SHA-256:
  `9983075bf5c00f2c1fc98be9492e08111f8f436081b6abf0e6c4f263d8f57fad`.

All eleven `DIAG2_FROZEN_NUMERICAL_ENTRIES` remain byte-identical. No optimizer,
objective, adapter, endpoint, scaling, trace-annotation, input-bundle, or native
reference producer file may change.

## Command-buffer execution policy

For every DIAG3 GPU snapshot child, the parent constructs `XLA_FLAGS` with the
pure host helper `command_buffer_disabled_xla_flags`. It preserves unrelated
tokens, removes only the exact command-buffer selector and its bare form, and
appends exactly `--xla_gpu_enable_command_buffer=`. The child checks that the
entire environment is already canonical before importing JAX. The authority
requires:

- `JAX_PLATFORMS=cuda` in children and CPU-only parent policy;
- `JAX_ENABLE_X64=true`;
- compilation cache disabled;
- `XLA_PYTHON_CLIENT_PREALLOCATE=true` in children;
- `command_buffer_enabled=false`; and
- exact canonical `XLA_FLAGS` equality at the pre-import boundary.

No alternative flag spelling, fallback, retry, driver reset, warm run, tuning,
or replacement cold is authorized.

## Exact schedule and publication

The authority is validated before output creation. The requested final root and
every matching `.partial-*` sibling must be absent. The supervisor first opens
and takes exclusive nonblocking Linux `flock` locks on the complete root-down
inode chains for the authority directory and resolved output-parent directory,
then on the exact canonical authority file. Every directory descriptor is
bound through its already-locked parent, and every directory and authority
pathname/device/inode binding is revalidated before unlock after the
diagnostic. Validation consumes only the bytes read from that locked authority
descriptor; a renamed/replaced directory or authority inode, or a competing
same-output claimant, fails closed. The legacy `--diagnostic-only` CLI and
every composition of a supervisor mode with `--snapshot-child` fail before
importing JAX. Exact separated and `name=value` successor-authority forms are
recognized at that boundary, argparse long-option abbreviations are disabled,
and protected-option abbreviations fail closed before JAX import. No GPU
diagnostic can be reached without this authority. The existing DIAG2 parent
then performs its unchanged publication protocol:

1. create one sibling staging root;
2. publish the exact source snapshot, then compare every allowlisted file in
   that sealed snapshot with the qualified hashes held by the locked authority,
   including the exact authority bytes and the full qualification-ledger bytes;
3. publish the frozen numerical subset, copied sealed reference, and
   independently derived policy authority;
4. prove the exact supervisor PID has zero bytes on the frozen GPU;
5. launch exactly one lower/compile-only preflight with no solver dispatch;
6. independently validate the complete preflight evidence and setup authorities;
7. prove the exact supervisor PID has zero bytes again;
8. launch at most one pristine cold child;
9. derive the receipt solely from raw evidence;
10. seal files `0444`, directories `0555`, fsync, deep-load staging, publish with
   Linux `renameat2(RENAME_NOREPLACE)`, fsync the parent, and deep-load final.

Any gate or child failure seals a truthful nonpromoting DIAG2 artifact when the
existing schema can represent it. A filesystem/publication failure that prevents
construction of the typed evidence required by that schema instead leaves only
the visible `.partial-*` staging root and never creates a final artifact. No
scientific route, parity, or speed claim is inferred from either failure form.

## Success and comparison criteria

Physics/numerical parity is established only by a complete DIAG2 receipt whose
raw terminal coordinates, objective, all 255 equalities, multipliers, KKT
evidence, accepted ledger/mask, 300-row history, first-hit replay, phase
attribution, source/runtime identities, and policy authorities pass independent
reconstruction against the sealed native reference.

Speed is established only after parity. The compared quantity is the synchronized
cold solve interval recorded by the complete execution evidence versus the
sealed native C++ reference timing under its frozen definition. Compile time,
parent time, profiler finalization, replay, endpoint audit, and D2H finalization
are reported separately and are not silently mixed into the solve comparison.

If the cold is incomplete, both parity and speed are `NOT_PRODUCED`. If parity
fails, speed is nonpromoting even if a duration exists.

## Qualification and rollback

Before authorization, the controlling CPU qualification must include the runner,
snapshot, receipt, GPU monitor, successor-authority, and their contract tests;
Ruff check, Ruff format check, compileall, and `git diff --check`; exact hashes of
all change-authorized files; all eleven frozen numerical hashes; the sealed
native-reference manifest; the absent output root; and fresh independent
implementation, receipt, atomic-publication, and coverage GO verdicts.

Rollback is no launch. After launch, the exact authorization is consumed when a
preflight child starts. The resulting final or visible partial root is retained
and never reused or mutated. Any later attempt requires another new authority,
root, and explicit authorization.

## Qualification-record protocol

The final record is append-only and excluded from the frozen plan hash. Its sole
EOF entry is canonical JSON under the
`single-stage-neq-gntr1-command-buffer-recovery-qualification-v1` schema. The
authority claimant rejects a missing, noncanonical, non-EOF, or inconsistent
entry. It must bind the exact authority SHA, output root, controlling CPU command
and exact pass count/duration with JSON-type-exact static and launch booleans,
qualified hashes, frozen numerical hashes, native reference, no-GPU
qualification fact, authorization cardinality, and at least two distinct
independent GO reviewers. No launch is authorized before that entry exists and
validates.

## Qualification Record
{"authority_sha256":"a554ad4c6105dd9c31d7247bf4d378b8c73dc4007f7de16fecce14bcc7124acc","authorization":{"maximum_cold_launches":1,"preflight_launches":1,"retry_allowed":false,"warm_allowed":false},"controlling_cpu":{"command":"env JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv-qn-cpu/bin/python -m pytest -q tests/benchmarks/test_process_gpu_monitor.py tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py tests/benchmarks/test_single_stage_fullspace_snapshot.py tests/benchmarks/test_single_stage_compute_graph_attribution_control.py tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py","duration_seconds":343.83,"passed":880},"frozen_numerical_entries":[{"relative_path":"benchmarks/single_stage_native_equivalent_reference.py","sha256":"faf7614ad827e3603b1ba8e4a792394e50fb8be2146bff5bb34f002cb41d96e6"},{"relative_path":"examples/jax/parity/cases/native_boozerqa.py","sha256":"3bf7c04ec64b340a7dbb8c08b0cd55cbc0bbd0cb41942976cd33451087894832"},{"relative_path":"examples/jax/parity/input_bundle.py","sha256":"303439ea4dcf9b444ad3410c088fb17bc25cd4701dabc88bcf5106faf9a8e87b"},{"relative_path":"src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py","sha256":"c28a598a56eae109b3e61f846ae58c34b97a2cdc5fe92fdb15af0a668eb380de"},{"relative_path":"src/simsopt_jax/objectives/single_stage_fullspace.py","sha256":"ca3a09f57fcabe4e448b9c50256bf28cc3750005cf52199ace2061d3e55f19fd"},{"relative_path":"src/simsopt_jax/runtime/trace_annotations.py","sha256":"9d50e5fca9dddc8b933f5039beb0ed5f25339dea78e2c5a12bacf67489881ea7"},{"relative_path":"src/simsopt_jax/solve/fullspace.py","sha256":"475cb63ddc183e343c1ae40faf7e0abf8bad5e6c288eabe38d31ce416e18cde4"},{"relative_path":"src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py","sha256":"62b7dec2194f7c381d676abeed852ff1c4acba9e1a5f8d764a845abcd040f436"},{"relative_path":"src/simsopt_jax/solve/fullspace_native_equivalent_quality.py","sha256":"abf9726e487eb4bda9f82c6092415e988e5a346383c89cec732fe7185b6e6fac"},{"relative_path":"src/simsopt_jax_adapters/geo/single_stage_fullspace.py","sha256":"910b59131cc9137fee65a8d14222eeccbc0cf3d61d300a63250a95469c413e4e"},{"relative_path":"src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py","sha256":"bad745833c598072e3b205599fd55eb4e35dec61e87fa6679552b8343d9d2934"}],"independent_reviews":[{"reviewer":"runner-implementation","session":"/root/diag_runner_map","verdict":"GO"},{"reviewer":"atomic-authority","session":"/root/diag_runner_map/ssot_atomic_review","verdict":"GO"},{"reviewer":"snapshot-receipt","session":"/root/diag_runner_map/ssot_atomic_review/diag3_snapshot_audit","verdict":"GO"}],"native_reference_manifest_sha256":"5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db","no_gpu_used_for_qualification":true,"output_root":"/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag3-cb0-20260811T150010Z","plan_sha256":"3d46564297f6f18a04a69152eb71bfcb45796aa662f34d4b8b3e74a82a08a9b1","qualified_files":{"benchmarks/process_gpu_monitor.py":"6f2e6d3c144a5e31b45533ef288e0bd1dba84066780f0bdffd443f89c3316f39","benchmarks/run_single_stage_native_equivalent_quality_campaign.py":"cd067432bd0f779359dc89fd55b6c695ce8fb4ea83b6ed4e47d2f028afbf8899","benchmarks/single_stage_fullspace_process_gpu_monitor.py":"57918ef564eb66705dd8789cc30f18bc095d3a66b4224888d84daa583f9549e3","benchmarks/single_stage_fullspace_snapshot.py":"d21c9edb02c458f66fa54c0c11ca06a6b71395d1d7aa9a43e0b618d3e20f4f4f","benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py":"59a56bd4589d6b743c9f1852630a4f687b0486962b72d8239df5ebfd24ea74da","benchmarks/single_stage_native_equivalent_quality_successor_authority.py":"ef358393bac8daee46fc4954db5b656c1d3bd4dd6299ef707571fada95dc4f82","docs/single_stage_jax_gpu_native_equivalent_quality_diag2_implementation_plan.md":"1ebcdeb59289235062a004fe8bf2bdafe593bdb6aecb8bdcef7526c6f0b03da4","docs/single_stage_jax_gpu_native_equivalent_quality_no_hit_diagnostic_implementation_plan.md":"f86897888b1c92baab791bf1d411e97fc177adda248e0ade902bc30a71215133","tests/benchmarks/_diag2_fixture.py":"bad70f7ac5d6021ad0bf3c99efcc1e8d7f9023360265bc67b24c4e43b67c275d","tests/benchmarks/test_process_gpu_monitor.py":"f776670184493012293638e32711a49340dccdc83a77e008cbd590a1dbad16e1","tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py":"9a468d91c29f05cc375f07e2935cb9f06c8516c6081930a3deb97a4324d419c0","tests/benchmarks/test_single_stage_fullspace_snapshot.py":"348c02beb924a2d559327d9e2a0e2904f723d1becc54306a7ec17e9b9ad8bcfb","tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py":"e82179c675b9ca372433a096d03e4c2b861e22472fc81c4ffada8005f85dbd11","tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py":"f111b94f4d039517cdfb66e09a342b6e34119d2fcfef48c97211b33c558b3275"},"schema_version":"single-stage-neq-gntr1-command-buffer-recovery-qualification-v1","static_checks":{"compileall":true,"git_diff_check":true,"ruff_check":true,"ruff_format_check":true}}
