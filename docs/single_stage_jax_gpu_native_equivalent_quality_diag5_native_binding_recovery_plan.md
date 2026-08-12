# NEQ-GNTR3 DIAG5 Native-Binding Recovery Contract

Status: unqualified successor implementation contract. This document does not
authorize a CPU or GPU launch. No DIAG5 authority JSON exists. The failed DIAG4
partial is immutable terminal input, not a scientific result.

## Objective and claim boundary

Recover only from the DIAG4 precompile native-extension binding defect while
preserving the NEQ-GNTR3 numerical problem, policy, options, physics, endpoint
gates, and one-shot execution discipline. The successor must first produce one
current-byte CPU trajectory qualification and may authorize one RTX 5090
preflight plus one cold solve only after independent qualification.

The numerical route remains `NEQ-GNTR3`. The evidence route is identity-distinct
`NEQ-GNTR3-DIAG5`; no DIAG4 object may be relabeled or accepted as DIAG5.

CPU qualification, successful compilation, finite values, zero exit, sealed
files, and timing do not establish physics parity. Only the complete scientific
receipt gates may do so.

## Terminal predecessor input

The failed predecessor root is exactly:

`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag4-cpu-qualification-20260811T214932Z.partial-claim`

It is retained permanently and must never be deleted, renamed, completed,
published, or used as a launch root. The intended DIAG4 final root remains
absent and is mechanically unretryable.

The successor qualification and authority bind all of these facts:

- independently reconstructed failure stage `NATIVE_EXTENSION_RUNTIME_BINDING`,
  before compilation and before any optimizer or physics evaluation, derived
  from the preserved partial, its copied qualifier source, the observed native
  topology, and the independent postmortem control described below;
- copied execution-source qualifier SHA-256
  `fbe302885c5b392958fb69ed5081edc0d69104573f19843c5be480c37af44c51`;
- copied execution-source manifest SHA-256
  `386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604`;
- exact copied execution-source membership: 603 source entries plus the one
  execution-source manifest, with no missing, extra, or differing entry;
- absence of `source-snapshot`, `scientific-evidence.json`,
  `artifact-manifest.json`, `history.json`, `terminal-numerical.json`,
  `policy.json`, `endpoint-audit.json`, `safeguard-telemetry.json`, and
  `arrays` at the partial root; and
- predecessor source-snapshot review verdict retracted because it did not
  inspect live native hardlink topology. The retraction is not a claim that
  any copied source byte differed.

The predecessor live CPU extension evidence is:

- resolved loaded path
  `/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-cpu/lib/python3.11/site-packages/simsoptpp.cpython-311-x86_64-linux-gnu.so`;
- loader `_ScikitBuildLoaderWrapper`;
- device `66306`, inode `50480769`, size `2883776`, link count `2`;
- SHA-256
  `41b2ca791a720f325ffa9b382b31d29bade73f6516693805d41adc0de6f6ed4b`;
  and
- a second installed hardlink at the corresponding `.venv-qn-gpu` path with
  the same device, inode, size, and bytes.

Every predecessor physical fact is revalidated read-only before qualification
and before authority creation. The reconstructed stage, class, and message are
adjudicated against copied source and the postmortem; they are not facts directly
encoded by the partial. Drift closes the successor without a launch.

The predecessor partial root remains mode `0755` because the failed producer
never sealed or published it. Descriptor-relative `lstat` enumeration must prove
that its exact sole directory entry is the real, nonsymlink directory
`execution-source`; a dangling symlink, symlink occupancy, hidden entry, socket,
device, FIFO, or any second name fails closed. The `execution-source` root and
every descendant directory are real mode-`0555` directories. Every leaf is a
real regular mode-`0444`, link-count-one file. The embedded canonical
execution-source manifest is the complete leaf authority: exactly 603 entries
plus the manifest itself, with no missing or additional leaf anywhere under
`execution-source`. DIAG5 never upgrades the partial root mode or treats it as a
finalized artifact. The intended predecessor final root and every scientific
path named above must be absent under descriptor-relative `lstat`; a dangling
symlink counts as present.

The predecessor full-tree map contains exactly 604 keys: each POSIX path relative
to `execution-source` maps to an object with exactly
`{sha256,size_bytes,mode,link_count}`. `mode` is the string `"0444"` and
`link_count` is integer `1`; directories are excluded. For every one of the 603
manifest entries, path, SHA-256, and size must equal the embedded manifest. The
604th key is
`benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json`,
whose SHA-256 is
`386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604`.
The aggregate is
`sha256(canonical_json_bytes(predecessor_full_tree)).hexdigest()`, exactly
`c04cbbb79650990ab38e497bd48d6d7ab9cc2714941c58e3ce91e4147997436a`.
The postmortem reconstruction and every later qualification, review, and
authority contain the exact field `predecessor_full_tree_sha256` with that
value. Validation reads and hashes every leaf through retained descriptors; a
path-only count or manifest-only check is insufficient.

Because the partial itself contains no exception record, DIAG5 also requires a
separately published canonical
`single-stage-neq-gntr3-diag4-independent-postmortem-v1` control. It explicitly
records `original_stdout_retained=false`, `original_stderr_retained=false`, and
`original_process_receipt=NOT_PRODUCED`. Its reconstruction map binds session
reference `74963`, retained command text, failure stage, exception class/message,
copied qualifier predicate and hashes, manifest membership, native topology,
final/scientific absence, and review retraction. It makes no original timestamp,
stderr digest, or byte-exact process-output claim. Four post-run reviewers hash
and adjudicate this reconstruction.

The postmortem path is exactly
`docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json`.
It is created by reviewed `apply_patch` before source-manifest freeze, then
included in qualified and execution-source maps. Its exact top-level keys are
`schema_version`, `session_reference`, `original_stdout_retained`,
`original_stderr_retained`, `original_process_receipt`, and `reconstruction`.
The reconstruction exact keys are `command_text`, `partial_root`,
`failed_stage`, `exception_class`, `exception_message`, `qualifier_sha256`,
`execution_manifest_sha256`, `execution_entries_sha256`,
`execution_source_entry_count`, `copied_tree_entry_count`,
`predecessor_full_tree_sha256`,
`copied_qualifier_predicate`, `native_binding`, `final_root_absent`,
`scientific_paths_absent`, `prior_reviews_retracted`, and
`retracted_reviews_sha256`. `prior_reviews_retracted` is the exact four-entry
array below; its canonical JSON SHA-256 is `retracted_reviews_sha256`.
`native_binding` has exact keys `path`, `loader`, `sha256`, `size_bytes`,
`device`, `inode`, and `link_count`. Canonical JSON is sorted, compact UTF-8
with one trailing newline; its SHA-256 is bound by every review, qualification
record, and authority.

The repository postmortem JSON is itself a qualified execution-source entry.
The CPU artifact copies its exact bytes to
`control/predecessor-postmortem.json` and publishes the scientific-evidence
field `predecessor_postmortem` as an artifact reference with exactly
`{relative_path,sha256,size_bytes,schema_version}`. The relative path and schema
are exactly `control/predecessor-postmortem.json` and
`single-stage-neq-gntr3-diag4-independent-postmortem-v1`; hash and size equal the
qualified repository JSON. No inline duplicate or alternate path is accepted.

## Route, schemas, and roots

The successor identities are:

- numerical route: `NEQ-GNTR3`;
- evidence route: `NEQ-GNTR3-DIAG5`;
- loader generation: `v5`, with `DiagnosticReceiptV5`; legacy v4 remains
  behaviorally unchanged and no v5 object dispatches through a v4 loader;
- CPU qualification:
  `single-stage-neq-gntr3-cpu-trajectory-qualification-v2`;
- CPU qualification manifest:
  `single-stage-neq-gntr3-cpu-trajectory-qualification-manifest-v2`;
- scientific receipt:
  `single-stage-neq-gntr3-trace-free-diagnostic-v2`;
- artifact manifest:
  `single-stage-neq-gntr3-trace-free-artifact-manifest-v2`;
- cold producer: `single-stage-neq-gntr3-trace-free-cold-result-v2`;
- preflight producer: `single-stage-neq-gntr3-trace-free-preflight-v2`;
- committed numerical bundle:
  `single-stage-neq-gntr3-trace-free-numerical-bundle-v2`;
- solve timing: `single-stage-neq-gntr3-solve-timing-v2`;
- safeguard telemetry:
  `single-stage-neq-gntr3-step-bound-safeguard-telemetry-v2`;
- execution record: `single-stage-neq-gntr3-trace-free-execution-v2`;
- supervisor terminal:
  `single-stage-neq-gntr3-trace-free-supervisor-terminal-v2`;
- runtime evidence: `single-stage-fullspace-runtime-evidence-v2`;
- successor authority:
  `single-stage-neq-gntr3-diag5-authorization-v1`;
- qualification record:
  `single-stage-neq-gntr3-diag5-qualification-v1`; and
- authority consumption:
  `single-stage-neq-gntr3-diag5-authority-consumption-v1`.

The fixed CPU qualification root is:

`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-cpu-qualification-20260812T090000Z`

The fixed GPU output root is:

`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-rtx5090-20260812T030000Z`

The CPU root and every CPU `partial-*` sibling must be absent before CPU
qualification. Its sealed final must exist and pass deep-load before authority.
The GPU root and every GPU `partial-*` sibling must be absent before authority.
Each root is one-shot: any attempted use, visible partial, final directory, or
consumption marker makes that root permanently ineligible for another launch.

The sole CPU staging path is the CPU root plus `.partial-claim`. The sole GPU
staging path is the GPU root plus `.partial-claim`; the sole post-publication
rollback path is the GPU root plus `.partial-rollback`. The durable consumption
marker is the GPU-root sibling
`.neq-gntr3-diag5-rtx5090-20260812T030000Z.diag5-authority-consumed.json`;
its only pending form is that name plus `.pending-<supervisor-pid>`. Exact-path
`O_EXCL|O_NOFOLLOW`, descriptor binding, no-replace publication, file and parent
fsync, and typed `CONSUMPTION_UNCERTAIN` semantics are required. No alternate
root-level partial, marker, rollback, retry, or cleanup path is allowed. The one
additional permitted sibling is the physical-publication-failure record
`.neq-gntr3-diag5-rtx5090-20260812T030000Z.diag5-physical-publication-failure.json`;
its sole pending form appends `.pending-<supervisor-pid>`. It is evidence, not a
campaign output, retry root, terminal rewrite, or authority-consumption marker.

Within the claimed GPU staging root, the child may write scientific output only
at `cold/.numerical-result.pending`. After a zero exit, the parent either commits
it by no-replace rename to `cold/numerical-result` or, when validation fails,
quarantines it by no-replace rename to
`cold/uncommitted-numerical-result`. These are the sole pending, committed, and
quarantine paths. “No alternate quarantine path” refers to root-level campaign
siblings; the one internal opaque quarantine path is mandatory. A successful
quarantine followed by parent fsync yields `PENDING_RESULT_INVALID`; quarantine
rename/collision/fsync/deep-load failure yields `QUARANTINE_FAILED`. Those
reasons are mutually exclusive. Quarantined bytes populate no typed evidence
slot. An absent pending tree after a required zero-exit producer yields
`PENDING_RESULT_ABSENT`. After a timeout, nonzero exit, monitor failure, launch
protocol failure, or any other outer `COLD` failure, an absent pending tree
leaves that exact outer `COLD` reason unchanged. If a pending tree is present
after any such failure, the parent must no-replace quarantine it to the sole
internal quarantine path, fsync, and deep-load it as opaque noncommittable
bytes; successful quarantine preserves the exact outer `COLD` reason rather
than replacing it with `PENDING_RESULT_INVALID`. Failure or ambiguity of that
mandatory quarantine records `QUARANTINE_FAILED` as the pending-disposition
failure while preserving the exact outer `COLD` reason as the terminal reason;
it prevents staging seal and publication. `QUARANTINE_FAILED` is the primary
`NUMERICAL_COMMIT` terminal reason only in a branch with no earlier `COLD`
failure. Commit is forbidden outside the zero-exit valid pending branch. Before
any staging seal or publication, the pending path must be absent: every present
pending tree is exactly committed or quarantined, and no pending bytes may
remain in a sealed artifact.

## Native-extension identity

The cross-runtime binary identity is exactly the pair
`native_extension_sha256` and `native_extension_size_bytes`. CPU and GPU loaded
paths are separate authority fields:

- `cpu_native_extension_path` is the absolute resolved path actually loaded by
  the decisive CPU qualifier;
- `gpu_native_extension_path` is the absolute resolved path that the sealed GPU
  supervisor and child actually load.

Each live path is independently opened root-down with `O_NOFOLLOW`, retained
under a shared lock, and descriptor-bound to path device, inode, size, and
digest. It is rechecked before and after every child boundary and through final
deep-load. CPU and GPU paths need not be equal and their inodes need not be
equal. Their SHA-256 and size must be equal.

For each live path, link count is a required positive integer
telemetry field. It is recorded and revalidated for that path but is not a
cross-runtime identity component and is not required to equal one. A positive
link-count change during a held claim is drift and fails closed; an initial
value greater than one is valid.

The CPU qualification v2 scientific-evidence top level contains the nested
object `cpu_native_binding` with exactly the six keys
`{cpu_native_extension_path,native_extension_sha256,native_extension_size_bytes,cpu_native_extension_link_count,cpu_native_extension_device,cpu_native_extension_inode}`.
No former flat native field may coexist. The authority copies this same exact
six-key object without renaming a key. The GPU binding has the exact keys
`gpu_native_extension_path`, `native_extension_sha256`,
`native_extension_size_bytes`, `gpu_native_extension_link_count`,
`gpu_native_extension_device`, and `gpu_native_extension_inode`. Device and inode are
per-path ownership/revalidation telemetry and are not cross-runtime identity.

The same CPU scientific-evidence top level contains `predecessor_postmortem` as
the exact typed reference defined above. The CPU qualification manifest v2
closes `scientific-evidence.json`, `control/predecessor-postmortem.json`, the
source snapshot, and every other artifact leaf. Predecessor evidence and native
binding are mandatory qualification identity, not optional diagnostics.

The CPU source snapshot contains one copied native extension at the constant
`native/simsoptpp.cpython-311-x86_64-linux-gnu.so`. The GPU snapshot reuses that
sealed CPU-qualified copy as
evidence, but the GPU child imports the separately bound GPU-venv installed
extension. The runtime receipt proves that its actual loaded path is exactly
`gpu_native_extension_path`.

Runtime-evidence v2 preserves every v1 runtime-identity field and adds exactly
`native_extension_sha256`, `native_extension_size_bytes`, and
`native_extension_link_count`. Its `native_extension_path` is the actual
absolute resolved installed module path observed from `simsoptpp.__file__`, not
the snapshot evidence path. Preflight and cold runtime documents must carry the
same authority-bound GPU path/hash/size/link-count tuple. CPU qualification
runtime carries the authority-bound CPU tuple. The snapshot manifest separately
binds the constant copied-native leaf and never substitutes that relative
evidence path into runtime identity.
Every copied native leaf is regular, non-symlink, mode `0444`, link count one,
and equal to the cross-runtime SHA-256 and size. Live installed-file link-count
policy must never be reused for sealed artifact leaves.

## Source, plan, and snapshot closure

`DIAG5_QUALIFIED_FILE_PATHS` is the exact following 24-path frozenset, with no
additional or missing member:

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
12. `docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json`;
13. `tests/benchmarks/_diag2_fixture.py`;
14. `tests/benchmarks/test_process_gpu_monitor.py`;
15. `tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py`;
16. `tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py`;
17. `tests/benchmarks/test_single_stage_fullspace_snapshot.py`;
18. `tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py`;
19. `tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py`;
20. `tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py`;
21. `tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py`;
22. `tests/benchmarks/test_single_stage_native_equivalent_reference.py`;
23. `tests/geo/test_fullspace_native_equivalent_quality.py`;
24. `tests/geo/test_projected_gauss_newton_trust_region.py`.

`DIAG5_FROZEN_NUMERICAL_PATHS` is the exact following 11-path frozenset:

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

The canonical `qualified_files` and `frozen_numerical_entries` objects map each
lexicographically ordered path to its lowercase exact-byte SHA-256 string, with
exact set equality to those literals. Their aggregate fields are respectively
`qualified_files_sha256` and `frozen_numerical_entries_sha256`, each computed as
`sha256(canonical_json_bytes(map)).hexdigest()`. Throughout DIAG5,
`canonical_json_bytes(value)` means UTF-8 of
`json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"),
sort_keys=True) + "\n"`. The same algorithm governs execution-source entries,
review maps, postmortem maps, qualification records, and authority. No alternate
whitespace, omitted trailing LF, path array, inherited DIAG4 map, or map-with-size
variant is accepted.

DIAG5 regenerates the canonical execution-source manifest after all successor
changes settle. Its entry set is exact no-ignore `Path.rglob("*.py")` regular,
nonsymlink membership beneath `benchmarks`, `examples`, and `src`, union the 24
qualified and 11 frozen paths, minus the manifest itself and this plan. The
future authority JSON and all generated review/qualification JSON are excluded;
the repository postmortem JSON is included. Membership discovery validates the
frozen manifest but never selects runtime copy inputs. The no-ignore Python set
is exactly 591 paths (113 `benchmarks`, 156 `examples`, 322 `src`); union with
the 12 qualified test paths and one repository postmortem yields exactly 604
deduplicated execution entries. The separately copied manifest is not an entry.

This plan is a separate blank prequalification control with exact schema
`single-stage-neq-gntr3-prequalification-plan-control-v2` and keys
`{schema_version,snapshot_relative_path,source_relative_path,sha256,size_bytes,plan_prefix_sha256}`.
The snapshot path is `control/prequalification-plan.md`; the source path is this
plan's repository path. The completed plan is separately authority-bound after
the Qualification Record append. Authority-module constant `DIAG5_PLAN_SHA256`
is exactly the SHA-256 of all bytes physically preceding the sole
`## Qualification Record` marker, including the newline immediately before that
marker; it is frozen only after the blank-plan independent audit.

The decisive CPU snapshot contains exactly 607 leaves: 604 execution entries,
the separately copied manifest, blank plan control, and one copied native
extension. The copied-native relative path is the constant
`native/simsoptpp.cpython-311-x86_64-linux-gnu.so`, derived once from the CPU
binding and frozen here. The GPU snapshot contains exactly 606 leaves: the
sealed CPU-qualified 604 execution entries, manifest, and copied native
extension at that same constant relative path even if the installed GPU basename
differs. It does not re-enumerate or copy live repository Python files. CPU and
GPU runtime documents separately record their installed absolute loaded paths.

Every source, plan, native, predecessor, qualification, review, authority, input,
reference, interpreter, marker, root, and output identity used for admission is
opened root-down with `O_NOFOLLOW`, retained by descriptor, and held under a
nonblocking shared `flock` for immutable leaves or exclusive `flock` for mutable
claims. The CPU qualifier retains its CPU-native descriptor and all admitted
source/control descriptors from pre-copy validation through final rename, parent
fsync, and final deep-load. The GPU authority claim retains both CPU and GPU live
native descriptors and all authority leaves from claim through final or rollback
deep-load and final revalidation. Observation helpers that close descriptors,
path reopens, and later pathname hashing are telemetry only and never identity
authority. Descriptor/path inode, initial bytes, size, device, inode, link count
where specified, and locks are revalidated at every frozen boundary.

## Qualification and execution gates

The sole CPU command is:

```text
env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv-qn-cpu/bin/python benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py --output-root /home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-cpu-qualification-20260812T090000Z
```

It produces only CPU schema
`single-stage-neq-gntr3-cpu-trajectory-qualification-v2` and manifest schema
`single-stage-neq-gntr3-cpu-trajectory-qualification-manifest-v2`. The output,
`.partial-claim`, and every same-prefix sibling must be absent first. Exactly one
invocation is permitted; any process start spends the root, and no failure,
partial, timeout, or `NO_HIT` may be retried or replaced.

The first CPU qualification root,
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-cpu-qualification-20260812T022000Z`,
was spent on 2026-08-12T06:43:42Z by an out-of-band bootstrap defect: the
direct-bootstrap phase referenced module names bound only after the worker
re-exec imports (`_EXECUTION_SOURCE_ENTRY_COUNT`,
`_PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH`), raising `NameError` after the
staging claim and before any byte gate. The empty `.partial-claim`, the sealed
stderr transcript
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-terminal-bootstrap-nameerror-20260812T0643Z.txt`
(SHA-256
`e7a8a347f3bf388a2efdb345d4b345d889447907d2b3abb1611029546c361068`), and the
retracted first pre-run review root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-reviews-20260812T022000Z`
are immutable evidence and may never be reused, replaced, or deleted. This
revision makes the bootstrap phase self-contained — the execution-entry count
is parsed from the admitted authority bytes inside membership validation, and
the plan source path is a pre-bootstrap literal cross-checked against the
authority constant after re-exec — adds a full pre-exec bootstrap dry-run
regression test, and reopens the tranche on the fresh `20260812T071500Z`
namespaces frozen above. All schemas, numerics, thresholds, and gates are
otherwise unchanged; the four pre-run `GO` reviews must be re-obtained against
the recovered bytes.

The second CPU qualification root,
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-cpu-qualification-20260812T071500Z`,
was spent on 2026-08-12T08:12:59Z: the repaired bootstrap succeeded and the
worker re-exec reached production import binding, where
`_validate_imported_source_bindings` failed closed because the CPU virtual
environment's scikit-build-core editable install registers a meta-path
redirecting finder that resolves repository packages to the live worktree
ahead of every `sys.path` entry, so
`simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region` escaped the
sealed execution-source tree. The retained `.partial-claim` staging tree, the
sealed stderr transcript
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-terminal-worker-import-escape-20260812T0813Z.txt`
(SHA-256
`c1fc5f40fc0ba7e1d932f7133bfd43328169030457cc551af5ec896dcb64f105`), and the
retracted second pre-run review root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-reviews-20260812T071500Z`
are immutable evidence and may never be reused, replaced, or deleted. This
revision imports the native extension through its installed loader first —
preserving the frozen `_ScikitBuildLoaderWrapper` binding — then removes the
redirecting finders (`_neutralize_editable_source_redirection`) before any
production-source import, extends the bootstrap dry-run regression test with a
sealed-tree subprocess resolution probe, and reopens the tranche on the fresh
`20260812T090000Z` namespaces frozen above. The four pre-run `GO` reviews must
be re-obtained once more against the recovered bytes.

Before the one CPU qualification:

1. validate the predecessor partial and failed stage exactly;
2. validate absent new CPU/GPU roots and siblings;
3. validate current qualified/frozen/execution-source bytes;
4. bind the CPU live native descriptor and record positive link-count telemetry;
5. obtain four distinct pre-run `GO` reviews: numerical-controller,
   receipt-schema, source-snapshot, and atomic-lifecycle; and
6. run the complete current-byte CPU trajectory and endpoint qualification once.

After CPU qualification, deep-load its sealed artifact and source snapshot and
obtain four new, distinct post-run record-review `GO` verdicts for the same four
roles. Only those post-run reviews can authorize appending the qualification
record and creating GPU authority.

The old DIAG4 source-snapshot GO is explicitly not one of those four reviews.
All four prior conversational DIAG4 pre-run GOs—numerical-controller,
receipt-schema, source-snapshot, and atomic-lifecycle—are retracted as
non-authorizing because the live hardlink topology was not covered and no
physical DIAG4 qualification record was ever produced. Each new pre-run review
binds the full blank-plan SHA-256 and frozen-prefix SHA-256. Each post-run review
binds the same two hashes plus CPU qualification and postmortem hashes. Only the
later authority binds the completed plan full-file SHA-256.

The invalidated conversational review array has exactly four objects in this
order: `numerical-controller`, `receipt-schema`, `source-snapshot`, then
`atomic-lifecycle`. Each object has exactly the keys
`{role,reviewer,session,verdict,reviewed_qualified_files_sha256,reviewed_frozen_numerical_entries_sha256,reviewed_execution_source_manifest_sha256,reviewed_execution_source_entries_sha256,reviewed_plan_full_sha256,reviewed_plan_prefix_sha256}`.
Every `verdict` is `RETRACTED`. Every row has these six complete hashes:

- `reviewed_qualified_files_sha256 =
  e1938b81503c696bd5dc796045cdd8164e14453420b48fb38fb0f89b35ddbcc8`;
- `reviewed_frozen_numerical_entries_sha256 =
  57a3bf08fad41871812322b516f994a8e66abe2104c0e8ed0055688e3209f7e0`;
- `reviewed_execution_source_manifest_sha256 =
  386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604`;
- `reviewed_execution_source_entries_sha256 =
  7b921fed75c8a0154833ee4acf16a82922ce11b4d93dc52154dc54cc71d248b2`;
- `reviewed_plan_full_sha256 =
  5c27a90047291774955858f1b86502bfeb0aec900c733f53d8a29c0dbe41a770`;
  and
- `reviewed_plan_prefix_sha256 =
  987dd67227431a90dd851d4d8ab78f639f9964c57de5ac093fc15f5aac504e5c`.

The four exact `(role, reviewer, session)` triples in array order are:

1. (`numerical-controller`, `codex-numerical-controller-current-manifest`,
   `numerical-controller-20260811T220006-manifest386698c5`);
2. (`receipt-schema`, `codex-receipt-schema-a55a4fac`,
   `5c87cc42-3234-4b9f-bcd8-3eee3e0ea01d`);
3. (`source-snapshot`, `/root/ftr_runner_receipt`,
   `source-snapshot-final-20260811-ftr01`); and
4. (`atomic-lifecycle`, `codex-atomic-lifecycle-current-manifest`,
   `/root/diag_runner_map/ssot_atomic_review@2026-08-12T02:01:09Z`).

The exact canonical array SHA-256 is
`062e35d183f9618d5b0ca6cf7011c0c500ed3f6c7c0c1685e01262feeb5a4111`.
The postmortem must equal this array and hash, not merely accept four self-hashed
rows. No physical DIAG4 qualification record or authority existed.

New review records live beneath exact root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-reviews-20260812T090000Z`,
which and all siblings sharing that basename must be absent before the first
pre-run review publication. The eight exact files are `pre-run/<role>.json` and
`post-run/<role>.json` for the
four roles in the order above. A pre-run record has schema
`single-stage-neq-gntr3-diag5-pre-run-review-v1` and exact keys
`{schema_version,phase,role,reviewer,session,verdict,qualified_files_sha256,frozen_numerical_entries_sha256,execution_source_manifest_sha256,execution_source_entries_sha256,predecessor_postmortem_sha256,predecessor_full_tree_sha256,blank_plan_sha256,plan_prefix_sha256}`.
A post-run record has schema
`single-stage-neq-gntr3-diag5-post-run-review-v1` and those exact keys plus
`cpu_qualification_manifest_sha256` and `cpu_scientific_evidence_sha256`.
Phases are exactly `PRE_RUN` and `POST_RUN`; verdict is exactly `GO`. Review maps
map the four ordered roles to exact artifact references
`{relative_path,sha256,size_bytes,schema_version}`. Their aggregates are
`sha256(canonical_json_bytes(map)).hexdigest()` in exact authority fields
`pre_run_reviews_sha256` and `post_run_reviews_sha256`. No reviewer identity may
appear in more than one role or both phases under a different role.

## Qualification record and complete authority

The sole authority path is
`docs/single_stage_jax_gpu_native_equivalent_quality_diag5_native_binding_recovery_authorization.json`.
It must be absent through blank-plan review, CPU qualification, post-run review,
and Qualification Record append. The sole Qualification Record is one canonical
JSON object appended beneath the physical EOF marker in this plan. Its schema is
`single-stage-neq-gntr3-diag5-qualification-v1` and its exact keys are
`{schema_version,route,plan_prefix_sha256,blank_plan_sha256,qualified_files_sha256,frozen_numerical_entries_sha256,execution_source_manifest_sha256,execution_source_entries_sha256,predecessor_postmortem,predecessor_full_tree_sha256,cpu_qualification,pre_run_reviews,pre_run_reviews_sha256,post_run_reviews,post_run_reviews_sha256,verdict}`.
The two artifact references have exactly
`{relative_path,sha256,size_bytes,schema_version}`; review maps are as defined
above; `verdict` is `GO`. The hash of the canonical record object is
`qualification_record_sha256`. The completed-plan hash is deliberately absent
from the record to avoid self-reference and is computed and bound only after the
append.

The authority schema is
`single-stage-neq-gntr3-diag5-authorization-v1` with exactly these top-level
keys:

`{schema_version,route,numerical_route,scientific_evidence_schema,plan_prefix_sha256,completed_plan_sha256,qualification_record_sha256,qualified_files,qualified_files_sha256,frozen_numerical_entries,frozen_numerical_entries_sha256,execution_source_manifest_sha256,execution_source_entries_sha256,predecessor_postmortem,predecessor_full_tree_sha256,decisive_cpu_qualification,pre_run_reviews,pre_run_reviews_sha256,post_run_reviews,post_run_reviews_sha256,cpu_native_binding,gpu_native_binding,native_reference,input_bundle,consumed_diag3,numerical_identity,interpreter,roots,gpu_uuid,execution_policy,launch}`.

`qualified_files` and `frozen_numerical_entries` are the exact canonical maps
above. The two artifact references `predecessor_postmortem` and
`decisive_cpu_qualification` have exactly
`{relative_path,sha256,size_bytes,schema_version}`. `interpreter` has exact keys
`{absolute_path,sha256,size_bytes}` because it is a held external executable;
its absolute path is exactly
`/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-gpu/bin/python`.
`cpu_native_binding` and `gpu_native_binding` are the exact six-key
objects above. `roots` has exactly
`{cpu_qualification_root,gpu_output_root,gpu_staging_root,gpu_rollback_root,consumption_marker}`
with the literal paths frozen in this plan. `launch` has exactly
`{preflight_exact,cold_max,warm_exact,retry_allowed}` equal to `1`, `1`, `0`,
and `false`. `native_reference` has exact keys
`{absolute_root,manifest_sha256}` and root
`/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag3-cb0-20260811T150010Z.partial-56a1ec6d730cc005db84f99e9965b868/native-reference`.
`input_bundle` has exact keys
`{absolute_root,input_fingerprint,configuration_fingerprint}` and root
`/home/jungdaesuh/simsopt-campaigns/.single-stage-speed-20260804.partial-20260805T052535Z-2add24ec/inputs`.
`execution_policy`, `consumed_diag3`, and `numerical_identity` retain their exact
already-qualified typed schemas and hashes; aliases or inline untyped
replacements are forbidden. `gpu_uuid` is exactly
`GPU-7951f78e-c05d-e01c-303f-d644f4341fe1` and cannot be filled from a child
after consumption.

The authority constructor is pure over already held bytes and refuses creation
unless the completed plan, record, eight reviews, CPU artifact, postmortem,
predecessor tree, source maps, native bindings, roots, interpreter, reference,
input, consumed DIAG3 evidence, and GPU UUID all validate. The validator
`validate_diag5_successor_authority` enforces exact keys, canonical bytes, maps,
aggregates, schemas, paths, cardinalities, and cross-joins. The context manager
`claim_diag5_successor_authority` performs unlocked discovery only to enumerate
candidate paths, acquires root-down directory and retained leaf descriptors with
the lock policy above, rereads every discovery byte, calls the validator on held
bytes, claims the exact absent output/staging/rollback/marker names, and retains
all descriptors until lifecycle completion. No reopened path can replace a held
leaf.

`consume_diag5_successor_authority` publishes exactly one durable marker with
schema `single-stage-neq-gntr3-diag5-authority-consumption-v1` and exact keys
`{schema_version,route,authority_sha256,plan_prefix_sha256,completed_plan_sha256,output_root}`.
The state machine `diag5_authority_lifecycle` is exactly `UNCONSUMED ->
CONSUMPTION_UNCERTAIN -> CONSUMED`; it never transitions backward. Pending
create/write/chmod/fsync failure is `AUTHORITY_CONSUMPTION_FAILED` only after
proving no final marker and removing the held pending name. Any ambiguity at or
after possible final-marker publication is `AUTHORITY_CONSUMPTION_UNCERTAIN`
and spends the authority forever. `validate_diag5_successor_snapshot` requires
exact GPU snapshot equality to the sealed CPU entries, manifest, and constant
native relative path, plus the live GPU-native identity join. This is the sole
public GPU snapshot validator. Each invocation operates on the already retained
root-down `O_NOFOLLOW` descriptors and locks, rereads and hashes every governed
leaf, proves exact membership/modes/link counts and descriptor-to-path inode
binding, and accepts no path reopen as identity authority.

GPU authority is a separate canonical JSON document created only after the
qualification record is appended. It binds the completed plan, CPU artifact,
predecessor postmortem, source closure, numerical identity, CPU and GPU native
bindings, exact GPU UUID, roots, interpreter, and launch cardinality.

The GPU supervisor consumes authority durably immediately before the first
preflight child. It may launch exactly one preflight and, only after all
preflight gates pass, one cold child. Warm timing and retries are forbidden.
The scientific outcome taxonomy is exactly `INCOMPLETE`, `NO_HIT`, or
`QUALITY_HIT`; a speed comparison exists only for `QUALITY_HIT`.
`NOT_PRODUCED` is a claim/evidence status, never a scientific outcome.

The authority claim opens and locks root-down directory descriptors, every
source/plan/qualification/postmortem/predecessor leaf, both live native paths,
the output parent, and its one exact staging inode. It revalidates identities
after claim, after staging/setup deep-load, immediately before authority
consumption, before preflight `Popen`, after preflight, before cold `Popen`,
after cold, before staging final rename, after final rename and parent fsync,
and after final deep-load inside the physical-success finalizer before its
pending reservation is removed. At every boundary after successful source
publication, and in every finalization branch selected as `PublishedSnapshot`,
it calls `validate_diag5_successor_snapshot`. A `PreSourceFailure` branch
instead proves the exact no-source namespace state and revalidates the held
authority; invoking the snapshot validator there is forbidden. The CPU and GPU
native bindings are recaptured and compared at each applicable boundary,
including link-count telemetry. A change is identity failure, never a
rediscovery opportunity.

Snapshot-validation failure mapping is exact and boundary-owned. Before durable
authority consumption it is `SETUP/SETUP_DEEP_LOAD_FAILED`. After consumption
but immediately before preflight `Popen`, it is
`BEFORE_PREFLIGHT/SOURCE_REVALIDATION_FAILED`; the five-slot prefix includes the
already published `supervisor_before_preflight`, the consumption marker is
durable, and no child is launched. Immediately after the preflight child and
before its gate decision, failure is `PREFLIGHT/PREFLIGHT_PROTOCOL_INVALID` with
the reserved source-revalidation detail below; the lawful launched prefix is
eight or twelve as specified below. Immediately before cold `Popen`, it is
`BEFORE_COLD/SOURCE_REVALIDATION_FAILED`. Immediately after a successfully
reaped and fully published cold supervision chain, snapshot failure is
`COLD/COLD_PROTOCOL_INVALID` with its distinct source-revalidation detail; an
already selected earlier cold child failure retains precedence and is not
replaced by this boundary result. At the staging-final validation before final
rename, failure is `PUBLICATION/STAGING_DEEP_LOAD_FAILED`. After final rename,
including the post-rename parent-fsync/final-deep-load revalidation boundary, it
is the out-of-band physical reason
`POST_FINAL_AUTHORITY_REVALIDATION_FAILED` and triggers exactly the one fixed
descriptor-bound final-to-rollback attempt. The final post-rename validation is
also the pre-unlock validation and occurs inside
`finalize_diag5_physical_evidence_success` while the exact pending physical
evidence inode is still bound and named; no snapshot/authority revalidation is
permitted after that function removes the reservation. These mappings are
mutually exclusive; no generic identity, setup, protocol, or publication reason
from a different boundary may replace them.

The two child-protocol source-revalidation branches are closed by exact detail
preimages. The post-preflight branch uses UTF-8 bytes without a trailing newline
`GPU_SNAPSHOT_REVALIDATION_FAILED_AFTER_PREFLIGHT`, whose SHA-256 is
`b0201988e5421a54500000ee56d2a836585f49b62a7a8d689d0c7f516316222e`.
The post-cold branch uses UTF-8 bytes without a trailing newline
`GPU_SNAPSHOT_REVALIDATION_FAILED_AFTER_COLD`, whose SHA-256 is
`320b43d84c82b9be812cdf389da4c89f74e548748922d8356a35d51a09192fa4`.
No child-protocol detail may equal either reserved hash.

Before durable consumption, a reachable failure may publish exactly one sealed
zero-child terminal at the claimed GPU staging/final lifecycle. After
consumption begins,
the marker makes the authority spent even when launch fails or consumption is
ambiguous. After final rename, any fsync, deep-load, revalidation, or authority
finalization failure permits exactly one no-replace rename to the fixed rollback
path; ambiguity leaves visible bytes and forbids cleanup or retry.

Post-rename physical publication failures are typed out-of-band and never
rewrite the already sealed schema-visible terminal. Their exact original reason
enum is `FINAL_FSYNC_FAILED`, `FINAL_DEEP_LOAD_FAILED`,
`POST_FINAL_AUTHORITY_REVALIDATION_FAILED`, or
`POST_FINAL_AUTHORITY_FINALIZATION_FAILED`. The supervisor attempts exactly one
descriptor-bound, inode-checked, no-replace final-to-`.partial-rollback` rename.
Rollback success requires final absent, rollback visible, sealed bytes unchanged,
and rollback-parent fsync/deep-load complete; the result is invalid,
unadjudicated, nonpromoting, and `speed=NOT_PRODUCED`. Rollback rename collision,
rename failure, fsync failure, deep-load failure, or ambiguous visibility is a
rollback-hard failure. Its wrapper retains exactly one original reason plus
exact `rollback_cause`, `rollback_state`, and observed final/rollback path
lifecycle. It creates no second terminal/original reason.

That wrapper is canonical JSON schema
`single-stage-neq-gntr3-diag5-physical-publication-failure-v1` with exactly the
keys
`{schema_version,route,authority_sha256,original_reason,rollback_cause,rollback_state,final_path,final_path_state,rollback_path,rollback_path_state,evidence_namespace_state_at_seal,sealed_artifact_manifest_sha256}`.
`route` is `NEQ-GNTR3-DIAG5`; the two paths are the exact fixed GPU final and
rollback roots; `sealed_artifact_manifest_sha256` is the manifest digest already
validated before final rename. `original_reason` is exactly one member of the
four-value enum above. `rollback_state` is exactly `SUCCEEDED`, `FAILED`, or
`AMBIGUOUS`. State `SUCCEEDED` pairs only with rollback cause `NONE`; state
`FAILED` pairs only with `ROLLBACK_COLLISION`, `ROLLBACK_RENAME_FAILED`,
`ROLLBACK_PARENT_FSYNC_FAILED`, or `ROLLBACK_DEEP_LOAD_FAILED`; and state
`AMBIGUOUS` pairs only with `ROLLBACK_VISIBILITY_AMBIGUOUS`. No other
state/cause combination is valid. Each path-state
field is exactly `ABSENT`, `VISIBLE_VALIDATED`, `VISIBLE_INVALID`, or
`VISIBILITY_AMBIGUOUS`, as established by retained-descriptor and
descriptor-relative `lstat` checks. Success requires final `ABSENT`, rollback
`VISIBLE_VALIDATED`, and rollback state `SUCCEEDED`; all other combinations are
failed or ambiguous and nonpromoting. `evidence_namespace_state_at_seal` is
exactly `PENDING_BOUND`, `PENDING_UNLINKED`, or `PENDING_AMBIGUOUS`, describing
the physical-evidence pending name when the wrapper bytes are sealed; it is not
a prediction of the later no-replace evidence publication.

Before the final campaign rename, the supervisor creates the exact pending
evidence name with `O_CREAT|O_EXCL|O_NOFOLLOW`, mode `0600`, binds its device and
inode through the retained output-parent descriptor, and keeps that descriptor
open. On a physical failure it writes this wrapper only after the one rollback
attempt, fsyncs and chmods the held inode to `0444`, publishes it no-replace to
the exact final evidence name, and fsyncs the parent. Finalized evidence must be
canonical, mode `0444`, link count one, and deep-loadable.

The public physical-success finalizer never accepts an unchecked optional
snapshot. Its source input is the exact discriminated union
`Diag5FinalizerSourceInput = PublishedSnapshot | PreSourceFailure`:

- `PublishedSnapshot` has exactly `{kind,snapshot}`, with
  `kind=PUBLISHED_SNAPSHOT` and a descriptor-bound `SnapshotPublication` for
  the complete atomically published exact `source-snapshot` directory. This
  discriminant asserts publication and retained identity, not successful
  authority validation; the finalizer performs that validation itself. Any
  branch after the atomic source rename selects this variant. The publisher
  must construct and retain the descriptor-bound publication at that rename
  boundary before any later validation can raise; path reopen, reconstruction,
  or an unchecked optional local is not an admissible substitute.
- `PreSourceFailure` has exactly
  `{kind,outcome,supervisor_terminal,diagnostic_receipt}`, with
  `kind=PRE_SOURCE_FAILURE`, one exact `StructuredFailureV5`, and exact
  four-key ArtifactRefs to the sealed supervisor terminal and diagnostic
  receipt. `supervisor_terminal` is exactly
  `{relative_path="supervisor-terminal.json",sha256,size_bytes,schema_version="single-stage-neq-gntr3-trace-free-supervisor-terminal-v2"}`;
  `diagnostic_receipt` is exactly
  `{relative_path="diagnostic.json",sha256,size_bytes,schema_version="single-stage-neq-gntr3-trace-free-diagnostic-v2"}`.
  Each hash and size binds the referenced sealed bytes. The receipt contains
  the ordered evidence vector; no separate or caller-supplied vector is
  accepted.

`PreSourceFailure` is production-valid for exactly two outcomes:
`AUTHORITY/IDENTITY_REVALIDATION_FAILED` at the claimed-staging,
post-claim/pre-source boundary, or `SETUP/SOURCE_PUBLICATION_FAILED`. Both have
zero present nonterminal evidence slots: `source_manifest` is the first absent
slot, carries the exact selected failure reason, every later absent slot
serializes the exact closed reason `NOT_REACHED`, and `supervisor_terminal`
alone closes the vector. Source
publication is atomic/no-replace: an exception qualifies for this variant only
after descriptor-relative rollback of every unpublished exact
`.source-snapshot.staging-*` name created by that attempt and proof that both
the final `source-snapshot` name and every such staging name are lexically
absent. `SETUP/SOURCE_PUBLICATION_FAILED` selects `PreSourceFailure` only when
the exception occurred before the atomic source rename. If that rename already
occurred, a failure in the mechanical publication tail—opening, fsyncing, or
closing the destination parent, or checking the exact copied entry set—retains
the descriptor-bound publication and seals the same
`SETUP/SOURCE_PUBLICATION_FAILED` outcome with exactly the one-slot prefix:
`source_manifest` is PRESENT and every later nonterminal slot is ABSENT with
reason `NOT_REACHED`. That branch requires `PublishedSnapshot`, has
`launched_children=[]`, and is distinct from the zero-slot pre-rename branch by
the source-manifest slot and descriptor-relative source namespace occupancy;
neither parser nor finalizer may accept an unchecked union of the two prefixes.
The mechanical publisher constructs and retains the publication at the rename
boundary before any tail operation can fail. It does not perform the first
semantic authority snapshot validation. The supervisor performs that validation
only after the frozen numerical subset, native reference, and policy authority
have been published, at the normal post-setup boundary; failure there remains
`SETUP/SETUP_DEEP_LOAD_FAILED` with the exact four-slot prefix and a
`PublishedSnapshot`. A visible partial, dangling symlink, or ambiguous name that
cannot be represented by a complete descriptor-bound `SnapshotPublication` is
a hard pre-final publication failure and cannot be sealed as either source
publication branch.

No other SETUP reason is pre-source. `FROZEN_NUMERICAL_SUBSET_INVALID`,
`NATIVE_REFERENCE_INVALID`, `POLICY_AUTHORITY_INVALID`, and
`SETUP_DEEP_LOAD_FAILED` occur only after a complete, atomically published,
descriptor-bound snapshot exists; validation may itself select the latter
failure. They therefore require `PublishedSnapshot`; all child, numerical,
receipt, publication, and scientific outcomes require it as well. If the source manifest
slot is PRESENT or descriptor-relative `lstat` finds any `source-snapshot`
occupancy, `PreSourceFailure` is forbidden. Conversely, a
`PublishedSnapshot` input requires the slot PRESENT with the exact manifest
ArtifactRef and the exact snapshot directory identity for any sealed artifact
that can pass finalization; an absent path/slot or mismatch is a finalization
failure, never a fallback to `PreSourceFailure`. The finalizer derives and
checks this discriminant against sealed bytes and namespace state rather than
trusting the caller's tag.

On clean success the public `finalize_diag5_physical_evidence_success` operation
owns the indivisible final sequence while the still-bound empty pending-evidence
inode remains named. For `PublishedSnapshot`, it deep-loads the sealed
terminal/receipt/vector, validates the snapshot through the public
descriptor-bound validator, and joins its manifest slot. For
`PreSourceFailure`, it deep-loads the sealed terminal/receipt/vector, proves the
exact outcome and zero-prefix rules above, proves the `source-snapshot` path
and every attempt-owned `.source-snapshot.staging-*` name lexically absent
through the retained final-root dirfd, and revalidates the held authority
without invoking a snapshot validator on nonexistent bytes.
It then completes authority finalization, and only afterward
descriptor-relatively unlinks the exact pending inode, fsyncs the parent, and
proves the campaign final is
`VISIBLE_VALIDATED` while both the pending and final physical-evidence names are
`ABSENT`. Only that state may return `QUALITY_HIT` or `NO_HIT`. An unlink failure
is `POST_FINAL_AUTHORITY_FINALIZATION_FAILED`, requires the same one campaign
final-to-rollback attempt, seals the wrapper with
`evidence_namespace_state_at_seal=PENDING_BOUND`, and uses the retained pending
name for normal no-replace wrapper publication. An unlink-parent-fsync failure
or namespace ambiguity after successful unlink has the same original reason,
triggers the same one rollback attempt, and is nonpromoting. If the exact pending name is
still descriptor-proven to denote the retained inode, it carries and publishes
the wrapper normally. If the name is absent or its persistence is ambiguous,
the retained unlinked inode carries the canonical wrapper for process evidence,
but no namespace name is recreated and no publication retry occurs; the wrapper
records `PENDING_UNLINKED` or `PENDING_AMBIGUOUS` respectively, and the campaign
remains unadjudicated/nonpromoting.

Snapshot or authority revalidation failure inside that public finalizer is
`POST_FINAL_AUTHORITY_REVALIDATION_FAILED`; authority-finalization failure after
successful revalidation but before unlink is
`POST_FINAL_AUTHORITY_FINALIZATION_FAILED`. Either occurs while the reservation
is still named and therefore performs the one rollback/wrapper path. Successful
pending unlink is the last identity-sensitive operation: there is no subsequent
pre-unlock revalidation that could require an impossible rollback without the
reservation.

Malformed sealed terminal/receipt/vector bytes, discriminant/outcome/prefix
mismatch, snapshot-presence mismatch, or final-root deep-load failure is
`FINAL_DEEP_LOAD_FAILED`. Held-authority or published-snapshot identity drift is
`POST_FINAL_AUTHORITY_REVALIDATION_FAILED`. Both are detected before pending
unlink and trigger exactly one final-to-rollback attempt plus the canonical
physical-failure wrapper. There is no unchecked `None`, fallback path reopen,
or third finalizer-source variant.

If evidence publication itself is uncertain while its pending name remains, the
pending name is retained without cleanup and the campaign is
unadjudicated/nonpromoting; final and pending cannot both be treated as valid.
Authority revalidation accepts the final wrapper only for failure adjudication,
never for promotion or retry. Any visible final, rollback, or pending evidence
after ambiguity is invalid and nonpromoting; no cleanup or retry is allowed.

DIAG5 has exactly this ordered 26-slot evidence vector: `source_manifest`,
`frozen_numerical_subset`, `native_reference`, `policy_authority`,
`supervisor_before_preflight`, `preflight_producer`, `preflight_terminal`,
`preflight_process`, `preflight_memory`, `preflight_memory_samples`,
`preflight_runtime`, `preflight_policy`, `supervisor_before_cold`,
`cold_producer`, `cold_terminal`, `cold_process`, `cold_memory`,
`cold_memory_samples`, `cold_runtime`, `cold_policy`, `cold_history`,
`cold_terminal_numerical`, `cold_solve_timing`, `cold_safeguard_telemetry`,
`execution`, and `supervisor_terminal`. Every slot is exactly one typed
`PRESENT` reference or one closed `ABSENT` reason. The four cold scientific
slots remain one atomic subgroup. The v5 receipt/vector is a deliberate mixed
schema map: the slot mapping below contains shared unchanged v1 schemas, exact
DIAG5 v2 schemas, and DIAG5 supervisor v1 schemas. “v5” names the receipt loader
generation, not a rule that every child artifact has a v5 or v2 suffix. The v5
loader accepts only this exact per-slot mapping. Legacy loaders reject the v5
receipt/vector and every DIAG5-only schema, while shared v1 artifacts remain
valid shared artifacts when encountered in their own legacy schema contexts.
DIAG5 does not make the shared v1 native reference, policy, history, or result
artifact DIAG5-only merely by referencing it.

The exact slot schema mapping is:

- `source_manifest`: `single-stage-fullspace-source-manifest-v1`;
- `frozen_numerical_subset`:
  `single-stage-neq-gntr3-frozen-numerical-subset-v2`;
- `native_reference`: `single-stage-native-equivalent-reference-v1` at exact
  path `native-reference/reference.json`;
- `policy_authority`: `single-stage-neq-gntr3-policy-authority-v2`;
- both supervisor-before slots:
  `single-stage-neq-gntr3-diag5-supervisor-gpu-zero-v1`;
- `preflight_producer`: `single-stage-neq-gntr3-trace-free-preflight-v2`;
- `cold_producer`: `single-stage-neq-gntr3-trace-free-cold-result-v2`;
- both child terminal slots:
  `single-stage-neq-gntr3-trace-free-child-terminal-v2`;
- both process slots: `single-stage-neq-gntr3-trace-free-process-v2`;
- both memory slots: `single-stage-neq-gntr3-memory-v2`;
- both memory-sample slots:
  `single-stage-neq-gntr3-trace-free-memory-samples-v2`;
- both runtime slots: `single-stage-fullspace-runtime-evidence-v2`;
- both policy slots:
  `single-stage-native-equivalent-quality-policy-v1`;
- `cold_history`: `single-stage-fullspace-neq-gntr3-history-v1`;
- `cold_terminal_numerical`:
  `single-stage-fullspace-neq-gntr3-result-v1-terminal`;
- `cold_solve_timing`: `single-stage-neq-gntr3-solve-timing-v2`;
- `cold_safeguard_telemetry`:
  `single-stage-neq-gntr3-step-bound-safeguard-telemetry-v2`;
- `execution`: `single-stage-neq-gntr3-trace-free-execution-v2`; and
- `supervisor_terminal`:
  `single-stage-neq-gntr3-trace-free-supervisor-terminal-v2`.

The preflight and cold producer schemas each have one additional exact
discriminated variant for supervision closure. The parent publishes that
variant at `preflight/producer.json` or `cold/producer.json` only after `Popen`
succeeded and the launched child produced no valid producer document. It uses
the existing slot schema for its mode—respectively
`single-stage-neq-gntr3-trace-free-preflight-v2` or
`single-stage-neq-gntr3-trace-free-cold-result-v2`—and has exactly these sixteen
keys:

`{schema_version,route,numerical_route,numerical_result_schema_version,plan_sha256,child_mode,document_origin,execution_status,promotion_eligible,selected_failure_reason,child_pid,child_start_time_ticks,process_started_monotonic_ns,process_stopped_monotonic_ns,process_evidence,child_terminal_evidence}`.

Its values are exact: `route=NEQ-GNTR3-DIAG5`,
`numerical_route=NEQ-GNTR3`,
`numerical_result_schema_version=single-stage-fullspace-neq-gntr3-result-v1`,
and `plan_sha256=DIAG5_PLAN_SHA256`, the frozen prefix constant defined above.
`schema_version` is the selected mode's exact existing v2 schema. `child_mode`
is `PREFLIGHT` or `COLD` consistently with that schema and path,
`document_origin=PARENT_SUPERVISOR`,
`execution_status=SUPERVISION_FAILURE`, and `promotion_eligible=false`.
`selected_failure_reason` is exactly one of `PREFLIGHT_TIMEOUT`,
`PREFLIGHT_MONITOR_FAILED`, `PREFLIGHT_EXIT_NONZERO`,
`PREFLIGHT_PROTOCOL_INVALID`, or `PREFLIGHT_PRODUCER_INVALID` for preflight,
and the corresponding five `COLD_*` literals for cold. It equals the selected
supervisor-terminal reason and the reason on the first absent evidence slot.
The two evidence fields are exact four-key ArtifactRefs
`{relative_path,sha256,size_bytes,schema_version}` to that mode's
`terminal.json` and `process.json`; their paths and schemas equal the frozen
slot table. The process document in turn contains exact ArtifactRefs to the
byte-for-byte `stdout.bin` and `stderr.bin`, and validation resolves and hashes
both streams. Thus malformed or absent child producer bytes remain raw process
evidence and are never reinterpreted as child-authored typed evidence.

The four child identity/timing scalars equal the referenced process document
exactly: `child_pid` is positive; `child_start_time_ticks` is nonnegative and
may be zero only for a `*_MONITOR_FAILED` terminal whose monitor failure kind is
`BINDING`; and positive monotonic start/stop values satisfy start less than or
equal to stop. The referenced child terminal and process must cross-validate
their terminal status, monitor failure kind, return code, identity, and selected
reason. The supervision variant contains no `runtime`, `runtime_evidence`,
policy, numerical-bundle, solve, endpoint, profiler, or physics field. It is
nonpromoting process-lifecycle evidence, never invented runtime or scientific
evidence.

A launch failure means no child existed: its exact present prefix remains five
slots for `PREFLIGHT_LAUNCH_FAILED` or thirteen slots for
`COLD_LAUNCH_FAILED`, and producer, terminal, and process are all absent. For a
launched child, a valid child-authored producer always occupies the producer
slot unchanged; the parent may never replace or wrap it. For each of the five
preflight supervision reasons above, whether the producer slot contains that
valid child document or the parent supervision variant, the exact present
prefix is eight slots, ending with `preflight_process`. For each of the five
cold supervision reasons above, the exact present prefix is sixteen slots,
ending with `cold_process`. Any child-produced artifacts referenced by a valid
producer but lying beyond that failure prefix remain manifest-closed raw
artifacts and do not populate later typed vector slots. `PREFLIGHT_GATE_FAILED`
retains its exact twelve-slot valid-producer prefix; all later previously frozen
prefixes are unchanged except the exact snapshot-revalidation branches. The
additive DIAG5-only `BEFORE_PREFLIGHT/SOURCE_REVALIDATION_FAILED` reason has
exactly the five-slot setup prefix ending with `supervisor_before_preflight`,
exact expected-child tuple `launched_children=()` serialized as JSON `[]`, a
durable authority-consumption marker, and absent producer/terminal/process
because `Popen` was never called. The reserved
post-preflight `PREFLIGHT_PROTOCOL_INVALID` detail above has exact
`launched_children=["preflight"]` and permits only prefix eight, ending with
`preflight_process`, when the launched raw supervision chain is complete but
later typed child evidence is unavailable, or prefix twelve, ending with
`preflight_policy`, when the full valid preflight chain was published. An
earlier selected child supervision failure retains its own reason and prefix.
The reserved post-cold `COLD_PROTOCOL_INVALID` detail above has exactly the
sixteen-slot prefix ending with `cold_process`, exact
`launched_children=["preflight","cold"]`, and a valid unchanged child-authored
cold producer, terminal, and process; any generated memory, runtime, policy, or
numerical bytes beyond that prefix remain reason-conditioned manifest-only raw
custody under the table below. No supervision variant is used for either
reserved protocol branch. No supervision variant is permitted for an unlaunched
mode, a valid child producer may not be relabeled `SUPERVISION_FAILURE`, and no
prefix of five or thirteen is valid after a child was launched.

The frozen parser tables encode these exceptions directly:
`DIAG5_STAGE_REASON_PRESENT_PREFIXES[(SETUP,
SOURCE_PUBLICATION_FAILED)]=(0,1)` with expected-child tuple `()`; prefix zero
is lawful if and only if the source slot and source namespace are absent and
the finalizer input is `PreSourceFailure`, while prefix one is lawful if and
only if the exact source-manifest slot and descriptor-bound source namespace
are present and the finalizer input is `PublishedSnapshot`. The table is
discriminator-checked against physical occupancy and therefore is not a
permissive alternative-prefix union. Also,
`DIAG5_STAGE_REASON_PRESENT_PREFIXES[(BEFORE_PREFLIGHT,
SOURCE_REVALIDATION_FAILED)]=(5,)` and the expected-child table returns `()`;
for the reserved post-preflight detail it permits `(8,12)` with expected child
tuple `("preflight",)`; and for the reserved post-cold detail it permits `(16,)`
with expected child tuple `("preflight","cold")`. General
`PREFLIGHT_PROTOCOL_INVALID` and `COLD_PROTOCOL_INVALID` entries keep their
supervision-failure prefix rules for every nonreserved detail. Receipt parsing
first matches the exact stage/reason/detail discriminator and then the matching
prefix/child rule; it cannot select a permissive union across variants.

Raw child custody is exact and non-destructive. If `<mode>/producer.json`
exists but fails canonical decoding or the selected mode's producer validator,
the parent retains its descriptor and no-replace renames that exact inode to
`<mode>/invalid-producer.bin`, fsyncs the mode directory, then publishes the
supervision variant to the now-absent `producer.json` with `O_EXCL|O_NOFOLLOW`.
It never truncates, overwrites, reparses, or deletes the invalid bytes. The
opaque file is present if and only if child producer bytes existed and were
invalid; a truly absent child producer creates no `invalid-producer.bin`.

Every regular child artifact is mode `0444`, link count one, nonsymlink, and an
exact artifact-manifest-v2 entry `{relative_path,role,sha256,size_bytes}`. Every
child directory is mode `0555`; admission is descriptor-bound and hashes the
sealed bytes. The following table governs the complete direct supervision
custody surface under `preflight/` and `cold/`, including manifest-only
auxiliaries that lie beyond the typed-vector prefix. It does not replace the
26-slot table or the already frozen committed numerical-bundle contract:

| Relative path | Exact role | Exact logical schema | Admission |
| --- | --- | --- | --- |
| `<mode>/producer.json` | `<mode>_producer` | selected preflight/cold producer v2 | Required for every launched child; valid child bytes or the supervision variant, never both. |
| `<mode>/invalid-producer.bin` | `<mode>_invalid_producer` | `opaque-invalid-child-producer-v1` | Allowed only with a supervision variant when invalid child producer bytes existed. |
| `<mode>/terminal.json` | `<mode>_terminal` | `single-stage-neq-gntr3-trace-free-child-terminal-v2` | Required for every launched child. |
| `<mode>/process.json` | `<mode>_process` | `single-stage-neq-gntr3-trace-free-process-v2` | Required for every launched child. |
| `<mode>/stdout.bin` | `<mode>_stdout` | `raw-child-stdout-v1` | Required and byte-bound by `process.json`. |
| `<mode>/stderr.bin` | `<mode>_stderr` | `raw-child-stderr-v1` | Required and byte-bound by `process.json`. |
| `<mode>/gpu-memory.json` | `<mode>_memory` | `single-stage-neq-gntr3-memory-v2` | Required when its typed slot is present; otherwise allowed only as one valid manifest-only pair with the next row after a launched child. |
| `<mode>/gpu-memory-samples.json` | `<mode>_memory_samples` | `single-stage-neq-gntr3-trace-free-memory-samples-v2` | Required when its typed slot is present; otherwise allowed only as one valid manifest-only pair with the previous row. |
| `<mode>/runtime-evidence.json` | `<mode>_runtime` | `single-stage-fullspace-runtime-evidence-v2` | Required when its typed slot is present; otherwise allowed as an independently valid manifest-only auxiliary generated before the selected supervision failure. |
| `<mode>/policy.json` | `<mode>_policy` | `single-stage-native-equivalent-quality-policy-v1` | Required when its typed slot is present; otherwise allowed as an independently valid manifest-only auxiliary generated before the selected supervision failure. |
| `cold/uncommitted-numerical-result/**` | `uncommitted_cold_numerical_result` | `opaque-uncommitted-cold-numerical-result-v1` | Allowed only after mandatory quarantine for one of the five launched COLD supervision reasons or `PENDING_RESULT_INVALID`; never typed evidence. |
| `cold/uncommitted-numerical-result.empty.json` | `empty_uncommitted_cold_numerical_result` | `single-stage-neq-gntr3-empty-quarantine-v1` | Required only when the quarantined pending tree is exactly empty. |

Here `<mode>` expands only to `preflight` or `cold`, and the selected producer
schema and role use that same mode. Optional memory, runtime, and policy rows
are reason-conditioned auxiliary manifest members for the five selected
supervision reasons even though the typed vector stops at eight or sixteen;
they cannot fill later evidence slots. A valid producer's referenced runtime or
policy bytes must equal the corresponding auxiliary row when present. Every
file beneath the opaque cold quarantine is individually manifest-listed under
the one quarantine role, mode `0444`, link count one, and beneath mode-`0555`
directories; the pending name is absent.

An exactly empty pending directory is still present and invalid. The parent
no-replace renames its unchanged empty contents to the quarantine root, seals
that root mode `0555`, fsyncs `cold/`, and publishes the sibling marker
`cold/uncommitted-numerical-result.empty.json` no-replace. That canonical marker
has exact keys
`{schema_version,route,quarantine_relative_path,selected_failure_reason}` with
schema `single-stage-neq-gntr3-empty-quarantine-v1`, route
`NEQ-GNTR3-DIAG5`, quarantine path
`cold/uncommitted-numerical-result`, and the selected COLD supervision reason or
`PENDING_RESULT_INVALID`. The file-only artifact manifest lists and hashes the
marker; the loader separately proves by descriptor-relative enumeration that
the quarantine directory exists, is real, mode `0555`, and has zero entries.
The marker is forbidden when the quarantine has any entry. Marker creation,
fsync, mode, link, or empty-directory validation failure is
`QUARANTINE_FAILED`; the parent never inserts a marker inside the quarantined
tree, so its empty contents remain exact.

The no-other-path rule in this paragraph applies to the direct supervision
custody paths and opaque quarantine rows in the table. A lawful PRESENT typed
slot remains governed by the 26-slot path/schema table, and
`cold/numerical-result/**` remains governed by the exact committed numerical
bundle schema, four-slot atomicity, ArtifactRefs, modes, hashes, and manifest
rules frozen above; neither is excluded by this custody table. Outside those
two existing authorities, no other child-relative path, schema, role, special
file, symlink, hardlink, empty directory, or unlisted leaf is admissible. An
unknown extra makes sealing fail closed and remains preserved in the one-shot
staging tree; the supervisor performs no deletion or alternate quarantine.

The artifact root manifest is
`single-stage-neq-gntr3-trace-free-artifact-manifest-v2` and the receipt is
`single-stage-neq-gntr3-trace-free-diagnostic-v2`. `DiagnosticReceiptV5` and
its manifest loader require this exact mapping and 26-slot order. The v5 loader
rejects every alternate schema at every slot, missing/extra slots, all v1--v4
receipt/vector schemas, and invented v6 labels. Every legacy loader rejects the
v5 receipt/vector, v2 artifact manifest, and every DIAG5-only slot schema; it
does not reject a shared v1 artifact solely because v5 also references that
unchanged schema.

Except for their schema, route, plan, and authority identities, every v2 slot
payload retains the exact corresponding DIAG4 key set and semantics. The only
scientific-document extension is native binding: runtime v2 adds the three
fields defined above; execution v2 adds exact `gpu_native_binding` plus
`authority_sha256`; receipt v2 adds exact `native_bindings` with `cpu` and `gpu`
objects using the six-key authority binding sets above, and one typed
`predecessor_postmortem` artifact reference. A valid child-authored producer
continues to reference runtime evidence rather than duplicate native fields;
the exact parent-origin `SUPERVISION_FAILURE` variant above deliberately has no
runtime field because none is authorized by that failure closure. CPU
scientific evidence carries exact `cpu_native_binding` and
`predecessor_postmortem` fields. No other v2 key addition, omission, alias, or
fallback is permitted.

The exact terminal stage order and reason sets are:

1. `AUTHORITY`: `AUTHORITY_INVALID`, `OUTPUT_ROOT_NOT_ABSENT`,
   `LOCK_CLAIM_FAILED`, `IDENTITY_REVALIDATION_FAILED`,
   `AUTHORITY_ALREADY_CONSUMED`;
2. `SETUP`: `SOURCE_PUBLICATION_FAILED`,
   `FROZEN_NUMERICAL_SUBSET_INVALID`, `NATIVE_REFERENCE_INVALID`,
   `POLICY_AUTHORITY_INVALID`, `SETUP_DEEP_LOAD_FAILED`;
3. `BEFORE_PREFLIGHT`: `SUPERVISOR_GPU_OBSERVATION_INVALID`,
   `SUPERVISOR_GPU_NONZERO`, `AUTHORITY_CONSUMPTION_FAILED`,
   `AUTHORITY_CONSUMPTION_UNCERTAIN`, `SOURCE_REVALIDATION_FAILED`;
4. `PREFLIGHT`: `PREFLIGHT_LAUNCH_FAILED`, `PREFLIGHT_TIMEOUT`,
   `PREFLIGHT_MONITOR_FAILED`, `PREFLIGHT_EXIT_NONZERO`,
   `PREFLIGHT_PROTOCOL_INVALID`, `PREFLIGHT_PRODUCER_INVALID`,
   `PREFLIGHT_GATE_FAILED`;
5. `BEFORE_COLD`: `SUPERVISOR_GPU_OBSERVATION_INVALID`,
   `SUPERVISOR_GPU_NONZERO`, `SOURCE_REVALIDATION_FAILED`,
   `IDENTITY_REVALIDATION_FAILED`, `CONSUMPTION_MARKER_INVALID`;
6. `COLD`: `COLD_LAUNCH_FAILED`, `COLD_TIMEOUT`,
   `COLD_MONITOR_FAILED`, `COLD_EXIT_NONZERO`, `COLD_PROTOCOL_INVALID`,
   `COLD_PRODUCER_INVALID`;
7. `NUMERICAL_COMMIT`: `PENDING_RESULT_ABSENT`, `TIMING_INVALID`,
   `SAFEGUARD_TELEMETRY_INVALID`, `NUMERICAL_IDENTITY_MISMATCH`,
   `QUARANTINE_FAILED`, `PENDING_RESULT_INVALID`, `COMMIT_COLLISION`,
   `COMMIT_RENAME_FAILED`, `COMMIT_FSYNC_FAILED`,
   `COMMITTED_DEEP_LOAD_FAILED`;
8. `RECEIPT`: `EVIDENCE_VECTOR_INVALID`, `GROUP_PREFIX_INVALID`,
   `SCIENTIFIC_RECONSTRUCTION_FAILED`, `RECEIPT_SCHEMA_INVALID`;
9. `PUBLICATION`: `MANIFEST_INVALID`, `MODE_OR_LINK_INVALID`,
   `STAGING_DEEP_LOAD_FAILED`, `FINAL_COLLISION`, `FINAL_RENAME_FAILED`; and
10. `SCIENTIFIC`: `INCOMPLETE`, `NO_HIT`, `QUALITY_HIT`.

Order is precedence. No reason from a later stage can replace an earlier-stage
failure, and opaque/quarantined evidence cannot populate a typed slot.

`BEFORE_PREFLIGHT/SOURCE_REVALIDATION_FAILED` is additive DIAG5 v5 vocabulary
only. V5 receipt/vector parsing accepts it only with the exact consumed,
no-child five-slot branch above. Every v1--v4 loader and terminal parser rejects
this new stage/reason pair; no shared legacy artifact acquires the new meaning.

The five `AUTHORITY` reasons are closed receipt/parser vocabulary. Before a
staging namespace is safely and exclusively claimed, every authority-validation,
path/output, consumption-state, lock-claim, or identity failure is a typed
out-of-band prelaunch CLI/claim failure because no authorized artifact root
exists. `main` emits or raises the exact typed
`StructuredFailureV5(AUTHORITY, <one of the five AUTHORITY reasons>,
detail_sha256)` through its prelaunch CLI failure channel, exits nonzero,
launches no child, and performs no output-root, staging, rollback,
consumption-marker, or physical-evidence namespace mutation.

`run_diag5` requires an already active retained-descriptor staging claim. After
that claim, exactly one `AUTHORITY` branch is production-reachable: held
authority/control/plan/native identity drift outside the GPU snapshot validator
at the mandatory post-staging/setup, pre-consumption revalidation seals a
zero-child `AUTHORITY/IDENTITY_REVALIDATION_FAILED` artifact in the claimed
staging/final lifecycle. GPU snapshot validation at that same boundary is the
more specific `SETUP/SETUP_DEEP_LOAD_FAILED` mapping frozen above. The other
four `AUTHORITY` reasons remain pre-staging out-of-band only. A GPU-zero
observation failure after that revalidation maps to its exact
`BEFORE_PREFLIGHT` reason, and marker publication failure maps to its exact
authority-consumption reason. After preflight, GPU snapshot/source drift maps to
`SOURCE_REVALIDATION_FAILED`, while other held-identity drift maps to
`IDENTITY_REVALIDATION_FAILED`, at `BEFORE_COLD`; later boundaries retain their already frozen COLD,
NUMERICAL_COMMIT, RECEIPT, PUBLICATION, or out-of-band post-final reason. Parser
and schema fixtures may instantiate all five closed `AUTHORITY` reason values,
but only the exact claimed-staging identity-revalidation branch may validate as
a production artifact.

## Done criteria

This successor tranche is complete only when one of these is true:

- a complete, provenance-valid DIAG5 artifact passes native-equivalent quality
  and reports synchronized RTX 5090 solve time against `287.304 s`; or
- a complete fail-closed bounded-negative record identifies the first failed
  gate without relabeling incomplete evidence as a physics or speed result.

No A100 run is authorized by this recovery tranche.

## Qualification Record
