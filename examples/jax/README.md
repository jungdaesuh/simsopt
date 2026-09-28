# JAX-first examples

Official upstream coverage is pinned to `hiddenSymmetries/simsopt`, branch
`master`, commit `9e027eac38028d57aa23777be52a781aa860e347`. Only examples
present in that official catalog count toward coverage.

Every eligible official native example owns exactly one JAX mirror at the same tier and
the same filename: `examples/<tier>/<name>.py` is mirrored by
`examples/jax/<tier>/<name>.py`. The mirror teaches the public JAX interfaces
directly — immutable state, compiled computations, batching, and explicit
host/device boundaries — while executing the scientific workflow of the one
native source it names.

The generated
[`NATIVE_TO_JAX_INDEX.md`](NATIVE_TO_JAX_INDEX.md) lists the native sources,
their exact JAX mirrors or blockers, device scope, execution scale, and latest
evidence status. Registered branch-only experiments would appear in a separate
section and not count toward official coverage. Regenerate it from the validated manifests and compact
authority record with
`python -m examples.jax.native_to_jax_index --write`; use `--check` in
validation.

## Retained shipped-workload validation

The 2026-09-22 UTC verification ran all 25 eligible official mappings at
`native_default` on local validation snapshot
`998148365e8db55577b4372c4c791ef6bc20cc35`. Each JAX lane was compared with
native CPU separately:

| Comparison | Case-contract pass | Qualified quality-band | Failed |
| --- | ---: | ---: | ---: |
| Native CPU / JAX CPU | 20 | 5 | 0 |
| Native CPU / JAX GPU | 19 | 4 | 2 |

The four qualified GPU cases are BoozerQA, coil forces, finite-build stage two,
and minimal stage two. A quality-band result does not establish convergence;
coil forces retains its raw GPU solver failure. QFM's GPU endpoint-quality
failure and the NCSX GPU long-trajectory comparison failure remain unresolved.

The independent official reference checks selected construction, input and count
quantities. Endpoint acceptance uses the individual native/JAX case contracts;
this does not establish equivalence of every official trajectory or setting.
PM4Stell's backtracking deviation and the planar-coil upstream-gradient
limitation remain disclosed in the case contracts. These are two native-referenced
comparisons, not a complete three-lane comparison matrix.

The source-bound reports and raw packets are retained locally under
`.artifacts/reconciliation-execution-20260921/`, which is not distributed in
Git. These counts describe that snapshot, not a new run on the current checkout.
Verification jobs overlapped, so their elapsed times do not establish speedups.

Install the CPU runtime from the repository root with:

```console
python -m pip install -e ".[JAX]"
```

Use `.[JAX_GPU]` in a supported CUDA environment.

## Running the mirrors

Runtime selection is process-wide and must happen before importing JAX-heavy
modules, so use the isolated runner rather than executing several examples in
one Python process. The ordinary runner includes every ready registered record;
it is not the official mirror verification batch. Every ready record supports
both devices and both intents:

```console
python examples/jax/run_examples.py --device cpu --scale bounded
python examples/jax/run_examples.py --device gpu --scale bounded
python examples/jax/run_examples.py --device cpu --intent parity --scale bounded
python examples/jax/run_examples.py --device gpu --intent parity --scale bounded
```

`--scale` is a typed selector, not an inference from the command name. It
defaults to `bounded`, but state it explicitly in CI and authority commands so
the emitted child argv, canonical input, receipt, and artifact scale are all
attributable. Only `bounded` maps to the child `--smoke` flag; `native_default`
emits no scale flag and runs the example's native-default step budget.
Callbacks passed to the public `run_example` helper receive
`(output_directory, max_steps, execution_scale)`. The scale is independent of
the step budget; custom callbacks must not recover it from iteration-count
thresholds.

Omitting `--intent` selects `fast`. Fast is the ordinary default and is not
certifying. The fully unset repository default remains `native_cpu`; JAX is
never selected implicitly.

Both ordinary-runner intents pin FP64 (`SIMSOPT_PRECISION=fp64`,
`JAX_ENABLE_X64=1`), strict backend selection, the same public objective and
custom SIMSOPT JAX solver family, and the same scientific-success checks.
Parity additionally selects the stable numerical policy. Only
[`run_parity.py`](run_parity.py) can publish certification evidence; ordinary
fast and parity example runs are diagnostic.

The strict device-to-host transfer guard applies to the GPU parity profile
only: it sets `JAX_TRANSFER_GUARD=disallow` and
`SIMSOPT_JAX_TRANSFER_GUARD=disallow`. GPU fast and both CPU profiles run a
non-fatal guard and must not be read as strict-purity evidence. Under
`disallow` the guard stays in force across example setup, SIMSOPT
orchestration, result publication, and all transfers. The custom
operator-GMRES implementation retains one documented, host-to-device-only
allowance scoped inside JAX's `gmres` call because that upstream routine lowers
internal scalar literals through host-to-device conversion. It does not permit
device-to-host materialization, and the surrounding numerical path remains
guarded.

The retained `--lane cpu-smoke` and `--lane gpu-strict` aliases still select
their historical parity profiles and emit a deprecation warning. They cannot be
combined with `--intent` and support only `--scale bounded`. New callers should
use `--device`, `--intent`, and `--scale`.

Device capability and bounded arguments remain in
[`manifest.json`](manifest.json); execution intent is a suite-wide runtime
policy and is not copied into every example record. Device selection changes
placement, not the algorithm or the public result contract. Ready JAX examples
normally use a JAX optimizer. Case-bound policies also permit a **CPU SciPy
outer optimizer over JAX physics and derivatives** where explicitly declared:

| Scope | Example | CPU controller | JAX numerical region |
| --- | --- | --- | --- |
| Official upstream mirror | Standard, minimal, planar, stochastic and finite-build stage-two; coil forces | SciPy L-BFGS-B | Coil physics, objective and derivatives |
| Official upstream mirror | QFM surface (`1_Simple/qfm.py`) | SciPy L-BFGS-B, then SciPy SLSQP | QFM residual, labels and their derivatives |
| Official upstream mirror | `just_a_quadratic`, `minimize_curve_length`, `surf_vol_area` | SciPy `least_squares` (TRF) | Residuals and Jacobians |

These are the providers the upstream scripts call, with upstream's own options. The minimal and finite-build
stage-two mirrors additionally accept `--device-solver`, an opt-in performance mode that replaces the host
provider by the in-tree device-resident L-BFGS-B (a JAX reimplementation of SciPy's algorithm); that mode is not
the upstream mirror and is never what the parity cases run.

The `outer_optimizer_policy` manifest field declares the policy owned by that
example. The declaration is bound to the registered example and, for parity,
the exact case and driver. A driver name alone grants no exception, and another
example cannot borrow this policy. Missing declarations retain the default ban.
This permission does not relax transfer guards, source provenance, scientific
acceptance, or parity comparisons. CPU control is not a GPU-resident optimizer;
performance claims must state whether they time the JAX region or the complete
workflow. Experimental policies authorize explicit experimental execution;
they do not contribute official upstream coverage.

Optimistix and Optax remain explicit optional driver choices and are never
selected implicitly by the serial example APIs. The serial wrappers publish `problem.x` and their
bounded log only after a successful solve; a failed result raises and leaves
caller-owned state unchanged. The deprecated least-squares `optimizer="lm"`
alias still selects explicit Optimistix LM; the legacy `gauss_newton` and
scalar `bfgs` spellings are rejected because the typed API exposes no
behavior-equivalent Optimistix driver for them.

Applications can use the same typed selection before importing JAX-heavy
modules:

```python
import simsopt_jax.config as simsopt_config

simsopt_config.set_backend("jax", device="cpu")
simsopt_config.set_backend("jax", device="gpu", intent="parity")
```

Passing a canonical mode such as `jax_cpu_float32_smoke` remains supported and
cannot be combined with `device` or `intent`. An unavailable requested GPU
fails; it never falls back to CPU.

## Native/JAX parity evidence

The paired parity runner reconstructs matched native SIMSOPT CPU and JAX
workflows from one serialized input bundle. Run the applicable official
upstream cases at their native-default scale on CPU with:

```console
python examples/jax/run_parity.py \
  --case all-applicable \
  --lanes native-cpu,jax-cpu \
  --scale native_default \
  --artifact-root .artifacts/jax-example-parity
```

In a CUDA environment, use the full matched lane set:

```console
python examples/jax/run_parity.py \
  --case all-applicable \
  --lanes native-cpu,jax-cpu,jax-gpu \
  --scale native_default \
  --artifact-root .artifacts/jax-example-parity
```

Then audit the published run independently and require authority explicitly:

```console
python -m examples.jax.parity.audit \
  --run .artifacts/jax-example-parity/<run-id> \
  --repo-root "$PWD" \
  --require-authoritative
```

`--case` also accepts individual case IDs, repeated. The legacy `--smoke` flag
is still accepted but no longer selects scale, and is rejected together with
`--scale native_default`; use `--scale`.

The runner fixes FP64 and platform selection before importing JAX and pins both
transfer guards to `disallow` in the `jax-gpu` lane; a missing GPU, CPU
fallback, wrong precision, implicit transfer, or forbidden host solver fails the
run. Each lane executes in a fresh process.
The published directory contains canonical input JSON/NPY files, one hash-bound
lane receipt per case, and an aggregate `summary.json` that records the loaded
manifest version pair and the selected scale. Failed or interrupted runs retain
a diagnostic `.partial` directory and are never published as passing evidence.

Only a clean committed run whose lane receipts are all marked authoritative may
promote a parity classification. Dirty-checkout runs remain useful exploratory
evidence and record the tracked diff hash plus untracked-file inventory.

Bounded and native-default scale are independent of workflow coverage: a
bounded `full` case is not native-default evidence. Every tracked parity
relationship records `scale_tier: bounded`, and native-default authority runs
only from the manual `run_native_default` dispatch input of the GPU parity
workflow. Treat native-default evidence as not run rather than inferring it
from a bounded pass. Current authority bundles are local-only: `.artifacts/` is
ignored, is not a durable shared archive, and cannot by itself support a
remotely reproducible retention claim.

Memory receipts use `XLA_PYTHON_CLIENT_PREALLOCATE=false`, synchronize the JAX
publication boundary, and report one combined import/compile/warmup/bounded-run
peak. They do not claim a separate steady-state peak and support no speed
claim. The receipts pin that setting explicitly rather than relying on the
runtime: the `jax_gpu_parity` and `jax_gpu_fast` modes already default
`xla_gpu_preallocate` to `False`
(`src/simsopt_jax/backend/_runtime_policy.py`, `_GPU_MEMORY_MODE_DEFAULTS`),
so preallocation is off under any supported GPU mode unless the user sets
`SIMSOPT_JAX_GPU_PREALLOCATE` or passes `xla_gpu_preallocate=True`. A script
that selects a device through `JAX_PLATFORMS` alone, without going through
`set_backend`, gets JAX's own preallocating default instead.

Generate a results table from an independently audited authority bundle with:

```console
python examples/jax/parity/report.py \
  --summary <run>/summary.json \
  --output <reviewed-output-path>
```

The intended documentation path is not tracked yet; do not overwrite a
pre-existing worktree file without reviewing it first.

## The one-to-one identity contract

The authoritative inventory is [`manifest.json`](manifest.json) and
[`parity_manifest.json`](parity_manifest.json). Nothing else in this directory
defines coverage.

`manifest.json` separates official coverage from branch-only registrations:

- `source_catalog` — 53 official upstream source rows: 25 `eligible`, 1 `hybrid`,
  25 `blocked`, and 2 `not_applicable`. Membership is checked against the pinned
  official catalog, not whatever Python files happen to be in the local tiers.
- `experimental_sources` — branch-only source rows, which contribute zero
  official upstream coverage. There are none.
- `jax_examples` — 26 executable records, 25 `ready` and 1 `planned` (the
  hybrid VMEC single-stage mirror). Each owns one official source.

An owned record must sit at the identical tier and filename as its source, must
be typed `one_to_one`, and cannot be a tutorial. Each mirror is owned by at
most one source. `parity_manifest.json` separates 26 official relationships
(25 `full`, 1 `unsupported`) from experimental relationships, of which there
are none. Execution scale and verified evidence are separate from source
coverage.

`run_parity.py --case all-applicable` selects the 25 executable official
relationships. Experimental cases remain available by explicit case ID;
registration and safety checks still apply, but their results do not count
toward official coverage.

List the pairs from the manifest rather than from a hand-maintained table:

```console
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("examples/jax/manifest.json").read_text())
examples = {record["id"]: record for record in manifest["jax_examples"]}
for source in manifest["source_catalog"]:
    mirror_id = source["mirror_example_id"]
    if mirror_id is None:
        continue
    mirror = examples[mirror_id]
    print(
        f"{source['disposition']:12s} examples/{source['source']}"
        f" -> examples/jax/{mirror['path']}"
        f" [{mirror['classification']}/{mirror['status']}]"
    )
PY
```

The generated native-to-JAX index linked above replaces this ad hoc listing
when a stable, citable source-to-mirror inventory is needed.

## Classification vocabulary

Executable records use the typed `classification` vocabulary:

- `mirror` — the workflow runs entirely on public JAX surfaces and declares no
  host boundary.
- `adapter` — the workflow begins with native SIMSOPT objects, snapshots their
  state through `simsopt_jax_adapters`, runs the numerical region in JAX, and
  publishes accepted state explicitly. It must name at least one host boundary.
- `hybrid` — the workflow retains a named native or external computation. Its
  GPU device scope must be declared `jax_slice_only`.
- `tutorial` — a combined lesson. It owns no native source and contributes
  zero one-to-one mirror coverage.

`teaching_kind` is orthogonal: `one_to_one` for every owned mirror, and
`combined` for tutorials.

Catalog rows use the typed `disposition` vocabulary: `eligible` (owns a
`mirror` or `adapter`), `hybrid` (owns the hybrid executable), `blocked`, and
`not_applicable`. Blocked and not-applicable rows are not placeholders: each
states the missing public boundary or external limitation required for
reconsideration.

Parity relationships use their own vocabulary. `full` covers every declared
scientific stage, `reduced` explicitly omits at least one scientific stage, and
`unsupported` names a concrete blocker. `bounded`, `native_default`, and
`not_applicable` describe scale independently of workflow coverage. Live oracle
kinds are `native_source_owned_simsopt` for executable relationships and
`pending_native_oracle` for unsupported relationships. An oracle kind describes
the comparator, not membership in official upstream; that comes from the
pinned source catalog.

## Official VMEC hybrid single-stage

`examples/jax/3_Advanced/single_stage_optimization.py` mirrors
`examples/3_Advanced/single_stage_optimization.py`. VMEC and its
finite-difference equilibrium derivatives stay on the CPU/MPI host; only the
coil-field, quadratic-flux, penalty, and mixed-derivative slice runs in JAX.
Its declared GPU scope is therefore `jax_slice_only`, and any GPU statement
about this example is a statement about the JAX slice, never the whole
workflow. Its CPU scope is `host_and_jax_slice`.

It runs under `mpiexec` with the same bounded arguments:

```console
mpiexec -n 1 python examples/jax/3_Advanced/single_stage_optimization.py \
  --smoke --json --output-dir <output-dir>
```

Its manifest status is `planned` and its parity classification is
`unsupported`, with no case ID, so `--case all-applicable` does not select it.
It is not promoted and holds no parity claim. Promotion is blocked on the
workflow-dispatch-only [VMEC hybrid authority
workflow](../../.github/workflows/jax_vmec_hybrid_authority.yml) lane proving
immutable VMEC/MPI build identity, the recorded MPI world size, and matched
CPU and GPU slice provenance on an approved runner.

## Manifest schema

The only accepted contract pair is example-schema-v3 plus parity-schema-v2,
read atomically; any other version of either document is rejected. Every parity
`summary.json` records `manifest_schema_version`,
`parity_manifest_schema_version`, and `used_legacy_manifest_adapter`, so an
audited bundle states which contract produced it. No legacy manifest reader
exists, so `used_legacy_manifest_adapter` is always `false`; the key stays so
summary schema 2 keeps its shape, and the auditor rejects any other value.

The active v3/v2 documents separate `experimental_sources` and
`experimental_relationships` from official coverage. Those arrays may be omitted
when empty. Historical v3 documents that mixed local extensions into
`source_catalog` must be split before use with the current validator; historical
receipts retain their original files and must be audited at their recorded
source revision.

[`one_to_one_inventory.json`](one_to_one_inventory.json) is a dated record of
the earlier 52-source migration inventory, not the current official catalog.
The current index is regenerated with
`python -m examples.jax.native_to_jax_index --write`.

## Author contract

Every runnable script must:

1. import a public `simsopt_jax` or `simsopt_jax_adapters` surface directly;
2. support `--smoke --json` and keep bounded work deterministic; the runner
   emits `--smoke` for `--scale bounded` and omits it for
   `--scale native_default`;
3. emit a final JSON line containing `example_id`, `backend_mode`, `platform`,
   `precision`, `status`, and an `observables` object;
4. return nonzero unless its independent scientific checks pass;
5. name every host boundary in both its opening comment and the manifest;
6. accept `--output-dir` or use a temporary directory if it writes artifacts;
7. reuse canonical repository inputs read-only and never forward to a native
   example module;
8. declare both `cpu` and `gpu` while it is `ready`, and use the same public JAX
   solver and algorithm on both devices.

A one-to-one mirror must additionally sit at the exact tier and filename of the
native source it mirrors, and must be typed `mirror` or `adapter`, or `hybrid`
when it retains a named native or external computation.

Add the behavioral correctness test first and preserve its authentic
RED → GREEN → REFACTOR commands in
[`docs/jax_examples_tdd_receipts.md`](../../docs/jax_examples_tdd_receipts.md).
Then mark the manifest record `ready`; the validator rejects a ready record
whose script or listed correctness tests do not exist, and the example tests
reject a ready record without both devices or a public JAX import.
