# JAX-first examples

Every eligible native example owns exactly one JAX mirror at the same tier and
the same filename: `examples/<tier>/<name>.py` is mirrored by
`examples/jax/<tier>/<name>.py`. The mirror teaches the public JAX interfaces
directly — immutable state, compiled computations, batching, and explicit
host/device boundaries — while executing the scientific workflow of the one
native source it names.

A separate set of combined and compatibility programs also lives here. They are
typed as `tutorial`, they own no native source, and they contribute zero
one-to-one mirror coverage. The manifest validator enforces that: a tutorial
record cannot own a catalog source.

The generated
[`NATIVE_TO_JAX_INDEX.md`](NATIVE_TO_JAX_INDEX.md) lists the native sources,
their exact JAX mirrors or blockers, device scope, execution scale, and latest
authority status. Regenerate it from the validated manifests and compact
authority record with
`python -m examples.jax.native_to_jax_index --write`; use `--check` in
validation.

Install the CPU runtime from the repository root with:

```console
python -m pip install -e ".[JAX]"
```

Use `.[JAX_GPU]` in a supported CUDA environment.

## Running the mirrors

Runtime selection is process-wide and must happen before importing JAX-heavy
modules, so use the isolated runner rather than executing several examples in
one Python process. Every ready record runs on both devices under both
intents:

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
normally use a JAX optimizer. Two explicitly declared Boozer workflows permit a
**CPU SciPy outer optimizer over JAX physics and derivatives**:

| Example | CPU controller | JAX numerical region |
| --- | --- | --- |
| `native-single-stage-boozer-vacuum-optimization` | SciPy BFGS | Exact Boozer Newton solve and implicit objective gradient |
| `native-boozerqa-ls` | SciPy L-BFGS-B | Reduced-Schur nested-LS solve and implicit objective gradient |

The optional `outer_optimizer_policy` manifest field declares
`scipy-bfgs-over-jax-exact-boozer` or `scipy-lbfgsb-over-jax-reduced-boozer` for
these respective examples. The declaration is bound to the registered example
and, for parity, the exact case and driver. A driver name alone grants no exception, and another
example cannot borrow this policy. Missing declarations retain the default ban.
This permission does not relax transfer guards, source provenance, scientific
acceptance, or parity comparisons. CPU control is not a GPU-resident optimizer;
performance claims must state whether they time the JAX region or the complete
workflow. Other native SciPy solves remain CPU reference oracles.

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
workflows from one serialized input bundle. Run every currently applicable
bounded case on CPU with:

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

`manifest.json` holds two tables:

- `source_catalog` — 53 rows, one per tracked native example in the native
  tiers. The validator checks that set exactly: 27 `eligible`, 1 `hybrid`,
  23 `blocked`, and 2 `not_applicable`.
- `jax_examples` — 41 executable records, 39 `ready` and 2 `planned`.
  Twenty-eight own a catalog source; the other 13 combined or compatibility
  programs own no native source.

An owned record must sit at the identical tier and filename as its source, must
be typed `one_to_one`, and cannot be a tutorial. Each mirror is owned by at
most one source. `parity_manifest.json` holds 28 relationships: 26 `full`
relationships with a case ID and 2 `unsupported` relationships. Execution scale
and verified evidence are separate from source coverage.

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
- `tutorial` — a combined or compatibility lesson. It owns no native source
  and contributes zero one-to-one mirror coverage.

`teaching_kind` is orthogonal: `one_to_one` for every owned mirror, and
`combined` or `compatibility` for tutorials. A `compatibility` tutorial must
name its successor mirror ID and carry a warning text that names both itself
and that successor.

Catalog rows use the typed `disposition` vocabulary: `eligible` (owns a
`mirror` or `adapter`), `hybrid` (owns the hybrid executable), `blocked`, and
`not_applicable`. Blocked and not-applicable rows are not placeholders: each
states the missing public boundary or external limitation required for
reconsideration.

Parity relationships use their own vocabulary. `full` covers every declared
scientific stage, `reduced` explicitly omits at least one scientific stage, and
`unsupported` names a concrete blocker. `bounded`, `native_default`, and
`not_applicable` describe scale independently of workflow coverage. Live oracle
kinds are `native_source_owned_simsopt` for the 26 executable relationships and
`pending_native_oracle` for the two unsupported relationships.

## Pairs that need their own command

### Boozer/vacuum single-stage

The native source and its mirror take the same arguments and can be run
side by side:

```console
python examples/3_Advanced/single_stage_boozer_vacuum_optimization.py \
  --smoke --json
python examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py \
  --smoke --json
```

Its parity relationship covers the `full` workflow at `native_default` scale, and its cost tier is
`scheduled`, so run it on its own rather than expecting it inside a quick loop
(drop `jax-gpu` from `--lanes` outside a CUDA environment):

```console
python examples/jax/run_parity.py \
  --case native-single-stage-boozer-vacuum-optimization \
  --lanes native-cpu,jax-cpu,jax-gpu \
  --scale native_default \
  --artifact-root .artifacts/jax-example-parity
```

A `ready` executable and a `full` workflow relationship do not certify parity.
The fresh full-default run `20260917T025023Z-8c8a5461` passed the source/build
audit at commit `9764474ab161767098a56c9655fc442c6dc94a51`. All three endpoint
objectives meet the declared `1e-7` quality bound, but 12 of 57 strict comparisons
failed and all three lanes exhausted 1000 iterations. This establishes endpoint
quality only, not strict parity or optimizer convergence. The superseded run
`20260916T194954Z-e3bd7376` used invalid provenance and remains historical; its
raw receipts are preserved locally and it no longer supplies the index record.

The native build receipt binds the extension binary to recorded source and
build inputs. It is a local, unsigned, operator-writable record, not a signed
or hermetic build attestation. A tracked evidence pointer alone cannot verify
the run: auditing requires the retained lane receipts and their clean recorded
checkout. Neither provenance nor an endpoint quality band proves convergence.

### Serial nested-LS Boozer

`2_Intermediate/boozerQA_ls.py` uses the approved CPU SciPy L-BFGS-B controller
with JAX reduced-Schur physics. Its native CPU reference is
`native_reference/boozerQA_ls.py`; both support `--smoke --json`. This is a
combined tutorial, not a one-to-one replacement for the MPI example. The
validated three-step default run establishes improving feasible steps, not
outer convergence or a whole-workflow speedup.

### VMEC hybrid single-stage

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

## Manifest schema deprecation interval

The active contract pair is example-schema-v3 plus parity-schema-v2. The
legacy example-schema-v2 plus parity-schema-v1 pair is still readable through a
legacy adapter for one documented deprecation interval. The pair is accepted
atomically: mixed `v2/v2` and `v3/v1` combinations are rejected. Every parity
`summary.json` records `manifest_schema_version`,
`parity_manifest_schema_version`, and `used_legacy_manifest_adapter`, so an
audited bundle states which contract produced it.

Regenerate and inspect the migration candidate without writing either active
manifest (`--dry-run` is required, so the generator cannot activate anything):

```console
python examples/jax/build_manifest_v3_candidate.py \
  --examples examples/jax/manifest.json \
  --parity examples/jax/parity_manifest.json \
  --inventory examples/jax/one_to_one_inventory.json \
  --dry-run
```

The [manifest v3 migration
note](../../docs/jax_manifest_v3_migration_candidate.md)
records the approved candidate SHA-256 pair, the byte-identical candidate files
checked in under `docs/`, and the rollback gate: activation must prove an
atomic rollback of both contracts, their activation readers and tests, artifact
observability, and compatibility behavior. That document describes the
pre-activation candidate, and its prose row counts trail the tracked 53-row
catalog; read the counts from the manifests, not from the migration note.

Compatibility tutorials carry their own removal gate. Each declares a successor
mirror ID, a warning naming both itself and that successor, and
`removal_after: "one documented deprecation interval"`, which the validator
requires verbatim.

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
without an executable script, CPU device, correctness owner, or public JAX
import.

## Serial nested-LS coil optimization

The [JAX example](2_Intermediate/boozerQA_ls.py) and its
[native CPU reference](native_reference/boozerQA_ls.py) optimize moving coils
around one NCSX Boozer least-squares surface without MPI. The JAX example uses the
reduced-Schur Newton inner solve and its implicit coil gradient. The native
example uses SIMSOPT's BFGS-then-Newton inner policy. Both use host SciPy
L-BFGS-B for the outer optimization; native setup and composition of objective
terms also have explicit host boundaries. These are different inner solver
policies, so matching physics or gradients does not establish identical optimizer
trajectories.

From the repository root with the native extension and JAX dependencies installed:

```bash
# Small 7x7 example, one outer iteration.
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
  python examples/jax/native_reference/boozerQA_ls.py --smoke --json
SIMSOPT_BACKEND_MODE=jax_gpu_fast SIMSOPT_PRECISION=fp64 \
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONPATH=src \
  python examples/jax/2_Intermediate/boozerQA_ls.py --smoke --json

# The full example uses mpol=ntor=6 and 48x48 quadrature.
SIMSOPT_BACKEND_MODE=jax_gpu_fast SIMSOPT_PRECISION=fp64 \
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONPATH=src \
  python examples/jax/2_Intermediate/boozerQA_ls.py --max-steps 3 --json
```

Read the reported inner-solve result, accepted coil movement, and outer optimizer
status separately. `status=ok` means the bounded workflow completed an accepted,
feasible step that lowered the objective; `optimizer_success` separately reports
convergence. A result after an iteration budget is not a converged coil optimum.
No end-to-end speedup is claimed for this new example;
the historical inner-solve measurements do not measure this complete workflow.
The serial JAX program is registered as a combined tutorial, with direct example
and gradient tests listed in the manifest. It does not count as a one-to-one MPI
port or as authority-harness parity evidence. The original MPI example remains a
separate, MPI-dependent workflow. The native reference uses native field and
BFGS/Newton solvers, but its standard SIMSOPT coil objectives require CPU JAX;
it therefore lives alongside the JAX examples and requires the JAX extra.
