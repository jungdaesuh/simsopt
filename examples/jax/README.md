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
their exact JAX mirrors or blockers, and device scope. Regenerate it from the
validated manifest with
`python -m examples.jax.native_to_jax_index --write`; use `--check` in
validation.

## Known limitations

- In-tree tests compare each mirror with native SIMSOPT at small scale. Endpoint
  contracts at `native_default` scale, including the PM4Stell backtracking
  deviation and the planar-coil upstream-gradient limitation, belong to the
  native/JAX parity harness, which is maintained outside this repository.

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
defaults to `bounded`, but state it explicitly in CI commands so the emitted
child argv and the scale each child reports are attributable. Only `bounded` maps to the child `--smoke` flag; `native_default`
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
Parity additionally selects the stable numerical policy. Fast and parity
example runs are both diagnostic; neither is certification evidence.

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

These are the providers the upstream scripts call, with upstream's own options.

The `outer_optimizer_policy` manifest field declares the policy owned by that
example. The declaration is bound to the registered example and its exact
driver. A driver name alone grants no exception, and another
example cannot borrow this policy. Missing declarations retain the default ban.
This permission does not relax transfer guards, source provenance, scientific
acceptance, or parity comparisons. CPU control is not a GPU-resident optimizer;
performance claims must state whether they time the JAX region or the complete
workflow.

The serial wrappers publish `problem.x` and their bounded log only after a
successful solve; a failed result raises and leaves caller-owned state
unchanged.

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

## Native/JAX same-state parity

[`tests/jax/test_mirror_same_state_parity.py`](../../tests/jax/test_mirror_same_state_parity.py)
builds each covered native SIMSOPT problem and its JAX mirror at one shared
starting state, at a small size, and compares the objective and, where it is
defined, the gradient there: native against JAX CPU, and against JAX GPU when
CUDA is present. It stores no reference data; the tolerances are the
same-state values of `simsopt_jax.parity_tolerances`.

```console
python -m pytest tests/jax/test_mirror_same_state_parity.py
```

## The one-to-one identity contract

The authoritative inventory is [`manifest.json`](manifest.json). Nothing else
in this directory defines coverage.

`manifest.json` holds the official coverage:

- `source_catalog` — 53 official upstream source rows: 25 `eligible`, 1 `hybrid`,
  25 `blocked`, and 2 `not_applicable`. Membership is checked against the pinned
  official catalog, not whatever Python files happen to be in the local tiers.
- `jax_examples` — 26 executable records, 25 `ready` and 1 `planned` (the
  hybrid VMEC single-stage mirror). Each owns one official source.

An owned record must sit at the identical tier and filename as its source, must
be typed `one_to_one`, and cannot be a tutorial. Each mirror is owned by at
most one source. Execution scale and verified evidence are separate from source
coverage.

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

Its manifest status is `planned`, so the example runner does not select it.
It is not promoted and holds no parity claim. Promotion is blocked on a
VMEC hybrid authority lane proving immutable VMEC/MPI build identity, the
recorded MPI world size, and matched CPU and GPU slice provenance on an
approved runner.

## Manifest schema

The only accepted example manifest is schema v3; any other version is
rejected. Every runner invocation first prints one JSON line to stderr with
`examples_manifest_schema_version` and `used_legacy_manifest_adapter`. No
legacy manifest reader exists, so `used_legacy_manifest_adapter` is always
`false`.

The active v3 document holds official coverage only; a document with any other
root field is rejected.

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

Add the behavioral correctness test first, then mark the manifest record
`ready`; the validator rejects a ready record
whose script or listed correctness tests do not exist, and the example tests
reject a ready record without both devices or a public JAX import.
