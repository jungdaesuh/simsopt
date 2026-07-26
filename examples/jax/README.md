# JAX-first examples

These examples teach the public JAX interfaces directly. They are inspired by
the native examples but may combine or reorganize them to make immutable state,
compiled computations, batching, and host/device boundaries explicit.

Install the CPU runtime from the repository root with:

```console
python -m pip install -e ".[JAX]"
```

Use `.[JAX_GPU]` in a supported CUDA environment. Runtime selection is
process-wide and must happen before importing JAX-heavy modules, so use the
isolated runner rather than executing several examples in one Python process:

```console
python examples/jax/run_examples.py --lane cpu-smoke
python examples/jax/run_examples.py --lane gpu-strict
```

The CPU lane selects strict FP64 `jax_cpu_parity` and disables mpi4py
auto-initialization because these are serial examples. The GPU lane selects
strict FP64 `jax_gpu_parity`, disables preallocation, enables the transfer
guard, and fails on a skip, unsupported result, CPU fallback, wrong precision,
or malformed output. Lane membership and bounded smoke arguments live only in
[`manifest.json`](manifest.json).

## Choosing an example

- `pure` examples consume immutable JAX arrays or frozen PyTrees and have no
  host boundary.
- `adapter` examples begin with native SIMSOPT objects, snapshot their state
  through `simsopt_jax_adapters`, run the numerical region in JAX, and publish
  accepted state explicitly.
- `hybrid` examples retain a named native or external computation in the
  demonstrated workflow.

The manifest records each example's kind, public JAX surfaces, remaining host
boundaries, source inspiration, dependencies, correctness owners, and lanes.
It also catalogs every native Python example as either a candidate or a
deliberately deferred concept. Deferred entries are not placeholders: each
states the missing public boundary or external limitation required for
reconsideration.

## Author contract

Every runnable script must:

1. import a public `simsopt_jax` or `simsopt_jax_adapters` surface directly;
2. support `--smoke --json` and keep smoke work deterministic and bounded;
3. emit a final JSON line containing `example_id`, `backend_mode`, `platform`,
   `precision`, `status`, and an `observables` object;
4. return nonzero unless its independent scientific checks pass;
5. name every host boundary in both its opening comment and the manifest;
6. accept `--output-dir` or use a temporary directory if it writes artifacts;
7. reuse canonical repository inputs read-only and never forward to a native
   example module.

Add the behavioral correctness test first and preserve its authentic
RED → GREEN → REFACTOR commands in
[`docs/jax_examples_tdd_receipts.md`](../../docs/jax_examples_tdd_receipts.md).
Then mark the manifest record `ready`; the validator rejects a ready record
without an executable script, CPU lane, correctness owner, or public JAX
import.
