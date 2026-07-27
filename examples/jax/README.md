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
guard through both the SIMSOPT policy and JAX's process-wide
`JAX_TRANSFER_GUARD=disallow`, and fails on a skip, unsupported result, CPU
fallback, wrong precision, or malformed output. Lane membership and bounded
smoke arguments live only in [`manifest.json`](manifest.json).

`gpu-strict` means the process-wide JAX guard remains `disallow` across example
setup, SIMSOPT orchestration, result publication, and all device-to-host
transfers. The custom operator-GMRES implementation retains one documented,
host-to-device-only allowance scoped inside JAX's `gmres` call because that
upstream routine lowers internal scalar literals through host-to-device
conversion. It does not permit device-to-host materialization, and the
surrounding numerical path remains guarded.

Every ready example runs in both lanes through the same JAX implementation.
Device selection changes placement, not the algorithm or public result
contract. Native SciPy solves may be used only by correctness tests as CPU
reference oracles; a ready example must not use `scipy.optimize` as its JAX
lane implementation. Optimistix and Optax remain explicit optional driver
choices and are never selected implicitly by the serial example APIs.
The serial wrappers publish `problem.x` and their bounded log only after a
successful solve. A failed result raises and leaves the caller-owned state
unchanged. The deprecated least-squares `optimizer="lm"` alias still selects
explicit Optimistix LM; legacy `gauss_newton` and scalar `bfgs` spellings are
rejected because the typed API does not expose behavior-equivalent Optimistix
drivers for those algorithms.

## Native/JAX parity evidence

The paired parity runner reconstructs matched native SIMSOPT CPU and JAX
workflows from one serialized input bundle. Run every currently applicable
bounded case on CPU with:

```console
python examples/jax/run_parity.py \
  --case all-applicable \
  --lanes native-cpu,jax-cpu \
  --smoke \
  --artifact-root .artifacts/jax-example-parity
```

In a CUDA environment, add `jax-gpu` to `--lanes`. The runner fixes FP64,
strict CUDA selection, and both transfer guards before importing JAX; a missing
GPU, CPU fallback, wrong precision, implicit transfer, or forbidden host solver
fails the run. Each lane executes in a fresh process. The final directory
contains canonical input JSON/NPY files, one hash-bound lane receipt per case,
and an aggregate `summary.json`. Failed or interrupted runs retain a diagnostic
`.partial` directory and are never published as passing evidence.

Memory receipts use `XLA_PYTHON_CLIENT_PREALLOCATE=false`, synchronize the JAX
publication boundary, and report one combined import/compile/warmup/bounded-run
peak. They do not claim a separate steady-state peak or support a speed claim.
This diagnostic policy is distinct from the supported production default,
which leaves JAX GPU preallocation enabled unless the user explicitly changes
it for their workload.

[`parity_manifest.json`](parity_manifest.json) classifies every ready
`inspired_by` relationship independently:

- `full` covers every declared scientific stage, while `reduced` explicitly
  omits at least one scientific stage and `unsupported` names a concrete
  blocker.
- `bounded` and `native_default` describe scale independently of workflow
  coverage. A bounded full case is not native-default evidence.
- `native_python_scipy` means normal native Python/SciPy orchestration;
  `native_simsoptpp` additionally binds the loaded extension path and binary
  hash. Analytic and external references remain distinct oracle kinds.

Only a clean committed run whose lane receipts are all marked authoritative
may promote a parity classification. Dirty-checkout runs remain useful
exploratory evidence and record the tracked diff hash plus untracked-file
inventory. Audit a published run independently with:

```console
python -m examples.jax.parity.audit \
  --run .artifacts/jax-example-parity/<run-id> \
  --repo-root "$PWD"
```

The generated results table lives in
[`docs/jax_native_example_parity_results.md`](../../docs/jax_native_example_parity_results.md).

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
   example module;
8. declare both `cpu-smoke` and `gpu-strict` while it is `ready`, and use the
   same public JAX solver and algorithm in both lanes.

Add the behavioral correctness test first and preserve its authentic
RED → GREEN → REFACTOR commands in
[`docs/jax_examples_tdd_receipts.md`](../../docs/jax_examples_tdd_receipts.md).
Then mark the manifest record `ready`; the validator rejects a ready record
without an executable script, CPU lane, correctness owner, or public JAX
import.
