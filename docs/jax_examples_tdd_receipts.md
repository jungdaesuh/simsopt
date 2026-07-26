# JAX Examples TDD Receipts

This log preserves the failing-first and passing commands required by
`docs/jax_examples_implementation_plan.md`. Commands run from the repository
root. The CPU environment is `../.venv-simsopt-linux-x86/bin/python` (Python
3.11, JAX 0.10.0 CPU). The system Python failure caused by its missing declared
`monty` dependency was an environment preflight failure and is not counted as
a RED receipt.

## Backend-neutral CPU/GPU serial solvers (2026-07-26)

### RED

The following focused contract command was run before replacing the implicit
optional backend and before assigning every ready example to both lanes:

```console
MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
  tests/solve/test_serial_jax.py::test_serial_jax_has_no_implicit_optional_optimizer_backend \
  tests/solve/test_serial_jax.py::test_serial_jax_numerical_solves_do_not_use_host_callbacks \
  tests/integration/test_jax_examples.py::test_every_ready_repository_example_runs_on_cpu_and_gpu
```

Observed result: 3 failed. The serial wrapper imported Optimistix directly,
used `jax.debug.callback`, and the traceable least-squares example declared
only `cpu-smoke`. A second failing-first check found
`scipy.optimize` in the curve-length GPU implementation.

### GREEN

After routing least-squares through `Driver.SIMSOPT_LM_GMRES`, scalar
minimization through `Driver.SIMSOPT_BFGS`, converting the curve and surface
examples to those public APIs, and making optional solvers explicit:

```console
MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
  tests/solve/test_serial_jax.py \
  tests/integration/test_jax_examples.py::test_gpu_examples_do_not_import_host_scipy_optimizers \
  tests/integration/test_jax_examples.py::test_every_ready_repository_example_runs_on_cpu_and_gpu
```

Observed result: `9 passed in 4.37s`.

The real CPU and CUDA FP64 executions of `traceable_least_squares.py` returned
identical parameters, objective, residual norm, gradient norm, driver, status,
iteration count, function-evaluation count, and Jacobian-evaluation count:

```text
driver=simsopt_lm_gmres status=1 success=true
iterations=4 function_evaluations=5 jacobian_evaluations=5
solution=[0.9999999999999845, 1.999999999999998, 2.9999999999999996]
objective=2.5016751456821337e-28
residual_norm=1.5816684689536343e-14
gradient_inf_norm=1.554312234475219e-14
```

The first RTX 5090 run was not accepted as strict-transfer evidence: it set the
SIMSOPT policy variable but did not set JAX's own process-wide transfer guard.
Adding `JAX_TRANSFER_GUARD=disallow` was the next RED step. It exposed implicit
transfers in example setup, QFM and Boozer closure constants, post-solve Newton
assembly, tracing state, and native derivative publication. Those transfers
were removed or made explicit at named host boundaries. This is not evidence
that no scoped guard override exists anywhere below the examples: the custom
operator-GMRES path retains a pre-existing host-to-device-only allowance around
JAX's `gmres`, whose internal scalar literals otherwise trip the guard. The
allowance does not cover device-to-host transfers or the surrounding SIMSOPT
numerical path.

The final CPU and RTX 5090 validations each ran all ten current-checkout
scripts with JAX's real transfer guard set to `disallow`. The CUDA command
template below was instantiated once for each manifest-ready path, replacing
`<ready-example>` with that exact path and retaining `--smoke --json`:

```console
GPU_ENV=/home/jungdaesuh/code/columbia/simopt-jax-clean-local/.pixi/envs/default
PYTHONPATH="$PWD/src:$GPU_ENV/lib/python3.11/site-packages" \
MPI4PY_RC_INITIALIZE=false \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_JAX_TRANSFER_GUARD=disallow SIMSOPT_PRECISION=fp64 \
JAX_TRANSFER_GUARD=disallow JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
"$GPU_ENV/bin/python" -S <ready-example> --smoke --json
```

Every CPU child reported `platform="cpu"`, and every CUDA child reported
`platform="gpu"`; all twenty results reported `precision="fp64"` and
`status="ok"`. The least-squares example matched CPU and GPU exactly for its
published parameters, objective, residual, gradient, driver, status, and
evaluation counts. Other examples matched their independent scientific
oracles and CPU/GPU tolerances but were not bitwise identical; for example,
the curve directional-gradient errors were `1.70e-10` on CPU and `1.61e-9`
on GPU, both below the `1e-6` contract.

The GPU environment's normal site initialization contains a stale editable
mapping to `/home/jungdaesuh/code/columbia/simopt-jax-clean-local`. Therefore
the current-source GPU receipt used `python -S` with this checkout's `src/`
first on `PYTHONPATH`. The normal runner's failure is retained as an
environment-provenance failure rather than misreported as a product failure.

The reviewed implementation started from source HEAD
`6547da3a4350dcbc67ce7188886c2d353e6ddbbc`. Two unrelated parity-plan
documentation commits advanced the live branch while validation was running;
the final scoped index was therefore based on
`2b352c436c9d1a62cbff88f1f1d88ded9384719c`. Excluding this receipt itself,
the staged implementation diff SHA256 was
`7397f8799cf53db630522dc5d580ac7ff877d5b352b33f30a391e237610e6bec`.
The CPU and GPU JSONL result SHA256 values were respectively
`82843570e6efe6ed72afe3ecaeb9cfbf54e6b1db601ecd31c09e138f2e225152`
and `700d863f0a30fcc462c12cd18e9d56e38d885979d3f23cf12814c0d70bb9bc3c`.
The hash-bound payloads are preserved in
[`jax_examples_cpu_strict_results.jsonl`](jax_examples_cpu_strict_results.jsonl)
and
[`jax_examples_gpu_strict_results.jsonl`](jax_examples_gpu_strict_results.jsonl).

### REFACTOR

Host logging is bounded to initial and final records after each numerical
solve. `constrained_serial_solve_jax()` now fails explicitly until a
SIMSOPT-owned backend-neutral constrained solver exists. The manifest is the
single owner of the two-lane assignment, and the example author contract now
forbids a hidden SciPy optimizer in ready JAX examples.

The broader `tests/jax/solve` refactor gate then exposed one strict-transfer
regression in the shared host-boundary owner: an explicit `jax.device_get` was
wrapped in a temporary device-to-host `allow` context. The existing contract
test was RED because strict mode must not be weakened around an explicit
materialization. Removing that redundant override made the focused test GREEN;
the full solver suite passed with `65 passed, 2 skipped`.

The first clean-room audit caught two fail-open behaviors. The QFM example had
accepted a five-iteration unsuccessful result, and the serial wrappers had
published unsuccessful result state. The repaired QFM smoke contract requires
solver success and a final gradient norm at most `1e-8`; its CPU and GPU runs
both converged in 54 iterations with gradient norms near `1.36e-9`. Both serial
wrappers now raise before logging or changing `prob.x` when a typed result is
unsuccessful. Focused tests also reject legacy optimizer spellings when the
typed API has no behavior-equivalent driver rather than silently changing the
algorithm.

After those review fixes, the final serial/manifest/example gate passed with
`57 passed in 141.19s`. The public solver suite passed with
`71 passed, 2 skipped`; the remaining import-boundary tests passed with
`8 passed, 1 deselected`, excluding an unrelated untracked parity artifact's
benchmark import.
Ruff `F` and format checks, compileall, and `git diff --check` were rerun after
the final source changes. The fresh standalone `uvx mypy` attempt was not accepted
as a gate because its isolated environment lacked the project's NumPy/JAX
dependencies; the earlier dependency-complete focused mypy receipt remains the
type-check evidence rather than being misreported as a fresh rerun.

## Manifest schema and source catalog

### RED

Command:

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/test_jax_examples_manifest.py
```

Expected and observed failure: pytest exited 2 during collection because the
public manifest boundary did not exist:

```text
ModuleNotFoundError: No module named 'examples.jax'
```

### GREEN

The identical command exited 0 after adding the immutable typed parser and
complete catalog:

```text
.................                                                        [100%]
17 passed in 0.89s
```

### REFACTOR

Commands:

```console
uvx ruff check examples/jax/_manifest.py tests/test_jax_examples_manifest.py
uvx ruff format --check examples/jax/_manifest.py tests/test_jax_examples_manifest.py
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/test_jax_examples_manifest.py
```

Observed result:

```text
All checks passed!
2 files already formatted
.................                                                        [100%]
17 passed in 0.92s
```

## Isolated runner and structured result contract

### RED

Command:

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py
```

Expected and observed failure: pytest exited 2 during collection because the
runner interface did not exist:

```text
ModuleNotFoundError: No module named 'examples.jax.run_examples'
```

### GREEN

The identical command exited 0 after adding deterministic selection, exact
bounded argv, lane-owned pre-import environments, real subprocess execution,
structured-result validation, and fail-closed diagnostics:

```text
..........                                                               [100%]
10 passed in 0.66s
```

### REFACTOR

Commands:

```console
uvx ruff check examples/jax/run_examples.py tests/integration/test_jax_examples.py
uvx ruff format --check examples/jax/run_examples.py tests/integration/test_jax_examples.py
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
```

Observed result:

```text
All checks passed!
2 files already formatted
...........................                                              [100%]
27 passed in 1.26s
```

## Traceable least-squares vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k traceable_least_squares_example
```

The behavioral test reached the manifest entry and failed because the example
was not executable yet: `AssertionError: assert 'planned' == 'ready'` (1
failed, 10 deselected).

### GREEN

The identical command passed after adding the public traceable solver example:

```text
1 passed, 10 deselected in 0.98s
```

### REFACTOR

The example uses a temporary output directory in smoke mode so the public
solver's progress log cannot dirty the checkout. The lane runner reported:

```text
PASS traceable-least-squares
```

## Curve-length optimization vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k curve_length_example
```

The behavioral test failed at the absent delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 11 deselected).

### GREEN

The identical command passed after adding the adapter-owned optimization,
analytic circle-length oracle, and independent directional finite difference:

```text
1 passed, 11 deselected in 1.36s
```

## Surface-geometry optimization vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k surface_geometry_example
```

The behavioral test failed at the absent delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 12 deselected).

### GREEN

The identical command passed after adding the two-metric optimization and
independent axisymmetric-torus area and volume formulas:

```text
1 passed, 12 deselected in 13.37s
```

## Three-example refactor gate

The runner now prepends the checkout's `src/` directory in one environment
owner and hides CUDA devices in the CPU lane. The CUDA-isolation assertion was
first observed RED as `KeyError: 'CUDA_VISIBLE_DEVICES'`, then GREEN with the
identical focused command (1 passed, 13 deselected).

Commands:

```console
uvx ruff check examples/jax/run_examples.py examples/jax/1_Simple/traceable_least_squares.py examples/jax/1_Simple/curve_length_optimization.py examples/jax/1_Simple/surface_geometry_optimization.py tests/integration/test_jax_examples.py
uvx ruff format --check examples/jax/run_examples.py examples/jax/1_Simple/traceable_least_squares.py examples/jax/1_Simple/curve_length_optimization.py examples/jax/1_Simple/surface_geometry_optimization.py tests/integration/test_jax_examples.py
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
../.venv-simsopt-linux-x86/bin/python examples/jax/run_examples.py --lane cpu-smoke
```

Observed result:

```text
All checks passed!
5 files already formatted
30 passed in 17.60s
PASS traceable-least-squares
PASS curve-length-optimization
PASS surface-geometry-optimization
```

## Permanent-magnet optimization vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k permanent_magnet_example
```

The behavioral test failed at the absent delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 13 deselected).

### GREEN

The identical command passed after adding the immutable two-dipole payload,
public greedy solve, exact selected-coordinate oracle, and residual certificate:

```text
1 passed, 13 deselected in 1.15s
```

## Field-line and particle tracing vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k fieldline_example
```

The behavioral test failed at the absent delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 14 deselected).

### GREEN

The example now checks an analytic pure-toroidal Reiman field-line endpoint,
normal termination, zero unexpected events, and full-orbit energy conservation.
The focused example plus runner-environment regression passed:

```text
2 passed, 13 deselected in 4.94s
```

## Coil-flux optimization vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k coil_flux_example
```

The behavioral test failed at the absent delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 15 deselected).

### GREEN

The identical command passed after adding the immutable adapter snapshots,
directional finite-difference gradient oracle, exact one-current line
minimizer, and analytic circular-coil length oracle:

```text
1 passed, 15 deselected in 4.07s
```

## QFM-surface optimization vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k qfm_example
```

The behavioral test failed at the absent delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 16 deselected).

### GREEN

The identical command passed after adding canonical NCSX input reuse, initial
and final penalty/gradient certificates, and explicit final surface
publication:

```text
1 passed, 16 deselected in 12.10s
```

## Serial child import isolation

During refactoring, adapter examples revealed that importing `simsopt.field`
auto-initialized mpi4py and blocked serial children. The runner contract test
was first observed RED as `KeyError: 'MPI4PY_RC_INITIALIZE'`. Setting
`MPI4PY_RC_INITIALIZE=false` in the single lane-environment owner made the
identical test GREEN and reduced the full coil-flux smoke from a bounded
five-minute timeout to 4.07 seconds.

## Wave 1 runner gate

```console
../.venv-simsopt-linux-x86/bin/python examples/jax/run_examples.py --lane cpu-smoke
```

Observed result:

```text
PASS traceable-least-squares
PASS curve-length-optimization
PASS surface-geometry-optimization
PASS coil-flux-optimization
PASS qfm-surface-optimization
PASS permanent-magnet-optimization
PASS fieldline-and-particle-tracing
```

## Strict GPU lane assignment

The initial traceable least-squares GPU selection was rejected fail-closed.
Under the exact strict lane environment, Optimistix's debug callback required a
local CPU device while `JAX_PLATFORMS=cuda` intentionally exposed only CUDA:

```text
RuntimeError: jax.debug.callback failed to find a local CPU device
```

That solver is therefore CPU-only in the manifest rather than weakening the
strict lane. The GPU-required coil-flux example ran in a fresh CUDA process
with transfer guard `disallow` and reported:

```json
{"backend_mode":"jax_gpu_parity","example_id":"coil-flux-optimization","platform":"gpu","precision":"fp64","status":"ok"}
```

The local GPU environment has a stale editable SIMSOPT mapping, so this source
checkout was injected in a `-S` validation process. CI installs the checkout
directly and invokes the normal strict runner command.

## Wave 1 refactor gate

```console
python -m compileall -q examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
uvx ruff check examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
uvx ruff format --check examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
```

Observed result:

```text
All checks passed!
13 files already formatted
35 passed in 50.38s
```

## CI reachability

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/test_jax_examples_manifest.py -k workflow_reaches_examples
```

The static workflow test failed because `examples/jax/**` was absent from the
push filter; the pull-request filter and runner steps were absent as well.

### GREEN

The identical command passed after adding both path filters and commands to the
existing CPU integration and strict-GPU jobs:

```text
1 passed, 17 deselected in 0.10s
```

## Boozer-surface optimization vertical slice

### RED

```console
../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k boozer_example
```

The behavioral test failed at the delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 17 deselected).

### GREEN

The identical command passed after adding a bounded NCSX surface solve,
solver-success and stationarity certificates, residual evaluation through
`BoozerResidualJAX`, and explicit accepted-surface publication:

```text
1 passed, 17 deselected in 32.21s
```

## Wireframe optimization vertical slice

### RED

```console
MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k wireframe_example
```

The behavioral test failed at the delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 18 deselected).

### GREEN

The identical command passed after adding native `ToroidalWireframe`
construction, a public RCLS solve checked against an independently assembled
KKT system, explicit current publication, and a bounded public GSCO multistep
transition:

```text
1 passed, 18 deselected in 3.44s
```

## Force and finite-build vertical slice

### RED

```console
MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/integration/test_jax_examples.py -k force_finite_build_example
```

The behavioral test failed at the delivery contract with
`AssertionError: assert 'planned' == 'ready'` (1 failed, 19 deselected).

### GREEN

The first implementation run exposed the public framed-curve constructor
contract (`ZeroRotationJAX` must be supplied explicitly). After correcting
that contract, the identical command passed with a native force-integral
oracle, directional finite-difference check, orthonormal-frame check, and the
zero-torsion planar limit:

```text
1 passed, 19 deselected in 5.34s
```

## Single-stage Wave 3 readiness gate

The single-stage record remains `planned`. Two bounded live probes—a symmetric
two-coil fixture and the convergent NCSX fixture—successfully constructed
`TraceableObjectiveSession` instances and returned finite outer values, but
their representative derivative evaluations returned the contract's
fail-closed all-NaN gradient. The plan requires a representative derivative
oracle before implementation, so no script was added and no invalid accepted
state was published. This is readiness evidence, not a RED/GREEN receipt.

## Final refactor and public-surface gate

Commands:

```console
uvx ruff check examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
uvx ruff format --check examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
python -m compileall -q examples/jax tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q tests/test_jax_examples_manifest.py tests/integration/test_jax_examples.py
../.venv-simsopt-linux-x86/bin/python examples/jax/run_examples.py --lane cpu-smoke
```

Observed result:

```text
All checks passed!
16 files already formatted
38 passed in 86.98s
PASS traceable-least-squares
PASS curve-length-optimization
PASS surface-geometry-optimization
PASS coil-flux-optimization
PASS qfm-surface-optimization
PASS permanent-magnet-optimization
PASS fieldline-and-particle-tracing
PASS boozer-surface-optimization
PASS wireframe-optimization
PASS coil-force-and-finite-build
```

Focused underlying public-surface regressions then passed: 4 Boozer
solver/objective cases in 27.77 seconds, 4 wireframe RCLS/GSCO cases in 3.72
seconds, and 5 force/Frenet-frame cases in 2.07 seconds.
