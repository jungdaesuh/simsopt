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

## Native/JAX end-to-end parity harness

These receipts belong to the parity implementation plan. The first aggregate
run was intentionally fail-closed as exploratory because its parity sources
were still uncommitted in the shared dirty checkout at commit
`b6775bf23030bafbd602ce3131dea948e7b8bb4b`. The final subsection records the
subsequent authoritative rerun from a clean isolated revision.

### Case RED -> GREEN receipts

Each case test was introduced before its registry implementation and run with
the project development Python:

```console
MPI4PY_RC_INITIALIZE=false python -m pytest -q \
  tests/integration/test_jax_example_parity_runner.py \
  -k '<case-owned test name>'
```

The case-owned tests and authentic initial RED diagnostics were:

| Case | Test selector | Initial RED | GREEN contract |
|---|---|---|---|
| traceable least squares | `traceable_least_squares_case_runs_native_and_jax_cpu_end_to_end` | missing matched case/runner behavior | residual, Jacobian, objective/gradient convention, solve outcome |
| curve length | `curve_length_case_runs_native_and_jax_cpu_end_to_end` | missing matched case/runner behavior | scalar value/gradient and bounded final solve |
| surface geometry | `surface_geometry_case_runs_native_and_jax_cpu_end_to_end` | missing matched case/runner behavior | area/volume residual and Jacobian plus final state |
| coil flux | `coil_flux_case_runs_native_and_jax_cpu_end_to_end` | `unknown parity case: coil-flux-optimization` | flux, gradient, length, and accepted current |
| permanent magnet | `permanent_magnet_case_matches_cpp_and_jax_cpu_end_to_end` | `unknown parity case: permanent-magnet-optimization` | native `simsoptpp`/JAX moments, residual, objective, selection |
| wireframe RCLS | `wireframe_rcls_case_matches_native_and_jax_cpu_end_to_end` | `unknown parity case: wireframe-optimization` | constrained RCLS currents, residual, objective, feasibility |
| coil force/frame | `coil_force_fixed_state_matches_native_and_jax_cpu_end_to_end` | `unknown parity case: coil-force-and-finite-build` | force value/gradient, Frenet frame, orthonormality, torsion |
| QFM surface | `qfm_case_matches_native_and_jax_cpu_original_residuals` | `unknown parity case: qfm-surface-optimization` | original QFM residual/gradient, constraint, penalty, accepted surface |

After the individual GREEN slices and refactors, the complete focused contract
suite passed:

```text
65 passed in 86.49s
```

The CI/ignore slice separately produced an authentic two-test RED (missing the
narrow `.artifacts/jax-example-parity/` ignore and parity CLI invocations),
then passed `2 passed, 18 deselected`. The parent-memory slice failed because
`parent_peak_rss_bytes` was `None`, then passed after the parent sampled each
child's Linux `VmHWM`. The complete runner/arbiter suite subsequently passed
`28 passed in 73.69s`.

### Real strict-CUDA GREEN receipt

Exact aggregate command:

```console
/home/jungdaesuh/simsopt_mixed_artifacts/v0c_62a262b09c_20260715T2142Z/runtime-env/bin/python \
  examples/jax/run_parity.py \
  --case all-applicable \
  --lanes native-cpu,jax-cpu,jax-gpu \
  --smoke \
  --artifact-root .artifacts/jax-example-parity
```

Published run: `.artifacts/jax-example-parity/20260726T213545Z-549ea1a9`.
Independent audit result:

```json
{"authoritative":false,"case_count":8,"comparison_count":228,"lane_receipt_count":24,"run_id":"20260726T213545Z-549ea1a9","verdict":"pass"}
```

All eight GPU receipts report `jax_gpu_parity`, FP64, both effective transfer
guards at `disallow`, and `NVIDIA GeForce RTX 5090`. All 228 direct pairwise
comparisons passed; two cases are `full, bounded` and six are
`reduced, bounded`. Host peaks are parent-observed with method
`parent-sampled /proc child VmHWM`; device peaks use JAX's validated
`peak_bytes_in_use` counter. The repository did not change during the run.

Hash bindings:

```text
summary.json sha256 435a83f752029a5bc6f9f382d47b174d3655aa8796281e1314a7f43d2a751010
examples/jax/run_parity.py sha256 fcc34355704e567aa67cc99e904362190f6fd52246db61fa3a5c1a1888170d68
examples/jax/parity_manifest.json sha256 0d26a114a61991f4492ec536988831c59f5d343a4db9a7712725c76cb795fd25
```

The generated human-readable table is
[`jax_native_example_parity_results.md`](jax_native_example_parity_results.md).

### Clean authoritative rerun

The predecessor baseline was committed as `b6775bf23030bafbd602ce3131dea948e7b8bb4b`.
Only the reviewed parity paths were then materialized through a temporary Git
index as isolated commit `4e21d9d0151af8e5a6fa873c20750d408bc86bde`;
the shared branch, index, and unrelated dirty files were not changed. The exact
checkout was installed editably, including a freshly built `simsoptpp` binary,
and the resolved package reported version `1.10.7.dev545+g4e21d9d01`.

The first clean full-run RED exposed a relative-artifact-root defect used by
the planned CI command. The children changed working directories and could not
open `.artifacts/.../input_bundle.json`. The public CLI regression
`test_run_parity_cli_resolves_relative_artifact_root` reproduced that failure,
then passed after the parent resolved the artifact root before constructing
child argv.

Final exact command, without the smoke flag:

```console
SIMSOPT_PARITY_SIMSOPTPP_BUILD_COMMIT=4e21d9d0151af8e5a6fa873c20750d408bc86bde \
  /home/jungdaesuh/simsopt_mixed_artifacts/v0c_62a262b09c_20260715T2142Z/runtime-env/bin/python \
  examples/jax/run_parity.py \
  --case all-applicable \
  --lanes native-cpu,jax-cpu,jax-gpu \
  --artifact-root .artifacts/jax-example-parity
```

Published run:
`.artifacts/jax-example-parity/20260726T215531Z-c735bab0`.
Independent fail-closed audit with `--require-authoritative` returned:

```json
{"authoritative":true,"case_count":8,"comparison_count":228,"lane_receipt_count":24,"run_id":"20260726T215531Z-c735bab0","verdict":"pass"}
```

All 24 lane receipts are authoritative. The checkout was clean and unchanged
during the run; all eight CUDA lanes used the RTX 5090 in FP64 with both
transfer guards at `disallow`; all 228 comparisons passed. The generated
setuptools-SCM version source records compatible commit `g4e21d9d01`. Every
loaded `simsoptpp` receipt records compatible build commit
`4e21d9d0151af8e5a6fa873c20750d408bc86bde` and binary SHA-256
`73fafa71cf28c6c0212ee8037676bdcedb09baa462747317d148bd64226b2e6a`.
The authoritative aggregate SHA-256 is
`9e73f85626be108fe14b33ecb430f963aeb5e9812642f004d9bf21cc23bb6994`.

### Hardened audit/schema RED -> GREEN and final authority

The final review found four receipt-level gaps not exposed by the first clean
numeric run: no serialized observable applicability map, no explicit combined
compile/run versus steady-state memory scope, no arbiter gate on scientific
success, and an independent auditor that validated stored pass flags without
recomputing the comparisons. Authentic focused REDs reported:

```text
3 failed in 0.34s
AttributeError: 'LaneObservation' object has no attribute 'applicability'
AttributeError: 'LaneProvenance' object has no attribute 'memory_measurement_scope'
Failed: DID NOT RAISE ArbitrationError (scientific failure)

1 failed in 3.42s
Failed: DID NOT RAISE ValueError (tampered value with self-consistent sidecar hash)
```

Separate adversarial REDs proved that a float32 required observable, absent
effective JAX transfer-guard receipt, and case-local terminal thresholds were
not yet rejected or centrally owned. They failed respectively with `DID NOT
RAISE ArbitrationError`, missing `jax_effective_transfer_guards`, and missing
`terminal_relative_reduction`.

GREEN added schema-validated applicability, explicit N/A optimizer counters,
scientific-success and normalized-status gates, per-observable FP64 checks,
runtime-effective JAX guard fields, synchronized combined memory scope, central
terminal gates, and full comparison recomputation by the independent auditor.
The real-case construction receipt regression also proves parameter, weight,
dtype, seed, stopping-option, quadrature, and target/constraint mutations
change the effective fingerprint. The complete focused suite then passed:

```text
120 passed in 207.50s
```

Only reviewed parity paths were materialized through an alternate Git index as
detached source snapshot `799c656e186642bdb7e296e46ad1c6cd61277839`.
The shared branch/index and unrelated `docs/jax_upstream_final_upgrades_implementation_plan.md`
and `.Codex/` work were not changed. A fresh editable build reported
`simsopt 1.10.7.dev542+g799c656e1`; the clean snapshot independently repeated
the focused gate with `120 passed in 210.50s`.

Final exact non-smoke command:

```console
SIMSOPT_PARITY_SIMSOPTPP_BUILD_COMMIT=799c656e186642bdb7e296e46ad1c6cd61277839 \
  /home/jungdaesuh/simsopt_mixed_artifacts/v0c_62a262b09c_20260715T2142Z/runtime-env/bin/python \
  examples/jax/run_parity.py \
  --case all-applicable \
  --lanes native-cpu,jax-cpu,jax-gpu \
  --artifact-root .artifacts/jax-example-parity
```

Published run:
`.artifacts/jax-example-parity/20260726T225943Z-09dfdc3e`.
The strengthened auditor reloaded all 24 lane receipts and NPY sidecars,
recomputed all routes through the live central tolerance owner, and returned:

```json
{"authoritative":true,"case_count":8,"comparison_count":228,"lane_receipt_count":24,"run_id":"20260726T225943Z-09dfdc3e","verdict":"pass"}
```

All lane receipts report scientific success. The three fixed-state lane
receipts mark `optimizer_outcome=false` and publish null optimizer counters.
All eight CUDA receipts identify `NVIDIA GeForce RTX 5090`, FP64, and effective
JAX host-to-device, device-to-host, and device-to-device guards at `disallow`.
Memory receipts are explicitly combined import/compile/warmup/bounded-execution
peaks and set `steady_state_memory_measured=false`; they make no speed or
steady-state-memory claim. The `simsoptpp` SHA-256 remains
`73fafa71cf28c6c0212ee8037676bdcedb09baa462747317d148bd64226b2e6a`.
The final aggregate SHA-256 is
`1f21ea8851bedfe5f7830ee340e0a1718c4d87d17492dfd0db279b97433edf0b`.

## Fast-default and explicit-parity runtime RED -> GREEN

The backend/API/environment/workflow contract tests were committed alone as
immutable pre-GREEN revision `dd37ea93c`. A detached clean worktree of that
revision ran:

```console
PYTHONPATH=/tmp/simsopt-fast-red-dd37ea93c/src:/tmp/simsopt-fast-red-dd37ea93c \
  MPI4PY_RC_INITIALIZE=false \
  /home/jungdaesuh/code/columbia/.venv-simsopt-linux-x86/bin/python \
  -m pytest -q \
  tests/test_backend_precision_policy.py \
  tests/test_jax_import_smoke.py::test_legacy_jax_environment_defaults_to_fast \
  tests/test_jax_examples_manifest.py::test_jax_workflow_reaches_examples_from_both_events_and_existing_jobs \
  tests/test_jax_example_parity_manifest.py::test_parity_workflows_reach_cpu_and_strict_gpu_without_case_duplication
```

The authentic RED was `19 failed, 27 passed in 4.27s`. It exposed the old
legacy JAX CPU/GPU parity defaults, absent typed `device`/`intent` API and
`use_runtime` export, and CI's retained deprecated-lane commands. Failures
were behavioral assertions and unsupported public keywords, not collection or
environment failures.

GREEN uses one typed profile resolver for `(device, intent)`, keeps fully
unset selection at `native_cpu`, defaults explicitly selected JAX to fast,
retains every canonical mode, scrubs inherited child selectors, and validates
the child's exact mode/device/FP64 result. Both ordinary GPU profiles reject a
CPU fallback result. Legacy lane aliases remain parity selectors and warn.
`run_parity.py` remains the only certification publisher.

The exact RED command above, rerun against the GREEN worktree, passed
`46 passed in 3.57s`.

Focused GREEN evidence:

```text
tests/test_backend_precision_policy.py: 42 passed in 2.40s
tests/test_jax_import_smoke.py: 96 passed, 8 skipped in 150.05s
tests/integration/test_jax_examples.py: 38 passed in 131.71s
manifest/runtime/workflow contract: 81 passed in 5.06s
Ruff check and format: passed for all changed runtime/runner/test modules
```

All ten ready examples passed the ordinary runner on CPU in both profiles:

```console
python examples/jax/run_examples.py --device cpu
python examples/jax/run_examples.py --device cpu --intent parity
```

All ten also passed both profiles on the real NVIDIA GeForce RTX 5090. The
current checkout was forced ahead of environment site packages with explicit
source paths; CUDA JAX came from the pinned 0.10.0 GPU environment and
`simsoptpp` from the Linux project environment:

```console
CHECKOUT=/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed
GPU_ENV=/home/jungdaesuh/simsopt_mixed_artifacts/b5a0_external_toolchain_20260717_r1/pixi-default
PROJECT_ENV=/home/jungdaesuh/code/columbia/.venv-simsopt-linux-x86
PYTHONPATH="$CHECKOUT/src:$CHECKOUT:$GPU_ENV/lib/python3.11/site-packages:$PROJECT_ENV/lib/python3.11/site-packages" \
  MPI4PY_RC_INITIALIZE=false "$GPU_ENV/bin/python" -S \
  examples/jax/run_examples.py --device gpu
PYTHONPATH="$CHECKOUT/src:$CHECKOUT:$GPU_ENV/lib/python3.11/site-packages:$PROJECT_ENV/lib/python3.11/site-packages" \
  MPI4PY_RC_INITIALIZE=false "$GPU_ENV/bin/python" -S \
  examples/jax/run_examples.py --device gpu --intent parity
```

These runs prove profile placement, fallback rejection, and the existing
per-example scientific gates. They are not certification or performance
evidence. The manifest-v2 sign-off, matched fast/parity benchmark, Phase 11
scientific/artifact repairs, and new clean authority run remain separate open
gates.

## QFM feasibility, status, and counter repair RED -> GREEN

The QFM authority regressions were committed without implementation changes as
immutable revision `9d43398df`. Against the old behavior, the focused RED was:

```text
2 failed in 0.21s
retained final constraint_value=1.2315874794985946e-8 was accepted
_normalized_driver_status was absent
```

The public end-to-end regression also exposed the same defects through the
serialized lane contract: the JAX receipt reported `raw_status=solver_failed`
as normalized `converged`, and emitted null `nfev`/`njev`.

Two solver designs were measured before implementation. Switching to the exact
SciPy-SLSQP/custom augmented-Lagrangian pair satisfied feasibility, but reached
materially different surface coefficients and gradients. Retaining the matched
penalty formulation with weight `12` instead preserved the common minimizer,
while a shared `5e-8` gradient stop and zero relative-objective stop gave both
drivers a genuine termination margin. This keeps the native and JAX algorithms
comparable without weakening a central comparison tolerance.

GREEN now applies the centrally owned `1e-10` constraint and `1e-7`
stationarity gates, keeps driver termination in `normalized_status` and
scientific acceptance in `success`, and publishes finite host-materialized
`nit`, `nfev`, and `njev`. The QFM adapter's result dictionary additively
exposes `status`, `nfev`, and `njev`; existing keys and behavior are unchanged,
so callers can ignore the new diagnostics and rollback is a one-commit revert.

Focused CPU evidence:

```text
3 passed in 30.48s
```

A dirty-tree diagnostic run exercised native CPU, JAX CPU, and strict JAX GPU
on the real NVIDIA GeForce RTX 5090. All three lanes reported scientific
success and normalized convergence, all 42 declared comparisons passed, and
the GPU receipt recorded FP64 with all effective transfer guards at `disallow`:

```text
/tmp/simsopt-qfm-phase11-gpu/20260727T051313Z-a8a0cced
native-cpu: nit=135, nfev=153, njev=153
jax-cpu:    nit=95,  nfev=140, njev=140
jax-gpu:    nit=95,  nfev=141, njev=141
```

This run is explicitly non-authoritative because the repair was uncommitted;
it proves the repaired CPU/GPU behavior but does not replace the final clean
authority run required after all Phase 11 safety repairs.
