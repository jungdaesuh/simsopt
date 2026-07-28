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

The optimizer-provenance regression was committed separately at immutable
revision `6dbda83de`. Its public QFM-case RED failed in `39.21s` because the
receipt claimed `simsopt_jax_lbfgs_qfm_penalty`, while the implementation uses
the full inverse-Hessian BFGS runner. Correcting only the receipt driver to
`simsopt_jax_bfgs_qfm_penalty` made the same test GREEN (`1 passed in 36.79s`).

## Complete comparison routes and independent input audit RED -> GREEN

The route-completeness and retained-input regressions were committed without
production changes as immutable revision `c0f1b3372`. Against its immediate
parent, the manifest/arbiter RED produced three focused failures: the manifest
accepted a partial lane-pair matrix, the traceable least-squares final Jacobian
had no declared routes, and arbitration accepted an applicable observable with
no routes. The publication-audit RED separately failed with `DID NOT RAISE`
after both `input_bundle.json` configuration and an input NPY sidecar were
changed while the retained lane fingerprint strings were left untouched.

GREEN makes a complete three-pair matrix mandatory for every applicable
observable, both while parsing the parity manifest and while arbitrating the
actual lane receipts. Non-unique raw surface parameters remain published as a
diagnostic but are explicitly inapplicable; invariant surface quantities stay
certified. Traceable and surface least-squares receipts now publish the
solver's distinct half-squared cost alongside SIMSOPT's public residual
sum-of-squares convention. The missing final-Jacobian routes and the permanent
magnet initial discrete-selection routes are declared for all three pairs.

The independent auditor now reloads the canonical input bundle and every
sidecar through the existing input-bundle owner. That operation revalidates
metadata and hashes, recomputes the input and configuration fingerprints, and
binds those recomputed values to the aggregate before numerical arbitration.
No case-local tolerance or audit-only comparison policy was introduced.

Focused GREEN evidence:

```text
route, arbiter, case, and tamper regressions: 27 passed in 21.70s
complete manifest plus integration suites: passed (exit 0)
Ruff check and format, JSON validation, compileall, git diff --check: passed
```

## No-replace publication and descriptor-safe artifact I/O RED -> GREEN

The deterministic publication and sidecar race tests were committed without
production changes at immutable revision `fc718aee9`. Replaying that revision
from a detached worktree produced the authentic focused RED:

```text
6 failed, 18 deselected in 0.33s
```

The failures proved all reopened defects directly: an empty destination was
silently replaced; a publish-time symlink produced an unnormalized OS error;
no completion marker existed; a hash-matching external NPY leaf substituted
after validation was loaded successfully; a substituted parent directory
received a write; and a second sidecar write replaced existing bytes.

GREEN centralizes artifact leaves behind descriptor-relative Linux I/O. Every
path component is opened relative to its already-open parent with
`O_NOFOLLOW`; leaves are opened once, reads hash and load through that same
descriptor, and writes use `O_EXCL`, fsync the file, then fsync the parent.
Unsupported platforms or missing descriptor flags fail explicitly rather than
falling back to pathname reopening. Input bundles, lane receipts, aggregate
summaries, the independent auditor, and the report reader use the same byte
primitive.

Directory publication now uses Linux `renameat2(RENAME_NOREPLACE)`, so neither
an empty directory nor a symlink can be replaced. The complete partial tree is
descriptor-walked and fsynced first. After the no-replace rename, publication
creates `COMPLETED.json` exclusively; that marker binds the run ID and SHA-256
of `summary.json`. The marker, final directory, and artifact parent are fsynced
in durability order. Readers require the valid marker/summary pair and reject
an incomplete final directory.

Making leaves immutable exposed one pre-existing parent rewrite: after a lane
child published its receipt, the parent rewrote all arrays merely to replace
the child RSS provenance. GREEN removes that rewrite. Scientific lane receipts
remain append-only; the completion-hash-bound aggregate is the sole owner of
parent-sampled peak RSS, while each child receipt retains its independently
measured fallback. The auditor requires positive parent RSS from the aggregate
without pretending the child measured the parent's value.

Focused GREEN evidence:

```text
publication suite: 10 passed in 0.63s
artifact suite: 14 passed in 0.98s
publication + artifact + manifest suites: 46 passed in 3.31s
complete parity runner integration suite: 38 passed in 64.40s
focused mypy: Success: no issues found in 8 source files
Ruff check and format, compileall, git diff --check: passed
```

## Manifest-v2 dual reader and read-only migration RED -> GREEN

The schema-migration and observability tests were committed without production
changes at immutable revision `3feaf521a`. Replaying that revision in a
detached worktree produced two independent authentic RED groups:

```text
manifest dual-reader/converter/dry-run contract: 8 failed, 19 deselected in 0.50s
runner schema/adapter observability contract: 2 failed in 0.19s
```

GREEN accepts absent-schema v1 through an explicitly observable read-only
adapter and accepts explicit schema v2 with `devices`. It rejects explicit
version 1, unknown versions, mixed `lanes`/`devices`, and per-example
`intents`. Both schemas normalize to the same typed example records; the
semantic comparator independently covers source catalog, IDs and order,
readiness, lineage, paths, device capability, and the complete normalized
record.

`examples/jax/migrate_manifest.py --dry-run` is the only candidate writer. It
prints canonical candidate bytes, their SHA-256, normalized semantic diff,
observed reader metadata, the one-release compatibility interval, and rollback
command. It has no mutating mode. The ordinary example runner emits schema and
legacy-adapter metadata to CI logs, and parity aggregates retain and audit the
same fields.

The committed canonical v1 input and complete worktree diff hashes were
identical before and after the live dry run. The retained review artifact is
`docs/jax_manifest_v2_dry_run_report.md`; its candidate digest is
`2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05`.
The canonical manifest was deliberately not changed because activation still
requires explicit user sign-off.

GREEN evidence:

```text
focused migration contract: 8 passed, 19 deselected in 0.59s
focused runner observability: 2 passed in 0.17s
manifest plus complete example integration suites: 67 passed in 104.84s
focused mypy for new migration owners: Success: no issues found in 2 source files
focused mypy with pre-existing runner codes excluded: Success in 3 source files
canonical input and repository diff hashes unchanged by dry run: passed
```

## Audit-gated parity report RED -> GREEN

The report-gate regression was committed without production changes at
immutable revision `037eca88d`. The focused RED command was:

```console
MPI4PY_RC_INITIALIZE=false ../.venv-simsopt-linux-x86/bin/python -m pytest -q \
  tests/integration/test_jax_example_parity_artifacts.py::test_results_report_cli_audits_receipts_before_rendering
```

It failed with exit status 1 because the report CLI did not accept the required
`--repo-root` audit context and therefore could not invoke the independent
auditor before rendering. This exposed a real evidence-boundary defect: a
caller could hand the report generator an aggregate JSON document directly,
bypassing receipt validation such as rejection of a fast JAX backend in a
parity lane.

GREEN at immutable revision `682c8b979` makes the report CLI call
`audit_published_run()` on the containing published run before it reads or
writes report content. The auditor remains the single owner of lane backend,
FP64, transfer-guard, source, input, sidecar, comparison, and publication
validation. The report adds no second comparison policy.

Focused GREEN evidence:

```text
parity artifact and report suite: 15 passed in 0.97s
```

## Matched fast/parity execution-mode benchmark RED -> GREEN

The benchmark was built in four failing-first slices. Immutable RED revision
`25e384017` defined the evidence contract before the validator existed; GREEN
`3f167cbaa` implemented the dependency-light schema and promotion arbiter. RED
`838ed2d62` required real isolated cold, warmup, and seven balanced warmed-pair
measurements; GREEN `970965a6f` added the runner. RED `727f96037` added invalid
cache, synchronization, resource, and GPU-load cases; GREEN `dd44d371b` made
those contexts fail closed. RED `9374b3d83` rejected incompatible runtime
provenance; GREEN `cadeb7e2b` bound supported JAX/JAXLIB versions before
admission. The complete contract suite currently passes 41 tests.

Valid JAX 0.10.2 full artifacts were collected without discarding repetitions:

```text
CPU: .artifacts/jax-example-execution-modes/20260727T065241Z-cpu-dacecfd63ed8
GPU: .artifacts/jax-example-execution-modes/20260727T064728Z-gpu-829eb1cbf037
```

Both artifacts pass scientific, cache, scheduling, synchronization, runtime,
and memory-evidence validation. Neither promotes the fast policy. All five CPU
workloads miss the warm-speed threshold or its one-sided lower bound. On the
RTX 5090, fieldline/tracing is the only workload that passes every gate;
traceable least squares, curve, and surface miss the warm-speed rule, while
surface and coil also record GPU-memory ratios of about `1.467` and `1.469`,
above the unchanged `1.25` ceiling.

A separately labeled diagnostic disabled the single-GPU sharding policy:

```text
.artifacts/jax-example-execution-modes/20260727T070236Z-gpu-8aa146a1699c
```

That run still failed all five warm-speed gates and left the surface/coil GPU
memory ratios unchanged. The sharding hypothesis is therefore rejected; no
default, threshold, repetition, or artifact label was changed to manufacture a
promotion. These artifacts are local non-certifying performance evidence.

Three later diagnostics isolated the remaining policy hypotheses without
changing the production branch. Stable chunks with chunk autotuning disabled
in
`.artifacts/jax-example-execution-modes/20260727T081128Z-gpu-bd2f10af1fe1`
left the surface/coil memory ratios unchanged, rejecting chunk selection as the
owner. XLA GPU autotuning level `0` in
`.artifacts/jax-example-execution-modes/20260727T081718Z-gpu-2ec7998ca32f`
reduced the fast cache from `58` files to the same `8` compiled-cache entries
as parity and made every peak GPU-memory ratio exactly `1.0`; all five
warm-speed gates nevertheless failed. Level `1` in
`.artifacts/jax-example-execution-modes/20260727T082532Z-gpu-db5c2f94e870`
restored `58` fast cache files, the approximately `1.467`/`1.469` surface/coil
memory ratios, and four warm-speed failures. Enabled GPU autotuning therefore
owns the measured memory excess, but disabling it does not supply the required
speedup.

The independent CPU diagnostic
`.artifacts/jax-example-execution-modes/20260727T083038Z-cpu-f4881864e568`
removed the fast-only `FAST_COMPILE` preset. It is not contract-valid evidence:
persistent-cache warm reloads emitted XLA CPU AOT target mismatches for
`prefer-no-gather` and `prefer-no-scatter`. Its raw paired medians were also
slower than parity on four of five workloads (`0.9999`, `0.8591`, `0.9564`,
`0.9083`, and `0.9782` in manifest order), with every lower bound below `1.0`.
The candidate is rejected rather than repairing the validator or weakening the
promotion rule.

Two final bounded diagnostics rejected measurement-control and search-width
hypotheses. Running the CPU benchmark under a fixed physical-core affinity in
`.artifacts/jax-example-execution-modes/20260727T084209Z-cpu-4c43607e1949`
changed XLA's perceived target features and invalidated persistent-cache AOT
reloads; four of five raw medians still failed. On the RTX 5090,
`xla_gpu_autotune_max_solutions=1` in the surface-only diagnostic
`.artifacts/jax-example-execution-modes/20260727T084814Z-gpu-74775db78ca4`
left the peak GPU-memory ratio at `1.467153`, with median speedup `0.959943`
and lower bound `0.883137`. That knob limits GEMM solutions, while the installed
XLA runtime already reports a one-candidate limit for non-GEMM fusion
autotuning. Neither diagnostic is promotion evidence.

## Accepted optimizer device-state retention RED -> GREEN

The BFGS regression was committed without production changes at immutable
revision `c95120d68`. The focused RED raised `AttributeError: x_device` because
the custom SIMSOPT BFGS result converter exposed only host-compatible NumPy
copies. The same defect appeared end to end on the real RTX 5090: the strict
GPU parity runner passed 9 of 10 ready examples, then rejected Boozer's
implicit host-to-device conversion while unpacking the accepted optimizer
state.

GREEN at `b3ca878fe` keeps the existing host-compatible `x` and `jac` result
fields and additively retains `x_device` and `jac_device` from the accepted JAX
state. `BoozerSurfaceJAX` consumes those device fields for follow-up unpacking
and gradient publication, with the legacy fields retained as a compatibility
fallback. The focused BFGS scope passed 34 tests with 1 skip.

The equivalent limited-memory regression was committed separately at
`6df21f913`; its focused RED also raised `AttributeError: x_device`. GREEN at
`eead0641e` adds the same device-resident fields to the custom SIMSOPT L-BFGS-B
result without changing its public host fields. A clean CPU-pinned run passed
the exact regression, and the complete private L-BFGS class passed:

```text
targeted L-BFGS device-state regression: 1 passed in 2.18s
private L-BFGS class: 29 passed, 1 skipped in 86.88s
```

The local JAX 0.10.2 runtime's global XLA garbage-collection callback stalled
when cyclic collection ran amid other concurrent JAX jobs. The clean focused
runs disabled cyclic GC in the pytest harness; Python reference counting and
the solver path were unchanged. The independent isolated example runners did
not require that harness setting.

At committed GREEN `eead0641e`, all four production profiles then passed every
ready example through isolated children using JAX 0.10.2 and FP64:

```text
CPU fast:    10 passed, 0 failed
CPU parity:  10 passed, 0 failed
GPU fast:    10 passed, 0 failed
GPU parity:  10 passed, 0 failed
```

The two GPU profiles ran on the real NVIDIA GeForce RTX 5090 with the required
CUDA runtime libraries. The parity profile kept strict transfer guards and no
CPU fallback; Boozer passed in that complete 10-example run. The canonical
manifest was still v1, so every runner also emitted
`manifest_schema_version=1` and `used_legacy_manifest_adapter=true`. This is
current functional evidence, not manifest-v2 activation or fast-performance
promotion evidence.

## Manifest-v2 activation RED -> GREEN (2026-07-27)

The user explicitly approved candidate SHA-256
`2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05`,
the one-release legacy-reader interval, observability fields, semantic replay,
and rollback command. Applying those exact bytes made the pre-activation test
fixtures fail because they still treated canonical v1 as their source:

```text
tests/test_jax_examples_manifest.py: 10 failed, 17 passed
```

GREEN reverses the fixture direction: canonical v2 is authoritative, while an
absent-schema v1 document is derived only for compatibility and migration
tests. It adds an exact canonical digest assertion and proves the v1 converter
replays byte-identical canonical v2 output:

```text
tests/test_jax_examples_manifest.py: 28 passed in 2.61s
```

The broader manifest/runner/parity-safety suite then found one additional stale
end-to-end assertion: a newly published Wave-A receipt correctly reported v2,
while the test still required schema v1 and legacy-adapter use. The run was
`1 failed, 157 passed`; after changing that assertion to require schema v2 and
`used_legacy_manifest_adapter=false`, the focused activation slice passed
`29 passed`.

The complete manifest, ordinary-runner, parity-manifest, artifact, input,
publication, runner, and runtime matrix then passed `158 passed` with seven
expected legacy-lane deprecation warnings.

The canonical file's live SHA-256 matched the approved digest. No production
solver, numerical tolerance, example mapping, device capability, or execution
intent changed in this migration.

The user separately chose to retain fast as the CPU/GPU JAX default despite
the negative performance characterization. This is a policy choice, not a
promotion: the unchanged benchmark failures remain recorded above, and no
speed or performance-qualification claim is allowed. Authority evidence is
also explicitly local-only; its absolute host path and digest may be recorded,
but no durable shared-retention claim may be made.

Post-activation execution used canonical v2 without the legacy adapter. Both
CPU profiles passed all ten ready examples through `run_examples.py`. Direct
current-checkout validation on the NVIDIA GeForce RTX 5090 also passed all ten
examples in fast and strict-parity FP64 modes. The GPU environment's editable
finder still points at a different checkout, so these GPU children used
`python -S` with this checkout and the environment's site-packages explicitly
ordered on `PYTHONPATH`; every accepted JSON result reported `platform=gpu`,
the selected `jax_gpu_fast` or `jax_gpu_parity` mode, FP64, and `status=ok`.
The installed JAX runtime emitted a driver-version parsing diagnostic while
still enumerating and executing on `CudaDevice(id=0)`; it did not trigger CPU
fallback or alter the validated result contract.

## Surface endpoint Jacobian quotient contract RED -> GREEN (2026-07-27)

The first fresh clean authority attempt at committed manifest-v2 revision
`eb8385cbb` was authoritative in all three lanes and passed seven of eight
cases, but correctly retained a failing `.partial` bundle:

```text
.artifacts/jax-example-parity/20260727T130725Z-e7ab9ba2.partial
verdict: fail
surface-geometry-optimization:
  final residual_jacobian native-cpu:jax-cpu: fail
  final residual_jacobian native-cpu:jax-gpu: fail
```

The native and JAX final parameter pairs differed by about `2.46e-6` while
their sum/product quotient coordinates agreed to about `1e-11`; area, volume,
residual, objective, and terminal status all passed. At the circular optimum,
the two ellipse semi-axes are interchangeable and the residual Jacobian is
nearly rank deficient. Therefore, the raw terminal Jacobians were evaluated at
different coordinate representatives and were not a valid same-state
cross-lane observable.

Two repairs were considered. Forcing a shared polished endpoint would change
the native and JAX solve workflows and conceal a legitimate algorithmic
difference. Relaxing the derivative tolerance would misclassify a coordinate
artifact as numerical parity. The first selected design retained raw terminal
parameters and Jacobians as non-applicable diagnostics and compared row-wise
sum/product coordinates of the interchangeable Jacobian columns. No tolerance
changed.

The failing-first surface integration test required the new final invariant
and failed with:

```text
KeyError: 'final:residual_jacobian_invariants'
1 failed in 11.39s
```

After both lanes published the same invariant contract and the manifest routed
all three direct pairs, the focused surface plus manifest suite passed
`23 passed`; the endpoint case and adversarial permutation/drift test then
passed `2 passed`. The broader parity manifest, runner, artifact, input,
publication, and runtime suite passed `91 passed in 97.83s` from an isolated
runtime whose source path was pinned to the current checkout.

The first post-repair authority launch then failed closed before arbitration
with `applicable observables require a complete direct lane-pair matrix`: the
shared state publisher emits the symmetric invariant at both initial and final
states, while the first manifest patch routed only the final state. The fix
declares the missing initial native/JAX CPU/GPU three-pair matrix and asserts
both its value and applicability in the surface integration test. No observed
value, comparator, or tolerance changed.

The next clean launch repeated the same schema failure at the first
`traceable-least-squares` case. Inspection showed the structurally repeated
JSON patch had placed the three initial routes on that case rather than on
surface geometry. The routes were moved to the surface relationship and a
manifest regression now proves that surface owns all six initial/final routes
and traceable least squares owns none of them.

### Global column-exchange repair RED -> GREEN

Review found that row-wise sum/product coordinates accept a broader symmetry
than the physics permits: independently swapping the two columns in only one
residual row leaves every row invariant even though the axis exchange must be
one global parameter-column permutation. The new adversarial test first failed
against the committed helper:

```console
/tmp/simsopt-parity-v2.3szEgy/runtime/bin/python -m pytest -q \
  tests/integration/test_jax_example_parity_runner.py \
  -k surface_jacobian_invariants_ignore_only_global_axis_exchange
```

```text
1 failed, 38 deselected
```

An unnormalized outer product of the column-difference vector correctly
encoded the global orbit, but its off-diagonal coupling was about `2.39e-8`
near the circular optimum and therefore remained invisible under the unchanged
absolute tolerance. Canonically orienting the raw difference rejected the
adversarial swap but incorrectly compared the differing endpoint
eccentricities, causing the real surface end-to-end test to fail. The GREEN
representation keeps the row sums/products and adds the outer product of the
max-norm-normalized column-difference vector. It is unchanged by one global
column exchange, couples all residual rows, rejects an independent row swap at
the central `native_workflow` tolerance, and remains finite when the columns
are identical. The focused adversarial and native/JAX CPU end-to-end tests pass
without changing a tolerance.

## Historical manifest-v2 authority (superseded 2026-07-27)

Revision `3e7ecb58eeb75e763f823deb631c9ee2b0ea0f9c` was checked out detached and
clean at `/tmp/simsopt-parity-v2-fix.SLPdTa/checkout`. A Python 3.11.15
environment built `simsopt==1.10.7.dev580+g3e7ecb58e` and its native extension
from that exact checkout. The extension SHA-256 was
`43205604c01308c147f9d9e7f771d3efa0b5d9533e3a315cc7828bdfb87c3b0c`.

The full `all-applicable` matrix ran `native-cpu,jax-cpu,jax-gpu` without
`--smoke` and published the host-local bundle:

```text
/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/
  .artifacts/jax-example-parity/20260727T133326Z-04db9b25
```

The independent required-authority audit returned at that revision:

```json
{"authoritative": true, "case_count": 8, "comparison_count": 252,
 "lane_receipt_count": 24, "run_id": "20260727T133326Z-04db9b25",
 "verdict": "pass"}
```

All eight cases passed on native CPU, JAX CPU, and strict JAX GPU in FP64. The
GPU was an NVIDIA GeForce RTX 5090 with UUID
`GPU-7951f78e-c05d-e01c-303f-d644f4341fe1` and driver `595.84`. The canonical
manifest schema was v2 and the legacy adapter was not used. The summary
SHA-256 is
`21c89ec296c3b08b76131a8422ddf90efc7218d22a6162affc65e52a72c4feaf`;
the freshly generated report SHA-256 is
`4e233fdb20cb016015789a69069f0eaeabdb6ea40c0f2dac056525ec1183b160`.
The ignored authority bundle is local-only and has no durable shared retention
guarantee.

This bundle no longer certifies the surface derivative contract: the later
adversarial review proved that its row-wise invariant admitted independent
row swaps. It remains historical execution evidence only. Fresh authority must
be generated from the corrected executable revision before the publication
gate can be checked again.

The freshly generated report was written to the isolated temporary work area,
not over the pre-existing untracked
`docs/jax_native_example_parity_results.md`. That worktree file describes the
older run `20260726T225943Z-09dfdc3e` and has SHA-256
`0164eab54209aae1226bfab22025f74ed5b8cb9d278792f20ccb5534f144a8dc`;
replacing user-owned untracked content requires explicit approval.

After the case-ownership correction, the final broad parity manifest, runner,
artifact, input, publication, and runtime regression suite passed
`92 passed in 85.48s`.

## Corrected global-orbit authority GREEN (2026-07-27)

The corrected executable revision
`3b401b54bf4b12d1a35b70dd9621080ca9620ff6` was checked out detached and
clean at `/tmp/simsopt-parity-v2-fix.SLPdTa/checkout`. Python 3.11.15 rebuilt
`simsopt==1.10.7.dev583+g3b401b54b` from that checkout. The native extension
SHA-256 remained
`43205604c01308c147f9d9e7f771d3efa0b5d9533e3a315cc7828bdfb87c3b0c`.

The full `all-applicable` matrix ran `native-cpu,jax-cpu,jax-gpu` without
`--smoke` and published the host-local bundle:

```text
/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/
  .artifacts/jax-example-parity/20260727T141144Z-cb97f4d1
```

The independent required-authority audit returned:

```json
{"authoritative": true, "case_count": 8, "comparison_count": 252,
 "lane_receipt_count": 24, "run_id": "20260727T141144Z-cb97f4d1",
 "verdict": "pass"}
```

All eight cases passed on native CPU, JAX CPU, and strict NVIDIA GeForce RTX
5090 JAX GPU in FP64. The aggregate binds the clean executable revision,
reports no repository change during execution, records an empty tracked diff
and untracked inventory, uses canonical manifest v2 without the legacy
adapter, and contains 24 authoritative lane receipts and 252 comparisons. The
summary SHA-256 is
`f317f5c69fcb52a83ebb598df897273160537ecae421f6a15c1db1c2574b0819`.
The fresh report was generated only in the isolated temporary work area and
has SHA-256
`297d13deaa6f1500c966aed2795bca8725d76c1fa8235378189e5d885875d43f`;
the pre-existing untracked worktree report was not replaced.

After the repair, all 120 manifest, runner, artifact, input, publication, and
runtime tests passed. Ruff, formatting, JSON parsing, compileall, and Git diff
checks passed. The isolated runtime did not contain mypy, so no fresh mypy
result is claimed for this repair.

## Exact `qfm.py` mirror RED -> GREEN -> REFACTOR (2026-07-28)

The source-owned parity test was committed first at RED revision
`ecf0c1ee9c1a93fbcada594ddda2fe8bbde26427` and failed because the
`native-qfm` case did not exist. GREEN revision
`5845ffcc9de02cd941a01cc3584a2076bc68d7ea` added the matched native and JAX
workflow:

- the fitted bounded NCSX surface and fixed coil construction are hash-bound;
- volume, toroidal flux, and area each run a penalty solve followed by an exact
  equality-constrained solve;
- native uses the source's LBFGS-B then SLSQP route;
- JAX uses the SIMSOPT-owned BFGS then augmented-Lagrangian route;
- initial parameters, QFM value/gradient, label value/gradient, all three
  stage endpoints, feasibility, final parameters, and volume-persistence
  diagnostics are retained.

The first GREEN run exposed that native derivative arrays were views into
reused SIMSOPT buffers. The receipt publisher now takes owned copies at each
phase; the initial native/JAX QFM value and gradient then matched at the
floating-point reduction floor. The unchanged parity test passed in
`106.82 s`.

The strict-device RED at
`346e35ee06ba84e1c19cf09e730e58c4fd38d880` proved that the user-facing
example still invoked adapter methods with repeated host publications. GREEN
revision `52f9071fdb29ce0bb4b533565a3550ed77899bcb` moved the complete
three-stage numerical sequence behind `solve_qfm_sequence`, retained the
result as a device PyTree, and made one final `jax.device_get` publication.
It also repaired RZ-Fourier scatter indices to inherit the reference DOF
array's device and made example platform metadata report the selected runtime
device rather than the first enumerated JAX device.

The strict NVIDIA GeForce RTX 5090 FP64 test passed `2 passed in 51.88 s`
under `jax.transfer_guard("disallow")`. A direct strict example run reported
`platform=gpu`, `status=ok`, and:

```text
initial QFM       1.6614172567241663e-02
final QFM         1.1461611986416241e-03
area feasibility  6.7853656192419110e-23  (0.5 * raw_residual^2)
```

The matched native final QFM was `1.1461612104134106e-03`; the JAX CPU result
was `1.1461611493776677e-03`. These differences are approximately
`1.18e-11` and `6.10e-11` absolute respectively and passed the committed
source-owned parity thresholds without weakening a tolerance. The combined
CPU parity plus strict-transfer suite passed `3 passed in 104.78 s`.

REFACTOR revision `113a62c7b760135137716e3c95c7ffd575487d03` uses
`jax.jacrev(..., has_aux=True)` so each diagnostic state shares one
field/geometry primal evaluation between its values and Jacobian. No iterate
history is retained, the BFGS state remains fixed-size, and only the final
result tree crosses to the host. Ruff and Pyright reported no issues in the
touched QFM implementation.

Both new immutable receipt slices replayed successfully. The complete
17-receipt structural audit passed, while replaying every historical command
is currently blocked by an older infrastructure receipt whose unpinned
`python` command now resolves to host Python 3.14 and returns its normalized
unexpected-outcome code. That stale historical-toolchain issue is separate
from the environment-pinned QFM receipts.

This closes the fifth of eight Wave-A exact mirrors and the fifth of 25
external-solver-free source mirrors overall. Performance and peak-memory
qualification remain a later matched-workload phase; the execution times
above include compilation and are validation timings, not speed claims.

## Exact `permanent_magnet_simple.py` mirror RED -> GREEN -> REFACTOR (2026-07-28)

The source-owned parity RED at
`205a27e9a5594668388aa318aaf1dfff61091cd5` failed because the
`native-permanent-magnet-simple` case was not registered. GREEN revision
`b7df9cf15bc56108a0bbaa8427f11c44e6f2565d` replaced the former two-dipole
toy evidence with the source workflow's bounded NCSX/FAMUS problem:

- the fixed-boundary WOUT and FAMUS inputs are SHA-256 bound;
- the bounded grid contains 574 dipole locations and a `(4, 1722)` response
  matrix;
- the native lane calls `simsoptpp.GPMO_baseline`;
- the JAX lanes call the SIMSOPT-owned `GPMO_baseline_jax`;
- construction arrays, initial state, all final physical moments, residual,
  objective, selected-magnet mask, and nonzero fraction are retained.

The native CPU and JAX CPU final moment arrays matched under
`rtol=1e-13, atol=1e-14`. The native final residual sum of squares was
`3.6520685774992756e-01`.

The strict-transfer RED at
`afa4d42ef0068a57984baf001c119402ad0300d0` found four separate host
publications in the user-facing example. GREEN revision
`1f5272fca7f734dd50f4259889ed1cd412afe4b4` moved every diagnostic reduction
onto the selected device and retained one explicit final `jax.device_get`.
REFACTOR revision `ca6fc50800c4b85e3b24a30275e058ad027703e2`
fused those reductions behind one JIT boundary.

The memory RED at
`a65ccd387bcd55f7d738fba99855e6dc441f6480` proved the public solver had no
final-state-only mode and therefore retained the full
`K x ndipoles x 3` moment history. GREEN revision
`23e8fd58adf464676938347604c085d60a0528f4` added
`retain_history=False`. The example and exact parity lane now retain only the
final state and selected-dipole trace. On the bounded RTX 5090 compilation,
JAX executable memory analysis reported:

```text
history-retaining output bytes   566184
final-state-only output bytes     14824
output reduction bytes           551360
history-retaining temp bytes      32344
final-state-only temp bytes       58704
combined output+temp reduction   525000
```

The strict RTX 5090 FP64 suite passed `3 passed in 3.29 s` with the production
preallocation policy. A direct matched native/GPU execution reported
`platform=gpu`, matching construction fingerprints, identical final physical
moments, and maximum absolute differences of `2.22e-16` in both the final
residual and objective.

The focused CPU parity, strict-transfer, and memory-contract tests passed.
Nine core baseline tests and thirteen solve-level baseline/history tests also
passed. A broader permanent-magnet test invocation later terminated in the
pre-existing native `relax_and_split` test with `SIGFPE`; it was outside the
modified baseline GPMO path and is not reported as a passing regression run.

All three new immutable receipts replayed successfully. The complete receipt
document now contains 20 structurally valid behaviors. Matched cold/warm
performance and process/device peak-memory comparison against the native
example remain part of the later claim-eligible measurement phase; the
validation timings above are not speed claims.

## Exact `wireframe_rcls_basic.py` mirror RED -> GREEN -> REFACTOR (2026-07-28)

The source-owned parity RED at
`ccfdcae9f0323a245367d290fde28de475e10c58` failed because the exact
`native-wireframe-rcls-basic` case was not registered. GREEN revision
`01f093297126e1b4d57dcdc699baf141673d072a` implemented the bounded source
workflow using the Landreman-Paul QA boundary, offset toroidal wireframe,
poloidal-current constraint, area-weighted normal-field response, feasible
initial currents, constrained RCLS solve, and full final wireframe field
diagnostics.

The bounded case contains 48 wireframe segments, a `(256, 48)` response
matrix, and 256 final magnetic-field vectors. Native SIMSOPT and JAX CPU
matched all construction arrays and source-level initial/final observables.
The direct matched native CPU/JAX GPU execution reported `platform=gpu`,
`precision=fp64`, and scientific success in both lanes. Representative maximum
absolute differences were:

```text
final currents                    2.60770320892334e-08
final normal-field residual       5.08273978461205e-16
final normal objective            8.67361737988404e-19
final magnetic field              4.71844785465692e-15
final mean relative normal field  1.38777878078145e-17
```

The current difference is approximately `3.8e-14` of the maximum solved
current. The final mean relative normal field is
`3.13300741480366e-02`, below the source-owned `4e-2` success gate.

The strict-transfer RED at
`72d646db2ae0c2e77963b7c3d55f01a5857e9e72` failed because no public
device-resident example workflow existed. GREEN revision
`eb5fa15c7fcc5d68f9746e222b75a68641a25bad` introduced
`solve_wireframe_rcls`, kept RCLS and final field diagnostics on the selected
device, returned a final-state-only immutable result, and reduced the
user-facing example to one explicit final `jax.device_get`.

The performance RED at
`a9ea641728d00c5eaabe086ef83f5a6871b52493` measured two independent
constructions of the same host constraint system. GREEN revision
`97d1b85e32d7f57c0e24a67eaac5bbb5282eef38` reuses one immutable constraint
snapshot for both the RCLS solve and diagnostics. REFACTOR revision
`57aeaf26b2cbac0522d457c5bf1477e64b428105` fuses the solve, free-current
expansion, field evaluation, and reductions into one JIT entrypoint.

On the RTX 5090 bounded compilation, the fused executable reported:

```text
argument bytes          127584
result bytes             13505
executable output bytes  13657
temporary bytes       33795480
result leaves               19
```

The result contains only initial/final states and final source diagnostics; it
retains no iteration history or dense intermediate factorization. The
temporary allocation is compiler-managed workspace for the constrained dense
linear algebra and remains separate from the small persistent result.

The focused CPU parity, strict-transfer, and efficiency suite passed
`4 passed in 3.57 s`. Strict RTX 5090 FP64 execution passed
`3 passed in 3.81 s` with production preallocation. All three immutable
wireframe receipts replayed successfully, and the complete receipt document
contains 23 structurally valid behaviors.

This closes the seventh of eight Wave-A exact mirrors and the seventh of 25
external-solver-free source mirrors overall. The later five-profile
measurement phase still owns claim-eligible cold/warm timing, process-tree
RSS, and process-attributed VRAM comparisons against the native source.

## Exact `strain_optimization.py` mirror RED -> GREEN -> REFACTOR (2026-07-28)

The source-owned parity RED at
`6e8a5c34c0cc92d7a2e89b348daf084b2cc6fff0` failed because the exact
`native-strain-optimization` case was not registered. GREEN revision
`438efddd2136fd61711d824a7ba0ec378f9717a6` implemented the bounded source
workflow with the scaled HSX coil, order-10 tape-frame rotation, centroid-frame
torsional and binormal-curvature strains, the source penalty widths and
thresholds, and the source L-BFGS-B budget.

The native and JAX lanes retain the fixed construction, initial/final
parameters, objective, gradient, both strain fields, maxima, raw solver status,
and evaluation counts. The unchanged parity test passed on JAX CPU. The
focused Wave-A execution, strain parity, and boundary suite passed
`9 passed in 72.10 s`.

The device-boundary RED at
`1cb24012d2cedbb7bd3b831b1672fbd86263c138` proved that the prior user-facing
example duplicated private strain kernels and performed seven separate
`jax.device_get` publications. GREEN revision
`8e4a2806d5351c7cdfb0af672eadc0a1f5b42d7d` moved the workflow behind the
public `solve_strain_rotation` contract and reduced reporting to one batched
result-tree publication. The solver uses the SIMSOPT-owned JAX L-BFGS
implementation in memory-bounded stepwise mode and disables optimizer-state
history; it does not JIT a monolithic outer loop. REFACTOR revision
`8b8965fce23175724560ab67391b11bcc7ab60c2` reuses the quadrature arc length
inside each objective evaluation instead of computing the same norm twice.

A direct strict NVIDIA GeForce RTX 5090 execution used FP64, reported
`platform=gpu`, and completed without CPU fallback under the strict transfer
configuration. Native CPU and JAX GPU both used 50 iterations and 54 function
and gradient evaluations. Representative maximum absolute differences were:

```text
initial objective                  6.776263578034403e-21
initial gradient                   3.218725199566341e-20
final parameters                   2.553512956637860e-14
final objective                    5.293955920339377e-23
final gradient                     4.229721887840903e-19
final torsional strain             2.655861641720492e-15
final binormal-curvature strain    3.308117668687772e-15
```

Both new immutable receipts replayed successfully from detached clean
worktrees. The complete receipt document contains 25 structurally valid
behaviors. A full historical replay currently stops on an older infrastructure
receipt whose normalized RED wrapper now returns `99` instead of its recorded
`2`; the two environment-pinned strain receipts are independently replay-clean,
and the stale historical receipt remains an explicit follow-up rather than
being silently relabeled.

This implements all eight Wave-A exact mirrors. Formal Wave-A readiness still
requires activating the new manifest identities and running the complete
ready-mirror CPU/GPU matrix. Matched cold/warm timing, process-tree RSS, and
process-attributed VRAM comparison against native remain owned by the later
claim-eligible measurement phase.
