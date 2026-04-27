# Single-Stage Parity Revalidation - 2026-04-28

## Status

The H100 production target-lane run is end-to-end successful and fast. The
strongest parity result is same-state JAX CPU vs H100 value-and-gradient parity:
the objective and gradient agree to floating-point roundoff.

Full optimizer final-state parity is close but not identical. That is expected
until all lanes record and compare the same optimizer trajectory. Optimizers are
path-sensitive: a tiny difference in one value, gradient, or Wolfe line-search
decision can send later accepted steps to a slightly different valid point.

CPU/C++ full-trajectory parity is not yet proven because the current CPU/C++
artifact is an initialization/reference artifact, not a matching full optimizer
run under the same trajectory contract.

## Source Artifacts

Current H100 Git-clone production E2E:

- Artifact: `.artifacts/runpod_single_stage_continuation/20260428-h100-gitclone-e2e1/continuation-20260428-h100-gitclone-e2e1`
- Commit: `7e3f2eb5e5462c7d3cc989ce8bf1fe010a04f3a2`
- Branch: `gpu-purity-stage2-20260405`
- Remote status before launch: clean

Current H100 memory diagnostic:

- Artifact: `.artifacts/runpod_single_stage_continuation/20260428-h100-gitclone-memory1/continuation-20260428-h100-gitclone-memory1`
- Commit: `7e3f2eb5e5462c7d3cc989ce8bf1fe010a04f3a2`
- Point chunk size: default

Current parity matrix inputs:

- Merged parity report: `.artifacts/parity/20260427-cpu-jax-gpu-parity-report.json`
- Parity matrix: `.artifacts/parity/20260427-single-stage-parity-matrix.json`
- Native CPU/C++ reference init: `.artifacts/parity/20260427-fresh2-cpu-reference-init/mpol=10-ntor=10-e78dcc9e/results.json`
- JAX CPU full optimizer artifact: `.artifacts/parity/20260427-fresh3-jax-cpu-m20/mpol=10-ntor=10-b53e6701/results.json`

## Lay Interpretation

The optimizer loop is:

```text
current coil shape
  -> compute physics score and gradient
  -> pick a search direction
  -> try step sizes until Wolfe line search accepts one
  -> repeat until the improvement is tiny
```

Same-state parity checks whether two backends compute the same score and
gradient when handed the same exact input. That currently passes for JAX CPU vs
H100.

Final-state parity checks whether two full optimizer runs land at the same final
coil shape. That is stricter. It requires the same value, gradient, search
direction, trial step, line-search accept/reject decision, and termination mode
at every iteration.

## Current H100 E2E Result

| Metric | Value |
| --- | ---: |
| Return code | 0 |
| Stage wall time | 462.010 s |
| GPU memory high-water | 14353 MiB |
| Optimizer success | true |
| Optimizer status | 4 |
| Termination | `Optimization terminated successfully (ftol)` |
| Optimizer evaluations | 17 value / 17 gradient |
| Final objective | 0.0008324085304084091 |
| Field error | 0.0003787698258266065 |
| Final iota | 0.24993804793782737 |
| Final volume | 0.03996462582481037 |
| Max curvature | 94.08851557341214 |
| Hardware constraints OK | true |

The measured production stage was about 7.7 minutes. The human-visible session
was longer because it also included clone/install/setup, runtime seed
compilation, a separate memory diagnostic, artifact transfer, and pod shutdown.

## Current Performance Comparison

| Lane | Artifact type | Script time | Optimizer time | Boozer solve time |
| --- | --- | ---: | ---: | ---: |
| Native CPU/C++ | init/reference only | 558.476 s | n/a | 329.263 s |
| JAX CPU | full `maxiter=20` optimizer | 11676.612 s | 10309.253 s | 300.386 s |
| H100 GPU | full `maxiter=20` optimizer | 452.657 s | 336.827 s | 25.620 s |

Derived speedups for the comparable JAX CPU full optimizer artifact:

| Metric | H100 speedup |
| --- | ---: |
| Total script time | 25.8x |
| Outer optimizer time | 30.6x |
| Boozer solve time | 11.7x |

## Current Memory Result

The current-code H100 memory diagnostic completed successfully.

| Metric | Value |
| --- | ---: |
| Diagnostic wall time | 621.945 s |
| GPU memory high-water | 14907 MiB |
| `optimizer_value_and_grad` XLA temp memory | 4551.75 MiB |
| `value_and_grad_pipeline` XLA temp memory | 4551.74 MiB |
| `forward_result` XLA temp memory | 4548.51 MiB |
| `solved_total_gradient` XLA temp memory | 2445.78 MiB |

This is no longer close to the previous 56-76 GiB failure envelope.

## Current Parity Results

### JAX CPU vs H100 Same-State Value/Gradient

This is the strongest result and currently passes.

| Metric | Delta |
| --- | ---: |
| Objective absolute delta | 8.673617379884035e-19 |
| Objective relative delta | 1.0417742202199725e-15 |
| Gradient max absolute delta | 5.828670879282072e-15 |
| Gradient allclose gate | true |

Interpretation: the JAX CPU and H100 physics objective agree to roundoff when
evaluating the same state.

### JAX CPU vs H100 Full Optimizer Final State

Both lanes succeed, but they land at slightly different valid final points.

| Metric | H100 | JAX CPU | Absolute delta |
| --- | ---: | ---: | ---: |
| Final objective | 0.0008324085304084091 | 0.0008323486959847253 | 5.983442368385616e-08 |
| Field error | 0.0003787698258266065 | 0.0003786184210519463 | 1.514047746602339e-07 |
| Final iota | 0.24993804793782737 | 0.2499444945641246 | 6.446626297229585e-06 |
| Final volume | 0.03996462582481037 | 0.039964503699568193 | 1.2212524217597442e-07 |
| Max curvature | 94.08851557341214 | 94.09060749137089 | 0.002091917958750855 |

Interpretation: this is close scientific parity, but not trajectory parity.
The parity matrix marks full trajectory parity blocked because matched
`optimizer_state_trace` entries are missing.

### CPU/C++ vs JAX CPU Same-Seed Metrics

The current matrix shows small drift in derived metrics:

| Metric | JAX CPU | CPU/C++ | Absolute delta |
| --- | ---: | ---: | ---: |
| Initial field error | 0.00037841223768129143 | 0.0003782765649675703 | 1.3567271372111117e-07 |
| Field error | 0.00037841223768129013 | 0.0003782765649675703 | 1.3567271371981013e-07 |
| Curve-surface min distance | 0.03915873312373569 | 0.03915781333940798 | 9.197843277089501e-07 |
| Surface-vessel min distance | 0.049955567179262125 | 0.04995444012383012 | 1.1270554320028103e-06 |

The same report shows JAX CPU and H100 agree on the corresponding initial
metrics to roundoff. That points the CPU/C++ drift at CPU/C++ vs JAX solver/path
differences, not at GPU arithmetic.

## How To Improve Parity

### 1. Treat same-state value/gradient as the first gate

Do not judge backend correctness first by final optimizer position. Always start
with same-state value-and-gradient parity. If the same input produces the same
objective and gradient, the physics kernel is aligned. That gate currently
passes for JAX CPU vs H100.

Next tightening:

- Add the same same-state value/gradient gate for CPU/C++ vs JAX CPU at the
  exact restored target-lane state.
- Compare objective terms and gradient components term-by-term, not only the
  composite scalar.
- Keep the existing strict transfer/device sync behavior so the comparison is
  explicit.

### 2. Compare optimizer traces before comparing final states

Final-state differences are downstream symptoms. The root comparison is the
first iteration where the optimizer diverges.

The parity matrix already supports this through `outer_optimizer_progress.json`
inputs. It compares:

- trial `x`
- search direction
- trial objective
- trial gradient
- step scale
- gradient infinity norm
- line-search status

Required next artifacts:

```bash
python benchmarks/single_stage_parity_matrix.py \
  --parity-report-json .artifacts/parity/20260427-cpu-jax-gpu-parity-report.json \
  --cpu-progress-json <cpu-full-run>/outer_optimizer_progress.json \
  --jax-cpu-progress-json <jax-cpu-full-run>/outer_optimizer_progress.json \
  --gpu-progress-json <h100-full-run>/outer_optimizer_progress.json \
  --output-json .artifacts/parity/<new-matched-trajectory-matrix>.json
```

Today this is blocked because matched progress traces are missing.

### 3. Make JAX CPU and H100 run the exact same trajectory contract

For JAX CPU vs H100, use:

- same commit
- same donor/run seed
- same runtime seed spec
- same Biot-Savart JSON
- same `mpol`, `ntor`, `nphi`, `ntheta`
- same optimizer method and `maxiter`
- same `outer_maxls`
- same `ftol`, `gtol`, and initial step policy
- same target-lane Boozer tolerances and max iterations
- same compile diagnostics off/on choice
- same progress trace recording

If same-state value/gradient passes but trajectory diverges, inspect the first
trace row that differs. The likely categories are:

- search direction differs
- Wolfe step scale differs
- line-search status differs
- termination mode differs
- final sync changes the reported derived metrics

### 4. Tighten CPU/C++ vs JAX CPU at the Boozer solve/adjoint boundary

The current CPU/C++ vs JAX CPU drift is small and concentrated in derived
Boozer/geometry metrics. The root fix is not to loosen tolerances. The root fix
is to isolate and align the Boozer solve and adjoint/operator contract.

Required checks:

- Freeze the same coil dofs, surface dofs, `G`, iota, quadrature grid, weights,
  and symmetry settings.
- Compare Boozer residual values before optimization.
- Compare Jacobian-vector products on fixed basis vectors.
- Compare transpose/adjoint solves on fixed RHS vectors.
- Compare the final original residual after any preconditioning. Preconditioning
  only preserves parity if the original unpreconditioned residual also passes.
- Compare each objective component before summing the composite objective.

JAX exact adjoints are intentionally operator-backed. Dense/PLU paths are
reference oracles and metadata producers, not production fallbacks. If CPU/C++
and JAX disagree at the operator solve, fix the operator contract or the solve
tolerances; do not add fallback paths.

### 5. Only after trace parity passes, tighten final-state tolerances

Final-state tolerances should be tightened only after:

- same-state value/gradient parity passes,
- first-step optimizer trace parity passes,
- accepted-step trace parity passes,
- termination mode matches,
- final sync metrics are compared from the same accepted state.

Otherwise tighter final-state assertions will mostly measure optimizer
path-sensitivity, not physics correctness.

## What Is Needed To Prove Full CPU/C++ Trajectory Parity

The missing artifact is a native CPU/C++ full optimizer run under the same
trajectory contract as the JAX CPU and H100 runs.

Minimum requirements:

1. Launch CPU/C++ with the same donor, constraints, resolution, targets, and
   optimizer limits as the JAX/H100 production run.
2. Record `outer_optimizer_progress.json` with non-empty `optimizer_state_trace`.
3. Launch JAX CPU and H100 with the same trace recording and the same contract.
4. Feed all three progress files into `benchmarks/single_stage_parity_matrix.py`.
5. Require:
   - same-state value/gradient parity,
   - first-step trace parity,
   - compatible accepted-step trace parity,
   - same termination mode or a documented reason for a different one,
   - final metrics inside the tightened envelope.

If the native CPU/C++ lane cannot run the identical production target-lane
optimizer, then the correct claim is weaker: CPU/C++ remains a reference for
fixed-state physics quantities, while JAX CPU and H100 own the production
target-lane trajectory.

## Current Conclusion

What is proven:

- H100 production E2E succeeds on the GitHub branch.
- H100 is much faster than JAX CPU for the comparable full optimizer artifact.
- H100 memory use is under control.
- JAX CPU and H100 same-state value/gradient parity is excellent.
- H100 reproduces the previous accepted H100 run to roundoff.

What is not yet proven:

- CPU/C++ full optimizer trajectory parity.
- JAX CPU vs H100 full optimizer trace parity, because matched trace artifacts
  have not yet been generated and compared.

Next concrete work:

1. Generate matched full-run progress traces for JAX CPU and H100.
2. Generate a native CPU/C++ full-run progress trace if that lane supports the
   same optimizer contract.
3. Run `benchmarks/single_stage_parity_matrix.py` with all progress files.
4. If the first divergence is CPU/C++ vs JAX CPU, debug the Boozer
   residual/adjoint/operator boundary term-by-term.
