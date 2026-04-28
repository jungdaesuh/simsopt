# Single-Stage Matched Trajectory Findings - 2026-04-28

## Verdict

JAX CPU vs H100 parity is now proven for the production target-lane trajectory
when both lanes use the same runtime seed spec and the same equilibrium file.

The earlier JAX CPU vs H100 final-state drift was not a backend arithmetic
problem. It came from comparing runs that were not launched from the same exact
input contract:

- old JAX CPU run: different runtime seed spec and
  `wout_nfp5ginsburg_000_014417_iota15.nc`
- H100 production run: H100-generated runtime seed spec and
  `wout_nfp5ginsburg_desc_iota21.nc`

With those inputs pinned, the JAX CPU and H100 optimizer traces agree to
roundoff.

## Artifacts

Matched JAX CPU artifact:

- `.artifacts/parity/20260428-jax-cpu-h100spec-m20/mpol=10-ntor=10-f5fde9e8`

H100 artifact:

- `.artifacts/runpod_single_stage_continuation/20260428-h100-gitclone-e2e1/continuation-20260428-h100-gitclone-e2e1/stage-01-final/mpol=10-ntor=10-856215fc`

Parity matrix:

- `.artifacts/parity/20260428-jaxcpu-h100-matched-trajectory-matrix.json`

Runtime seed spec:

- `.artifacts/runpod_single_stage_continuation/20260428-h100-gitclone-e2e1/continuation-20260428-h100-gitclone-e2e1/stage-01-final/single_stage_jax_runtime_spec.json`

Equilibrium file:

- `wout_nfp5ginsburg_desc_iota21.nc`

## Numerical Result

The matched trajectory matrix reports:

- `optimizer_state_trace_pairs.status = pass`
- `full_trajectory_parity.status = pass`
- line-search statuses: `[0, 0]`
- termination: `Optimization terminated successfully (ftol).`

Key matched deltas:

| Quantity | Absolute delta |
| --- | ---: |
| Initial objective | `6.51e-19` |
| Initial gradient infinity norm | `5.56e-15` |
| Wolfe step scale | `1.84e-17` |
| Trial objective | `5.42e-19` |
| Trial gradient infinity norm | `1.52e-13` |
| Final objective | `5.42e-19` |
| Field error | `6.99e-18` |
| Final iota | `5.55e-17` |
| Final volume | `2.08e-17` |
| Max curvature | `2.84e-14` |

Interpretation: for the JAX target lane, CPU and H100 are computing the same
optimizer trajectory to floating-point precision.

## Performance Result

| Lane | Script time | Outer optimizer time | Boozer solve time |
| --- | ---: | ---: | ---: |
| JAX CPU matched run | `12151.064 s` | `10738.886 s` | `349.401 s` |
| H100 run | `452.657 s` | `336.827 s` | `25.620 s` |

H100 speedups:

| Region | Speedup |
| --- | ---: |
| Total script | `26.8x` |
| Outer optimizer | `31.9x` |
| Optimizer main loop | `34.3x` |
| Boozer solve | `13.6x` |
| Final sync | `19.3x` |

Interpretation: H100 is not just passing parity; it is the practical execution
lane for this run shape.

## Lay Explanation

Think of the optimization as a route-planning app for coil shapes.

Each run starts from a current coil shape, asks "how bad is this shape?", gets a
gradient that says "which direction improves it?", tries a step size, and either
accepts or rejects that move. This repeats until improvement becomes tiny.

The earlier comparison was like comparing two route planners after giving them
slightly different starting addresses and slightly different maps. They both
found good destinations, but the destinations were not identical. That did not
prove the H100 math was different from CPU math.

The matched run fixed the comparison. Both lanes got the same starting address,
same map, same rules, and same stopping criteria. Then they chose the same first
direction, same step size, same accepted point, same final answer, and same
termination mode up to floating-point roundoff.

So the JAX GPU port is behaving correctly for this target-lane path. The GPU is
not changing the physics answer; it is computing the same path much faster.

## Remaining Open Issue

CPU/C++ full optimizer trajectory parity is still not proven.

The current native CPU/C++ reference lane routes through SciPy. SciPy's public
callback does not expose the same rich optimizer internals that the JAX
target-lane trace records, such as search direction, Wolfe trial step, line
search status, and trial gradient. Treating SciPy callback data as equivalent
would be a shortcut.

The remaining CPU/C++ work is therefore separate:

1. Either add a trace-capable CPU/C++ reference optimizer contract.
2. Or scope CPU/C++ to fixed-state physics parity and keep production
   trajectory parity as JAX CPU vs H100.
3. Debug the CPU/C++ vs JAX drift at the Boozer residual, JVP, transpose-solve,
   and objective-component boundary.

## Code Fix Made During This Pass

`benchmarks/single_stage_parity_matrix.py` previously compared fresh optimizer
trace files but took termination messages from the older merged parity report.
That could falsely report stale termination drift.

The matrix now uses termination messages from supplied progress JSON files when
they are provided, so the trace and termination source of truth are the same
artifact.

Regression coverage:

- `tests/test_benchmark_helpers.py::test_single_stage_parity_matrix_uses_progress_terminations`

Validation:

- `git diff --check`
- `python -m py_compile benchmarks/single_stage_parity_matrix.py tests/test_benchmark_helpers.py`
- `pytest tests/test_benchmark_helpers.py -k single_stage_parity_matrix -q`

Runpod status after the pass:

- `runpodctl pod list` returned `[]`
