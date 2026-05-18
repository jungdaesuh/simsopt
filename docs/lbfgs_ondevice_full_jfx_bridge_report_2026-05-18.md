# L-BFGS On-Device Full JF.x Bridge Report

Date: 2026-05-18

## Summary

The full `JF.x` contract was tested for `lbfgs-ondevice`.

Result: `lbfgs-ondevice` now matches CPU/SciPy fullgraph to about `1.2e-9` on the reduced comparison run.

| lane | final objective | diff vs CPU | nfev/njev |
|---|---:|---:|---:|
| CPU/SciPy | 1.1132110846645535 | 0 | 4/4 |
| scipy-jax-fullgraph | 1.1132110846645542 | +6.7e-16 | 4/4 |
| lbfgs-ondevice fullgraph bridge | 1.113211083463847 | -1.20e-9 | 4/4 |

## Before And After

Before this change, `lbfgs-ondevice` was effectively following the compact coil-only path and landed at:

```text
1.120947417946756
```

That was about `7.7e-3` away from the CPU/SciPy fullgraph reference.

With the full 51D `JF.x` path, `lbfgs-ondevice` follows the same printed line-search sequence as CPU/SciPy:

```text
reject 1.00e+00
reject 8.52e-01
accept 1.41e-01
```

## Important Caveat

This is not a pure on-device fullgraph objective yet.

The current implementation routes full `JF.x` value/gradient evaluation through an ordered JAX `io_callback` into the private JAX L-BFGS-B driver. The L-BFGS-B state machine is the on-device JAX port, but objective and gradient evaluation are host-backed to preserve exact `JF.x` and Boozer semantics.

## Runtime Impact

The fullgraph bridge completed instead of being killed.

The run is still expensive:

| metric | value |
|---|---:|
| wall time | 932.7 s |
| max RSS | about 10.4 GB |

Sampling showed the long first phase was:

```text
JAX PjitFunction -> CompileAndLoad
```

## Implementation Touch Point

Primary touched path:

```text
examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1606
```

The bridge entry point is:

```text
build_single_stage_full_graph_host_callback_value_and_grad(...)
```

## Validation

Validation passed:

- `py_compile`
- `ruff check`
- `ruff format --check`
- 3 targeted single-stage tests
