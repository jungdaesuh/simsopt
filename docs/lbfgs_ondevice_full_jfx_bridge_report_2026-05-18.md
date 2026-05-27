# Retired L-BFGS Full JF.x Host-Callback Bridge Report

Date: 2026-05-18
Updated: 2026-05-26

## Summary

The full `JF.x` host-callback bridge was tested for `lbfgs-ondevice`.

Result: the bridge matched CPU/SciPy fullgraph to about `1.2e-9` on the
reduced comparison run.

| lane | final objective | diff vs CPU | nfev/njev |
|---|---:|---:|---:|
| CPU/SciPy | 1.1132110846645535 | 0 | 4/4 |
| scipy-jax-fullgraph | 1.1132110846645542 | +6.7e-16 | 4/4 |
| retired lbfgs-ondevice fullgraph bridge | 1.113211083463847 | -1.20e-9 | 4/4 |

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

## Retirement

This bridge is no longer the production GPU proof path.

It routed full `JF.x` value/gradient evaluation through an ordered JAX
`io_callback` into the private JAX L-BFGS-B driver. Per the JAX external
callback contract, callbacks execute Python on the host. That made the bridge a
useful diagnostic for fullgraph parity, but not a valid pure CUDA objective
proof.

The production `lbfgs-ondevice` single-stage path now uses the traceable JAX
target-lane value/gradient path. Fullgraph CPU-order parity remains covered by
the `scipy-jax-fullgraph` lane.

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

## Historical Implementation Touch Point

The retired bridge lived at:

```text
examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1606
```

## Validation

Historical validation passed:

- `py_compile`
- `ruff check`
- `ruff format --check`
- 3 targeted single-stage tests
