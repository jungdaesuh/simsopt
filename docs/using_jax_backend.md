# Using the JAX Backend

This document is the user-facing contract for the Columbia `simsopt-jax`
backend lane as of 2026-04-01.

It is intentionally narrower than the architecture notes in
[`gpu_jax_pro.md`](/Users/suhjungdae/code/columbia/simsopt-jax/gpu_jax_pro.md).
Use this file for:

- backend selection
- parity expectations
- strict-mode expectations
- reporting terminology
- current supported workflow lanes

## Quick start

Programmatic selection is the primary interface:

```python
import simsopt.config

cfg = simsopt.config.set_backend("jax_cpu_parity")
policy = simsopt.config.get_backend_policy()

print(cfg.mode)
print(policy.jax_platform)
```

Current public modes:

- `native_cpu`
- `jax_cpu_parity`
- `jax_gpu_parity`
- `jax_gpu_fast`

## Mode guide

### `native_cpu`

- backend: CPU reference path
- role: oracle / baseline
- expected use: correctness comparison, non-JAX workflows

### `jax_cpu_parity`

- backend: JAX on CPU
- role: algorithmic parity oracle for the JAX backend
- contract:
  - x64 required
  - stable chunking policy
  - `transfer_guard="log"` by default

### `jax_gpu_parity`

- backend: JAX on GPU
- role: device-parity lane
- contract:
  - x64 required
  - parity-oriented chunking policy
  - `transfer_guard="log"` by default

### `jax_gpu_fast`

- backend: JAX on GPU
- role: performance-oriented target lane
- contract:
  - may use more aggressive chunking
  - not the primary parity oracle

## Strict mode

Strict mode is for catching unsupported compatibility fallbacks early:

```python
import simsopt.config

simsopt.config.set_backend("jax_gpu_parity", strict=True)
```

Use `strict=True` to fail immediately instead of silently dropping to known
forbidden compatibility seams.

## Reporting contract

Parity reporting is split into three categories. Do not mix them in one bucket.

### Algorithmic parity

Compare:

- `native_cpu` vs `jax_cpu_parity`

Meaning:

- same formulas
- same quadrature and chunking intent
- same dtype policy
- same reference objective/gradient contracts

Typical artifacts:

- objective value parity
- gradient or adjoint parity
- finite-difference checks
- kernel-level invariant checks

### Device parity

Compare:

- `jax_cpu_parity` vs `jax_gpu_parity`

Meaning:

- same JAX algorithm on different devices
- cross-device agreement under parity mode

Typical artifacts:

- CPU JAX vs GPU JAX value comparisons
- CPU JAX vs GPU JAX gradient comparisons
- reduced-fixture GPU regression smoke tests

### Physics parity

Compare:

- final workflow outcomes, not just local numerics

Meaning:

- optimization-level or solver-level agreement
- invariants and final objective quality still hold

Typical artifacts:

- final solver outcome quality
- final objective / residual agreement
- physics invariants on representative fixtures

## Current supported lanes

The currently strongest JAX-backed lanes are:

- Stage 2 fixed-surface target/objective paths
- single-stage traceable target lane
- grouped Biot-Savart forward and derivative validation paths
- Boozer and single-stage objective slices already routed through immutable
  specs where documented in the roadmap docs

The current CPU reference lane remains the oracle for broad workflow trust.

## Recommended usage pattern

### 1. Validate on CPU parity first

```python
import simsopt.config

simsopt.config.set_backend("jax_cpu_parity", strict=True)
```

Use this to validate:

- algorithmic parity
- finite-difference checks
- objective/gradient consistency

### 2. Move to GPU parity

```python
import simsopt.config

simsopt.config.set_backend("jax_gpu_parity", strict=True)
```

Use this to validate:

- device parity
- reduced-fixture GPU smoke coverage

### 3. Move to the fast lane only after parity is green

```python
import simsopt.config

simsopt.config.set_backend("jax_gpu_fast", strict=True)
```

Use this for:

- profiling
- warm-run timing
- throughput experiments

Do not use `jax_gpu_fast` as the first proof lane.

## Runtime inspection

Use the runtime helpers to record the current backend contract:

```python
import simsopt.config

policy = simsopt.config.get_backend_policy()

print(policy.mode)
print(policy.backend)
print(policy.jax_platform)
print(policy.requires_x64)
print(policy.transfer_guard)
print(policy.debug_nans)
print(policy.chunk_policy)
print(policy.tolerance_tier)
print(policy.provenance_label)
```

## Current caveats

- `native_cpu` is still the default and the broadest trusted lane.
- Routine GPU regression CI is still intentionally minimal.
- Not every legacy object family is fully routed through immutable specs yet.
- Some broader workflow families remain planned rather than fully implemented.

## What this file does not claim

This file does not claim:

- universal full-GPU completion
- bitwise-identical cross-device results
- full PM / wireframe / greedy coverage
- that every legacy mutable compatibility path is already removed

Use the roadmap docs for the broader status:

- [`/Users/suhjungdae/code/columbia/analysis/jax_backend_master_updates_2026-03-31.md`](/Users/suhjungdae/code/columbia/analysis/jax_backend_master_updates_2026-03-31.md)
- [`/Users/suhjungdae/code/columbia/analysis/jax_combined_backend_10_10_plan.md`](/Users/suhjungdae/code/columbia/analysis/jax_combined_backend_10_10_plan.md)
- [`/Users/suhjungdae/code/columbia/simsopt-jax/gpu_jax_pro.md`](/Users/suhjungdae/code/columbia/simsopt-jax/gpu_jax_pro.md)
