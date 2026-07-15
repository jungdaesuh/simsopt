---
date: 2026-07-15
problem: A tiny FP64 A100 diagnostic was slower than native C++ despite older records showing GPU wins.
tags: [benchmarking, jax, gpu, scientific-validation]
---

# Resolve performance contradictions by locating the workload crossover first

## Problem

The current 15x15 diagnostic measured native C++ near 1.67 seconds and FP64 JAX
on A100 near 23 seconds, apparently contradicting older faster-than-C++ records.
The result could have been misclassified as a regression because hardware,
precision, and repository SHA alone did not identify the benchmark regime.

## Dead ends

- Comparing only headline ratios mixed tiny synthetic solves with production-size
  `J+dJ` measurements.
- Blaming compilation was inconsistent with the harness: first-call compilation
  was reported separately and the 23-second value was a warmed fresh solve.
- Reusing the historical A100 7.3x headline was invalid. Its raw CPU reference
  failed, used a different seed contract, reached a different outcome, and marked
  `supports_performance_headline=false`.
- Treating an unconverged timing as a performance verdict ignored that C++ and
  JAX ended at different objective and iota values.

## Working approach

1. Anchor comparisons to the same SHA, GPU model, precision, configuration, and
   timed scope.
2. Recover the same-run size sweep rather than comparing isolated headlines.
3. Check convergence and final-state comparability before computing a claim-grade
   ratio.
4. Separate scaling evidence from performance claims.

The same SHA and A100 showed the crossover directly: JAX was slower at 15x15
(23.2714 versus 1.6590 seconds), faster at 64x64 (14.8778 versus 20.1380 seconds),
and faster at 128x64 (52.4649 versus 223.3622 seconds). The current tiny result
therefore reproduced prior behavior rather than introducing a slowdown.

## Why it worked

GPU execution has fixed launch, dispatch, and on-device optimizer overhead that
small problems cannot amortize. Increasing quadrature and problem size raises the
parallel work per dispatch, producing a crossover. This explains the size trend;
exact kernel-level attribution still requires profiling. Because the size-sweep
solves were unconverged, it establishes diagnostic scaling but not a scientific
replacement claim.

## Reusable rule

When a new GPU timing contradicts an older record, first match SHA, hardware,
precision, workload size, timed scope, convergence, and final state; then sweep
workload size. Do not call the difference a regression or speedup if any timed
solve failed or the compared endpoints differ.

## Pointers

- `benchmarks/run_code_benchmark_common.py:360`
- `benchmarks/run_code_benchmark_common.py:502`
- `.artifacts/perlmutter/fp64_cpp_5cff05449_20260714T2145Z/`
- `.artifacts/perlmutter/fixall_final6_5cff05449_60d9cc38_20260715/`
- Commit `ef4c86815`
