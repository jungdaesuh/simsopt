# JAX LS Newton persist — public API contract (2026-08-20)

**Status: documentation closure for `df0a46a9f`.** Not a nested-LS GPU
claim. The implementation is native-predicate-aligned persistence with
JAX-specific finite-state and no-move rules, not byte-identical C++ Newton.

## Observable behavior delta

`BoozerSurfaceJAX.minimize_boozer_penalty_constraints_newton` now persists
the polished iterate when `_boozer_iterate_is_persistable(success, ‖∇J_LS‖₂,
‖∇J_LS(x₀)‖₂)` holds **and** the iterate's `x` and `∇J` are finite, or when
the iterate is an exact no-move. Otherwise it restores the start surface,
`(ι, G)`, `fun`, and `jacobian`, reports `success=False`, and drops the
last-iterate Hessian (`hessian is None`).

JAX still differs from C++ in globalization (Armijo vs full step), Hessian
recompute-at-`x₀` on rollback, residual representation (`residual` is the
long LS vector), result fields, and unsuccessful no-move handling.

Rollback sets `success=False`, so `get_adjoint_runtime_state()` rejects the
operator VJP even though the callback objects remain on the result.

## Caller inventory

Production and test callers of the public Newton method (JAX and native
namesake):

| Caller | Action |
| --- | --- |
| `src/simsopt/geo/boozersurface.py` `run_code` | Native BFGS then Newton. Unchanged. |
| `src/simsopt_jax_adapters/geo/boozer_surface.py` `run_code` | JAX LS path. Observes the new persist rule. |
| `src/simsopt_jax_adapters/geo/nested_ls_newton_parity.py` | Reconstruct-bar pair API. No action. |
| `tests/geo/test_nested_ls_newton_parity.py` | Compatibility tests for persist and pair. No action. |
| `tests/geo/test_boozersurface_jax.py` | Invalid-Newton restore and public result identity. No action. |
| `tests/geo/test_boozersurface.py` | Native Newton. Unchanged. |
| `benchmarks/run_code_benchmark_common.py` | Benchmark Newton. No action: consumes `success` / `iota` / `G`. |
| `benchmarks/production_boozer_parity_probe.py` | Same. |
| `benchmarks/_cpp_compatible_probe.py` | Same. |
| `benchmarks/cpp_baseline_benchmark.py` | Native. Unchanged. |

In-graph `simsopt_jax.geo.optimizers.optimizer.newton_polish` is **not** a
caller of this persist wrapper and was not changed.

## Migration / no-action

Callers that already branch on `success` and read `iota`, `G`, `fun`,
`jacobian`, and the live surface need no code change. Callers that assumed
JAX always committed the last iterate even when the gradient worsened must
treat rollback as `success=False` plus restored start state. There is no
deprecation window: the previous always-commit behavior was the defect.

## Compatibility tests

`tests/geo/test_nested_ls_newton_parity.py` locks rollback, success-true
worse-gradient commit, NaN-`x` / NaN-`∇J` rollback, unmoved Hessian keep,
and persist on `‖∇J_LS‖₂` not `J_LS`.
`tests/geo/test_boozersurface_jax.py` locks invalid-Newton restore of the
LBFGS surface.

## Rollback plan

The persist **implementation** is commit `df0a46a9f`. Revert that commit
(`git revert df0a46a9f`) to restore always-commit JAX Newton. This
documentation file may ship in a later commit with the reduced nested-LS
track; reverting that later commit does **not** revert persist. No
persisted artifacts or on-disk state depend on the new rule.

## Ruff

The repository-pinned Ruff version is the CI gate. Ruff 0.16.1 reports
additional pre-existing categories beyond I001/FA102 on this tree; the pin
bump is deferred and is not part of this persist contract.
