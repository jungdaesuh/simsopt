# Public quasi-Newton routing inventory

Snapshot: `9c64c2ef6cee45eb7eb1989bd5a41e2adf8bfc26` (dirty worktree,
2026-08-01). The inventory covers the public method names and the callers that
forward their options or consume their results.

| Surface | Owner/callers | Contract covered |
| --- | --- | --- |
| `Driver.SIMSOPT_BFGS` / `Driver.SIMSOPT_LBFGSB` | `src/simsopt_jax/solve/driver.py`, `src/simsopt_jax/solve/dispatch.py` | typed-driver to `bfgs-ondevice` / `lbfgs-ondevice` mapping |
| `method="bfgs-ondevice"` / `method="lbfgs-ondevice"` | `src/simsopt_jax/geo/optimizers/optimizer.py:target_minimize` | target-lane validation, value-and-gradient requirement, callback and option forwarding |
| `method="lbfgs-trace"` | `src/simsopt_jax/geo/optimizers/reference.py:reference_minimize` | host/reference traced whole-solve route remains distinct |
| `lbfgs_run_mode` | `optimizer.py`, `private/_lbfgs.py`, `src/simsopt_jax_adapters/geo/boozer_surface.py` | `stepwise` default; explicit `monolithic_debug` compatibility route |
| `maxcor`, `maxfun`, `maxiter`, `ftol`, `gtol`, `maxls` | `private/_lbfgs.py`, `private/_lbfgsb_scipy.py`, `private/_types.py` | SciPy-compatible limits, line-search controls, and fixed-shape history |
| seeded value/gradient | `optimizer.py`, `private/_lbfgs.py`, `src/simsopt_jax_adapters/geo/boozer_surface.py` | initial objective/gradient are accepted without rebuilding the public objective |
| callback/progress/trace hooks | `runtime/host_boundary.py`, `private/_lbfgs.py`, `private/_bfgs.py` | one audited accepted-step host packet; traced callback route remains explicit |
| `status`, `success`, `nit`, `nfev`, `njev`, `hess_inv` | `private/_result_converters.py`, `private/_lbfgsb_scipy.py`, `solve/contracts.py` | result fields and inverse-Hessian adapter remain SciPy-shaped |
| direct application callers | `examples/jax/3_Advanced/single_stage_optimization.py`, `src/simsopt_jax/examples/strain_optimization.py` | both use the unchanged custom public method names |
| Boozer callers | `src/simsopt_jax_adapters/geo/boozer_surface.py` | limited-memory selection still resolves to `lbfgs-ondevice`; optional Optax/Optimistix lanes remain explicit |

The contract is exercised by `tests/jax/solve/test_driver_dispatch.py`,
`tests/jax/solve/test_compat_shim_translation.py`,
`tests/geo/test_boozersurface_jax.py`,
`tests/geo/test_boozersurface_jax_private.py`, and
`tests/jax/solve/test_custom_quasi_newton_step_runtime.py`. A repository
search for `lbfgs_run_mode`, `target_minimize(`, and the public method strings
found no additional production callers outside the rows above.

This is an inventory and routing-compatibility record, not a claim that full
application-scale endpoint parity or GPU qualification is complete.
