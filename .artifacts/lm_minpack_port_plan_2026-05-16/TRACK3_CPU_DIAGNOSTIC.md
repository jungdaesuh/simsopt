# Track 3 CPU Diagnostic

Date: 2026-05-17

Scope:

- Implement `least_squares_algorithm="optimistix-lm"` as an optional CPU/JAX
  target lane resolving to `method="optimistix-lm-ondevice"`.
- Use Optimistix `least_squares` with `LevenbergMarquardt` and Lineax `LSMR`.
- Keep Optimistix, Lineax, and Equinox behind the optional `JAX_OPTIMISTIX`
  extra because the packages require Python 3.11 while simsopt still advertises
  Python 3.8+.
- Skip CUDA/GPU validation per objective.

Evidence:

- Direct overdetermined linear least-squares fixture: PASS.
- Direct Rosenbrock residual fixture: PASS.
- Failure-path and unsupported-option contract fixtures: PASS.
- Boozer route resolution and `run_code`/`run_code_traceable` routing: PASS.
- Oversampled Boozer fixture (`ncoils=4`, `nphi=16`, `ntheta=8`):
  Optimistix/LSMR succeeds and matches the current in-tree LM objective and
  residual-norm scale, but does not match endpoint state or residual vector at
  the original `branch-stable-resolve` promotion gate.
- Default near-rank-deficient Boozer fixture: Optimistix/LSMR succeeds and
  remains finite, but does not reduce cost versus the current matrix-free LM.

Focused validation command:

```bash
.conda/jax/bin/python -m pytest \
  tests/geo/test_lm_optimistix_contract.py \
  tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_optimistix_lm_rejects_callbacks_at_option_normalization \
  tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_optimistix_lm_rejects_nondefault_tuning_at_option_normalization \
  tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_resolve_least_squares_optimizer_method_contract \
  tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_routes_lm_least_squares_contract \
  tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_run_code_traceable_ls_routes_lm_ondevice \
  -q
```

Result: `38 passed in 27.19s`.

Decision:

Track 3 is implemented as an experimental diagnostic lane, not a production
promotion. The original CPU endpoint-state and robustness gates are not met in
this tree. `TRACK3_OPTIMISTIX_CONTRACT.md` is the contract source for accepted
inputs, rejected inputs, result fields, and diagnostic-vs-parity meaning.
