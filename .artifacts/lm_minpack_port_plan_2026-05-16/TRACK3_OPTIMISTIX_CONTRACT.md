# Track 3 Optimistix Contract

Date: 2026-05-17

Scope: `least_squares_algorithm="optimistix-lm"` resolves to
`method="optimistix-lm-ondevice"` only on the target on-device least-squares
lane. The lane is diagnostic and is not promoted as a production parity lane.

Accepted inputs:

- `tol`: the single convergence tolerance. It is passed to Optimistix
  `LevenbergMarquardt(rtol=tol, atol=tol)` and Lineax
  `LSMR(rtol=tol, atol=tol)`.
- `maxiter`: the Optimistix nonlinear `max_steps` budget.
- `materialize_dense_linearization` and `max_dense_linearization_bytes`: final
  compatibility artifact controls only.

Rejected inputs:

- `callback` and `progress_callback` on `target_least_squares`.
- `stage_callback` and `progress_callback` on `BoozerSurfaceJAX` when combined
  with `optimizer_backend="ondevice"` and
  `least_squares_algorithm="optimistix-lm"`.
- Non-default `ftol`, `xtol`, or any explicit `gtol`. The lane does not expose
  MINPACK's three independent LM stopping knobs because Optimistix and Lineax
  use relative/absolute tolerances.

Result contract:

- `success` follows Optimistix `RESULTS.successful` and finite final residual,
  state, gradient, cost, and optional dense artifacts.
- Optimistix nonlinear max-step exhaustion maps to `info=5`.
- Non-finite final residual/state/gradient/cost maps to `status=2`.
- `optimistix_result` and `optimistix_result_message` preserve the upstream
  result for diagnostics.
- `nfev` and `njev` keep the legacy `nit + 1` compatibility shape and are
  diagnostic only for this lane; Optimistix 0.1.0 exposes `num_steps` but not
  separate residual/Jacobian evaluation counts in `Solution.stats`.
- `dense_linearization_kind="post_hoc"` means final Jacobian/Hessian artifacts
  were materialized after the Optimistix solve; this is not the same as the
  `lm-minpack-ondevice` dense-QR lane, whose dense linearization is part of the
  solver step.

Validation contract:

- Direct linear and Rosenbrock fixtures check that the wrapper solves ordinary
  problems.
- Failure-path fixtures check callback rejection, unsupported LM tuning
  rejection, nonlinear max-step mapping, and non-finite status mapping.
- Oversampled Boozer fixture checks the intended diagnostic state: objective
  and residual-norm scale match the current in-tree LM lane, but endpoint state
  and residual-vector parity do not meet the promotion gate.
