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

## Addendum — `fused_stepwise` route (2026-08-03, review iteration)

The performance campaign added a third `lbfgs_run_mode` value after the
snapshot above:

| Surface | Owner/callers | Contract covered |
| --- | --- | --- |
| `lbfgs_run_mode="fused_stepwise"` | `private/_lbfgs.py` (`_check_lbfgsb_run_mode`, fused preparation branch); benchmark runner `benchmarks/custom_quasi_newton_runtime.py:_solver_route` (fast intent only); `benchmarks/lbfgs_ondevice_compile_shape.py`; `benchmarks/lbfgs_warm_soak.py` | whole-solve on-device `lax.while_loop` driver over the existing macro-step transitions; callbacks are rejected fail-closed (callback-capable callers stay on `stepwise`); fused preparation compiles exactly `initial_state`, `value_and_grad`, and `fused_solve` (no result-payload executable) |

No public application surface routes to `fused_stepwise` implicitly: the
public optimizer (`optimizer.py`) and the Boozer adapter
(`boozer_surface.py`) still default to `stepwise`, and the fast route is
selected only by the benchmark runner's explicit `--intent fast` lane and
the two diagnostic scripts above (verified by repository search for
`fused_stepwise` on 2026-08-03; production callers are exactly the files
listed). Receipt validation enforces the route string per provider in
performance qualifications (`custom_quasi_newton_receipts.py`).

## Addendum — computed routing defaults supersede the 2026-08-03 closure (2026-08-14)

As of 2026-08-14, `src/simsopt_jax/solve/dispatch.py` is itself a
`fused_stepwise`-selecting public surface: its `_legacy_lbfgsb_options`
computes `lbfgs_run_mode="fused_stepwise"` for observer-free
`Driver.SIMSOPT_LBFGSB` solves (`"stepwise"` only when a `callback` is
attached), and `private/_bfgs.py`'s `_minimize_bfgs_private` routes
unobserved concrete BFGS solves to the fused whole-solve program (the eager
driver is retained only for `callback`/`progress_callback`/
`memory_analysis_callback` callers). This supersedes the 2026-08-03
addendum's closing claim that "production callers are exactly the files
listed" for `fused_stepwise` as of that date: the typed dispatch lane and
the concrete BFGS entry point are now themselves `fused_stepwise`-routing
production surfaces rather than opt-in callers. CPU bitwise evidence:
`tests/jax/solve/test_lbfgsb_dispatch_run_mode.py` and
`tests/jax/solve/test_bfgs_host_fused_gate.py`.

The benchmark BFGS lane (`benchmarks/custom_quasi_newton_runtime.py`,
`method == "bfgs"` branch) takes the compiled-step memory report from an
untimed one-iteration probe solve
(`_bfgs_compiled_step_memory_analysis`, executed inside the existing
`algorithm_memory_analysis` RSS phase — whose own `phase_rss` entry therefore
now includes one XLA compile plus a one-iteration solve where it previously
held pure arithmetic) and leaves both timed solves
unobserved, so the benchmark's custom BFGS lane measures the production fused
program. Measured on the `bfgs_quadratic` fixture at `maxiter=5`: the warm
transfer audit went from 5 `advance` observations to 0, and the objective's
private-solver cache gained the fused `bfgs` key it previously never built
(it held `bfgs-eager-runtime` alongside the shared value-and-gradient keys; after the probe was decoupled from
`_run_custom` the eager key is absent entirely — the cache now holds
`bfgs`, `bfgs-scalar-value-and-grad`, and
`bfgs-value-and-grad-closure-converted`). The persisted `solver_route` string
is unchanged — `custom_bfgs_private` names the emitting solver and its status
vocabulary, not the eager/fused driver within it — and that lane's
`algorithm_memory_contract` still carries every `compiled_step_*` field,
which is budget-independent because it measures one lowered step. The probe
is gated to the only rows that can carry its report: a `custom`-provider,
non-accepted-incumbent BFGS child (`provider == "custom" and not
accepted_incumbent_bfgs`). Native rows therefore neither execute a
one-iteration custom solve inside their `algorithm_memory_analysis` RSS phase,
nor mutate the shared fixture objective, nor carry `compiled_step_*` fields
(a native row has no compiled step to describe), and where the probe does run
it is
fail-closed: a probe that ends before its first transition raises rather than
silently dropping the fields. Provenance, mirroring the
`dense_update_compiled_memory_is_update_only` flag on the dense-update
numbers: the `compiled_step_*` fields describe one lowered step of the
eager/observer step program, while the same row's `cold_seconds`/`warm_seconds`
describe the fused whole-solve route. Timing caveat for that lane: the probe
runs before `solver_start_rss_kib` and outside every timed phase, and because
it drives `_minimize_bfgs_private` directly — without the timed lane's
cacheable-objective marker — it leaves the objective's solver cache untouched,
so `cold_seconds` still pays for the value-and-gradient programs and for the
fused whole-solve program compiled inside `cold_solver`. Solver peak RSS is
NOT insulated from the probe: `solver_peak_rss_kib` is absolute process RSS
over the `preparation`/`cold_solver`/`warm_solver` phases, and the untimed
probe raises the process floor those phases inherit (measured on the
unpublished `bfgs_quadratic` fixture: absolute peak +~4%,
`solver_peak_rss_delta_kib` collapses from ~145 MB to ~28 MB because the
probe pre-pays the solver's allocations). Both RSS fields are recorded
diagnostics, not gated on, and no published BFGS receipt row runs the probe
(`boozer` rows are accepted-incumbent).

Formulation note for the same-date Stage-II refactor: splitting the length
penalty out of the coil-forces stage objective
(`examples/jax/3_Advanced/coil_forces.py` and its aligned parity mirror
`examples/jax/parity/cases/native_coil_forces.py`) is bitwise invariant
against the pre-change formulation only while the total base-curve length
stays under the 17.4 m target, where the penalty is exactly zero. Once the
penalty is active the two spellings reassociate the same sum and their values
differ at the ~1 ULP level; the shipped start point is under the target, so
the bitwise receipt holds there
(`tests/jax/examples/test_duplicate_compile_fixes.py`).
