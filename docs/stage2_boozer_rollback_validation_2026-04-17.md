# Independent Validation: Stage 2 Boozer Rollback Hazard

Date: 2026-04-17

## Verdict

The core issue is validated.

The Stage 2 iota hot loop in the current `simsopt-surrogate` worktree lacks a
`boozerQA.py`-style rollback guard, while SIMSOPT's `BoozerSurface` solvers do
persist failed terminal iterates into `self.res` and `self.surface` and clear
`need_to_run_code`. That combination is a real correctness hazard for both the
soft and ALM Stage 2 iota paths.

However, two parts of the revised draft were too strong and needed correction:

1. `surface.x` is not a live alias in this codebase. `Optimizable.x` returns a
   freshly concatenated array at `src/simsopt/_core/optimizable.py:1039-1045`.
   Taking `prev_surface_x = surface.x` already snapshots values. Adding
   `.copy()` is harmless, but not required for alias safety.
2. Explicitly calling `boozer_surface.recompute_bell(None)` after restoring
   `surface.x` is not required in the current dependency graph. Assigning
   `surface.x` routes through `DOFs.free_x`, which calls `_flag_recompute_opt()`
   at `src/simsopt/_core/optimizable.py:309-324`, and that propagation already
   re-arms dependent recomputation, including `boozer_surface.need_to_run_code`.

I also did **not** validate one downstream claim in the draft as written:

- The statement that SciPy L-BFGS-B specifically retains rejected-step gradient
  information in its secant history "unless the step is explicitly flushed" is
  plausible, but was not directly proven in this pass from the `_lbfgsb` C core.
  What *is* validated is that SciPy maintains correction-pair workspace
  (`s`, `y`) across iterations in `_minimize_lbfgsb` at
  `scipy.optimize._lbfgsb_py`, so a bad gradient is a real quasi-Newton-state
  risk. The exact rejected-step update semantics were not traced to ground here.

## Core Claims: Validated

### 1. The Stage 2 monkey-patch is timing-only

Confirmed at `examples/single_stage_optimization/banana_opt/stage2_objectives.py:184-193`.

`build_stage2_iota_runtime(...)` wraps `boozer_surface.run_code` only to record
`stats.runtime_calls` and `stats.runtime_seconds`. There is no state snapshot,
no success check, no exception handling, and no restore path.

### 2. The Stage 2 soft path has no rollback guard

Confirmed at:

- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:609-626`
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1452-1463`
- `src/simsopt/geo/surfaceobjectives.py:955-978`

`make_stage2_fun(...).fun(...)` calls `J = JF.J()` and `grad = JF.dJ()`
unconditionally.

In soft mode, `soft_iota_objective` is already embedded directly into `JF`, so
the iota penalty is evaluated before the outer closure can inspect anything.
That evaluation flows through:

`QuadraticPenalty(Iotas(...)).dJ()` -> `Iotas.dJ()` -> `Iotas.compute()`

and `Iotas.compute()` reads:

- `booz_surf.res['iota']`
- `booz_surf.res['G']`
- `booz_surf.res['PLU']`
- `booz_surf.res['vjp']`

with no success check.

### 3. The Stage 2 ALM path has no rollback guard

Confirmed at `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1076-1088`.

`_evaluate_stage2_alm_problem(...)` calls:

- `evaluate_stage2_iota_state(stage2_iota_runtime)`
- `stage2_iota_runtime.penalty_objective.dJ()`

Both use the same unguarded `Iotas.compute()` path. So if the terminal Boozer
iterate is bad, `iota_state`, `iota_violation`, `iota_signed_value`, and
`iota_grad` are all derived from that failed state.

### 4. SIMSOPT persists failed terminal iterates and clears `need_to_run_code`

Confirmed at all of the following sites:

- `src/simsopt/geo/boozersurface.py:684-685`
- `src/simsopt/geo/boozersurface.py:767-768`
- `src/simsopt/geo/boozersurface.py:865-866`
- `src/simsopt/geo/boozersurface.py:955-956`
- `src/simsopt/geo/boozersurface.py:1129-1130`

These correspond to:

- `minimize_boozer_penalty_constraints_LBFGS`
- `minimize_boozer_penalty_constraints_newton`
- `minimize_boozer_penalty_constraints_ls`
- `minimize_boozer_exact_constraints_newton`
- `solve_residual_equation_exactly_newton`

All of them assign `self.res = res` and then set `self.need_to_run_code = False`
without conditioning that state transition on `res['success']`.

So the solver contract is effectively:

- write terminal iterate to object state
- record success/failure in `self.res['success']`
- expect the caller to inspect it

The Stage 2 hot loop does not inspect it.

### 5. `boozerQA.py` explicitly snapshots and restores

Confirmed at `examples/2_Intermediate/boozerQA.py:95-110`.

The canonical example saves:

- `sdofs_prev`
- `iota_prev`
- `G_prev`

then restores them and returns `J = 1e3` if `boozer_surface.res['success']` is
false.

### 6. The soft iota penalty is baked into `JF`

Confirmed at `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1452-1463`.

Because `soft_iota_objective` is added to `JF` before the call to `minimize(...)`,
the Stage 2 soft outer closure cannot intervene between:

- "Boozer solve failed"
- "gradient was consumed from failed state"

That failure already happens inside `JF.J()` / `JF.dJ()`.

## Why The Hazard Is Real

The core validated mechanism is:

1. A failed Boozer solve can leave `self.res` holding a terminal iterate with
   `success=False`.
2. `Iotas.compute()` still reads `res['PLU']` and `res['vjp']` from that state
   at `src/simsopt/geo/surfaceobjectives.py:964-978`.
3. The adjoint backsolve `forward_backward(P, L, U, dJ_ds)` then builds an
   iota gradient from that failed terminal linearization.

That is sufficient to establish a genuine gradient-quality hazard.

What remains an inference, not a directly validated local fact, is the exact
degree to which a single bad gradient pollutes SciPy's later quasi-Newton
correction history. SciPy's L-BFGS-B driver does maintain correction-pair
workspace across iterations, but I did not trace the precise rejected-step
update rule in the `_lbfgsb` core during this pass.

## Validated Fix Shape

The fix direction is correct, with two adjustments.

### 1. Add a guarded Stage 2 Boozer evaluator

Recommended.

Implement a Stage 2-local helper that:

- snapshots the last good surface dofs and iota/G guesses
- runs the Boozer solve
- restores the prior state if the solve raises or returns `success=False`
- exposes `solve_failed` as an explicit signal to the caller

This should live in the Stage 2 seam, not in global `BoozerSurface`.

### 2. Soft mode should stop embedding raw `penalty_objective` inside `JF`

Recommended.

Because the current soft path bakes the iota penalty into `JF`, the outer `fun`
closure cannot block bad-state gradient consumption.

The safer structure is:

- keep the non-iota objective in `JF`
- evaluate iota through the guarded helper inside `make_stage2_fun(...)`
- on solve failure, return a reject objective and avoid consuming a bad iota
  gradient

### 3. ALM should avoid reading `penalty_objective.dJ()` from a failed state

Recommended.

At minimum:

- set an explicit `solve_failed` flag
- avoid calling `penalty_objective.dJ()` on failure
- reject the subproblem iterate

If the ALM driver is widened, a dedicated `solve_failed` channel is cleaner than
encoding every failure as a fake huge violation.

## Corrections To The Drafted Refinements

### Snapshot contents

The draft was directionally right but overstated two details.

- A shallow `dict(res)` is not a strong snapshot of `res`, because it keeps
  references to objects such as `OptimizeResult`, `PLU`, and closures.
- But `surface.x` itself is already value-snapshotting in this codebase, so the
  specific aliasing warning was incorrect.

The cleanest restore target is not a deep copy of all of `res`. It is a minimal
restore payload:

- previous surface dofs
- previous `iota`
- previous `G`
- optionally previous `success`

After restoring `surface.x`, the recompute propagation re-arms the next solve,
so `PLU` / `vjp` / `info` can be rebuilt rather than snapshotted.

### Cache clearing after restore

The draft said an explicit `boozer_surface.recompute_bell(None)` should be
fired because assignment-driven bells might not run.

That claim is too strong for the current code:

- `surface.x = ...` routes through `Optimizable.x` ->
  `opt.local_x = ...` -> `DOFs.free_x = ...`
- `DOFs.free_x` always calls `_flag_recompute_opt()`

So restore-time recomputation is already wired.

An explicit `recompute_bell(None)` is still acceptable as belt-and-suspenders,
but it is not required by the current implementation.

## Additional Concerns: What Validated And What Did Not

### A. Basin-hopping is in scope

Validated.

Penalty-mode basin hopping at
`examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1761-1786`
reuses the same `fun` closure built by `make_stage2_fun(...)`.

So one rollback fix in the Stage 2 soft objective path covers both:

- plain penalty-mode L-BFGS-B
- penalty-mode basin hopping

There is no separate `fun_basin` path to patch.

### B. Mid-optimization Stage 2 checkpoint contamination

Not validated.

I did not find a current Stage 2 mid-optimization checkpoint/resume path in
`banana_coil_solver.py` or `run_stage2_alm.py`.

What I *did* confirm is:

- `banana_coil_solver.py` writes final `results.json` at
  `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:2031-2032`
- `run_stage2_alm.py` writes a wrapper summary at
  `examples/single_stage_optimization/run_stage2_alm.py:685-686`

So the checkpoint/resume concern may be reasonable as a future design review
item, but it is not a validated current-path bug in this pass.

### C. The probe path should stay report-oriented

Validated in spirit.

`probe_stage2_seed_bootability(...)` at
`examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py:550-634`
uses the same `attempt_initialize_boozer_surface(...)` seam as the Stage 2
runtime bootstrap, but it is a one-shot diagnostic probe, not a hot loop.

So the design guidance is correct:

- keep the hot-loop guard for optimization paths
- do not silently mask probe failures that should be reported as bootability
  failures

## Regression Test Guidance

The two original regression directions remain good:

1. a fake Boozer surface that mutates state and returns `success=False`,
   proving restore-on-failure works
2. one soft-mode and one ALM-mode Stage 2 test proving failed solves are
   rejected without consuming failed-state gradients

A third test is also worthwhile:

3. an alternating success/failure fake solve in soft mode, asserting the live
   Stage 2 state after multiple evaluations matches the last successful solve,
   not the last attempted failed solve

## Validation Limits

I validated the code paths statically in the current repo and checked the SciPy
Python driver for L-BFGS-B state retention behavior.

I did not complete a local pytest run for this note because the test environment
in this workspace is missing `monty`, which currently blocks test import.

## Bottom Line

The core rollback concern is real and validated.

The correct production fix is to add a Stage 2-local guarded Boozer evaluation
seam and route both soft-mode and ALM iota handling through it.

The revised draft was right on the main bug, but it overstated:

- `surface.x` alias risk
- the necessity of an explicit restore-time bell
- the current existence of a Stage 2 checkpoint/resume contamination path
- the exact L-BFGS-B rejected-step-history semantics as a directly validated fact
