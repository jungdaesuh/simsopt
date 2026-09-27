# Pitfalls

Each entry: the symptom, the cause, the fix.

1. **Sign convention.** A row is feasible when `g <= 0`. `q >= bound` is
   `g = bound - q` (`signed_lower_bound`), `q <= bound` is `g = q - bound`
   (`signed_upper_bound`). A flipped row drives the design into the region
   it should avoid, often with a `converged` result. `sign_check.py` catches
   it when each row is probed on both sides.
2. **Scales.** The penalty is shared by all rows and the tolerances are
   absolute. Rows in raw units (a distance in m next to a curvature in 1/m and
   a squared curvature) differ by orders of magnitude, so one row dominates
   each subproblem and the others converge late or never. Divide each row by
   its bound or typical size, and f by a reference value such as f(x0);
   `sign_check.py` warns when gradient norms spread over more than 1e3 or a
   value is far from O(1).
3. **`maxiter` is a whole-run budget.** `inner_options["maxiter"]` counts
   L-BFGS-B iterations over every subproblem of one `minimize_alm` call, not
   per subproblem. A small value ends the run early with the last step's
   action as the termination reason (`dual_update`, `penalty_increase`, ...,
   see [termination.md](termination.md)).
4. **Cached evaluator with stateful physics.** `cached_alm_evaluator` returns
   the physics of the first evaluation at a bitwise-equal x. A warm-started
   inner solve (Boozer surface, VMEC restart) can land elsewhere on a second
   solve, so caching hides that and the solver's accept/restore bookkeeping
   goes wrong. Stateful physics: evaluate fresh every call and pass
   `snapshot_accepted_state_fn` with `restore_incumbent_state_fn`.
5. **Stale cache.** A cached evaluator must be told when anything besides x
   that the physics reads changes (a threshold, a smoothing temperature, a
   target): call `evaluator.cache_clear()`.
6. **Hybrid quartet is all or none.** `hard_signed_constraint_values`,
   `hard_violation_values`, `surrogate_signed_constraint_values` and
   `hard_dual_update_values` come together (a missing one raises `KeyError`)
   with the shape of `constraint_values`. Expect `signal_mismatch_*` reasons
   near the boundary, where the conservative smooth value reads active while
   the hard one is feasible: smooth rows alone (no quartet) are the default.
7. **Cyclic metadata is rejected.** An evaluator dict, or `ALMPhysics`
   extras, that contains itself (a dict holding itself, a list inside itself)
   raises `ValueError` naming the path. A subtree shared by two keys is fine.
   `ALMPhysics` extras also may not set `total`, `grad`, the four physics
   fields or other multiplier-dependent keys.
8. **Non-finite values.** A NaN or inf at a trial point is rejected (the
   line search backtracks); at an outer iterate (the start point, a restored
   incumbent) `minimize_alm` raises `ValueError: ... produced non-finite ALM
   data`. Make x0 evaluate cleanly.
9. **The returned x may be an earlier iterate.** On failure the solver can
   return the best hard-feasible iterate (`result.restored_best_feasible`).
   Set your objects to `result.x` (and, for stateful physics, re-solve from
   the restored state) before saving; the templates' `finish` does this.
10. **Callbacks come in pairs.** `snapshot_accepted_state_fn` and
    `restore_incumbent_state_fn` are both given or both omitted
    (`ValueError`); `resume_from` cannot be combined with
    `initial_multipliers` or `initial_penalty`, and needs the checkpoint's x
    as x0.
11. **Taylor-testing only the augmented Lagrangian misses rows.** With zero
    multipliers an inactive row (`max(0, multiplier + penalty * g) = 0`)
    drops out of L and its gradient goes unchecked; `gradient_check.py` tests
    f and each row separately.
12. **Taylor steps and smoothing.** The smooth rows select points near the
    extremum; a step that changes the selection breaks the ratio test. Use
    steps far below the smoothing temperature (the templates use 1e-5 to
    2.5e-6).
13. **Three ways to pass the gradient check.** `gradient_check.py` reports a
    `verdict` per quantity: `ratio_test` (the error falls with the step),
    `no_ratio` (the error is at the round-off floor at every step, so no
    ratio exists: the difference is exact for a linear or quadratic
    quantity such as a linear row, or the truncation error at these small
    steps is already below the floor, as for a coil-length row) and
    `accuracy` (the ratio test fails at round-off
    but the smallest error is within 1e-6 relative). All three mean the
    gradient matches; `vacuous` (zero derivative) means nothing was tested,
    and `failed` means the gradient is wrong.
14. **Unique row names.** The runner reports multipliers and values keyed by
    name, so a repeated name hides a row.
15. **Conflicting constraints.** Thresholds no design can meet (e.g. a coil
    spacing and a coil-surface distance that exclude each other) show as a
    penalty that keeps rising, `penalty_cap_reached`, or `max_outer_after_penalty_increase`
    with one row's violation flat. Relax a threshold; a larger penalty does
    not fix infeasibility.
