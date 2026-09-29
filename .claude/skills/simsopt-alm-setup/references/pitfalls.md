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
   its bound or typical size, and f by a positive reference value such as
   |f(x0)| (a negative divisor turns minimization into maximization; the
   templates use `ZERO_OBJECTIVE_SCALE` when f(x0) = 0);
   `sign_check.py` warns when gradient norms spread over more than 1e3 or a
   value is far from O(1). Remove large constant offsets from f as well:
   `converged` certifies the implemented approximate-KKT tests at the
   returned point, not global optimality, not descent from the start, and
   not the same basin under a large constant offset. Step acceptance compares
   the reported totals with a tolerance of a fraction of the step's
   first-order size (no round-off allowance), and L-BFGS-B's own `ftol` stop
   is relative to max(1, |f|), so both see an offset once |f| dwarfs f's
   variation: at float resolution a tie can lead to a different local KKT
   point. Example: f = C + 0.001 x - sin(2 pi x)/pi on [0, 1] from x = 0.
   In the golden recording environment, with C = 0 the run converges in the
   interior, x = 0.2499; with C = 6e13 (spacing 0.0078125) f(0) and f(1) tie
   though f(1) - f(0) = +0.001, and it converges at the bound KKT point x = 1.
   Other SciPy builds may pick either local KKT point; the certificate holds
   at whichever point is returned. A snap onto a bound is kept only
   on a strict reported decrease, so a near-bound improvement the totals
   cannot represent stalls instead.
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
   the hard one is feasible by more than the feasibility gate (channels that
   agree do not mismatch): smooth rows alone (no quartet) are the default.
7. **Cyclic metadata is rejected.** An evaluator dict, or `ALMPhysics`
   extras, that contains itself (a dict holding itself, a list inside itself)
   raises `ValueError` naming the path. A subtree shared by two keys is fine.
   `ALMPhysics` extras also may not set `total`, `grad`, the four physics
   fields or other multiplier-dependent keys.
8. **Hand-built evaluator dicts.** A dict without `constraint_grads` (or
   with it None) raises `KeyError`, even when no row is active; one with the
   wrong number of rows or shapes raises `ValueError`. A
   `search_step_success` that is not a `bool` or `numpy.bool_` (0, None, a
   string) raises `ValueError`, and so does a negative
   `constraint_activity_tolerances` entry. Build the dict with `ALMPhysics.evaluation`
   or `augmented_inequality_objective`, as the templates do, and pass any
   step flag as `bool(...)`.
9. **Non-finite values.** A NaN or inf at a trial point is rejected (the
   line search backtracks); at an outer iterate (the start point, a restored
   incumbent) `minimize_alm` raises `ValueError: ... produced non-finite ALM
   data`. Make x0 evaluate cleanly.
10. **The returned x may be an earlier iterate.** On failure the solver can
    return the best hard-feasible iterate (`result.restored_best_feasible`).
    Set your objects to `result.x` (and, for stateful physics, re-solve from
    the restored state) before saving; the templates' `finish` does this.
11. **Callbacks come in pairs.** `snapshot_accepted_state_fn` and
    `restore_incumbent_state_fn` are both given or both omitted
    (`ValueError`); `resume_from` cannot be combined with
    `initial_multipliers` or `initial_penalty`, and needs the checkpoint's x
    as x0.
12. **Taylor-testing only the augmented Lagrangian misses rows.** With zero
    multipliers an inactive row (`max(0, multiplier + penalty * g) = 0`)
    drops out of L and its gradient goes unchecked; `gradient_check.py` tests
    f and each row separately.
13. **Taylor steps and smoothing.** A kernel's smoothing temperature must
    lie in [1e-100, 1e100], and every sample coordinate must be finite with
    |x| <= 1e100 (`ValueError` otherwise, naming the bound and the value; 0 is
    not the exact value, which is the kernel's third item). Values are
    accurate to a small multiple of machine epsilon times the row's scale,
    gradients to that over T (api.md). The smooth rows are a log-sum-exp over every sample
    (curvature) or point pair (distances), smooth everywhere, but they bend
    on the scale of the temperature, so only steps that move the constrained
    quantity far below the temperature see the gradient.
    `gradient_check.py` sweeps relative steps from 1 down to 1e-10; when
    larger steps plateau at the slope of the extremum and smaller
    ones converge to the gradient, the two ranges disagree and the row is
    NOT TESTED rather than failed.
14. **Reading the gradient check.** `gradient_check.py` sweeps relative
    steps from 1 to 1e-10 (each dof moves relative to its own size). Per
    direction it finds the step ranges that converge: three or more
    consecutive steps agreeing to 1e-4, or all zero to the round-off of the
    evaluated values (their machine epsilon times the largest value over the
    stencil, over the step; float32's epsilon when every value is a float32
    number). An FD check cannot know which range holds the derivative when
    ranges converge to different values, so then the verdict is `NOT
    TESTED`, listing the ranges: a float32 or quantized term, a warm-started
    inner solve that returns its start below its tolerance, or a kink makes
    small steps see a different function than large ones. Larger steps
    agreeing only to 5e-4 count as a range too, and so do two consecutive
    steps agreeing with the claim to 1e-6; a range of zeros cannot judge
    while any step resolves a nonzero difference. When the ranges agree on
    one value, the claim `FAIL`s if it misses that value by more than 5e-4
    plus the scatter of the smaller steps (noise of the evaluation can bias
    a range that much), `PASS`es if two consecutive steps inside a range
    agree with it to 1e-6 and every range converged to 1e-6 agrees with it
    to 1e-6, and is `NOT TESTED` otherwise. A zero value passes a claim
    inside its round-off only when that round-off is within 1e-6 of the
    problem's derivative scale; a nonzero claim never passes through a
    floor. So these are NOT TESTED whatever the claim: a clipped row whose
    bound the sweep reaches (larger steps see the active slope); `(F + x) -
    F` with F large enough that the computed quantity is flat at float
    resolution at small steps; and a quantity evaluated in float32 whose
    small steps are flat (its values are float32 numbers, so the check uses
    float32's round-off). Known limits: a kink exactly at x0 (e.g. `max` at
    a tie) FAILs, because the central difference converges to the average of
    the one-sided slopes. simsopt runs JAX in float64 (importing simsopt.geo
    enables x64), but your own evaluator may compute terms in float32 (your
    own JAX code without x64, float32 NumPy arrays, a GPU or ML-surrogate
    term): a float32 term inside a float64 quantity is not recognised: when
    the float32 term has a large value-to-slope ratio (a value of 1e3 or
    more at slope 1, or an O(1) value at slope 1e-2 or less) its computed
    contribution is flat at float32 resolution at small steps, so the right
    gradient can FAIL and one that leaves the float32 term out can PASS.
    Check gradients with that evaluator in float64 (in your own JAX code,
    `jax.config.update("jax_enable_x64", True)`). A steep quantity on a
    large offset (say `1e4 + sin(1e3 x)/1e3`) can be NOT TESTED, because
    round-off and truncation leave no pair of steps accurate to 1e-6;
    relative noise of 1e-4 or more (an inner solve with a loose tolerance)
    seldom leaves three steps converged to 1e-4, so a wrong gradient there
    is mostly NOT TESTED rather than FAIL; an exactly zero row cannot pass a
    round-off-size claim such as 1e-17 (NOT TESTED); and a problem whose
    objective has no gradient at x0 (a constant) has no scale for zero, so
    zero claims are NOT TESTED.
15. **Unique row names.** The runner reports multipliers and values keyed by
    name, so a repeated name hides a row; `gradient_check.py` and
    `sign_check.py` refuse one.
16. **`converged` is an approximate KKT point.** Success means feasibility
    within `feasibility_tol`, an augmented-gradient norm within
    `stationarity_tol`, and a complementarity gap Σ λ⁺_i max(0, -g_i)
    within `feasibility_tol`, not an exact KKT point. Both tolerances are
    absolute in f's units (an offset added to f changes nothing): scale f
    to O(1), as the templates do by dividing it by |f(x0)|. A large multiplier on a row just inside its bound
    blocks success until dual updates shrink it (`plateau_stall` or a
    `max_outer_*` reason; see [termination.md](termination.md)). Known
    limit: a nonconvex row steep enough that its multiplier times its slack
    stays under the tolerance while its multiplier times its gradient
    cancels f's gradient can still pass; check `result.multipliers`
    against the physics.
17. **Conflicting constraints.** Thresholds no design can meet (e.g. a coil
    spacing and a coil-surface distance that exclude each other) show as a
    penalty that keeps rising, `penalty_cap_reached`, or `max_outer_after_penalty_increase`
    with one row's violation flat. Relax a threshold; a larger penalty does
    not fix infeasibility.
