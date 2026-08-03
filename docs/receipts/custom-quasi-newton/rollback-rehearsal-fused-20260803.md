# Fused-route rollback rehearsal (2026-08-03)

Rehearses the documented `fused_stepwise` rollback lever — distinct from
`rollback-rehearsal-20260802.md`, which predates the fused route and
covers the prepared-runtime commits (`3b2b9f40a`, `41d95cf50`,
`fd200f564`) only.

## Setup

Clean detached worktree at `76f8fdf23` (`git worktree add --detach`,
porcelain empty). A whole-commit `git revert 8fbf50918` (the fast-intent
routing commit) no longer applies cleanly — the receipts module has since
been rewritten around it — so the rehearsal exercises the lever exactly as
the solver matrix documents it: drop the intent routing in
`_solver_route`, i.e. replace

```python
    return "fused_stepwise" if intent == "fast" else "stepwise"
```

with `return "stepwise"` (one line, `benchmarks/custom_quasi_newton_runtime.py`).

## Observed under rollback (CPU env prefix, `.venv-qn-cpu`)

- Route probe: `_solver_route("custom", "lbfgs", intent="fast")` returns
  `"stepwise"` — fast intent falls back to the callback-capable route.
- Parity oracle: `tests/jax/solve/test_lbfgsb_trajectory_parity.py`
  **4 passed** — SciPy-trajectory parity is unaffected by the rollback.
- Full runtime + fused-mode suites
  (`tests/benchmarks/test_custom_quasi_newton_runtime.py` +
  `tests/jax/solve/test_lbfgs_fused_stepwise_mode.py`): **103 passed,
  2 failed**, and the two failures are exactly the fused-contract pins
  that must fail under a rollback —
  `test_custom_lbfgs_route_and_prepared_program_follow_intent[fast-…]`
  and `test_fast_custom_lbfgs_has_zero_advance_observations`. The fused
  kernel tests themselves stay green (the kernel remains buildable; only
  routing reverts).

## Rollback caveats (part of the lever's contract)

- The two failing route pins must be reverted together with the routing
  line in a real rollback commit.
- New fast-intent performance lanes would then emit
  `solver_route="stepwise"` and be rejected by the performance
  qualification's fused-route requirement — a full rollback therefore
  also retires the fused receipt gate (or pauses fast-lane publication).
- Committed receipts are unaffected: validation replays hashes and
  qualification arithmetic, not solvers.

Worktree restored (`git checkout -- .`) and removed after the rehearsal.
