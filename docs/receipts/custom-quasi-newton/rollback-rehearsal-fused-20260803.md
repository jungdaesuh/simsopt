# Fused-route rollback rehearsal (2026-08-03, revised)

Rehearses the documented `fused_stepwise` rollback lever with
EXECUTION-level evidence — distinct from `rollback-rehearsal-20260802.md`,
which predates the fused route and covers the prepared-runtime commits
(`3b2b9f40a`, `41d95cf50`, `fd200f564`) only.

## Why this receipt was revised

The first version of this rehearsal was INVALID and was caught by external
review: before commit `2f23db25a`, preparation, execution, and the
persisted row each re-derived the route from intent independently, so the
one-line `_solver_route` lever only relabeled the measurement while the
solver still ran fused — `103/2` test results proved a mislabeling, not a
rollback. `2f23db25a` makes one route decision drive preparation
(`_prepare_custom`), execution (`_run_custom`), the fast-lane transfer
gate, and the persisted `solver_route`; this receipt records the rehearsal
repeated against that commit with a behavioral discriminator.

## Setup

Clean detached worktree at `2f23db25a` (`git worktree add --detach`,
porcelain empty). A whole-commit `git revert 8fbf50918` (the fast-intent
routing commit) no longer applies cleanly — the receipts module has since
been rewritten around it — so the rehearsal exercises the lever exactly as
the solver matrix documents it: drop the intent routing in
`_solver_route`, i.e. replace

```python
    return "fused_stepwise" if intent == "fast" else "stepwise"
```

with `return "stepwise"` (one line, `benchmarks/custom_quasi_newton_runtime.py`).

## Observed under rollback (CPU env, `.venv-qn-cpu`)

- EXECUTION probe at the real benchmark entrypoint
  (`benchmarks/custom_quasi_newton_runtime.py --device cpu --intent fast
  --providers custom --cases coil47 --method lbfgs --maxiter 3`, backend
  mode `jax_cpu_fast`): the emitted row records
  `solver_route="stepwise"` AND
  `work_counters.advance_observations = 5` — per-advance host-boundary
  packets audited by the transfer instrumentation, which a fused run
  cannot produce (the fused invariant is zero advance observations, and
  the runner's transfer gate — now keyed on the route, not the intent —
  fails closed if a fused-labeled run observes them). The solver
  demonstrably executed the stepwise driver.
- Intermediate finding, preserved: at the pre-SSOT commit `6c6eb1962`
  (gate still keyed on intent), the same probe FAILED CLOSED with
  "fused_stepwise advance observations exceed the runner transfer gate" —
  the run executed stepwise but the runner refused to emit a measurement.
  Mislabeled evidence was impossible in either state; honest emission
  requires the gate to follow the same route decision.
- Parity oracle: `tests/jax/solve/test_lbfgsb_trajectory_parity.py`
  **4 passed** under rollback.
- Full runtime + fused-mode suites: **104 passed, 2 failed**, the
  failures being exactly the two fused-contract pins that must revert
  together with the lever —
  `test_custom_lbfgs_route_and_prepared_program_follow_intent[fast-…]`
  and `test_fast_custom_lbfgs_has_zero_advance_observations`. The fused
  kernel tests stay green (the kernel remains buildable; only routing
  reverts).

## Rollback caveats (part of the lever's contract)

- The two failing route pins revert together with the routing line in a
  real rollback commit.
- New fast-intent lanes then emit honest `solver_route="stepwise"` rows,
  which the performance qualification's fused-route requirement rejects —
  a full rollback therefore also retires the fused receipt gate (or
  pauses fast-lane publication).
- Committed receipts are unaffected: validation replays hashes and
  qualification arithmetic, not solvers.

Worktree restored (`git checkout -- .`) and removed after the rehearsal.
