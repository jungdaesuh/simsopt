# Fused-route rollback rehearsal (2026-08-03, behavioral evidence)

Rehearses the documented `fused_stepwise` rollback lever with tracked
execution bytes — distinct from `rollback-rehearsal-20260802.md`, which
predates the fused route and covers the prepared-runtime commits
(`3b2b9f40a`, `41d95cf50`, `fd200f564`) only.

## Revision history (kept as the honest record)

- The FIRST version of this rehearsal was INVALID and was caught by
  external review: before the single-decision routing fix, preparation,
  execution, the fast-lane transfer gate, and the persisted row each
  re-derived the route from intent independently, so the one-line
  `_solver_route` lever only relabeled the measurement while the solver
  still ran fused. During development of the fix (`2f23db25a`) an
  intermediate state existed in which the transfer gate still keyed on
  intent: there, the same probe failed closed ("fused_stepwise advance
  observations exceed the runner transfer gate") — the solver executed
  stepwise but the runner refused to emit a measurement. That state was
  folded into `2f23db25a` by amend and is not a reachable commit; it is
  recorded here as prose history only.
- The SECOND version proved behavioral rollback but carried its numbers
  as prose only. This revision re-runs the rehearsal at `280624e80` and
  commits the raw bytes.

## Setup

The worktree began clean and detached at `280624e80` (`git worktree add
--detach`). Applying the rollback lever necessarily made the execution tree
dirty. The raw runner payload records that dirty state but did not capture the
complete `git status` or full-diff hash, so it cannot prove that the separately
tracked `rollback-rehearsal-fused-20260803/raw/lever.patch` was the only dirty
change. The raw bundle is retained as behavioral evidence, not exclusive
source-tree attestation.

## Tracked evidence (`rollback-rehearsal-fused-20260803/raw/`, hashes in `SHA256SUMS`)

- `measurements.json` — the emitted row from the REAL benchmark
  entrypoint under rollback (`benchmarks/custom_quasi_newton_runtime.py
  --device cpu --intent fast --providers custom --cases coil47
  --method lbfgs --maxiter 3`, backend mode `jax_cpu_fast`; stdout in
  `probe-stdout.log`): `solver_route="stepwise"` with
  `work_counters.advance_observations = 5` — per-advance host-boundary
  packets audited by the transfer instrumentation. A fused run cannot
  produce this row: the fused driver performs no per-advance host
  transfers, and the route-keyed transfer gate fails closed when a
  fused-labeled run observes more than `iterations + 1` advance packets
  (the small allowance covers the terminal payload fetch; the gate is
  NOT zero-tolerance — see `custom_quasi_newton_runtime.py`, transfer
  gate in `_measurement`). Five observations at three iterations
  exceeds that allowance, so this row could only be emitted under the
  stepwise route label the lever produced.
- `parity-under-rollback.log` —
  `tests/jax/solve/test_lbfgsb_trajectory_parity.py`: **4 passed**.
- `suites-under-rollback.log` — runtime + fused-mode suites:
  **104 passed, 2 failed**, the failures being exactly the two
  fused-contract pins that must revert together with the lever
  (`test_custom_lbfgs_route_and_prepared_program_follow_intent[fast-…]`,
  `test_fast_custom_lbfgs_has_zero_advance_observations` — the latter
  pins the fused lane's zero-advance INVARIANT at the work-counter
  level, stricter than the gate's `iterations + 1` allowance). Fused
  kernel tests stay green: the kernel remains buildable; only routing
  reverts.

## Rollback caveats (part of the lever's contract)

- The two failing route pins revert together with the routing line in a
  real rollback commit.
- New fast-intent lanes then emit honest `solver_route="stepwise"` rows,
  which the performance qualification's fused-route requirement rejects —
  a full rollback therefore also retires the fused receipt gate (or
  pauses fast-lane publication).
- Committed receipts are unaffected: validation replays hashes and
  qualification arithmetic, not solvers.
- The clean-tree regression
  `test_solver_route_decision_drives_stepwise_execution_and_persistence`
  forces `_solver_route` to `stepwise` through the real `_measurement` path and
  asserts the persisted route plus discriminating per-advance observations.
  That regression, rather than the dirty rehearsal's source identity, is the
  durable guard for the single-decision routing bug.

Worktree restored (`git checkout -- .`) and removed after the rehearsal.
