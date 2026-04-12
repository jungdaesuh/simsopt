# GPU Continuation Backlog

This file tracks the remaining single-stage GPU throughput and validation work
after the completed callback-removal, lazy host-wrapper, cache-key, line-search
budget, and multi-donor campaign orchestration changes.

The ordering is intentional. Do not skip ahead to hardware tuning before the
profiling and continuation-policy work is done.

## Now

1. Select 2 to 4 representative donor runs.
   Use one strong donor, one borderline donor, and optionally one donor from a different basin.

2. Verify each donor run directory has the required reusable artifacts.
   Confirm `biot_savart_opt.json`, `surf_opt.json`, and `results.json` exist.

3. Freeze one reproducible continuation configuration.
   Lock the schedule, `trial-policy`, resolution, iteration budgets, backend lane, and run ID.

4. Run one real profiled multi-donor continuation campaign.
   Use `run_single_stage_continuation.py` with repeated `--campaign-donor-run-dir`,
   `--jax-profile-dir`, fixed `--run-id`, and explicit `--campaign-output-json`.

5. Capture exact provenance for the profiled run.
   Record commit SHA, GPU type, JAX and jaxlib versions, relevant environment variables,
   donor paths, and the full CLI command.

6. Collect all per-donor outputs from that run.
   Gather the campaign summary, per-donor continuation summaries, validation reports,
   stage outputs, and JAX or XProf trace directories.

7. Build one compact profiling report.
   Include compile or setup time, per-stage wall time, accepted-step count,
   rejected-step count if present, optimizer iteration count, and Boozer-backed
   objective evaluation count.

8. Compare donors at the right level.
   Focus on compile-heavy front half versus steady-state runtime, accepted progress
   per stage, and objective evaluations per accepted step.

9. Make the first branch decision from the trace.
   Decide whether the next bottleneck is reevaluation or host-stall dominated,
   device-throughput dominated, or donor-quality dominated.

## Next

10. If reevaluation or host stalls dominate, reduce outer-loop reevaluation.
    Tighten continuation-stage budgets, reduce line-search reevaluations, and
    cut unnecessary Boozer-backed objective calls.
    Current state: the validated-fast coarse scaled initial phase is already skipped.
    The next throughput cut is the non-final main outer-loop line-search budget.

11. If steady-state device kernels dominate, evaluate true batched multi-donor execution.
    The current campaign path is orchestration, not real batched GPU execution.

12. If donor quality dominates outcomes, implement donor ranking and seed-selection policy.
    Stop treating all donors as equivalent starting points.

13. Run real convergence campaigns after the first profile-guided fix.
    Use multiple donors and schedules, not just reduced-fixture or one-step proofs.

14. Build final candidate ranking on physics and hardware gates.
    Rank by field error, iota target, non-QS, curvature, coil-coil distance,
    coil-surface distance, vessel clearance, and convergence quality.

15. Build the candidate ledger.
    It should say which donor and schedule won, why it won, what failed for the others,
    and whether each result is research-usable.

16. Emit machine-readable validation summaries for every run.
    Make results indexable and comparable across campaigns.

## Later

17. Extend proof from reduced parity to long-horizon continuation behavior.
    Current parity is good for init, wrapper, and one-step outer-loop use, but not
    yet for multi-iteration basin-level continuation behavior.

18. Close the remaining Stage 2 structural gap.
    Fix the Stage 2 target-lane coil-distance scan so it respects the pairwise
    row-sharding contract.

19. Keep the current parity and strict-transfer-guard tests as guardrails.
    Use them to catch regressions while changing throughput and continuation behavior.

20. Keep Runpod and HF remote execution reproducible.
    Reuse the checked-in launcher and preserve provenance.

21. Tune hardware and runtime only after the structural work is done.
    That is when A100 choice, XLA flags, PGLE, and related tuning become worth doing.

22. Revisit whether more parity expansion is needed only after campaign data says so.
    Do not expand parity work without a concrete correctness signal.

## Decision Tree

1. Profile one real multi-donor continuation run.
2. Choose the highest-leverage branch:
   reevaluation reduction, batching, or donor ranking.
3. Implement that branch.
4. Run convergence campaigns.
5. Build ranking, ledger, and long-horizon validation.
6. Tune hardware last.

## Minimum Deliverables For The Current Phase

- One reproducible campaign command.
- One `campaign_summary.json`.
- One set of per-donor JAX traces.
- One short profiling memo with compile time, stage time, accepted steps, and
  objective-evaluation counts.
- One explicit recommendation for the next implementation branch.
