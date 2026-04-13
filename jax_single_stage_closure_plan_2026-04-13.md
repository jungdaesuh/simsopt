# simsopt-jax Single-Stage Closure Plan

**Date:** 2026-04-13
**Scope:** Single-stage CUDA closure, donor/seed/search policy follow-up, and shared GPU regression hardening
**Status:** Planning document
**Primary backlog source:** [jax_gpu_port_todos_2026-04-08.md](jax_gpu_port_todos_2026-04-08.md)

## Audit Update

This file was assembled from the current repo state, the checked-in GPU proof harness, the active TODO file, official JAX docs, and open-source references.

- Checked boxes reflect facts already confirmed in the current tree.
- Unchecked boxes are the ordered execution plan.
- This file is the single-stage closure runbook, not a replacement for the broader GPU-port backlog.

## Confirmed Baseline

- [x] The checked-in single-stage GPU proof path already exists in `benchmarks/single_stage_outer_loop_probe.py`, `benchmarks/validation_ladder_contract.py`, and `tests/integration/test_single_stage_physics_parity.py`.
- [x] The single-stage ship-critical backlog item is still `#40` in [jax_gpu_port_todos_2026-04-08.md](jax_gpu_port_todos_2026-04-08.md).
- [x] The current closure order is `#40 -> #41 -> #42 -> #46 -> #49`.
- [x] `_line_search.py` directional derivatives already use `_dot(..., precision=lax.Precision.HIGHEST)` via `src/simsopt/geo/optimizer_jax_private/_common.py`.
- [x] The CPU/reference single-stage path already has rollback-style failure behavior in `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`.
- [x] The shared compile-count harness already exists in `tests/subprocess/jax_runtime_cases.py` and `tests/test_jax_import_smoke.py`.

## Ordered Checklist

### Phase 1: CUDA Runtime Closure

1. [ ] Close `#40` on a clean CUDA environment by rerunning the existing outer-loop proof harness on the real target lane.
   - Goal: prove the true single-stage ondevice path runs for at least the required iteration count and returns a net objective decrease.
   - Exit criteria: `TestSingleStageOuterLoopGpuProof` passes on CUDA with `transfer_guard=disallow`.

2. [ ] Capture the first real failing or stalling region with `jax.profiler.trace()`, XProf, `jax_log_compiles`, and `jax_explain_cache_misses`.
   - Goal: replace guesswork with one concrete failing seam.
   - Exit criteria: one short diagnostic note with the first bad kernel region, compile behavior, and whether the failure reproduces.

3. [ ] Audit only the remaining real target-lane reductions that profiling still implicates.
   - Scope: residual objective aggregation, failure-penalty reductions, PLU-related paths, or any other reduction surfaced by the trace.
   - Non-goal: blanket-patching `_line_search.py` directional derivatives that are already hardened.

4. [ ] Run a diagnostic A/B with deterministic GPU reductions if the trace still points at reduction-order sensitivity.
   - Scope: treat `XLA_FLAGS=--xla_gpu_deterministic_ops=true` as an experiment, not as the planned production fix.
   - Exit criteria: record whether it changes the failure mode, compile behavior, or acceptance pattern.

### Phase 2: Recompilation and Stability Guards

5. [ ] Close `#41` by extending the existing compile-count harness to the real target-lane outer-loop path.
   - Goal: detect shape- or cache-driven `run_solver` recompilation regressions on the production single-stage path.
   - Exit criteria: repeated identical target-lane runs compile once under the smoke harness.

6. [ ] Re-run `#40` after the `#41` smoke is in place and confirm the proof lane stays stable with compile diagnostics enabled.
   - Goal: avoid promoting a proof that only passes without observability enabled.
   - Exit criteria: proof still passes or fails for the same localized reason.

### Phase 3: Donor / Seed / Search Policy

7. [ ] Implement donor-family-local feasible-start preservation for the single-stage continuation path.
   - Goal: preserve the last feasible inner-solve state per donor family instead of relying on a generic preserve-first rule.
   - Exit criteria: donor-local state can be restored deterministically when the trial point stays inside the donor-local neighborhood.

8. [ ] Implement restore-shrink-retry for invalid geometry or failed inner solves.
   - Goal: when the first large outer step leaves the feasible basin, restore the last feasible donor-local state, shrink the step, and retry once before accepting failure.
   - Exit criteria: the first oversized step no longer ejects a good donor without a local recovery attempt.

9. [ ] Tighten seed policy for single-stage runs.
   - Reject `init_only` Stage-2 artifacts by default.
   - Preserve same-family Stage-2 parents.
   - Improve bridge selection when no compatible same-family single-stage donor exists.
   - Exit criteria: seed selection is deterministic and covered by tests for accepted and rejected donor classes.

10. [ ] If restore-shrink-retry is not sufficient, make the failure penalty adaptive to solve quality or residual size.
    - Goal: provide a useful downhill signal toward the feasible basin when the inner solve fails.
    - Exit criteria: penalty scaling is tied to actual solve quality, not a flat fallback.

### Phase 4: GPU Regression Hardening

11. [ ] Close `#42` by broadening GPU `transfer_guard=disallow` coverage only after `#40/#41` are stable.
    - Goal: promote the existing strict CUDA workflows from curated slices toward fuller suite coverage.
    - Exit criteria: the broader GPU strict lane runs with live logging and produces actionable failures instead of silent hangs.

12. [ ] Close `#46` with transfer-guard fuzzing on real single-stage target-lane entry points.
    - Goal: inject host scalars into public kernel entry points and assert rejection under `disallow`.
    - Exit criteria: parameterized coverage exists for the production single-stage target lane, not only toy kernels.

13. [ ] Close `#49` last.
    - Scope: the two known FD-sensitive failures only.
    - Exit criteria: either loosen the tolerances with justification or mark them skipped with explicit reasons.

## Shared Performance Follow-Up

- [ ] `#31` Replace L-BFGS slice-and-concatenate history updates with a ring buffer.
- [ ] `#34` Default `SIMSOPT_JAX_COMPILATION_CACHE_DIR` to a real cache path.
- [ ] `#32` Remove the dead line-search re-evaluation path.
- [ ] `#26` Investigate a measured iterative-refinement heuristic after correctness closure.

## Deferred, Not Closed

- [ ] Keep `#24` open but profiling-gated.
- [ ] Keep `#28` open but profiling-gated.
- [ ] Keep `#29` open but profiling-gated.
- [ ] Keep `#33` open but profiling-gated.

These are lower priority than correctness closure, but they are not yet strong `WONTFIX` candidates from current repo evidence.

## Suggested Two-Week Sequence

### Week 1

- [ ] Day 1: rebuild the clean CUDA lane and rerun the direct `#40` probe.
- [ ] Day 2: capture XProf trace plus compile diagnostics and localize the first real failing seam.
- [ ] Day 3: implement the smallest runtime or reduction fix justified by the trace.
- [ ] Day 4: extend the compile-count smoke to the real target-lane outer-loop path and close `#41`.
- [ ] Day 5: rerun `#40` with the new smoke and confirm stable behavior.

### Week 2

- [ ] Day 6: implement donor-family-local feasible-start preservation.
- [ ] Day 7: implement restore-shrink-retry.
- [ ] Day 8: tighten single-stage seed selection and bridge policy.
- [ ] Day 9: add `#46` transfer-guard fuzzing against production entry points.
- [ ] Day 10: broaden `#42`, then clean up `#49`.

## Validation Commands

### `#40` Direct CUDA Probe

```bash
conda run --no-capture-output -n jax-0.9.2 env \
  SIMSOPT_BACKEND_MODE=jax_gpu_parity \
  SIMSOPT_BACKEND_STRICT=1 \
  SIMSOPT_JAX_TRANSFER_GUARD=disallow \
  JAX_PLATFORMS=cuda \
  JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_LOG_COMPILES=1 \
  JAX_EXPLAIN_CACHE_MISSES=1 \
  python benchmarks/single_stage_outer_loop_probe.py \
    --platform cuda \
    --optimizer-backend ondevice \
    --output-json benchmark_artifacts/single_stage_outer_loop_cuda.json
```

### `#40` Pytest Proof Gate

```bash
conda run --no-capture-output -n jax-0.9.2 env \
  SIMSOPT_BACKEND_MODE=jax_gpu_parity \
  SIMSOPT_BACKEND_STRICT=1 \
  SIMSOPT_JAX_TRANSFER_GUARD=disallow \
  JAX_PLATFORMS=cuda \
  JAX_ENABLE_X64=1 \
  python -m pytest tests/integration/test_single_stage_physics_parity.py \
    -k "TestSingleStageOuterLoopGpuProof" -v --tb=short
```

### `#41` Existing Compile-Count Baseline

```bash
conda run --no-capture-output -n jax-0.9.2 env \
  JAX_ENABLE_COMPILATION_CACHE=0 \
  python -m pytest tests/test_jax_import_smoke.py \
    -k "lbfgs_ondevice_reuses_compiled_solver or target_lbfgs_ondevice_reuses_compiled_solver" \
    -v --tb=short
```

### Donor / Restore Validation Slice

```bash
conda run --no-capture-output -n columbia-jax-0.9.2 \
  python -m pytest tests/geo/test_single_stage_example.py \
    -k "snapshot_restore_round_trip or evaluate_candidate_failure_restores_cpu_state_on_legacy_path or snapshot_accepted_step_state_can_skip_objective_refresh or resolve_warm_start_boozer_init_overrides" \
    -v --tb=short
```

### Broader Live Pytest Lane

```bash
scripts/run_pytest_live.sh tests
```

## References

- Official JAX profiling docs: https://docs.jax.dev/en/latest/profiling.html
- Official JAX config options: https://docs.jax.dev/en/latest/config_options.html
- Official JAX transfer guard docs: https://docs.jax.dev/en/latest/transfer_guard.html
- Official JAX GPU memory allocation docs: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- Official JAX `dot` API: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.dot.html
- JAXopt `LBFGS`: https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGS.html
- IPOPT restoration-phase reference: https://cepac.cheme.cmu.edu/pasi2011/library/biegler/ipopt.pdf
