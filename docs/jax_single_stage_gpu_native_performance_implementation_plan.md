# JAX Single-Stage GPU-vs-Native Performance Plan

**Status:** SUPERSEDED (2026-08-05) — retained as the pre-campaign historical
record. The operative documents are
`docs/single_stage_speed_campaign_protocol.md` (amendments r2–r5), its
frozen validator `benchmarks/validate_single_stage_speed_claim.py`, and the
bounded closeout in `docs/single_stage_speed_campaign_results.md`. The
convergence-ladder strategy, the single 10%-warm-median gate, and the
endpoint-certificate eligibility rule below are dead. Measured 2026-08-05
outcome: the host-driven GPU lane lost by ~26x (see the supersession record
at the end of this document).
**Last updated:** 2026-08-05

## Purpose

Make the custom-JAX GPU path for the VMEC-free single-stage Boozer example
faster than its matched native SIMSOPT/C++ CPU path while preserving the
existing scientific parity contract and bounded memory behavior.

## Goals

- Both native CPU and JAX GPU finish with valid endpoint certificates.
- `jax_gpu_fast` is at least 10% faster than `native_cpu` by warm median on the
  native-default workload.
- Native CPU, JAX CPU parity, and JAX GPU parity pass the existing observable
  tolerances without increasing memory budgets.

## Non-Goals

- VMEC/MPI single-stage optimization.
- A new optimizer, whole-solve JIT, directional line search, or dense
  Jacobian/Hessian implementation.
- A new benchmark framework or changes to unrelated examples.
- Claiming that uncached compilation is faster than native execution.

## Current Context

- The target is
  `examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py`, measured
  as `native-single-stage-boozer-vacuum-optimization`.
- Existing source-owned reduced-fixture evidence shows a large fixed-budget GPU
  speed advantage, but both the custom and native 1,000-step endpoints are
  iteration-limited. It is diagnostic evidence, not a valid speed claim.
- The exact example and parity case also use a 1,000-step native-default cap.
- Optimizer evaluations compute outer reporting terms whose curvature and three
  distance weights are zero in this example.
- The current host-driven optimizer keeps compilation and optimizer state
  bounded with iteration count. This plan preserves that structure.

## Rationale

The current blocker is a matched successful endpoint, not proof that the GPU
can execute the kernels quickly. The shortest path is therefore: converge the
existing route, measure it, and optimize only one confirmed source of wasted
work if the converged run loses the speed advantage.

## Execution Record

- The exact native-default 1,000-step native CPU and RTX 5090 JAX GPU
  diagnostics were launched together on 2026-08-03. The native lane was
  rejected by the canonical certificate: SciPy status 1 at exactly 1,000
  iterations, with terminal gradient infinity norm `4.685879032582642e-05`
  against the unchanged `1e-07` threshold. The GPU lane reached its declared
  two-hour timeout (exit 124) without emitting a terminal payload. Both
  2,000-step ladder lanes then started with four-hour bounds. The native lane
  was also rejected: SciPy status 1 at exactly 2,000 iterations, stopping
  reason `iteration-limit`, and terminal gradient infinity norm
  `1.745635582275606e-05`. The GPU 2,000-step and native 4,000-step lanes were
  stopped after a code audit proved that the public native and JAX solver
  termination contracts are not matched (`gtol=1e-15` versus `gtol=1e-8`).
  These runs are diagnostic-only; no budget is selected.
- The ladder was halted while five matched-workload defects were repaired.
  Native scale is now explicit and independent of the iteration budget; the
  native and JAX examples share the unchanged `1e-15` outer-gradient stopping
  policy; parity-native persists the public inner Newton options; parity-JAX
  uses the public accepted-incumbent controller and anchored final evaluation;
  and measurement profiles now execute L-BFGS-B for `fast` and BFGS for
  `parity`, with matching status conventions and enforced driver provenance.
  Host L-BFGS-B `ftol` termination is interpreted through the provider's own
  success vocabulary and remains independently gated by the endpoint's
  finiteness and terminal-stationarity checks.
- No prior exact resolution-6 public-example receipt was found for the
  1,000/2,000/4,000 ladder. Historical 1,000-step data use a reduced
  `mpol=ntor=1` fixture and are not substituted for this run.
- The native example and target parity path now fail closed on the shared
  endpoint certificate. The certificate and iteration budget are owned by
  dependency-neutral `simsopt` modules; the native oracle does not import the
  optional JAX package.
- The target parity relationship now authorizes `native_default`, and every
  emitted endpoint field has a complete three-lane comparison matrix.
- The existing measurement runner now distinguishes declared `fast` and
  `parity` execution intent while preserving truthful backend modes and GPU
  transfer-guard policies. A bounded live `jax_cpu_fast` parity-child run
  executed
  `simsopt_jax_host_lbfgsb_with_traceable_boozer_newton` and failed closed at
  its two-step diagnostic budget, as expected.
- Current validation: the integrated seven-file run passed 274 tests with one
  CUDA-only reporting skip. This includes 112 endpoint-certificate tests, 13
  public-example tests, 46 measurement-contract tests, the six-test
  single-stage parity module, and negative forged-driver/status regressions.
  Static lint, format, compile, and diff checks pass. No repaired
  native-default ladder or performance run has yet been promoted as scientific
  or timing evidence.

## Assumptions

- RTX 5090 is the primary performance target. A100 is a final portability
  check.
- Persistent-cache warm timing is the primary performance metric; cold compile
  time remains separately reported.
- A timing is ineligible unless every compared lane has a valid scientific
  endpoint.

## Implementation Plan

1. Find the smallest converged matched budget.
   - [ ] Run paired native CPU and JAX GPU parity diagnostics at 1,000, 2,000,
     then 4,000 maximum iterations, stopping at the first budget where both
     pass the unchanged endpoint certificate.
   - [ ] Do not loosen `gtol`, scientific tolerances, or endpoint rules. If
     neither lane converges by 4,000 iterations, stop and diagnose convergence
     instead of continuing this performance plan.
   - [ ] Update the native example, JAX example, and parity case to the smallest
     proven native-default cap.

2. Measure first; remove only confirmed waste if required.
   - [ ] Run the existing native-default measurement artifact on the RTX 5090.
   - [ ] If `jax_gpu_fast <= 0.90 * native_cpu`, make no optimizer change.
   - [ ] Otherwise, change the optimizer-forward path in
     `surface_objectives_traceable.py` to skip zero-weight outer terms during
     trial evaluations while still computing all final reporting metrics once
     from the accepted solution.
   - [ ] Prove the old and new objective and gradient agree before repeating the
     performance measurement.

3. Certify the final candidate.
   - [ ] Run focused example, accepted-incumbent, and reporting tests.
   - [ ] Run native CPU, JAX CPU parity, and JAX GPU parity at native-default
     scale with unchanged tolerances.
   - [ ] Produce the final RTX 5090 measurement artifact and one A100
     portability artifact, reporting cold time, warm time, parity, RSS, and GPU
     memory together.

## Validation Plan

```bash
env MPI4PY_RC_INITIALIZE=0 PYTHONPATH=src:. \
  .venv-qn-cpu/bin/python -m pytest \
  tests/geo/test_traceable_reporting_cache.py \
  tests/geo/test_traceable_trial_evaluator.py \
  tests/jax/examples/test_single_stage_boozer_vacuum_example.py \
  tests/integration/test_jax_mirror_single_stage_boozer_vacuum_parity.py \
  -q

env MPI4PY_RC_INITIALIZE=0 \
  .venv-qn-gpu/bin/python examples/jax/run_parity.py \
  --case native-single-stage-boozer-vacuum-optimization \
  --lanes native-cpu,jax-cpu,jax-gpu \
  --scale native_default \
  --artifact-root /tmp/simsopt-single-stage-parity

MEASUREMENT_ROOT=$(mktemp -d /tmp/simsopt-single-stage-measurement.XXXXXX)
env MPI4PY_RC_INITIALIZE=0 \
  .venv-qn-gpu/bin/python benchmarks/run_jax_native_example_measurements.py \
  --case native-single-stage-boozer-vacuum-optimization \
  --scale native_default \
  --artifact-root "$MEASUREMENT_ROOT" \
  --gpu-index 0 \
  --poll-interval-seconds 0.05 \
  --timeout-seconds 7200
```

- [ ] Every timed lane passes its endpoint certificate and scientific
  comparison.
- [ ] RTX 5090 `jax_gpu_fast/native_cpu <= 0.90` by warm median.
- [ ] Final objective, parameters, iota, volume, non-QS ratio, and Boozer
  residual pass the existing parity tolerances.
- [ ] Peak process-tree RSS and allocation-sensitive GPU memory do not exceed
  the pre-change baseline.
- [ ] Existing 8192 MiB host and 12288 MiB GPU grouped-adjoint budgets remain
  unchanged and pass.

## Risks and Mitigations

- Risk: more iterations still do not produce a certified endpoint.
  Mitigation: stop at 4,000 and diagnose convergence; do not hide failure with a
  larger arbitrary cap or weaker tolerance.
- Risk: skipping zero-weight terms changes final reporting.
  Mitigation: skip them only in optimizer trials and recompute full metrics once
  from the accepted solution.
- Risk: a speedup comes from a failed or scientifically different run.
  Mitigation: reject timing unless the existing endpoint and parity contracts
  pass unchanged.

## Completion Criteria

- [ ] Both matched lanes have valid endpoint certificates.
- [ ] RTX 5090 JAX GPU warm median is at least 10% faster than native CPU.
- [ ] Native/JAX CPU/GPU parity passes with unchanged tolerances.
- [ ] Peak memory does not regress, and existing memory budgets are unchanged.
- [ ] Final RTX 5090 and A100 artifacts are saved with source, environment, and
  device identity.

## Open Questions

- The retained 8192/12288 MiB grouped-adjoint budget belongs to the legacy
  stateful grouped-VJP adapter, while this target's production gradient uses a
  traceable scalar-gradient route. A new receipt must not claim they are the
  same path.
- The generic measurement artifact is intentionally descriptive and contains
  no performance or memory threshold. A new claim-bearing qualification schema
  is a Tier-4 integrity boundary and requires explicit design sign-off before
  implementation. *(Resolved 2026-08-05: designed, implemented, and frozen as
  `benchmarks/validate_single_stage_speed_claim.py` under the campaign
  protocol.)*
- If the single scoped optimization does not meet the speed gate, profile the
  remaining converged workload and write a separate follow-up plan.
  *(Superseded: the protocol's "Optimization ladder" section owns this.)*

## Supersession record (2026-08-05)

This plan was executed in amended form by the single-stage speed campaign and
is closed. What happened, against the plan's own claims:

- **Measured outcome (cold samples, RTX 5090 box, receipts preserved at
  `~/simsopt-campaigns/.single-stage-speed-20260804.partial-20260805T052535Z-2add24ec/`):**
  `native_cpu` finished 1,000 iterations in 287.30 s (objective 4.4822e-8);
  the custom GPU lane took 7,541.46 s (objective 1.6136e-7) — ~26x slower
  with a worse objective; the Optax GPU lane hit the 10,800 s timeout at
  iteration 588. The reduced-fixture "large fixed-budget GPU speed advantage"
  cited in Current Context did not survive native-default scale.
- **The Execution Record's 2-hour GPU timeout was later root-caused:** it
  masked a first-evaluation all-NaN gradient (Biot-Savart chunk padding —
  synthetic zero points × quadrature padding evaluated a 0×∞ reverse-mode
  path), identical on CPU and GPU; fixed in
  `src/simsopt_jax/core/biotsavart.py` with a fail-before/pass-after
  regression test. The fast lane had never completed a native_default run
  before that fix.
- **Strategy supersessions:** the 1,000/2,000/4,000 stationarity ladder
  (Implementation Plan step 1, Completion Criteria) was replaced by
  matched-quality metrics — time-to-quality primary, fixed-budget
  quality-and-time secondary, stationarity reported-not-gated (protocol
  amendment r2). The custom lanes now use a direct FP64 LU exact-adjoint
  because fast-mode operator-GMRES stagnates on this workload (r3). A
  divergent-lane physics contract (r4) governs any future
  proximal/coupled/batched lane. An Optax lane (`jax_gpu_optax`) is a gated
  comparison target, superseding the "no new optimizer" non-goal.
- **Validation Plan caveat:** command 3 is unexecutable as written — it
  routes to the legacy collector (no `--single-stage-speed-campaign` flag,
  which cannot emit the campaign receipt layout), uses `/tmp` artifact roots
  the protocol forbids for claim-bearing evidence, and its 7,200 s timeout is
  smaller than one measured custom-lane cold sample. Use the campaign
  invocation in the protocol's "Definition of done" instead.
- **Naming note:** this plan's `jax_gpu_fast` is the campaign's
  `jax_gpu_custom` lane id (`jax_gpu_fast` survives as its backend_mode).
