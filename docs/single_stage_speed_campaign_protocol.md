# Single-Stage Speed Campaign Protocol

**Campaign:** `single-stage-speed-20260804`
**Baseline snapshot tag:** `campaign-20260804-pre` (full tracked+untracked working state)
**Frozen-files baseline tag:** `campaign-20260804-frozen-r5` (amendment r5; supersedes r4)
**Status:** Active
**Created:** 2026-08-04
**Roles:** Claude (protocol owner, validator owner, terminal auditor) · Codex
(implementation worker, campaign runner). The agent that runs the campaign
never certifies it.

## Claim under test

On the VMEC-free single-stage Boozer vacuum example
(`native-single-stage-boozer-vacuum-optimization`, native-default scale), the
custom JAX GPU lane is faster than **both**:

1. the native SIMSOPT/C++ CPU path, and
2. the Optax L-BFGS GPU lane,

with every compared lane scientifically valid under the existing (unchanged)
parity tolerances.

## Why this protocol replaces the stationarity gate

Upstream simsopt never shipped converging single-stage examples: upstream
`single_stage_optimization.py` runs 10 iterations at `tol=1e-15`;
`boozerQA.py` runs 1,000 at `tol=1e-15`. Both always stop at the iteration
cap. Requiring terminal gradient ∞-norm ≤ 1e-7 demands something the problem
formulation has never delivered (measured decay: 4.7e-5 → 1.7e-5 from 1,000 →
2,000 iterations). Timing claims here are therefore anchored to **matched
quality**, not stationarity.

## Metrics

### Primary — time-to-quality (TTQ)

- The quality bar is set by the native lane itself:
  `J_target = 1.001 × median(final objective of native_cpu warm samples)`.
- Per lane and per warm sample, TTQ = wall-clock time from optimizer start to
  the first iteration whose objective ≤ `J_target`, read from the
  per-iteration trajectory record. A sample that never reaches `J_target`
  within its iteration budget has TTQ = +∞.
- Lane TTQ = median over warm samples.

### Secondary — fixed-budget quality-and-time

At the shared native-default iteration cap, per lane: warm-median wall time
(existing measurement runner schedule: 1 cold, 1 warmup, 7 warm, rotating
order) and final objective. A lane passes the quality clause iff its
warm-median final objective ≤ `J_target`.

### Pass thresholds (fixed before any measurement)

- `TTQ(jax_gpu_custom) ≤ 0.90 × TTQ(native_cpu)`
- `TTQ(jax_gpu_custom) ≤ 0.90 × TTQ(jax_gpu_optax)`
- `warm_median(jax_gpu_custom) ≤ 0.90 × warm_median(native_cpu)` with the
  quality clause satisfied by both lanes.
- Warm timing is primary; cold compile time is reported separately and never
  enters a pass gate.

### Verdicts

- **WIN** — all four gates pass.
- **TIE** — a gate fails but no failing ratio exceeds 1.05, the quality
  clause holds, and profile evidence identifying the dominant cost is
  attached in the sibling directory `<artifact-root>.profile/` (at least one
  non-empty file; the receipt root itself is immutable). There is no lower
  bound: a failing ratio is by definition above 0.90, so the partition has no
  gap and never grades a faster result worse than a slower one (r5).
- **LOSS** — otherwise. TIE and LOSS are valid terminal states; the campaign
  must never trade a gate for a weaker tolerance.

## Scientific validity gates (all unchanged from the existing contracts)

A lane's timing is ineligible unless:

- inner Boozer Newton solves report success under the existing (public,
  persisted) inner options;
- terminal objective, gradient, and observables are finite;
- final observables (objective, iota, volume, non-QS ratio, Boozer residual)
  pass the existing parity tolerances against the native lane;
- fp64 end-to-end; matched initial point; matched native-default scale and
  iteration cap; truthful backend mode and driver provenance (existing
  forged-driver regressions stay green).

Outer-gradient stationarity is **reported, not gated**.

## Lanes

| Lane id | What it runs |
|---|---|
| `native_cpu` | native example, unmodified optimizer path |
| `jax_gpu_custom` | existing `jax_gpu_fast` profile (host L-BFGS-B + traceable Boozer Newton) |
| `jax_gpu_optax` | new measurement profile over the already-supported `optimizer_backend='optax-lbfgs'` |
| `jax_cpu_custom` | existing `jax_cpu_fast` profile (reference, not a pass gate) |

Parity lanes (`jax_cpu_parity`, `jax_gpu_parity`) keep running for the
science gates exactly as today.

## Instrumentation contract (what Codex builds; no numerics changes)

Every lane records a per-iteration trajectory without altering the
optimization sequence: for each optimizer iteration, `iteration`, `objective`,
and `wall_seconds_from_start`, appended from the existing callback/reporting
path. Receipt layout under the durable artifact root
`~/simsopt-campaigns/single-stage-speed-20260804/`:

```
campaign.json                      # git describe, device identity, env, budgets, lane list
lanes/<lane_id>/measurement.json   # existing runner artifact (per-sample wall_seconds)
lanes/<lane_id>/endpoint.json      # terminal payload incl. certificate fields
lanes/<lane_id>/trajectory-<phase>-<sample>.jsonl
```

Raw per-sample records are authoritative. The validator recomputes every
median and ratio from them, cross-checks each sample's recorded wall time
against its own trajectory, re-derives every parity bound from tolerance
constants deliberately duplicated from the frozen SSOT (never from the
receipt's tolerance field), and cross-binds each parity row's native value to
the native lane's endpoint. Fields it cannot recompute (the observables
themselves) are guarded by provenance, fingerprints, and the finite-value
gates. `/tmp` is forbidden for anything the claim depends on.

## Definition of done (machine-checkable)

`benchmarks/validate_single_stage_speed_claim.py` — owned by Claude, run by
Codex, re-run by Claude at audit. PASS requires: receipt schema complete,
validity gates green, thresholds met on recomputed numbers, and the frozen
files (list lives in the validator, enforced against
`campaign-20260804-frozen-r5`) untouched. Codex must never edit the validator, the
frozen files, or this protocol; any such diff fails the campaign
unconditionally.

## Optimization ladder (only if gates fail; one rung at a time)

1. Skip zero-weight outer reporting terms in optimizer trial evaluations
   (full metrics still computed once from the accepted solution).
2. Remove host-sync/while-predicate overhead on the custom lane (the A100
   campaign's proven mechanism class).
3. Batch what is batchable inside the inner Newton chain.

Each rung requires an old-vs-new objective+gradient equivalence proof on the
target workload before its re-measurement counts.

## Amendments

**r2 (2026-08-04, user-approved).** Resolves the three worker-reported
conflicts in `.Codex/campaign-notes.md` without touching any frozen science
file:

1. Measurement-lane parity rows are computed directly against the native lane
   using the frozen tolerance table (`src/simsopt_jax/parity_tolerances.py`
   remains the tolerance SSOT). The arbiter is not part of the measurement
   path and stays frozen and unchanged; ordinary parity lanes are unaffected.
2. Campaign endpoint validity admits iteration-cap endpoints: the gates are
   finite fp64 observables, inner-solve success, and the parity rows — not
   the ordinary outer-success flag (consistent with "stationarity reported,
   not gated").
3. Validator upgraded to v2 (provenance, coverage, and shared-identity
   checks) and the frozen baseline re-tagged `campaign-20260804-frozen-r2`.

**r3 (2026-08-05, user-approved).** The campaign's custom lanes
(`jax_cpu_custom`, `jax_gpu_custom`, `jax_gpu_optax`) use the direct FP64 LU
exact-adjoint route instead of fast-mode operator-GMRES, which stagnates on
this workload's (valid, κ≈1e3) adjoint system. This strengthens numerics
(exact solve replacing a non-converging iterative one). Conditions: the same
route on all three custom lanes; the endpoint audit records the adjoint
route; a one-time direct-vs-parity gradient agreement check at the
native_default initial point precedes the campaign; the GMRES stagnation
gets a follow-up investigation outside the campaign's critical path.
Context: the underlying first-eval NaN was root-caused and fixed as a
Biot-Savart chunk-padding bug (synthetic zero points × quadrature padding →
0×∞ in reverse mode) with a fail-before/pass-after regression test.

**r4 (2026-08-05, user-approved). Divergent-lane physics contract.** Speed
lanes may use path-divergent algorithms (different optimizers, proximal
projection of the Boozer constraint, coupled formulation, batched
multi-start). Their claims are governed by:

1. **Two-tier comparison rule.** Parity lanes (matched algorithm, matched
   inputs, frozen tolerances) carry ALL port-correctness claims. Speed lanes
   carry ONLY certified-quality-versus-wall-clock claims, worded "reaches
   certified configurations of equal-or-better objective in X% of native
   wall-clock." Speed-lane receipts are admissible only while the parity
   lanes pass at the same source state. Per-iteration comparisons are void
   wherever evaluation semantics differ.
2. **Equation-anchored certification.** Every reported endpoint passes the
   unchanged endpoint certificate: converged inner-solve residual, finite
   fp64 state, observables computed from the certified state. Linearized or
   trial evaluations may steer an optimizer but never enter receipts;
   trajectory records contain only accepted, fully-projected iterates.
3. **Cross-evaluator endpoint audit.** Each lane's final DOFs are
   re-evaluated from scratch by the other implementation (native endpoint
   through the JAX evaluator; JAX endpoint through the native evaluator).
   Independently recomputed observables must agree within the frozen parity
   tolerances. This check is per-endpoint and path-free.
4. **Optimizer-independent physics validation.** Field-line tracing on the
   final configuration must confirm the claimed surface (Poincaré closure
   spot-check; iota from tracing versus iota from the solve within the
   existing tolerance). Geometry/feasibility gates apply unchanged.
5. **Basin ruling.** An endpoint failing native-endpoint parity but passing
   items 2–4 is classified "different-basin valid": reportable as a quality
   result, never as same-answer parity. Same-answer claims require the
   endpoint parity boxes.

Enforcement status (restated honestly by r5): the validator machine-enforces
schema, provenance identities, shared initial-point/fingerprint identity,
adjoint-route presence, trajectory contiguity and budget completeness,
wall-clock consistency, recomputed parity bounds, and the speed/quality
gates. Enforced by worker-owned (non-frozen) code and re-audited at
promotion: inner-solve success at the observation layer, endpoint
certificate fields, the gradient-agreement precondition, and observable
extraction. Purely declarative until a divergent lane ships: r4 items 1
(same-source parity-lane admissibility), 3 (cross-evaluator audit), 4
(field-line tracing), and 5 (basin classification). No divergent-lane
receipt may be promoted before those checks exist in the validator.

**r5 (2026-08-05, doc-review audit).** A four-agent adversarial audit of this
protocol against the implementation produced validator v3 and these
corrections; all changes strengthen or clarify — no gate is loosened except
the verdict-partition repair, which removes the dead zone where a 7%-faster
lane graded worse than a dead-even one:

1. Verdict partition made monotone (see Verdicts): TIE = quality holds, every
   failing ratio ≤ 1.05, profile evidence present in the sibling
   `<artifact-root>.profile/` directory. Verdict exit codes are part of the
   contract: 0 WIN, 3 TIE, 1 LOSS, 2 integrity error; every unexpected
   validator exception is an integrity error, never a LOSS.
2. Validator v3 hardening: parity bounds recomputed from duplicated frozen
   SSOT constants (rtol 2e-8, atol 2e-12, referenced to the NATIVE value so a
   drifting lane cannot widen its own tolerance) applied to the five named
   observables; parity native-values cross-bound to the native endpoint;
   per-sample wall time cross-checked against its trajectory; trajectories
   must be contiguous from iteration 1 to the full budget (the fixed-budget
   metric is now genuinely fixed-budget); shared identity includes the
   effective-construction fingerprint; `audit.adjoint_route` required on
   custom lanes; all four receipt lane directories required; NaN/zero ratio
   guards; the endpoint binds to warm sample 6 by index, not list order.
3. Receipt semantics carried by worker-owned files
   (`benchmarks/run_jax_native_example_measurements.py`,
   `benchmarks/single_stage_speed_campaign_receipt.py`) are re-audited at
   promotion time; the runner's decorative hardcoded
   `inner_solver_success=True` was replaced by the actual observation value.
4. Corrections of record: the WIN gate additionally requires the custom
   lane's TTQ to be finite; when the Optax lane never reaches `J_target`,
   `ttq_vs_optax` passes by domination and the validator says so; the
   "quality clause satisfied by both lanes" wording reduces to the custom
   lane (the native clause is tautological by construction); cold compile
   time is stored in receipts but not separately analyzed by the validator;
   `measurement.json` is a campaign-specific per-sample record, not the
   legacy runner artifact schema; the upstream single-stage example's two
   legs run `maxiter=10` at `tol=1e-12` (stage-2) and `tol=1e-15`
   (single-stage); the 4.7e-5 → 1.7e-5 gradient-decay figures are
   diagnostic-only numbers from the superseded plan's mismatched-contract
   ladder runs and carry that caveat; the r3 κ≈1e3 conditioning figure is a
   prose estimate from the July measurement series with no campaign
   artifact; the positive-objective assumption behind `J_target` is part of
   this protocol; there is no amendment r1 (the numbering starts at r2).
5. Canonical invocation (previously unstated): the campaign is produced by
   `benchmarks/run_jax_native_example_measurements.py
   --single-stage-speed-campaign --scale native_default --artifact-root
   ~/simsopt-campaigns/single-stage-speed-20260804 [--gpu-index N
   --timeout-seconds S]` and judged by
   `benchmarks/validate_single_stage_speed_claim.py` with its defaults.
6. The validator now has its own test suite
   (`tests/benchmarks/test_validate_single_stage_speed_claim.py`) covering
   every verdict path and tamper class.

## Process hygiene (all incident-derived, non-negotiable)

- One measurement child per process; never share a process between lanes; no
  environment leakage into reference lanes.
- Watchers are killed by recorded PID, never by `pkill -f` pattern.
- Concurrent sessions: the tree is snapshot-tagged; any unexplained working
  tree drift halts the campaign for re-verification.
- Historical reduced-fixture (`mpol=ntor=1`) results are never substituted
  for native-default evidence.
