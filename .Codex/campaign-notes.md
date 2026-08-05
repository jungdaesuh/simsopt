# Single-stage speed campaign implementation notes

## Amendment r2 resolution

The historical conflicts below are superseded by the user-approved r2
amendment in `docs/single_stage_speed_campaign_protocol.md`. Measurement-lane
parity is now a direct native comparison using the frozen tolerance table, so
the ordinary arbiter is outside the measurement path. Iteration-cap endpoints
are campaign-valid when their FP64 observables are finite, their inner solves
succeed, and their direct parity rows pass. Validator v2 independently checks
the receipt's lane identities and shared initial/input/configuration identity
against `campaign-20260804-frozen-r2` *(baseline later re-tagged: the live
validator enforces `campaign-20260804-frozen-r4` — see the r3/r4 entries
below)*.

The sections below remain as the historical rationale for amendment r2; they
are not current collector blockers.

## Frozen arbitration conflict

The protocol requires `jax_gpu_optax` parity rows to be filled from the existing
arbitration path while also requiring truthful driver provenance. The frozen
arbiter rejects every JAX observation whose driver contains `optax`
(`examples/jax/parity/arbiter.py`, lines 109-115), and its accepted execution
intents only resolve the existing `jax_gpu_fast` or `jax_gpu_parity` backend
modes (lines 82-104).

Therefore a truthful Optax observation cannot pass the existing arbitration
path without changing a frozen file. This implementation will not modify the
arbiter. The receipt writer must fail closed when Optax arbitration evidence is
absent; synthetic schema tests may use explicitly supplied fake parity rows but
do not constitute campaign evidence.

## Frozen validator coverage mismatch

The protocol requires truthful backend/driver provenance, a matched initial
point and iteration cap, and parity coverage for objective, iota, volume,
non-QS ratio, and Boozer residual. The frozen validator does not inspect the
backend mode or driver, does not compare the initial point, only rejects a
trajectory that exceeds the iteration budget, and accepts any non-empty set of
self-described scalar parity rows. Producer-side checks can fail closed for
these fields, but they cannot make the stated definition of done independently
machine-checkable. Changing the validator would violate the frozen-files
contract, so terminal certification requires the protocol owner to apply these
additional checks or publish and freeze a revised validator.

## Budget-exhaustion validity mismatch

The protocol explicitly reports rather than gates outer-gradient stationarity,
because the native-default workload normally reaches its iteration cap. The
existing parity observation marks that endpoint as unsuccessful, while both the
legacy measurement runner and frozen arbiter reject unsuccessful observations.
A campaign-specific validity gate can correctly require finite FP64 observables
and successful inner solves without weakening ordinary parity behavior, but the
Optax arbiter conflict above still prevents fully arbitration-derived Optax
parity evidence.

## Retained parity-lane and endpoint-arbitration mismatch

The protocol says the existing `jax_cpu_parity` and `jax_gpu_parity` lanes keep
running for the science gates. The four-lane timing collector intentionally
contains only `native_cpu`, `jax_cpu_fast`, `jax_gpu_fast`, and
`jax_gpu_optax`; it cannot truthfully substitute a timing observation for a
parity-lane receipt. Moreover, ordinary arbitration applies final-parameter
and endpoint-certificate routes in addition to the campaign's five required
final observables. Different outer optimizers are not expected to share final
parameters at a fixed budget.

A legitimate arbitration-derived Optax comparison would require a separately
specified endpoint-evaluation relationship that binds evaluator receipts to
the exact optimized endpoint and preserves the Optax optimizer provenance.
That is a new scientific contract, not instrumentation under the frozen
relationship. This implementation therefore does not create such a route and
does not rename the Optax driver, copy fast-lane rows, or force scientific
success. The real collector remains fail-closed until the protocol owner
supplies and freezes a compatible arbitration contract.

## Coordination notice — Claude session, 2026-08-04 19:35 EDT

Two campaign launches from the Claude session were SIGTERM'd by the
controller session (evidence: rollout-2026-08-04T14-03-55, kill -TERM of the
r2 collector's native child at 19:13). Meanwhile the r1 native failure was
root-caused: the collector inherited the launcher's OMP/BLAS threading env,
forking FP trajectories between launch contexts; the collector now scrubs
numerical-env prefixes and pins deterministic threading (all 77 affected
tests green). Any campaign launched from ANY context now produces identical
native trajectories.

Single-launcher rule, effective now: exactly one session launches the
campaign; the other validates. Options are recorded with the user. Until the
user designates the launcher, DO NOT launch and DO NOT kill campaign
processes. If you believe a campaign process must die, write the reason here
first, with the PID and evidence.

## Blocker diagnosis — Claude session, 2026-08-04 23:2x EDT

The campaign's current blocker is the custom fast lane itself, not machinery:

- native_cpu at native_default: healthy (inner Newton converges from the
  canonical bundle; residual 2.3e-29; full 1,000-iteration trajectory).
- jax_gpu_fast at native_default: first gradient evaluation returns the
  all-NaN sentinel (`final:gradient`), zero accepted iterations. Reproduced
  standalone twice; env verified (fp64, X64, cuda, strict backend).
- jax_cpu_fast at native_default: IDENTICAL failure — so this is
  scale-dependent solver behavior, not GPU numerics.
- Bounded-scale suites pass; the fast lane has apparently never completed a
  native_default run (the pre-repair 2h GPU timeout masked this).

Repro scripts: ~/simsopt-campaigns/native-inner-probe/run_probe_gpu_fast.py
and run_probe_cpu_fast_short.sh (both use the persisted canonical bundle in
that directory). The offending key was identified via the named-key
ArtifactValidationError now in examples/jax/parity/artifacts.py.

Fix constraints: the inner solve must legitimately succeed (or the config
divergence vs the native inner options be root-caused); no tolerance
weakening, no sentinel suppression, frozen files unchanged.

## Amendment r3 AUTHORIZED — Claude session, 2026-08-05

User approved. Protocol amended (see docs/single_stage_speed_campaign_protocol.md,
Amendments r3); frozen baseline re-tagged campaign-20260804-frozen-r3; the
validator now enforces against r3. Authorization scope:

1. Route ALL THREE custom lanes (jax_cpu_custom, jax_gpu_custom,
   jax_gpu_optax) to the direct FP64 LU exact-adjoint. Matched lanes stay
   matched — no lane keeps operator-GMRES.
2. Record the adjoint route in the endpoint audit block (truthful
   provenance).
3. Before relaunching the campaign: one direct-vs-parity gradient agreement
   check at the native_default initial point (finiteness is already proven;
   record the agreement numbers).
4. GMRES stagnation at κ≈1e3 is a follow-up investigation ticket, NOT a
   campaign blocker. _point_direction_chunk_reduce's remaining padding is a
   second follow-up (same bug class as the fixed one).

Then relaunch the campaign: GPU quiet gate enforced, cpuset isolation as
before, receipts to ~/simsopt-campaigns/single-stage-speed-20260804, frozen
validator chained. The Claude session audits the receipts.

## Amendment r4 FILED — Claude session, 2026-08-05

Divergent-lane physics contract added to the protocol (user-approved):
two-tier comparison (parity lanes = correctness, speed lanes =
certified-quality-vs-wall-clock), equation-anchored certification (only
accepted+projected iterates in receipts), cross-evaluator endpoint audit,
optimizer-independent field-line validation, and the different-basin ruling.
Frozen baseline re-tagged campaign-20260804-frozen-r4; validator enforces
against r4. No divergent lane may promote receipts until validator
enforcement of the cross-evaluator and tracing checks ships with it.

## Amendment r5 nonblocking follow-ups — 2026-08-05

- Endpoint certification is bound only to warm sample 6 of 7. Consider
  binding an endpoint to every measured sample in the next receipt-schema
  revision; this is not a blocker for the current r5 campaign.
- Receipt semantics remain implemented in the intentionally unfrozen
  collector and receipt-writer files. Promotion-time re-audit of both files
  is the compensating control; this is not a blocker for collection.

## Campaign closeout — 2026-08-05

The campaign is closed as `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING`. The
closeout record is
`docs/single_stage_speed_campaign_results.md`. No complete r5 `campaign.json`
was produced, so the frozen validator did not emit `WIN`, `TIE`, or `LOSS`;
the closeout must not be described as a protocol verdict. The preserved 5090
partial trajectories, the r3 direct-vs-parity gradient agreement, and the
hash-bound Landau A100 T1 findings are the bounded evidence. A full T2 Optax
run is declined. The two r3 follow-up tickets remain nonblocking and are not
silently folded into this campaign.
