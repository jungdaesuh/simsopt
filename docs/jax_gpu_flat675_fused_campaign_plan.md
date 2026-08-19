# Flat-675 fused single-stage GPU campaign (F3) — charter

Status: DRAFT r4 (round 1: 12 required edits, addressed in r2; round 2:
9 findings N1–N9, addressed in r3; round 3: 5 findings F1–F5 plus the F6/
status-line bookkeeping pair from round 4, addressed in this revision).
Freezes at its commit sha; execution begins only after the genuine-675
fair-bar chain releases the box.

Operator directive (2026-08-19): land the flat-675 objective in the production
tree, wrap it in the certified fused `dispatch.minimize` lane, gate with
unit/oracle tests only, then run ONE fused timing campaign whose receipt
supersedes the archived 9.8× claim regardless of the fair-bar verdict. The port
is complete and crucible-PASSED across three rounds (`06a7588eb` →
`e4ef23765` → `fb0ad88d9`). This charter governs the one campaign.

Inheritance rule, stated precisely: this charter inherits the fair-bar
campaign's mechanics (`docs/jax_gpu_genuine675_fair_bar_plan.md`, freeze
`7b6d69041` + amendments A1–A3a) ONLY where this document explicitly says
"inherited". Every clause this document writes itself — verdict rules, work
matching, quality gates, budgets, governance — is complete here and does NOT
fall back to the parent. (Round-1 review showed silent fallback creates
unsatisfiable inherited gates; the fix is explicitness, not silence.)

## Claim under test

The fused on-device L-BFGS-B flat-675 single-stage lane
(`simsopt_jax.examples.single_stage_flat675.solve_single_stage_flat675`,
fp64, RTX 5090) is ≥ 1.10× faster than the native C++/simsoptpp lane —
**equal-budget** at B3/B37 and **quality-matched** at BQ — at equal-or-better
natively-adjudicated endpoint quality, on warm/persistent-cache repeated
workloads, measured on **process wall seconds**. Cold start (XLA compile) is
disclosed per rung (below), never claimed.

Scope sentence (mandatory in the receipt): this campaign's verdict speaks to
optimization launched from this one archived mid-trajectory native iterate,
not to an ensemble of start candidates.

## Timer law (no timer shopping)

- **Primary, all rungs: `process_wall_seconds`** of the timed child process,
  measured by the runner exactly as the fair-bar harness measures it.
- Report-only secondaries: in-process solve wall (the fused lane's warm solve
  time; the native lane's optimizer wall), steady per-eval rates.
- Every ratio in the verdict section and the supersession sentence uses the
  primary timer. Secondaries never appear in a claim.

## Shared optimizer policy (fail-closed identity)

Both lanes run the archived policy
(`schema simsopt.single_stage.genuine_675_dynamic.lbfgsb_policy.v3`, read from
the archived lane record): **method L-BFGS-B, `maxcor=300`, `maxls=8`,
`ftol=0.0`, `gtol=0.001`, unbounded, analytic jacobian, `maxiter` = the rung
budget.**

Code prerequisite (chartered): `solve_fused_lane` currently constructs
`SimsoptLBFGSBOptions` without `maxls` (dataclass default 20) and exposes no
way to set it. Before any timed leg: thread an optional
`lbfgs_line_search_max_steps` through `solve_fused_lane` (default preserving
current behavior, 20) and pin `FLAT675_LBFGS_MAXLS = 8` in the flat-675 binder
with the same archived-record provenance sentence as `FLAT675_LBFGS_HISTORY`.
The three port gates (parity / liveness / fused strict-transfer) re-run green
at the new binder before the freeze commit; the GATE-3 endpoint value is
re-recorded (its certified 2.4245e-14 was measured at the test policy
`gtol=1e-12/ftol=1e-15/maxls=20` and held only because the 3-iterate cap
bound; the charter does not assume it survives the policy pin unchanged).

`maxfun`: the fused lane caps `maxfun = maxiter × 20`. With `maxls=8` an
L-BFGS-B iteration costs at most 9 evaluations, so the cap cannot bind at any
chartered budget (37 × 9 = 333 < 740); stated here so the extra cap is not a
hidden stopping rule.

Policy-identity check (fail-closed, per leg): the harness serializes the
constructed policy `{method, maxiter, maxfun, gtol, ftol, maxcor, maxls}` for
both lanes, sha256s it, compares against this charter's frozen policy constant
for the rung, and records it in the leg row. Mismatch voids the leg.

## Input bundle eligibility (dated, pre-evidence)

The frozen input bundle (`84febc05d195d84c0802205b2b4c85ea1fa38faa7ff856ef…`)
carries `performance_eligible: false` from its origin as a parity input. As the
fair-bar charter did for its own scope, this clause — dated 2026-08-19, before
any F3 timed evidence exists — reclassifies the bundle as timing-eligible
**for this campaign only**, on the same grounds: the bundle is the only
bit-exact carrier of the archived start candidate, and both lanes consume the
identical bytes. The F3 harness verifies the bundle fail-closed before every
L1/L2 leg by invoking the fair-bar campaign-manifest loader
(`benchmarks/genuine_675_fair_bar.py`'s frozen-bundle validator binding the
member shas) — not the production `load_flat675_bundle`, which stays
verification-free by design. A leg on an unverified or drifted bundle is
voided.

Cross-charter dependency, declared: that loader hard-binds the **fair-bar**
charter sha (`CHARTER_SHA256`) and source-manifest sha — F3 therefore
consumes the fair-bar campaign manifest (`genuine-675-fair-bar-input.v1`)
under the fair-bar's enforced reclassification; F3's own clause above states
intent, the fair-bar's constant enforces it. The fair-bar charter sha the
loader binds at execution time is a component of F3's per-row contract sha
(Governance), so the dependency is recorded, not silent. A fair-bar amendment
that moves that constant fail-closes F3 legs; this is accepted because the
fair-bar campaign completes before F3 executes and its charter freezes with
it.

## Lanes

- **L1 (verdict): fused GPU** — production tree at the campaign freeze commit,
  5090, fp64, one device program end to end (no host callbacks; the GATE-3
  transfer discipline). Child processes pin the production tree via a
  fail-closed import-origin guard.
- **L2 (verdict): native C++ CPU** — bit-identical fair-bar native lane
  (instrument worktree @ `1c23f6c5`, frozen bundle), with the fair-bar per-leg
  child-observed conformance gate (inherited). Import-origin guards are
  **per-child-process**: L2 children and oracle children pin the instrument
  worktree (the oracle's bundle validator demands a clean `1c23f6c5`
  checkout); L1 children pin the production tree. The two guards never apply
  to the same process, so there is no contradiction.
- **L3 (diagnostic, non-verdict): host-loop GPU** — the fair-bar GPU lane at
  the same budgets. Disclosed confound: L3 differs from L1 in BOTH the loop
  (host scipy vs fused) AND the tree (instrument vs production; the eval
  graphs are numerically equivalent at ~1e-15, not byte-identical), so L3
  bounds the fused-loop delta rather than isolating it. Report-only.

## Budgets (rungs) and denominators

- **B3** — continuity rung. Native config: the fair-bar **B3 matrix's own
  selection** (omp16 SMT-assisted, 52.70 s median, run dir
  `20260819T080502Z-native-matrix-b3-3720552`) — B3's own sweep, not an
  extrapolation.
- **B37** — headline rung. Native config: the fair-bar **B37 matrix's
  selection** (in flight at charter time — a forward reference).
  Contingency: if the fair-bar B37 matrix ends without a selection
  (`NOT_PRODUCED`), F3 runs its own five-config sweep (omp 1/2/4/8/16, three
  reps, fair-bar selection rule) before its B37 pairs.
- **BQ (time-to-quality)** — same native config as B37. Sweep proximity,
  disclosed: BQ's native legs run at `n*` ≤ 37 iterations (BQ protocol), so
  the B37 sweep is the nearest on-scale sweep; the extrapolation from 37 to
  `n*` is downward on the same config and is disclosed in the receipt rather
  than re-swept (a per-`n*` sweep would itself require knowing `n*`, which
  the searches produce after the B37 rung).

Program-law statement: the B3 and B37 native denominators are stated, swept
at their own rung, and pinned. BQ's denominator is the B37-swept config
applied at `n*` ≤ 37 — a downward on-config extrapolation, disclosed in the
receipt and not re-swept (per the BQ bullet above). The interleave law and
N=5 pairs-per-rung law apply to every rung.

## BQ protocol (selection-bias closed, symmetric)

BQ asks: how long does each lane take to reach a common natively-adjudicated
quality target? Both lanes receive the SAME minimized-budget treatment — no
callbacks anywhere, and neither lane runs a budget the other's discipline
would not grant it:

1. **Target.** After the B37 rung completes, `Q*` = the median
   oracle-evaluated endpoint objective of the five B37 native pair legs.
   Frozen before any BQ work. (Dependency, stated: **B37 `NOT_PRODUCED` ⇒ BQ
   `NOT_PRODUCED`** — this is the one chartered exception to "a voided rung
   never blocks the others".)
2. **Symmetric budget searches (untimed, disclosed).** For EACH lane
   independently, find the smallest `maxiter` whose oracle-evaluated endpoint
   objective ≤ `Q*`: fused `m*` and native `n*`, each by the same procedure —
   start at 37, double upward if short, then bisect. Every probe is an
   untimed solve (fused warm; native with its primer discipline); the
   endpoint test is always the **oracle's** evaluation, never a lane's
   self-reported objective. Caps per lane: ≤ 12 probes, `maxiter` ≤ 1024,
   search wall ≤ 2 h — any cap breached ⇒ BQ = `NOT_PRODUCED`. `m*` and `n*`
   are frozen before the first timed BQ leg and never re-searched per pair.
   (`n*` ≤ 37 by construction since `Q*` is native's iterate-37 quality;
   the search discipline is identical anyway.)
3. **Timed pairs.** Five interleaved pairs: fused at `maxiter = m*` vs native
   at `maxiter = n*`. Reported number per leg: the primary timer of that
   leg's fresh child process. Search solves are never eligible as timed legs
   (no min-over-search, on either side).
4. **Quality gate per pair:** BOTH endpoints' oracle objectives ≤ `Q*`
   (re-verified per timed leg, not assumed from the searches).

## Work matching (per rung, explicit — replaces any inherited clause)

- **Evaluation counters, defined once:** the native (L2) counter is
  `work_counts.compact_candidate_evaluations` — the lane record carries no
  `nfev`; the fused (L1) counter is `OptimizerResult.nfev`. Both count full
  value-and-gradient evaluations of the same 675-DOF objective (the archived
  policy runs with the analytic jacobian supplied, so each native compact
  evaluation is one (f, ∇f) pair, exactly what one fused `nfev` is). Every
  anchor formula below is stated in these units.
- **Counter liveness (fail-closed, per leg, all rungs):** L1 `nfev` > 0 and
  L2 compact evaluations > 0, or the leg is voided — `dispatch.py` populates
  `nfev` via a `getattr(..., 0)` default, and a silently-missing counter must
  void the leg rather than reach an anchor formula as 0.
- **B3/B37 (fixed-budget rungs):** both lanes must report `nit` equal to the
  rung budget, fail-closed. Both counters are recorded per leg.
  Evaluation-count **identity is NOT required**: the two implementations run
  independent line searches, and after the cross-implementation fork point
  their trajectories legitimately differ. Endpoint comparability is carried
  by the quality gate, not by trajectory identity.
- **BQ:** work is matched on quality by construction (`Q*`), with both
  budgets independently minimized under the same search discipline.

## Quality gate (fail-closed, every timed endpoint, oracle-adjudicated)

Every timed L1 endpoint is evaluated by
`benchmarks/genuine_675_fair_bar_oracle.py` — the native cross-evaluator whose
inner solve is anchor-invariant by construction. The harness computes the
endpoint's own `(iota, G)` host-side after timing (via the flat675 package's
y-solve at the endpoint `x`) and passes it as the oracle anchor; no production
code change is needed for this.

- **B3/B37:** oracle-evaluated fused endpoint objective ≤ oracle-evaluated
  paired-native endpoint objective × (1 + 1e-10). Deliberately **one-sided**:
  this campaign claims equal-or-better quality at equal iteration count for a
  minimization, and does not claim trajectory identity — a two-sided gate
  would demand identity the design explicitly forgoes. The 1e-10 matches the
  fair-bar endpoint tolerance (round-1's 1e-9 is withdrawn). Gradient clause
  (chartered bound): the oracle-evaluated gradient-∞ at the fused endpoint
  must be ≤ **2×** the oracle-evaluated gradient-∞ at the paired native
  endpoint (K=2, fixed pre-evidence: equal-budget endpoints share a
  stationarity scale; the factor tolerates legitimate post-fork line-search
  divergence while rejecting a lower-objective-but-non-descending endpoint).
  Both values are reported per pair.
- **BQ:** both endpoints' oracle objectives ≤ `Q*` per timed leg; the fused
  endpoint's gradient-∞ ≤ 2× the paired native endpoint's (same K — the BQ
  endpoints share a *quality* scale rather than a budget: both sit at the
  first crossing of the same `Q*`, so their stationarity scales are
  comparable for the same reason, and K=2 tolerates the same line-search
  divergence).
- Any oracle failure or gate miss voids the pair (`NOT_PRODUCED`).

## Verdict rule (dual, per rung — both must hold)

1. **Live rule:** five interleaved (L1, L2) pairs; median per-pair
   process-wall speedup ≥ 1.10 AND every pair > 1.00. Min/median/max of both
   lanes' walls are reported per rung.
2. **Anchor rule:** archived-anchor / L1-median ≥ 1.10, where (counters per
   the work-matching section: L2 = `compact_candidate_evaluations`, L1 =
   `nfev`, both counting (f, ∇f) evaluations)
   - B3: anchor = the archived B3 process wall, 58.702 s;
   - B37: anchor = archived sustained per-compact-evaluation mean
     (52.807/9 s) × **min(median L2 compact evaluations, median L1 `nfev`)**
     across the rung's pairs — the smaller of the two lanes' counts, so the
     GPU is never credited with native work it did not do, and never charged
     extra line-search evaluations at native prices;
   - BQ: anchor = archived per-compact-evaluation mean × **median L2 compact
     evaluations at `n*`** (native's minimal cost of producing `Q*`, priced
     at the uncontended archived rate — the time-to-quality currency, stated
     and disclosed).
   The anchors were timed on an uncontended box, so partition-era contention
   can only make the anchor rule harder to pass.

Terminal vocabulary (per rung): **WIN** (both rules hold),
**CLOSED_BOUNDED_NEGATIVE** (rules fail with all gates green),
**NOT_PRODUCED** (gates voided the rung). No other outcomes exist.

## Priming and cold-start (symmetric, per rung)

- Warm legs: BOTH lanes prime with a **separate discarded primer child
  process** immediately before each timed child (the fair-bar's native primer
  law, applied to both lanes). The fused primer child populates the
  persistent XLA compile cache; the timed fused child then contains exactly
  one solve plus cache-load startup, and the timed native child exactly one
  solve plus startup — each lane's primary timer contains one solve. No
  in-process primers anywhere.
- Cold disclosure: **one fresh-cache disclosure pair per rung** (three total):
  both lanes cold, no primers, fused with the XLA persistent cache directory
  cleared. Report-only, never part of the verdict — matching the finite-build
  precedent's cold-is-disclosed-not-claimed discipline.

## Governance (complete here; nothing inherited silently)

- **Amendment rule:** amendments are dated, pre-evidence for the clauses they
  touch, appended to this document, and never edit frozen text in place. The
  harness records the full append-only charter sha lineage (fair-bar A3a
  mechanism); rows bind the sha current at execution and `validate` accepts
  the lineage.
- **Per-row contract sha:** every leg row binds
  sha256(F3 charter sha ‖ rung policy sha ‖ bundle campaign-manifest sha ‖
  fair-bar charter sha bound by the bundle loader ‖ production-tree commit ‖
  instrument commit). `validate` refuses rows bound to a foreign contract.
- **Validate entrypoint:** the F3 harness ships a `validate <run-dir>`
  subcommand that recomputes every gate and the rung verdict from the run
  directory alone.
- **Caps and aborts** ("leg" defined): a *timed leg* is a child process whose
  primary timer enters a verdict or disclosure; caps: ≤ 51 timed legs
  (3 rungs × 5 pairs × 2 = 30, + 6 cold-disclosure, + 15 for the B37
  contingency sweep at 5 configs × 3 reps) and ≤ 130 solve-executing child
  processes total, with the arithmetic shown: 51 timed + 45 primers (one per
  timed leg, none for the 6 cold legs) + 1 fused budget-search child (the
  ≤ 12 fused probes run in ONE reusing child process — they are untimed, so
  process isolation buys nothing and `PreparedFusedLaneSolve` supports
  reuse) + ≤ 12 native probe children + ≤ 12 native probe primers = ≤ 121;
  ≤ 12 h campaign wall; three `NOT_PRODUCED` pairs in a rung abort that
  rung; a voided rung never blocks the others — with the single chartered
  exception B37 → BQ (BQ protocol step 1). Breaching any cap ends the
  campaign at the current rung boundary; completed rungs keep their
  verdicts, unstarted rungs are `NOT_PRODUCED`.
- **Non-amendable post-evidence:** thresholds (1.10, 1.00, 1e-10), pair count
  (5), budgets (3/37/BQ-as-defined), the timer law, and the anchor formulas.

## Instrument construction (chartered work, pre-freeze)

The L1 instrument does not exist yet. Before execution:
1. `benchmarks/flat675_fused_campaign.py` — the F3 harness: partition +
   conformance + policy-identity + bundle verification + oracle wiring +
   `validate`, reusing the fair-bar implementations by import where they are
   lane-agnostic. New file ⇒ execution-source manifest `--admit` + count
   twins, same commit.
2. The `maxls` threading + binder pin (Shared policy section) + port-gate
   re-run. Cascade: `fused_lane.py` and `single_stage_flat675.py` are both
   execution-source manifest members — the same commit regenerates the
   manifest (digest refresh for the two edited members; no count change) and
   `test_regenerate_execution_source_manifest.py` runs green.
3. An artifact-free contract test for the harness's pure logic (policy sha,
   anchor arithmetic, verdict rules), plus reviewer strict PASS on the
   harness before the first timed leg.

## Protocol inherited by reference (explicit list — nothing else)

From the fair-bar charter: the partition protocol (reserved CPUs {0–7,32–39},
confinement daemon, bracketed fail-closed partition-integrity gate with the
GPU ≤ 5% clause applying to native legs); the per-leg child-observed
conformance gate (env echoes, resolved `omp_get_max_threads`, granted
affinity, provenance shim); the primer law (applied to both lanes per the
Priming section); the interleave law; the run-dir row/lane/provenance file
schema; the **GPU campaign lock** (one GPU campaign owns the device at a
time; no foreign GPU work during L1 legs); and the **clean-tree
requirement** — the instrument worktree clean at `1c23f6c5` (the oracle's
bundle validator independently enforces this) AND the production tree clean
at the freeze commit for every L1 child, recorded per row. Sequencing: no F3
leg starts while the fair-bar chain owns the box.

## Evidence & receipt

Run dirs under `~/simsopt_mixed_artifacts/flat675_fused_campaign/`; campaign
manifest binds the charter lineage, harness commit, bundle campaign-manifest
sha, production commit, instrument commit. Receipt lands in `docs/receipts/`
with: per-rung dual verdicts with min/median/max; the policy-identity shas;
the L3 diagnostic with its confound named; cold-start disclosure per rung;
partition disclosure; the scope sentence; and the supersession section below.
Reviewer strict PASS before the receipt commit.

## Supersession and conflict rule

The archived 2026-07-21 r3 "9.8×" is an **optimizer-wall** number
(53.603/5.471 s); its process-wall basis is 7.469×. Two receipts touch it:

- The **fair-bar receipt** is the sole adjudicator of the archived number
  itself (B3 continuity rung, its own instrument, per its charter) — it owns
  the past.
- **This receipt** replaces the flat-675-formulation GPU-speed claim going
  forward (production instrument, fused lane) — it owns the present. It
  supersedes the archived 9.8× as the program's citable flat-675 number
  regardless of the fair-bar outcome.

If the two verdicts differ (e.g. fair-bar bounded-negative at B3, F3 WIN),
there is no contradiction to resolve: different instruments, different
claims, both receipts state this rule and cite each other. The scoreboard
carries the fair-bar verdict as the archived claim's fate and the F3 verdict
as the current claim, each with its timer named.

## Non-goals

- No mixed-precision lane (dropped at the port; fp64 only).
- No A100 lane (supplementary silicon stays out of verdicts).
- No projected-route / two-stage-example comparison (that bar needs
  P-as-matvec — separate track).
- No re-litigation of the fair-bar verdict.

---

## Amendment 1 (2026-08-19, pre-evidence — no F3 timed leg has run)

**Timer-tail resolution.** The frozen Quality-gate sentence "the harness
computes the endpoint's own `(iota, G)` host-side **after timing**" is in
tension with the Timer law ("`process_wall_seconds` of the timed child …
exactly as the fair-bar harness measures it"): the endpoint inner-state
computation is not a bare 2-column QR — it rebuilds the Boozer system
(grouped Biot–Savart over the surface grid plus a fresh XLA compile) and
measured 5.257 s (2.59 % of a B3 child's wall) on the instrument review's
dry run, while the fair-bar native child stops its clock only after ALL its
work, certificates included. Resolution, in the conservative (anti-GPU)
direction: **the endpoint (iota, G) computation is charged INSIDE the L1
primary timer** — the child stops its clock immediately before writing its
result payload, mirroring the instrument's native lane
(`genuine_675_dynamic_lane.py:418`). The Quality-gate sentence is amended
to read "after the solve" (it remains outside the *solve*, inside the
*timer*). The per-leg row additionally records the endpoint computation's
own duration, report-only.

**Disclosed counter-asymmetry (not engineered away):** the L1 child's
timer necessarily starts before its `import jax` (the import-origin guard
must run first), while the fair-bar native child's module-level imports
precede its clock start. This term is anti-GPU and inherent to the two
instruments; the receipt discloses it.

This amendment rewords a measurement clause in the conservative direction
before any timed evidence exists; thresholds, budgets, pair counts, anchor
formulas, and the timer's identity (process wall of the timed child) are
unchanged.

## Amendment 2 (2026-08-19, pre-evidence — no F3 timed leg has run)

**Disclosure verdict named.** The verdict vocabulary gains
**`FRESH_REPORTED`** for the per-rung cold-disclosure pairs (matching the
fair-bar campaign's token): a disclosure pair is *reported*, enters no
verdict, and cannot be `NOT_PRODUCED` (which means "gates voided the
rung" — a disclosure never attempted a verdict). `WIN` /
`CLOSED_BOUNDED_NEGATIVE` / `NOT_PRODUCED` remain the only *rung* verdicts;
`FRESH_REPORTED` labels disclosure evidence only.

**Process-cap arithmetic corrected.** The r4 derivation (≤121) omitted the
oracle children. The instrument's accounting is the chartered one: each
warm pair spawns **6** solve-executing children (2 primers + 2 timed +
2 oracle), each cold pair 4; full campaign shape = 3 rungs × 5 warm pairs
(90) + 3 cold pairs (12) + BQ searches (1 fused search child + ≤12 native
probe children + ≤12 probe primers = 25) = **127 ≤ 130**. The B37
contingency sweep cannot fire (the fair-bar B37 matrix selected omp16,
receipt `4d155174c`); were it ever revived, its ~30 additional children
would breach the cap and the rung-admission rule refuses the rung — a
dated amendment must precede any such run.

Both clauses are pre-evidence; thresholds, budgets, pair counts, anchor
formulas, and the timer law are unchanged.
