# Flat-675 fused single-stage campaign (F3) — terminal receipt

**Verdict: `WIN` at all three rungs** (2026-08-19, preregistered dual rule,
primary timer `process_wall_seconds` of the timed child — every ratio below
is that timer unless explicitly labeled report-only). The production-tree
fused on-device L-BFGS-B flat-675 lane
(`simsopt_jax.examples.single_stage_flat675`, fp64, RTX 5090) beats the
matched native C++/simsoptpp lane at equal budget and at matched quality:

| Rung | L1 fused (median) | L2 native omp16 (median) | Pair ratios | Live median | Anchor / L1 | Gates |
| --- | --- | --- | --- | --- | --- | --- |
| **B3** (equal budget 3) | 35.09 s | 59.05 s | 1.55 / 1.59 / 1.67 / 1.69 / 1.70 | **1.67×** | 58.702 / 35.09 = **1.67×** | all green |
| **B37** (equal budget 37, headline) | 39.82 s | 307.11 s | 7.63 / 7.66 / 7.70 / 7.73 / 7.75 | **7.70×** | 281.637 / 39.82 = **7.07×** | all green |
| **BQ** (quality-matched, m\*=n\*=37) | 38.69 s | 282.87 s | 7.28 / 7.28 / 7.36 / 7.36 / 7.37 | **7.36×** | 287.505 / 38.69 = **7.43×** | all green |

Five interleaved pairs per rung, alternating order, symmetric discarded
primer children, per-leg policy-identity shas (archived policy: L-BFGS-B,
`maxcor=300, maxls=8, ftol=0, gtol=1e-3`), counter-liveness and `nit ==
budget` fail-closed on both lanes, bracketed partition-integrity gates,
zero voided pairs anywhere. Anchors per the frozen formulas: B3 = the
archived process wall 58.702 s; B37 = 52.807/9 × min(median L1 nfev = 48,
median L2 compact = 49) = 281.637 s; BQ = 52.807/9 × median L2 compact at
n\* (49) = 287.505 s. Anchors were timed on an uncontended box, so box
contention only makes the anchor rule harder.

Charter: `docs/jax_gpu_flat675_fused_campaign_plan.md`, frozen `b7ec63b6e`,
amended pre-evidence A1 (`595b7da60`, endpoint inner-state charged inside
the timer) and A2 (`e8625f691`, `FRESH_REPORTED` vocabulary + cap
arithmetic), post-campaign A3 (`2181ecbf1`, implementation-state record:
status-line correction, the two ledger-accounting defects, the cap breach,
m\*=n\*=37) and A3a (`6d3b179d3`, counterfactual-admission correction).
Five crucible review rounds froze the charter; five more cleared the
instrument; the fused port itself carries its own three-round strict PASS
(`06a7588eb` → `e4ef23765` → `fb0ad88d9`).

## Supersession (exact scope)

This receipt supersedes the archived 2026-07-21 "flat-675 fp64-GPU 9.8×"
as the program's citable flat-675 GPU-speed claim going forward, **on
`process_wall_seconds`, for the production fused lane versus the July
host-loop instrument**. The archived number itself was separately
adjudicated by the fair-bar campaign (`docs/receipts/genuine675_fair_bar.md`,
sealed `4d155174c`), which owns the past: the July *host-loop* instrument
measures 8.07× at B3 (10.33× on the archived claim's own optimizer-wall
timer) and 25.87× at B37. Different instruments, both receipts state this
rule, no contradiction to resolve.

**The two GPU lanes side by side (same timer, same rungs — required
context, not a caveat):**

| | fused production lane (this receipt) | July host-loop lane (fair-bar, chartered L3 diagnostic) |
| --- | --- | --- |
| B3 | 35.09 s (1.67×) | 6.80 s (8.07×) |
| B37 | 39.82 s (7.70×) | 11.11 s (25.87×) |

The fused lane is 5.2× / 3.58× **slower** than the host-loop lane at these
budgets on process wall: its child pays a ~34–35 s per-process floor
(interpreter + jax import + tracing/lowering the whole-program graph — the
persistent XLA cache spares compilation only), while the host-loop child
starts in seconds. The L3 comparison is the chartered diagnostic and
carries a named confound (different loop AND different tree). 7.70× must
never be read as beating 9.8× or 25.87× — those are different instruments'
numbers on the same physics.

Report-only secondaries (never claims): the fused lane's marginal cost is
≈ 0.118 s per evaluation from the timed rungs' own delta
((39.82 − 35.09)/40 evals), corroborated at 0.120 s/eval by the untimed
budget-search child's in-process slope (a `PreparedFusedLaneSolve`-reuse
quantity, inadmissible as a timed leg by charter). Against native's
5.867 s per compact evaluation that is a derived ≈ 49× per-evaluation
rate — an in-process, marginal quantity, sourced from the search child and
the rung deltas, not a process-wall speedup. The per-leg
`endpoint_inner_state_seconds` ≈ 0.69 s is charged inside the primary
timer per Amendment 1.

## Physics (oracle-adjudicated, per rung — not flattened)

Every timed L1 endpoint was cross-evaluated by the instrument's native
oracle (anchor-invariant closed-form inner solve); oracle values are
bitwise-stable across all five pairs of each rung:

| Rung | fused oracle objective | native oracle objective | relative |
| --- | --- | --- | --- |
| B3 | 1.8133486877705736 | 1.8133486877704454 | +7.1e-14 (fused slightly worse; one-sided 1e-10 gate slack 1.81e-10 vs excess 1.28e-13) |
| B37 / BQ | 0.013957201998031181 | 0.013957201998345709 | −2.25e-11 (**fused better**) |

Endpoint gradient-∞ at B37/BQ: fused 4.0370817079 vs native 4.0370817173 —
far inside the K=2 gate. `Q*` = 0.013957201998345709 is bitwise identical
to the fair-bar campaign's B37 native endpoint and to the archived-lineage
native trajectory. The native B3 endpoint is bitwise identical to the
fair-bar's. The fused child's `lane.json` records `success: false` — that
is the `maxiter` stop of a budget-capped solve, by design, not a failed
gate.

## BQ rung details

The symmetric budget searches (untimed, capped, oracle-adjudicated,
frozen before the timed pairs) returned **m\* = n\* = 37**: probe 36
missed `Q*` by 4.5% (0.014584 vs 0.013957) on BOTH lanes — the two
implementations' probe-objective ladders are identical to six decimals at
every budget (37/18/27/32/34/35/36). BQ is therefore a quality-gated
remeasure of B37, per protocol — a consistency confirmation, not a new
physics result. Both endpoints re-verified ≤ `Q*` on every timed leg.

## Cold-start disclosures (`FRESH_REPORTED`, N=1 per rung — reported, not claimed)

| Rung | fused cold | native cold | ratio |
| --- | --- | --- | --- |
| B3 | 191.73 s | 90.31 s | **0.47× (fused loses)** |
| B37 | 184.3 s | 282.2 s | 1.53× (fused wins) |
| BQ | 184.4 s | 283.2 s | 1.54× (fused wins) |

The cold fused child pays the full XLA compile (~150 s). At budget 3 that
dwarfs the solve (finite-build precedent); at budget 37 one solve amortizes
it. N=1 cannot satisfy the five-pair rule, so no cold-start claim is
minted in either direction.

## Disclosures

- **Scope:** this campaign's verdict speaks to optimization launched from
  this one archived mid-trajectory native iterate, not to an ensemble of
  start candidates.
- **Denominator vs fair-bar:** F3's native legs ran ~7% slower than the
  fair-bar's at the same omp16 config (59.05 vs 54.97 s at B3; 307.11 vs
  287.74 s at B37) — a busier box (active desktop during part of the
  campaign; partition + confinement held for CPU, and the GPU ≤5% gate
  fail-closed the native legs repeatedly until quiet windows). Live ratios
  are correspondingly fatter; the anchor rule (uncontended archived bar)
  bounds this and passed at every rung. Never read 7.70× against the
  fair-bar's 287.74 s denominator.
- **Cap ledger — one cap was breached in fact (Amendment 3):** timed legs
  36 ≤ 51 and campaign wall ~7 h 10 m < 12 h were respected, but the
  solve-child cap was not. The executed ledger recorded 102 children (the
  six pair phases only); the budget-search phase — never accumulated, an
  accounting bug — spawned **35** more (7 fused probe children + 7 fused
  oracles + 7 native probes + 7 native primers + 7 native oracles, counted
  from the run directory), for a true total of **137 > 130**. The breach
  was invisible while it happened because the search was uncounted.
  Counterfactual under correct accounting (projected rung admission):
  before the BQ pairs the ledger reads 4+30+30+35 = 99, and 99+30 =
  129 ≤ 130 **admits the BQ pairs** — all three verdict rungs were within
  cap; only the two report-only cold disclosures (129+4 = 133 > 130) would
  have been refused as `NOT_PRODUCED` (verified against the live
  `admits()`: sole breach `solve_children_133_over_130`; timed legs 32/51
  and wall 6.93 h/12 h do not bind). The search phase itself carries no
  admission gate — it is charged after the fact by design (Amendment 3a). The breach therefore materialized
  entirely in disclosure evidence that enters no verdict. The accounting is
  fixed forward (wall accumulation, search and sweep child charging); the
  sealed `campaign_state.json` (102 / 0.0 s) is disclosed as undercounting,
  not retro-edited. Every executed leg passed its own gates; the cap is a
  machine-time budget, not a verdict input.
- **Operational incidents (all fail-closed, zero contaminated evidence):**
  four instrument seams surfaced at launch and were fixed pre-evidence
  (`c50531f69` legacy-precision env rejection; `b8db9df44` campaign/bundle
  manifest split; `580217e0c` oracle gradient key; plus the chartered
  `maxls` pin at `020fb0c8b`); the sequencer (session tooling, not the
  instrument) crashed once on a lane-key mismatch costing ~2 h 17 m idle
  between the search and the BQ pairs, with no evidence effect; aborted
  partial run dirs are quarantined under `discarded-blocked-launch/`, never
  validated.
- **Validation:** all six rung/disclosure run directories validate
  `valid: true` with recomputed verdicts equal to the recorded ones, via
  `benchmarks/flat675_fused_campaign.py validate <run-dir>` (per-row
  contract shas, policy shas, counter liveness, quality gates, and both
  verdict rules recomputed from bytes). The budget-search directory is
  documented by its own manifest and per-probe oracle records; `validate`
  fail-closes on it (`run directory holds no rows` — the search writes no
  F3 rows by design), a documented property rather than a gap: the search
  is untimed and non-verdict.

## Evidence

Tracked under `docs/receipts/evidence/flat675_fused_campaign/` — the
`manifest.json` of all seven run directories (smoke cold-B3, pairs-B3,
pairs-B37, budget-search, pairs-BQ, cold-B37, cold-BQ), each binding the
F3 charter lineage, the fair-bar campaign-input manifest (`2a381125…`,
minted per run root from the sealed bundle `84febc05…`), the production
commit (`580217e0c`, clean, all legs) and instrument commit (`1c23f6c5`,
clean), per-row contract shas, and per-leg policy shas. Raw run
directories at `~/simsopt_mixed_artifacts/flat675_fused_campaign/`.

Instrument: `benchmarks/flat675_fused_campaign.py` +
`flat675_fused_campaign_contract.py` + `flat675_fused_lane_child.py`
(commit chain `a3639915a` … `580217e0c` at execution; post-campaign
accounting fixes and charter amendments follow on the same branch; five
adversarial review rounds to strict PASS before the first timed leg, plus
live dry runs reproducing the fair-bar's native endpoint bitwise). Native
lane and oracle: the fair-bar machinery imported unmodified, instrument
worktree pinned clean at `1c23f6c5`.
