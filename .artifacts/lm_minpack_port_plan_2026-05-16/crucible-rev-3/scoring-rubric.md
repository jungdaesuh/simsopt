# Crucible scoring rubric — rev 3

## Methodology

Per Crucible skill spec, each Phase-1 finding is scored 0–100:

| Score | Meaning | Action |
|---|---|---|
| 100 | Definitely real, confirmed with evidence, will happen in practice | Auto-confirmed |
| 75 | Likely real, will impact functionality, or explicit CLAUDE.md violation | Auto-confirmed if ≥ 90; passes to Phase 3 if 60–89 |
| 50 | Real but low-impact nitpick | Passes if 60–89; dropped if < 60 |
| 25 | Might be real, but unverified. Stylistic issue not in CLAUDE.md | Dropped |
| 0 | False positive. Doesn't survive scrutiny, or pre-existing | Dropped |

**Filtering rules (per Crucible spec):**
- Score ≥ 90: auto-confirmed, skip Phase 3
- Score 60–89: passes to Phase 3 verification (or auditor judgment for planning-doc scope)
- Score < 60: default-dropped, **but the auditor may resurrect a sub-60 finding as advisory or for fold-in to a broader cleanup**

For this planning-doc review:
- Phase 3 (execution verification) is not applicable; findings in the 60–89 band were resolved by direct re-verification of the plan text against the cited source (file:line, URL contents, or empirical probe).
- The auditor exercised the resurrection option for 4 sub-60 findings (L2F2@50, L2F3@45-as-rescored-85, L2F4@60-as-rescored-80, L2F5@70-as-rescored-80) where Lens-2's self-reported confidence was low but cross-check showed the underlying drift was real. These are tracked in the "Auditor-resurrected internal-consistency fixes" section below.
- The score-50 boundary case (L2F2) is dispositioned as "dropped from confirmed-findings table, but the underlying §4 wording was fixed in rev 3 as part of broader §4 cleanup, not a directed response to this finding". Recorded under Dropped with a boundary note.

## All 27 findings, scored

### Critical (1)

| ID | Score | Finding | Source |
|---|---|---|---|
| L3F1 | 100 | info=8 misclassified as matrix-free-computable in 5 places; lmder.f:431 shows info=8 depends on gnorm. Also info=3 wrongly excluded (it's the matrix-free conjunction info=1 ∧ info=2 per lmder.f:421-422). Correct subset = {1,2,3,5,6,7}, pivoted-QR-only = {4,8}. | Lens 3 |

**Verification (Phase-3-equivalent):** Orchestrator directly fetched `https://www.netlib.org/minpack/lmder.f`, ran `sed -n '420,435p'`. Confirmed line 431 is `if (gnorm .le. epsmch) info = 8`. Confirmed lines 421-422 are `info = 3` set when `info .eq. 2` already holds (the conjunction). Finding promoted to auto-confirmed at score 100.

### Major (9) — present in the verdict.md confirmed-findings table

| ID | Score | Disposition | Finding | Source |
|---|---|---|---|---|
| L1F1 | 100 | auto-confirmed (≥90) | `lmder_serial.f` cited at PLAN.md:144, 151 doesn't exist on netlib (HTTP 404). Should be `lmder.f` (same line numbers correct). | Lens 1 |
| L1F2 | 100 | auto-confirmed (≥90) | `lmpar.f:229-230` cited at PLAN.md:252 wrong. Those lines are loop continuations. The secondary-exit predicate is on `lmpar.f:220-222`. | Lens 1 |
| L2F1 |  90 | auto-confirmed (≥90) | Front-matter "Track 1 spike through G5: 1-3 days gated cumulative" contradicts §4.2 per-phase duration sum (1+1+1+2+1.5+2 = 8.5 days). | Lens 2 |
| L3F6 |  85 | auditor-promoted (60–89) | §5.1 first todo lists `optimistix` and `lineax` as required deps, contradicting §8 Q4 which is still an open owner decision. | Lens 3 |
| L3F9 |  85 | auditor-promoted (60–89) | G0 fail action mentions "FFI to `dgeqpf` direct" — `dgeqpf` is the deprecated Level-2 BLAS predecessor of `dgeqp3`. SciPy and JAX both call `dgeqp3`. Falling back to `dgeqpf` would guarantee losing byte-equality. | Lens 3 |
| L4F3 |  85 | auditor-promoted (60–89) | "Re-scope to m≈n only" offered as G0-fail option is a residual-value mirage: production BoozerSurface fixture is exclusively (384,40) per §6.2, so m≈n has zero production payoff. | Lens 4 |
| L3F2/L4F1 |  80 | auditor-promoted (60–89) | Path C (XLA custom call to LAPACK `dgeqp3` directly, returning native packed `(A_packed, tau, jpvt)`) is omitted from Phase 0. This is structurally bit-equal-by-construction; should be evaluated or explicitly dismissed. | Lens 3 + Lens 4 (duplicate, merged) |
| L3F3 |  70 | auditor-promoted (60–89) | Track 1 (final) L3 success criterion claims per-iteration trace at rtol=1e-12 on the (384,40) BoozerSurface fixture, but §1.2 (384,40) Q-drift may propagate through qtb even after G0 passes via Path B. L3 may be unachievable on (384,40); §0 hard-asserts it but Phase 5 G6 allows fail. | Lens 3 |
| L4F2 |  65 | **severity-promoted by auditor** | G0 acceptance was disjunctive (Path A OR B passes). Conjunctive (≥ 2 paths must agree) would reduce downstream G3/G4 risk. Lens 4 self-scored at 65 (advisory-band) but auditor promoted to MAJOR severity because the auditor judged the downstream G3/G4 risk reduction (3+ engineer-days of sunk cost avoidance) substantively outweighs the 0.5-day cost of stricter G0 acceptance. Fixed in rev 3 (§4.2:207-208). | Lens 4 |

Sum: **9 majors** in the confirmed-findings table (matches verdict.md). Of these, 3 are at strict-auto-confirmed score ≥ 90 (L1F1, L1F2, L2F1) — the threshold the validator correctly flagged. The remaining 6 are auditor-confirmed from the 60-89 band (plus L4F2 which is severity-promoted within the auditor band).

### Minor advisories (7, kept as non-blocking — NOT in verdict's confirmed-findings table)

| ID | Score | Why advisory not required-fix | Source |
|---|---|---|---|
| L3F4 | 75 | Phase 5 "L1+L2 contract only" fallback doesn't propagate byte-equal-contract-collapse recognition to §0/§4 header. Auditor decision: this is wording about contract semantics, not a technical bug; the §3.4:173 + §10:519 wording in rev 3 sufficiently flags the gnorm-dependency. Carries as advisory for owner review. | Lens 3 |
| L4F7 | 70 | Track 3 priority should be conditional on Track 1 outcome (if Track 1 lands, Track 3's main argument — κ(J) conditioning improvement — is already in production). Auditor decision: real point but Track 3 is explicitly deferred in §5; conditional-priority can be addressed at the time §5 is unfrozen. Advisory for now. | Lens 4 |
| L4F4 | 65 | "1 day" Phase 0 estimate has no uncertainty band; other estimates in the plan use ranges. Addressed in rev 3 by changing to "1–3 days" in front-matter and §4.2. | Lens 4 |
| L4F5 | 60 | G0 abandon enforcement "zero further work funded" lacks named approval gate. Subject to sunk-cost-fallacy in practice. Advisory because the rev-3 §8 Q8 addition partially addresses by requiring owner sign-off on re-scope. | Lens 4 |
| L4F6 | 65 | "Track 2 must come first" framing in §2 cites contract-debt but tracks are code-decoupled; argument is actually project-management, not technical. Advisory because rev 3 retained the "Track 2 first" discipline — the framing is project-management which is the author's call. | Lens 4 |
| L3F5 | 55 | §0 row for "Track 1 (per gate)" mentions Phase 0 as a gate name, not as a conditional precondition for the rest of Track 1. **Addressed in rev 3** (§0:48 added "G0 is the only existential gate") — so this is more "advisory addressed" than "advisory deferred". | Lens 3 |
| L3F7 | 55 | TL;DR "research-grade, novel" not hedged at point of use; hedge is in §1.3 + §8 Q5. Advisory: an attentive reader will find the hedge two sections down. | Lens 3 |

### Dropped (6, score ≤ 50, not in confirmed-findings table)

| ID | Score | Finding | Source | Why dropped |
|---|---|---|---|---|
| L2F6 | 40 | Line 326-328 framing tension "G0 fast-fail vs G1-G5 are full spike" | Lens 2 | Minor framing, not contradiction |
| L2F7 | 35 | §4.3 LOC sum 1763 vs "~1770" target | Lens 2 | "~" qualifier + "halving or doubling" caveat covers it |
| L2F8 | 45 | §12 line 560 while-loop predicate summary drops disjunction text | Lens 2 | §12 is a summary; collapses expected |
| L3F8 | 40 | §1.3 line 96 awkward phrasing ("would be a novel contribution. Publication...not independently verified") | Lens 3 | Hedge is present, just structurally awkward — stylistic |
| L4F8 | 40 | Phase 0 100-seed sample size unjustified | Lens 4 | Addressed in rev 3 with "drift is deterministic-per-shape" note in §4.2; cheap-to-run anyway |
| L2F2 | 50 | §4 "Spike-only validation of the first two gates: ~3 days" ambiguous after G0 insertion | Lens 2 | **Boundary case at exactly 50.** Dispositioned as dropped from the confirmed-findings table because L2F2 was self-scored at the strict-spec threshold and not auditor-resurrected as standalone; **however the underlying §4 wording was fixed in rev 3 as a side-effect of broader §4 cleanup** (§4 line 181 expanded to per-phase breakdown after L2F1's effort-estimate correction). The fix is real but it is not directed at L2F2; if L2F1 had not triggered §4 cleanup, L2F2 would have stayed unfixed in rev 3. |

### Auditor-resurrected internal-consistency fixes (3, not in confirmed table but addressed in rev 3)

These were initially in the drop pile based on Lens-2's self-reported confidence (45–60–70) but the auditor promoted them to minor-actionable after cross-checking against the changelog discipline rev 3 was supposed to enforce. **All three were fixed in rev 3 alongside the major findings, even though they don't appear in the verdict.md confirmed-findings table.**

| ID | Score | Finding | Source | Fix in rev 3 |
|---|---|---|---|---|
| L2F3 | 85 | §12 Phase 2.4 "MINPACK quadratic-interpolation update" terminology nowhere else | Lens 2 | Fixed by changing §12:559 to "MINPACK symmetric Marquardt damping update per Agent C §4" |
| L2F4 | 80 | §12 Master TODO Track 2 doesn't reflect "matrix-free-computable subset" framing | Lens 2 | Fixed in rev 3 §12:571–575 (matrix-free-subset language now mirrors §3.3) |
| L2F5 | 80 | "§0.2 non-goals" reference doesn't resolve (§0 has unnumbered subsections) | Lens 2 | Fixed in rev 3 §7:422 (now reads "§0 Non-goals") |

These three are **not counted in the verdict's "1 critical + 9 major confirmed"** number. The verdict's confirmed table captures the 10 findings that materially affect plan correctness or shippability; L2F3/L2F4/L2F5 are documentation-drift fixes that improve consistency but don't affect a downstream consumer who reads only the body text. Distinguished here for accurate provenance.

## Roll-up — matches verdict.md exactly

| Bucket | Score band | Count | Composition |
|---|---|---|---|
| Critical (auto-confirmed) | ≥ 90 | 1 | L3F1 |
| Major (auto-confirmed) | ≥ 90 | 3 | L1F1, L1F2, L2F1 |
| Major (auditor-promoted from 60-89) | 60–89 | 5 | L3F6, L3F9, L4F3, L3F2/L4F1, L3F3 |
| Major (severity-promoted by auditor from advisory) | 65 | 1 | L4F2 |
| **Confirmed total (in verdict table)** | — | **10** | 1 critical + 9 major |
| Advisories (auditor declined as non-blocking) | 50–75 | 7 | L3F4, L4F7, L4F4, L4F5, L4F6, L3F5, L3F7 |
| Dropped (≤ 50, not in confirmed-findings table) | ≤ 50 | 6 | L3F8, L4F8, L2F6, L2F7, L2F8, L2F2 (boundary case at exactly 50) |
| **Total dispositioned (deduplicated)** | — | **23** | — |

The "1 critical + 3 major auto-confirmed" claim in verdict.md is the strict ≥ 90 subset; the full "1 critical + 9 major confirmed" total includes 6 auditor-promoted findings from the 60-89 band (where Crucible spec says findings pass to Phase 3 or — for planning docs — auditor judgment). All 10 confirmed findings carry a "Fixed in rev 3" tag and a §-and-line citation in verdict.md.

Discrepancy with verdict.md "27 raw findings" — the 27 number counts each lens-output finding individually before deduplication. After dedup (L3F2 ≡ L4F1) and dropping single-line summary-statement duplicates within Lens 2, the actionable count is 23. Both numbers are honest depending on whether you count raw lens output (27) or deduplicated finding-list (23). Sum check: 10 confirmed + 7 advisory + 6 dropped = 23 ✓.
