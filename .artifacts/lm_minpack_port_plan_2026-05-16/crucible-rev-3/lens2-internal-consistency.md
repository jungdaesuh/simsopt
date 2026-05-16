# Lens 2 — Internal Consistency

**Model:** Opus 4.7 (max effort)
**Scope:** Find cross-reference drift, numbering inconsistencies, terminology drift, and master-TODO-vs-body mismatches in `.artifacts/lm_minpack_port_plan_2026-05-16/PLAN.md` rev 2 after the Phase 0 insertion + Track 2 reframe edits.

## Audits performed

1. Section cross-references (§X.Y resolution)
2. Numbered phase references (Phase 0-7, Gates G0-G8)
3. Effort estimate consistency (front-matter vs body)
4. L1/L2/L3/L4 definition consistency across §3, §4, §5 vs §6.1
5. Acceptance gate consistency (§4.2 per-phase text vs §4.4 table)
6. "Track 2 first" ordering claim
7. Spelling / terminology drift
8. Master TODO checklist (§12) drift
9. Old claims that should have been retired but weren't
10. Frontmatter date / status consistency

## Findings

```
FINDING L2F1: Front-matter "Track 1 spike through G5: 1–3 days gated cumulative" contradicts §4.2 per-phase duration sum (~8.5 days)
Location: PLAN.md:11
Lens: internal-consistency
Confidence: 90
Severity: major (effort estimate misleading by 3-8x)
Evidence: Line 11 says "Track 1 spike through G5: 1–3 days gated cumulative". §4.2 phase durations explicitly state:
  - Phase 0: 1 day (line 196)
  - Phase 1: enorm port (1 day) (line 211)
  - Phase 2: qrfac wrapper (1 day) (line 223)
  - Phase 3: qrsolv Givens elimination (2 days) (line 236)
  - Phase 4: lmpar univariate Newton (1.5 days) (line 247)
  - Phase 5: Driver integration (2 days) (line 262)
Sum = 8.5 working days, not 1–3.
```

```
FINDING L2F2: §4 "Spike-only validation of the first two gates: ~3 days" inconsistent after G0 insertion
Location: PLAN.md:181
Lens: internal-consistency
Confidence: 65
Severity: minor (was correct in rev 1 when G1+G2 = enorm+qrfac were the "first two"; G0 insertion shifted what "first two" refers to)
Evidence: Line 181 "Spike-only validation of the first two gates: ~3 days." With G0 now Phase 0:
  - Phase 0 (G0) = 1 day
  - Phase 1 (G1) = 1 day
  - "First two gates" naturally reads as G0+G1 = 2 days
But the text still says ~3 days as if the old enumeration is in force.
```

```
FINDING L2F3: §12 Master TODO Phase 2.4 calls damping update "MINPACK quadratic-interpolation update" — terminology not used elsewhere
Location: PLAN.md:559
Lens: internal-consistency
Confidence: 85
Severity: minor (cosmetic)
Evidence: Line 559 "Phase 2.4: replace asymmetric damping with MINPACK quadratic-interpolation update per Agent C §4".
Compare:
  - §3.2 line 144 "Marquardt's symmetric × 2 / × 1/2 scheme with bracket-based escalation"
  - §3.3 line 151 "MINPACK-style symmetric Marquardt scheme (×2 / ×0.5, no asymmetric expand=4.0 / shrink=0.5 mismatch) plus the par escalation logic from lmder_serial.f:381-396"
"Quadratic-interpolation" appears nowhere in §3.2 / §3.3 / §4.2 / Agent C summary. Unique-to-§12 terminology.
```

```
FINDING L2F4: §12 Master TODO Track 2 list doesn't reflect "matrix-free-computable subset (1, 2, 5, 6, 7, 8)" framing introduced in §3.3 / §3.4
Location: PLAN.md:558, PLAN.md:562
Lens: internal-consistency
Confidence: 80
Severity: minor (downstream drift)
Evidence:
  - Line 558 "Phase 2.3: implement compute_info per Agent C §5" — drops §3.3 line 150 qualifier "restricted to the matrix-free-computable subset (codes 1, 2, 5, 6, 7, 8)".
  - Line 562 "Phase 2.7: tests/geo/test_lm_termination_parity.py — 5 MGH + 1 BoozerSurface fixture" omits §3.3 line 155 / §3.4 line 167 "when SciPy's run also exits on that same code" caveat that the revision explicitly added.
```

```
FINDING L2F5: Line 422 references "§0.2 non-goals" but §0 subsections aren't numbered 0.1/0.2/0.3
Location: PLAN.md:422
Lens: internal-consistency
Confidence: 70
Severity: minor (broken cross-ref)
Evidence: Line 422 "Documented in §0.2 non-goals". §0 header at line 27 has subsections "### Goals" (line 29), "### Non-goals" (line 36), "### Success criteria" (line 43) — none numbered as 0.1, 0.2, 0.3. The reference is unresolvable.
```

```
FINDING L2F6: Line 326-328 framing tension between "after G0 (fast fail) and after G5" vs "G1-G5 are the full spike"
Location: PLAN.md:326-328
Lens: internal-consistency
Confidence: 45
Severity: minor (framing, not contradiction)
Evidence: Line 326 "Spike commit/discard decision: after G0 (fast fail) and after G5 (final commit)". Line 328 "G1-G5 are the full spike: if G0 passes and G1-G5 all pass...". The "G1-G5 are the full spike" phrasing excludes G0 from "the full spike" while line 326 lists G0 as a decision checkpoint. Minor framing tension; internally readable.
```

```
FINDING L2F7: §4.3 LOC total "~1770" doesn't sum (1763)
Location: PLAN.md:310
Lens: internal-consistency
Confidence: 60
Severity: minor (arithmetic + "~" qualifier covers it)
Evidence: §4.3 table values sum to: 600 + 30 + 5 + 120 + 100 + 120 + 150 + 150 + 200 + 250 + 30 + 5 + 3 = 1763. Line 310 says "~1770 across 13 files." The "~" qualifier and table's own "Subject to halving or doubling" caveat make this borderline. Score: 60 (advisory).
```

```
FINDING L2F8: §12 line 560 "Phase 2.5: update while-loop predicate to `info == 0`" drops disjunction text from §3.3 line 153
Location: PLAN.md:560
Lens: internal-consistency
Confidence: 55
Severity: minor (summary collapse; semantics agree)
Evidence: Line 560 "Phase 2.5: update while-loop predicate to `info == 0`". §3.3 line 153 "Update levenberg_marquardt / levenberg_marquardt_traceable while-loop predicate to terminate when info != 0 (matrix-free subset) OR success fires on the legacy ‖∇‖_∞ ≤ tol criterion, whichever comes first. Backward-compat: existing callers passing only tol get identical termination behavior." The §12 collapses both clauses into a single condition.
```

## No-drift confirmed

The following checks ran clean (no finding to report):

- **Phase 0 insertion cross-refs**: every reference to Phase 0 / G0 / "Phases 1-5" / "Phases 0-5" / "Phases 6-7" / "Phases 1-7" resolves to the new structure consistently. §1.2 narrative (line 84), §4.2 Phase 0 block (line 196+), §4.3 file map row (line 299), §4.4 G0 row (line 316), §12 Track 1 G0 todo (line 572) all reference Phase 0 / G0 coherently. No leftover "Phase 1" references where "Phase 0" or "Phases 0-1" should appear.
- **§10 provenance trail**: line 511 explicitly cites "a second independent critic validation pass (2026-05-16, logged in session)" with the four blocking corrections enumerated and tagged to their fix locations (§3.3, §3.4, §4.2, Phase 5 fallback). The first-pass critic is also attributed at line 505.
- **G1-G5 references**: lines 321 (G5 row), 326-328 (commit/discard text), 502 (Agent C) properly use G1-G5 / G0-G8 without conflating G0 into the "G1-G5" subset.
- **L1/L2/L3/L4 definition consistency**: §6.1 table (line 371-374) defines all four levels; all in-body usages at lines 49, 272, 273, 280-282, 322, 357, 422, 537 align with §6.1 definitions.
- **"L1-only-byte-equal" leftover scan**: only 2 occurrences (line 273 "Do not call this L1-only-byte-equal..." and line 515 critic-attribution) — both correctly in explanatory/critic-attribution context, not as a recommended label.
- **"byte-equal by construction" leftover scan**: only 2 occurrences (line 101 critic-finding bullet and line 506 §10 critic-attribution) — both correctly attributed as critic flags, not as plan claims.
- **"Drops ~500" leftover scan**: zero occurrences. The TL;DR §5 wording in lines 23 and 334 is properly conditional.
- **"1.49012e-8" usage**: single occurrence at line 148, correctly framed as explanatory historical context (legacy `leastsq` MINPACK-direct default) and explicitly contrasted with chosen `1e-8` default. Also referenced at line 512 in critic-attribution paragraph.
- **"exact info int match" leftover scan**: no occurrence as a Track 2 gate. §3.4 line 167 says "exact int match **only when SciPy also exits on a matrix-free-computable code**" — correctly conditional. Line 173 explicitly disclaims it as a Track 2 gate.
- **Front-matter date / status consistency**: "Created: 2026-05-16" (line 5), "Status: DRAFT (rev 2, 2026-05-16)" (line 10), §10 "second independent critic validation pass (2026-05-16, logged in session)" (line 511) — all three dates agree.
- **Track 2 first claim**: §0 success-criteria table lists Track 2 first (line 47); §1.2 line 65 prioritizes the Phase-0 finding; §2 line 126 "Why Track 2 must come first"; §3 header "DO FIRST". §12 master TODO lists Pre-work → Track 2 → Track 1 → Track 3 → Post-implementation (lines 549, 554, 570, 629, 641). Track 2 todos do appear before Track 1 todos.
- **G0-G8 gate table ordering**: §4.4 lists G0, G1, G2, G3, G4, G5, G6, G7, G8 in order; §12 Track 1 master TODO follows the same order (G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7 → Phase 6/G8 → Phase 7).
- **Spelling / terminology consistency**: "MINPACK" used consistently throughout; no stray "Minpack" or "minpack-lm" casing (other than `cminpack` library name and `least_squares_algorithm="lm-minpack"` API string, which are correct). "Moré-Garbow-Hillstrom" used consistently with "MGH-1981" shorthand defined in §11 glossary. "byte-equal" / "bit-equal" appear in mixed but contextually appropriate forms.
- **LM family entry count**: Phase 7 says "add 4th entry: lm-minpack" (line 624); Track 3 says "fifth family entry: optimistix-lm" (line 349) — internally consistent.
- **§4.4 G0 row at the top of gate table**: yes, present.
- **§5.2 Track 3 gates and §12 Track 3 todos**: Track 3 unchanged by revision; no drift introduced.

## Lens summary

Eight findings on internal consistency, ranging from major (front-matter effort estimate, broken §0.2 cross-ref, downstream-drift from Track 2 reframe to §12 master TODO) to minor framing observations. The structural integrity of the plan post-Phase-0-insertion is largely sound; the drift is concentrated in the master TODO summary (§12) which didn't track the body edits and in two stale front-matter / table-row strings.
