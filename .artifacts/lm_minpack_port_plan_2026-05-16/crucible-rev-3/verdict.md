# Crucible Verdict — rev 3

**Result:** PASS WITH ADVISORIES (after rev-3 fixes applied)

**Discovery:** 27 findings from 4 lenses (mistake-book, code-comment, git-history, prior-PR-review lenses skipped per /crucible invocation scope — not applicable to brand-new planning markdown).

**Scored:** 27 raw lens findings, after dedup ≈ 23 distinct, dispositioned as follows:

| Bucket | Score band | Count | IDs |
|---|---|---|---|
| Auto-confirmed | ≥ 90 | 1 critical + 3 major = **4** | L3F1@100 (crit), L1F1@100, L1F2@100, L2F1@90 |
| Auditor-promoted to confirmed | 60–89 | 6 major | L3F6@85, L3F9@85, L4F3@85, L3F2/L4F1@80, L3F3@70, L4F2@65* |
| Advisories (auditor declined as non-blocking) | ≥ 50 | 7 | L3F4@75, L4F7@70, L4F4@65, L4F5@60, L4F6@65, L3F5@55, L3F7@55 |
| Dropped | ≤ 50 | 6 | L3F8@40, L4F8@40, L2F6@40, L2F7@35, L2F8@45, L2F2@50 (boundary case) |

\* L4F2 was self-scored 65 by Lens 4 (initially advisory-band) but auditor-promoted to MAJOR severity based on a downstream-risk argument: a single passing path at G0 creates unmodeled G3/G4 failure risk worth more than the 0.5-day-savings of accepting the disjunctive criterion. The score-vs-severity split is itself recorded for traceability — see scoring-rubric.md L4F2 note.

**Confirmed total:** 1 critical + 9 major = **10 findings**, all fixed in rev 3 (see table below).

Validator-checkable assertion: "1 critical + 3 major" at the strict ≥ 90 threshold matches the IDs above; the remaining 6 of the 9 confirmed majors are auditor-promoted from the 60-89 band (which is the legitimate auditor-decision band per the Crucible spec).

**Monotonicity:** SAFE — rev 3 only tightens claims; no prior valid assertion was weakened or replaced with a worse one.

**Checklist:** adversarial=yes; required_checks=COVERED (adapted for planning-doc scope per user instructions — toolchain/test gates explicitly skipped).

**Validation:** N/A (planning markdown; no code compiled, no tests run, no lint/format gates).

**Official docs/upstream:** CHECKED
- netlib MINPACK Fortran source (`lmder.f`, `lmpar.f`, `qrfac.f`) — fetched via `curl`, individual line numbers verified by `sed`
- SciPy 1.17.1 — runtime probe of `inspect.signature(scipy.optimize.least_squares)` and `inspect.signature(scipy.optimize.leastsq)`
- JAX 0.10.0 / jaxlib 0.10.0 — runtime probe; `jax.scipy.linalg.qr` and `jax.lax.linalg` API surface
- JAX PR #25955 metadata — via `gh pr view 25955 --repo jax-ml/jax`
- `src/simsopt/geo/optimizer_jax.py` HEAD — all 11 cited line refs verified by `grep -n`

## Confirmed findings → fix locations in rev 3

| # | Severity | Finding | Fix in rev 3 |
|---|---|---|---|
| 1 | **critical** | info=8 misclassified as matrix-free (`lmder.f:431` shows it depends on `gnorm`); info=3 wrongly excluded (it's matrix-free per `lmder.f:421-422`). Subset {1,2,5,6,7,8} → **{1,2,3,5,6,7}** | §0:47, §3.3:149-155, §3.4:167-173, §10:519, §12:571-575 |
| 2 | major | `lmder_serial.f` doesn't exist on netlib | §3.2:144, §10:520 |
| 3 | major | `lmpar.f:229-230` wrong → `lmpar.f:220-222` (fused compound `if`) | §4.2:252, §10:521 |
| 4 | major | Front-matter "Track 1 spike through G5: 1-3 days" contradicts §4.2 sum (~8.5 days) | Front-matter:11, §10:522 |
| 5 | major | §5.1 lists `optimistix` as required dep, pre-empting §8 Q4 | §5.1:340-341, §10:523 |
| 6 | major | `dgeqpf` typo (deprecated Level-2 BLAS); should be `dgeqp3` | Replaced by Path C in §4.2:203; §10:524 |
| 7 | major | Path C (XLA custom call to `dgeqp3`) omitted from Phase 0 | §4.2:203 (new Path C), §8:438 (new Q8), §10:526 |
| 8 | major | "Re-scope to m≈n only" residual-value mirage (production fixture is m≫n) | §4.2:208 reframed; §8 Q8:438 (owner sign-off required); §10:525; consistent across §0:48, §1.2:84, §4.4:316, §12:590 |
| 9 | major | L3 success criterion on (384,40) hard-asserted but Phase 5 G6 allows L3 fail | §0:49 reconciled; §10:527 |
| 10 | major | G0 acceptance was disjunctive (A∨B) vs conjunctive (A∧B) for downstream-risk | §4.2:207-208 tightened to ≥2 paths must agree; §10:528 |

All 10 confirmed findings were folded into rev 3 in a single iteration. No re-spawn of Phase 1 was required since every required fix was a text edit with explicit pre/post grep verification.

## Independent validator pass (post-rev-3)

After rev 3 was published, a third independent validator confirmed:
- All 10 listed rev-3 fixes are present in the live file
- Netlib MINPACK Fortran refs cross-check (info=4 uses gnorm at `lmder.f:313`, info=3 is conjunction at `:421-422`, info=8 uses gnorm at `:431`, lmpar fused exit at `:220-222`)
- JAX 0.10.0 source lowers CPU `geqp3` through `lapack.prepare_lapack_call("geqp3_ffi", ...)` — confirms the §1.2 claim about LAPACK FFI being the underlying call
- SciPy 1.17.1 `least_squares` defaults `ftol=xtol=gtol=1e-8` confirmed by runtime probe

Validator flagged two residuals after the first rev-3 pass:
- **§12:589 G0 checkpoint** still allowed "abandon spike immediately OR re-scope to m≈n shapes only" without owner-signoff constraint
- **§4.4:316 gate table row** had same residual phrasing
- **§1.2:84 narrative** still said byte-equality "drops to m≈n shapes only" without owner-signoff hedge

All three residuals fixed in the same rev 3; now all 5 G0-fail policy sites (lines 48, 84, 208, 316, 590) carry consistent "abandon at production scope, re-scope requires owner sign-off per §8 Q8" wording. Validator confirmed PASS.

## Advisories (rev 3 carries but does not block)

1. §0 row for Track 1 (per gate) now foregrounds G0 as the only existential gate ✓ addressed
2. TL;DR "research-grade, novel" doesn't carry inline hedge; §1.3 + §8 Q5 hedge it adequately for an attentive reader — minor
3. §12 Master TODO Phase 2.4 "quadratic-interpolation update" terminology is unique to that line; could be unified to "symmetric Marquardt update" — minor cosmetic
4. Phase 0 "100 random seeds" sample-size justification now includes "drift is deterministic-per-shape" reasoning in §4.2 — sufficient
5. Track 2 / Track 1 serialization framing in §2 — kept as project-management discipline rather than technical coupling; acceptable as documented
6. Track 3 priority conditional on Track 1 outcome — not formalized; remaining open question, but plan §5 already flags Track 3 as deferred so the conditional priority is implicit

## Phase coverage

| Phase | Status |
|---|---|
| Phase 1 — 4 parallel discovery lenses | RUN (Opus 4.7 max effort × 4) |
| Phase 2 — confidence scoring | COMPRESSED INLINE (orchestrator scored; not a separate subagent run) |
| Phase 3 — execution verification | SKIPPED per escape-hatch ("no test runner" for planning doc); critical finding L3F1 manually verified by orchestrator via direct `curl` + `sed` of `lmder.f` (see [lens3-overclaim-underclaim.md](lens3-overclaim-underclaim.md) Phase-3-equivalent verification block) |
| Phase 4 — dialectical audit | COMPRESSED INLINE (orchestrator adjudicated; not a separate subagent run) |
| Phase 5 — mistake book + verdict | Mistake book NOT updated per /crucible invocation scope (planning doc, not implementation pattern); verdict captured in this file |

## Iteration history

| Pass | Trigger | Result | Findings |
|---|---|---|---|
| 1 | /crucible on rev 2 | FAIL | 1 critical + 9 major + 7 minor advisories |
| 2 | Fixes applied, self-audit | PASS WITH ADVISORIES | 0 critical + 0 major + 6 minor advisories |
| 3 | Validator residual flag (§12:589, §4.4:316) | Fixed | 0 critical + 0 major |
| 4 | Validator residual flag (§1.2:84) | Fixed | 0 critical + 0 major |

## Plan state at close

- File: `.artifacts/lm_minpack_port_plan_2026-05-16/PLAN.md`
- Size: 664 lines
- Revision: 3
- Status: DRAFT, pending owner sign-off on §8 Q1–Q8 (notably Q8 Path C scope + re-scope authority)
- Git status: untracked

## What's not closed by this verdict

- Owner decisions on §8 Q1–Q8 — required before execution funding
- Track 2 implementation — 13 todos, ~2 days estimated
- Track 1 Phase 0 spike — 1–3 days, true abandon checkpoint
- Track 3 deferred — 1–2 weeks, conditional on Tracks 1+2 outcomes
- Publication / write-up — §8 Q5 not independently lit-reviewed
