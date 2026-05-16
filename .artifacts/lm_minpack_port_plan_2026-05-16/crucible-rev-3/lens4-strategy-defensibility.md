# Lens 4 — Strategy Defensibility

**Model:** Opus 4.7 (max effort)
**Scope:** Adversarially question whether the rev-2 revisions (especially Phase 0 insertion) are sound strategy.

## Investigations performed

1. Is Phase 0 real risk-mitigation or moved-goalpost theater?
2. Is "abandon at G0" the right binary?
3. Path C — direct LAPACK FFI via XLA custom call — unfairly omitted?
4. Spike commit/discard structure honesty
5. Track 2 / Track 1 serialization — actually independent code-wise?
6. Track 3 priority — conditionally dependent on Track 1 outcome?
7. Phase 0 100-seed sample size — justified?

## Findings

```
FINDING L4F1: Path C (XLA custom call to dgeqp3) is omitted from Phase 0 without justification, despite being structurally dominant over Paths A and B
Location: PLAN.md:200-208
Lens: strategy-defensibility
Confidence: 85
Severity: major
Evidence: Phase 0 enumerates exactly two paths: "A. Extract Householder vectors directly from Q... B. Use jax.lax.linalg.householder_product / ormqr to compute Q^T·fvec directly". The root cause of the (384,40) drift, per the plan itself (§1.2 line 81), is that "JAX jax.scipy.linalg.qr returns explicit (Q, R, P)... no geqrf analog that yields the packed form directly". This is precisely the artifact of JAX's *post-processing* — the underlying LAPACK call (lapack_dgeqp3_ffi, plan line 67) already produces the bit-exact packed form. Path A reconstructs *back* from the lossy explicit Q. Path B sidesteps Q but still depends on householder_product/ormqr to chain Householders, which is itself a JAX-implemented kernel whose bit-equality to LAPACK's ormqr is not asserted and not empirically verified anywhere in the plan. A Path C — XLA custom call dispatching to dgeqp3 directly and exposing (A_packed, tau, jpvt) — bypasses both reconstruction paths and is *by construction* bit-exact because it IS the LAPACK call. The plan author has a duty either to evaluate Path C in G0 or to explicitly dismiss it with stated cost (e.g., "XLA custom call adds C-extension build dep, violates §0 non-goal of preserving JAX-pure lane" — but no such non-goal is currently in §0). Counterargument: Path C might require non-trivial XLA FFI work that could itself consume more than 1 engineer-day, defeating Phase 0's "cheap discard" property. But the plan should *state* this tradeoff rather than silently omit a structurally-dominant option.
NOTE: This finding overlaps L3F2 from Lens 3. Treated as duplicate in scoring rollup.
```

```
FINDING L4F2: G0 acceptance criteria are disjunctive (either Path A *or* Path B passes), but should be conjunctive for risk reduction
Location: PLAN.md:206-207
Lens: strategy-defensibility
Confidence: 75
Severity: major (initially scored, promoted to required-fix during audit)
Evidence: Plan reads "G0 pass: proceed to Phase 1 with the chosen path (A or B) frozen as the spike's QR-frontend convention. G0 fail with path B but not A (or vice versa): proceed with the passing path". The disjunctive acceptance trades short-term discard speed for downstream brittleness. Path A and Path B exercise different JAX kernels (householder_product vs explicit Q construction), and downstream qrsolv / lmpar consume the chosen frontend's output. If only one path passes G0, the subsequent gates G3/G4 depend on the *consistency* between that frontend's packed form and what MINPACK assumes. The plan acknowledges this risk implicitly at §1.2 line 82-83 ("With Q bit-drifted, qtb will bit-drift, and the rest of the byte-equality argument collapses on the (384,40) fixture") but then commits to a path that may collapse at G3 even after G0 passes. A conjunctive G0 (both A and B agree to LAPACK bit-equal) would provide much stronger evidence that the JAX QR substrate as a whole is reliable. Counterargument: conjunctive G0 is a strictly stricter test, and a passing single path is still useful evidence. But the asymmetric path A vs B trade is precisely what creates *unmodeled* G3/G4 risk — the cost of a stricter G0 is bounded (additional ~0.5 day) versus the cost of getting through G2 only to fail at G3 (3+ days sunk).
```

```
FINDING L4F3: "Re-scope to m≈n only" is a residual-value mirage — production payload is exclusively m≫n shapes
Location: PLAN.md:208 and PLAN.md:576
Lens: strategy-defensibility
Confidence: 90
Severity: major
Evidence: G0 fail allows "ABANDON Track 1 spike at this checkpoint. Re-scope to m ≈ n only, or pivot to a different LAPACK call... or accept that Track 1 cannot byte-match MINPACK on production BoozerSurface shapes and reframe as 'MGH-test-suite-only' research lane." The Goals §0 explicitly state the W4.3 divergence target is on the BoozerSurface fixture which is m=384, n=40 (m≫n) — see §6.2 lines 379-381: "build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8) — oversampled, well-conditioned" and the empirical drift in §1.2 is also at (384,40). Goal 1 line 31 names "simsopt's CPU BoozerSurface.minimize_boozer_penalty_constraints_ls" — that fixture has no m≈n version in real use. Re-scoping to "m≈n only" or "MGH-test-suite-only" produces a research lane with **zero production payoff** for the simsopt project's actual workload. The plan's own §0 success criterion line 49 requires "the oversampled BoozerSurface fixture" parity — m≈n fallback fails this criterion by construction. Counterargument: MGH-test-suite parity is independently publishable per §1.3 ("A true Track 1 implementation would be a novel contribution"), and the engineering time spent through Phases 1-5 might still yield a publication even without simsopt-production utility. But that conflates two different project rationales (engineering improvement of simsopt vs. academic publication) and the plan §0 non-goal "Publication / write-up. Listed as a possible follow-up but not a deliverable" explicitly disclaims the publication path as a deliverable.
```

```
FINDING L4F4: "1 day Phase 0" estimate has no uncertainty band despite acknowledged extrapolation methodology
Location: PLAN.md:11, PLAN.md:196, PLAN.md:327
Lens: strategy-defensibility
Confidence: 70
Severity: minor
Evidence: Plan states "Phase 0 alone (cheapest abandon checkpoint): 1 day" (line 11), "Packed-fjac + qtb feasibility gate (1 day)" (line 196), and "G0 fail → spike abandoned at end of day 1, ~1 engineer-day spent" (line 327). All three references give the same point estimate with no range. By contrast, the plan applies appropriate uncertainty to other estimates: line 21 "Estimated 600–800 LOC and 30–60s first-trace; both are extrapolated estimates", line 181 "All effort estimates are extrapolations from _lbfgsb_scipy.py's ~3300 LOC, not measurements of unbuilt code", and line 310 "Subject to halving or doubling based on Phase 4-5 complexity". Phase 0 specifically requires householder_product / ormqr JAX low-level FFI familiarity (lines 202, 575), which is not used in the existing _lbfgsb_scipy.py reference implementation. The plan has no anchor for the 1-day estimate.
```

```
FINDING L4F5: "G0 fail → zero further work funded" lacks institutional enforcement mechanism, making it aspirational
Location: PLAN.md:326-328
Lens: strategy-defensibility
Confidence: 65
Severity: minor
Evidence: Plan asserts "G0 is a true gate: G0 fail → spike abandoned at end of day 1, ~1 engineer-day spent, zero further work funded. This is the cheapest possible discard checkpoint." But there is no named approval gate, no documented sign-off requirement, no checklist item in §12 "Pre-work" that establishes the enforcement mechanism. Compare to §8 "Open decisions" which does list owner-approval items as `[ ]` checkboxes — but G0 abandonment is not among them. In practice, "abandon" decisions are subject to sunk-cost-fallacy pressure. The plan's §4.1 already acknowledges this dynamic with "Any gate fails after 2 attempts to align with MINPACK behavior" — a retry policy that explicitly permits sliding the discard checkpoint.
```

```
FINDING L4F6: Track 2 / Track 1 serialization is asserted as "must come first" but the technical-coupling argument doesn't survive scrutiny
Location: PLAN.md:19, PLAN.md:126-128
Lens: strategy-defensibility
Confidence: 80
Severity: minor (project-management vs technical conflation)
Evidence: Plan §2 line 126-128 justifies Track 2 first: "It's small enough to land as a single PR with full validation, and the result is independently useful even if Track 1 is later abandoned. It also reduces the contract debt that any Track 1 / Track 3 work would inherit." TL;DR line 19 calls it "Track 2 first (low risk, high signal)". But examining the file targets: Track 2 modifies optimizer_jax.py:1416 (_lm_iteration) + :1403 (_lm_defaults). Track 1 creates a NEW module _lmder.py. These are non-overlapping file edits. The "contract debt" argument is vague: Track 2 updates docstrings at optimizer_jax.py:14-45 and jax_acceptance.rst:156-187 (line 157-158). Track 1 Phase 7 updates the same files (line 287-288). The "debt" reduction is approximately writing the docstring once instead of twice — bounded benefit. Meanwhile Track 1's G0 is the riskiest decision in the entire plan; if it fails at day 1, no Track 2 work was even started.
```

```
FINDING L4F7: Track 3 priority should be conditionally dependent on Track 1 outcome but plan treats it as unconditional follow-on
Location: PLAN.md:23, PLAN.md:120-124, PLAN.md:336
Lens: strategy-defensibility
Confidence: 75
Severity: minor
Evidence: Plan §1.2/§2 frame Track 3's value as: line 23 "Better numerical conditioning on near-rank-deficient fixtures (κ(J) not κ(J)²)", line 32 same claim, line 116 "For the documented near-rank-deficient default BoozerSurface fixture (κ(J) ≈ 10⁷), this is a 7-order-of-magnitude conditioning improvement." Both Track 1 (via pivoted QR) and Track 3 (via LSMR) reduce condition number from κ(J)² to κ(J) — the plan §2 line 123 acknowledges this: "Both pivoted-QR (Track 1) and LSMR (Track 3) reduce the effective condition number to κ(J)." If Track 1 lands successfully and provides the κ(J) conditioning improvement, Track 3's primary numerical-conditioning rationale disappears. The remaining Track 3 benefits would be: (a) GPU vmap-friendliness from Optimistix (real), (b) library maintenance burden tradeoff with a single-maintainer dep (debatable per §2 line 122), (c) tolerance-equivalent rather than byte-equivalent (a feature for cross-platform GPU work). Net: Track 3 priority should drop substantially if Track 1 lands; should stay high if Track 1 abandons at G0. Plan §5 line 336 says "Should not start until Tracks 1 and 2 land" — this serialization implicitly assumes Track 1 always lands, but G0 may abandon.
```

```
FINDING L4F8: Phase 0 G0 acceptance trial count (100 random seeds) lacks justification given the empirical evidence was 4-shape × 2-seed
Location: PLAN.md:204-205
Lens: strategy-defensibility
Confidence: 50
Severity: minor (cheap to over-test; no real harm but design unjustified)
Evidence: Plan §1.2 lines 72-76 record empirical drift evidence from exactly 5 trials (one per (shape, seed) combination). The G0 gate (line 205) escalates to "100 random seeds tested" at (384,40). The plan doesn't argue why 100 is sufficient or how it would catch failure modes that the 2-seed sampling missed. Reproducibility of "Q max diff = 4.163e-17" being identical at seeds 0 and 1 (line 75-76 both report 4.163e-17 exactly) suggests the drift may be a deterministic function of shape rather than data — in which case 100 seeds at the same shape provides no new information beyond 2 seeds. The interesting test is across shapes (2000×80 is good, but only one additional point), not across seeds at one shape.
```

## Strategy-sound (checked and found defensible)

- **Tolerance ladder L1–L4 structure (§6.1)**: well-grounded, maps cleanly to existing parity-ladder lanes, levels are properly orthogonal. L4 caveat for single-host single-build is correctly stated.
- **Track 2 scope (§3)**: closing the matrix-free-computable subset of info codes and deferring 4 to Track 1 G5 is the right factoring — exposing the algorithmic constraint (pivoted-QR data needed for info=4) rather than hiding it. (NOTE: see L3F1 which finds the subset *itself* is wrong; the factoring strategy is sound.)
- **G2 dependency on G0 (line 318)**: correctly noting that G2 is "already empirically verified on m≈n; G0 covers production shape" prevents wasted gate work.
- **§0 non-goal 1 (GPU byte-equality)**: correctly identified as impossible with cuSOLVER/MAGMA and removed from contract.
- **§4.1 abandonment criteria including compile-time and per-subroutine alignment**: operationalizable, not aspirational.
- **Critic-pass folding (§10 line 511-516)**: four 2026-05-16 critic findings reproduced empirically and folded into specific plan sections, with clear traceability.
- **§5 net-LOC additive correction**: TL;DR line 23 and §5 line 334 correctly state Track 3 net-LOC is additive and the "~500 LOC simplification" is conditional on future cleanup. Previous version conflated these.
- **Track 1 success criterion (line 49)**: "CPU same-machine only" qualifier is correctly placed; doesn't overclaim portability.
- **Open decision §8 Q5 (publication ambition)**: correctly flagged as "claim was not independently verified by a numerical-analysis literature review" with the "possible follow-up, not a guaranteed novelty" caveat.

## Lens summary

Eight findings — three major (Path C omission [duplicates L3F2], G0 disjunctive/conjunctive question [later promoted in audit], m≈n re-scope mirage) + five minor (Phase 0 1-day point estimate, G0 enforcement aspirational, Track 2 vs Track 1 serialization argument weak, Track 3 priority conditional on Track 1 outcome, 100-seed sample design unjustified). Strategy is largely defensible but has three substantive gaps: Path C, G0 acceptance rule, and re-scope authority.
