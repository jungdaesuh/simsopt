# Lens 3 — Overclaim / Underclaim Detection

**Model:** Opus 4.7 (max effort)
**Scope:** Find remaining overclaims that rev 2 missed, AND over-corrections (under-claims) introduced by the revision walking back the first critic pass's 4 blocking corrections.

## Critical finding upfront

The most important finding from this lens is the **info=8 misclassification** in Track 2's "matrix-free-computable subset". This is a critical technical error because:
- It would lead to Track 2 implementation setting `info = 8` incorrectly (claiming "gtol too small" termination on a path that cannot compute `gnorm`)
- The plan's own §3.4 wording correctly excludes info=4 for the same gnorm-dependency reason, so the inclusion of info=8 in the matrix-free subset is internally inconsistent

The orchestrator escalated this to critical and ran a direct lmder.f verification:

```bash
$ curl -s https://www.netlib.org/minpack/lmder.f | sed -n '425,435p'
            if (nfev .ge. maxfev) info = 5
            if (dabs(actred) .le. epsmch .and. prered .le. epsmch
     *          .and. p5*ratio .le. one) info = 6
            if (delta .le. epsmch*xnorm) info = 7
            if (gnorm .le. epsmch) info = 8         <-- info=8 SET HERE, depends on gnorm
            if (info .ne. 0) go to 300

$ grep -n "gnorm" /tmp/lmder.f
187: declaration
297: gnorm = zero
306: gnorm = dmax1(gnorm, dabs(sum/wa2(l)))   <-- COMPUTED here, requires pivoted QR
313: if (gnorm .le. gtol) info = 4
431: if (gnorm .le. epsmch) info = 8
```

Confirmed: info=8 at lmder.f:431 depends on gnorm, same as info=4. Conversely info=3 at lmder.f:421-422 is the conjunction `info=1 AND info=2`, both of which are matrix-free (actred/prered/ratio/delta/xnorm don't need pivoted QR).

**Correct matrix-free subset: {1, 2, 3, 5, 6, 7}. Pivoted-QR-required: {4, 8}.**

The plan's rev 2 subset of {1, 2, 5, 6, 7, 8} is wrong in BOTH directions: info=3 wrongly excluded, info=8 wrongly included.

## Findings

```
FINDING L3F1: info=8 listed as matrix-free-computable but actually requires gnorm (pivoted-QR data) — same as info=4
Location: PLAN.md:149, PLAN.md:150, PLAN.md:154, PLAN.md:155, PLAN.md:167, PLAN.md:173 (Track 2 section), PLAN.md:47 (success criteria table by transitive implication)
Lens: overclaim-underclaim
Confidence: 98
Severity: critical
Evidence: Verified against netlib lmder.f. The setting line is `if (gnorm .le. epsmch) info = 8` at line 431. `gnorm` is exactly the quantity the plan correctly identifies in §0/§3.4 as "a pivoted-QR-only quantity unavailable in the matrix-free GMRES inner solve" (line 47, 173). info=6 (`actred`/`prered`/`ratio`) and info=7 (`delta`/`xnorm`) genuinely do not need gnorm and are correctly classified. But info=8's gnorm dependency is identical to info=4's. The matrix-free-computable subset is therefore {1, 2, 5, 6, 7}, NOT {1, 2, 5, 6, 7, 8}. Plan repeats the wrong subset in five places:
  - Line 149: "Codes 1, 2, 5, 6, 7, 8 are computable from ftol/xtol bookkeeping alone; codes 3, 4 require pivoted-QR data"
  - Line 150: "compute_info ... restricted to the matrix-free-computable subset (codes 1, 2, 5, 6, 7, 8)"
  - Line 154: "info ∈ {0, 1, 2, 5, 6, 7, 8} in Track 2 — info=3, 4 only set by the Track 1 lm-minpack lane"
  - Line 155: "each matrix-free-computable info value (1, 2, 5, 6, 7, 8)"
  - Line 167: "info code parity on matrix-free-computable subset (1, 2, 5, 6, 7, 8)"
Plan also says info=3 belongs to the non-computable set (line 149: "codes 3, 4 require pivoted-QR data"). Per lmder.f line 421-422, info=3 = conjunction of info=1 AND info=2. Both predicates are matrix-free. info=3 is therefore actually matrix-free-computable. The plan's classification of info=3 with "requires pivoted-QR data" is wrong by the opposite mistake — info=3 is too pessimistic, info=8 is too optimistic.
Corrected subset: matrix-free-computable = {1, 2, 3, 5, 6, 7}; pivoted-QR-required = {4, 8}.
```

```
FINDING L3F2: Phase 0 omits a third option (Path C) — capture packed dgeqp3 output directly via the FFI, which would be bit-equal by construction
Location: PLAN.md:200-209 (Phase 0 candidate paths only enumerates A and B)
Lens: overclaim-underclaim
Confidence: 82
Severity: major
Evidence: §1.2 establishes that `jax.scipy.linalg.qr(pivoting=True)` dispatches to `lapack_<t>geqp3_ffi` (line 67). `dgeqp3` natively returns the MINPACK-packed Householder form; the (384,40) Q drift of 4e-17 arises in JAX's post-processing that converts the packed output to explicit (Q, R, P). If the spike captured the FFI output directly (or wrote a thin XLA custom call), the packed form would be byte-equal by construction. The plan's two enumerated paths both attempt to *reconstruct* the packed form from the lossy post-processed (Q, R) — they inherit the very drift the plan is trying to defeat. The plan should either (a) list Path C as a third evaluation option in Phase 0 with the caveat that it may require touching JAX private API, or (b) explicitly dismiss it with a stated reason (e.g., "private FFI; out of contract; spike risk too high"). The current omission could lead the spike to abandon Track 1 at G0 when a viable Path C exists. Note: G0-fail action list at line 208 already mentions "pivot to a different LAPACK call (e.g., FFI to dgeqpf direct)" — closely related but FFI-direct-to-dgeqp3 for the packed form is not stated as the obvious first pivot. **Compounding error:** the cited fallback `dgeqpf` is the deprecated Level-2 BLAS predecessor of `dgeqp3`; SciPy and JAX both use `dgeqp3`, so `dgeqpf` fallback would lose byte-equality (see L3F9).
```

```
FINDING L3F3: Track 1 (final) L3 success criterion may not be achievable on the (384,40) BoozerSurface fixture even if G0 passes via Path B
Location: PLAN.md:49 (success criteria table), PLAN.md:272 (Phase 5 gate)
Lens: overclaim-underclaim
Confidence: 70
Severity: major
Evidence: Plan §0 row "Track 1 (final)" claims L3 (per-iter trace at rtol=1e-12, atol=1e-14) on "the oversampled BoozerSurface fixture, CPU same-machine only" (line 49). The oversampled fixture is `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` per §3.1/§6.2; its Jacobian shape on the LS path is (384, 40) per §1.2 probe. §1.2 establishes Q drifts by ~4e-17 on (384,40), and that this drift flows into qtb (line 82: "With Q bit-drifted, qtb will bit-drift"). Even if Path B (`householder_product`/`ormqr`) eliminates the Q reconstruction, qtb is still computed via FFI-orthogonalization paths that JAX may schedule differently from LAPACK's dgeqp3 internals; the drift may persist at last-bit level. The per-iter trace at rtol=1e-12 may accumulate non-trivially over outer iterations even when L1 (1e-6) and L2 (exact int match) hold. Recommend downgrading the L3 claim for the (384,40) BoozerSurface fixture to L2-only, and reserving L3 for the m≈n MGH subset (Phase 6's "5 canonical problems" at line 282). The G6 row at PLAN.md:322 correctly says "acceptable to fail" — but the §0 success criterion at line 49 hard-asserts L3, which is inconsistent with the G6 fall-through. §0 and §4.4 should be reconciled: L3 on (384,40) is aspirational, not a gate.
```

```
FINDING L3F4: Phase 5 fallback says "L1+L2 contract only" but L1 is tolerance parity (not byte-equality), so the byte-equality contract collapses if L3 fails — the wording does not flag this collapse
Location: PLAN.md:273
Lens: overclaim-underclaim
Confidence: 75
Severity: major
Evidence: Phase 5 fallback wording (line 273) correctly disclaims "L1-only-byte-equal" as a contradiction. But the resulting "L1+L2 contract only" is itself a downgrade away from byte-equality (L4 per §6.1) toward tolerance parity (L1: rtol=1e-6, atol=1e-10 per §6.1 line 371). If L3 fails, Track 1's headline pitch of "CPU byte-exact MINPACK port" (§4 title, line 177) is no longer achieved on the BoozerSurface fixture — only the converged-state-tolerance-parity contract is held. This collapse is not currently flagged in §0 success criteria, in the TL;DR (line 21 still asserts "CPU-byte-exact"), or in the §4 section header. The wording "Do not call this L1-only-byte-equal" addresses the term, not the substance.
```

```
FINDING L3F5: §0 success-criteria row for "Track 1 (per gate)" mentions Phase 0 only as a gate name, not as a conditional precondition for the rest of Track 1
Location: PLAN.md:48
Lens: overclaim-underclaim
Confidence: 55
Severity: minor
Evidence: Line 48 reads "At each spike gate (Phase 0 feasibility, enorm, qrfac, qrsolv, lmpar, driver), the JAX kernel matches its SciPy/MINPACK oracle... Gate failure → abandon spike." Structurally honest but doesn't foreground that G0 is the only existential gate — others can fail individually with localized retry, but G0 failure invalidates Track 1 wholesale on production shapes (§4.2 line 198, §4.4 line 327: "G0 is a true gate"). For an executive-reading-§0-only audience, this asymmetry deserves explicit callout in §0.
```

```
FINDING L3F6: §5.1 first todo lists optimistix/equinox as required deps, contradicting §8 Q4 which is still an open question
Location: PLAN.md:340 vs PLAN.md:434
Lens: overclaim-underclaim
Confidence: 88
Severity: major (internal contradiction; pre-empts owner decision)
Evidence: Line 340 "Add optimistix>=0.0.10 + equinox>=0.11.0 to pyproject.toml as required deps (currently optimistix is an optional dep per pyproject.toml:92)." Line 434 (Q4): "Q4 — Optimistix as required dep: Track 3 currently has optimistix as an optional dep. Move to required for the JAX lane, or keep optional?" The §5.1 todo pre-commits to the answer Q4 leaves open. Per the revision's stance that Track 3 LOC is additive and the simplification is conditional on a future cleanup (line 23, line 334), the conservative default is "keep optional until Optimistix is proven in production." Same applies to lineax at line 341.
```

```
FINDING L3F7: TL;DR "research-grade, novel" is not hedged at point of use, even though §1.3 and §8 Q5 do hedge it
Location: PLAN.md:21
Lens: overclaim-underclaim
Confidence: 65
Severity: minor
Evidence: TL;DR line 21 says "Track 1 as gated spike (research-grade, novel)" with no inline caveat. §1.3 line 96 correctly hedges. §8 Q5 (line 435) repeats the hedge. But TL;DR — typically the most-read part — drops the hedge.
```

```
FINDING L3F8: §1.3 line 96 phrasing "A true Track 1 implementation would be a novel contribution" is asserted in the indicative even though the same sentence's next clause walks it back
Location: PLAN.md:96
Lens: overclaim-underclaim
Confidence: 50
Severity: minor (stylistic)
Evidence: Line 96 "A true Track 1 implementation would be a novel contribution. Publication-viability is asserted by Agent E but not independently verified by a numerical-analysis literature review; treat as a possible follow-up, not a guaranteed outcome." The first sentence is declarative; the second walks it back. Awkward but the hedge is present.
```

```
FINDING L3F9: §4.2 G0 fail action list mentions "FFI to dgeqpf direct" rather than "FFI to dgeqp3 direct" — dgeqpf is the obsolete BLAS-2 routine; dgeqp3 is what jax/scipy already use
Location: PLAN.md:208
Lens: overclaim-underclaim
Confidence: 78
Severity: major (technical incorrectness; falling back to dgeqpf would GUARANTEE losing byte-equality)
Evidence: Line 208 "...or pivot to a different LAPACK call (e.g., FFI to dgeqpf direct)...". dgeqpf is LAPACK's older Level-2 BLAS pivoted-QR routine (deprecated as of LAPACK 3.x in favor of dgeqp3's Level-3 BLAS version). The byte-equal target (SciPy MINPACK / `scipy.linalg.qr(pivoting=True)`) calls dgeqp3, not dgeqpf. Falling back to dgeqpf would *guarantee* losing byte-equality since the upstream oracle uses dgeqp3. The intended fallback is almost certainly direct FFI to the same dgeqp3 underneath jax/scipy, bypassing JAX's explicit-Q post-processing — which is essentially Path C above.
```

## Verified-balanced (checked and found appropriately hedged)

- **"CPU byte-equal by construction" walk-back**: TL;DR (line 21) now says "CPU-byte-exact" without "by construction"; §1.2 explicitly establishes Phase 0 as a gating feasibility check; §4 title says "GATED SPIKE". Walk-back is consistent.
- **"Drops ~500 LOC" → "additive LOC"**: TL;DR line 23 and §5 line 334 both correctly state net LOC is additive. Walk-back is correct and not over-corrected.
- **"L1-only-byte-equal" term**: §10 line 515 acknowledges the fix; Phase 5 fallback line 273 now uses "L1+L2 contract only" with explicit reminder that L1 is tolerance not bit-equality. Term is fixed (though see L3F4 about the substantive collapse).
- **"exact info int match" moved to Track 1 G5**: §3.4 line 167 limits Track 2 info-code parity to the matrix-free subset with "when SciPy also exits on a matrix-free-computable code" guard. Track 2 success criterion line 47 explicitly excludes general exact-info parity. Track 1 G5 line 321 owns it. Walk-back is correct (modulo L3F1 about the subset membership of info=8).
- **Compile time "30-60s" / "<90s" / "<60s"**: Carries "extrapolated estimates from existing _lbfgsb_scipy.py port, not measurements" in TL;DR line 21; §4 line 181 reiterates extrapolation. §4.4 G7 line 323 treats thresholds not predictions. Acceptable.
- **info=4 dependency on gnorm**: Correctly stated at line 47 and line 173 as a pivoted-QR-only quantity.
- **info=3 = conjunction of info=1 and info=2**: Confirmed via lmder.f line 421-422. Plan's claim that info=3 belongs to the non-computable set (line 149) is wrong in the opposite direction (see L3F1's note).
- **Fixture for "iteration count within 1.5× CPU MINPACK"**: §0 line 47 names `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)`; §3.4 line 168 and §6.2 line 380 confirm.
- **Track 2 "convergence semantics improved" as a qualitative gate**: §3.4 line 166 ties it to "LM stops on ftol/xtol disjunction where applicable, not only on ‖∇‖_∞". Concrete enough; backed by new termination tests in line 155.
- **Track 1 (final) success criterion mentions CPU same-machine only**: Line 49 carries "CPU same-machine only" explicitly.
- **TL;DR Track 3 net LOC framing**: Line 23 is appropriately conservative, not over-corrected.

## Lens summary

Nine findings — one critical (info=8 misclassification with parallel info=3 mis-exclusion, infecting 5 plan locations + transitively §0) + five major (Path C omission, L3-on-(384,40)-collapse, byte-equality contract collapse not propagated, §5.1 vs §8 Q4 contradiction, dgeqpf vs dgeqp3 technical error) + three minor (G0 not foregrounded in §0, TL;DR novelty hedge dropped, §1.3 awkward phrasing).
