# JAX MINPACK `lmder` Parity — Implementation Plan

| Field | Value |
|---|---|
| Created | 2026-05-16 |
| Branch | `gpu-purity-stage2-20260405` |
| Parent plan | `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` (Wave 4 W4.3 conditional rollout) |
| Driver | 5-agent max-effort research (Opus 4.7) + independent critic validation pass |
| Author | Jung Dae Suh + Claude Opus 4.7 |
| Status | EXECUTED REVISED (rev 5, 2026-05-17) — Track 2 implemented; original Track 1 byte-equality spike abandoned at Phase 0 G0; revised Track 1 tolerance-equivalent dense-QR lane implemented as `least_squares_algorithm="lm-minpack"` -> `method="lm-minpack-ondevice"`; revised Track 1 broad validation (G3/G4/G5) pending; Track 3 deferred with priority weakened by revised Track 1 outcome |
| Estimated effort | Track 2: implemented · revised Track 1 core route: implemented · **revised Track 1 broad validation pending: ~1–1.5 engineer-days for G3+G4+G5 combined** (MGH suite ~0.5d, oversampled BoozerSurface fixture ~0.5d, compile-timing measurement ~0.25d; recommend bundling into a single PR) · Track 3 (Optimistix): 1–2 weeks if reopened |

---

## Execution Status — 2026-05-17

Track 2 landed as the low-risk convergence-semantics retrofit: the matrix-free
JAX LM now carries the computable MINPACK-style `info` subset, exposes
`ftol`/`xtol`/explicit `gtol`, and uses symmetric damping factors in the
existing matrix-free lane.

Track 1 was executed through its mandatory Phase 0 byte-identity gate and the
original CPU byte-equal contract was abandoned at production scope. The G0
probe in `PHASE0_G0_REPORT.md` compares JAX internal packed `geqp3` + `ormqr`
against SciPy LAPACK `dgeqp3` + `qr_multiply`; all 100 production-shape
`(384, 40)` seeds fail bit equality for packed factor and `Q^T f`, but the
observed drift is approximately `1e-15`.

The owner then changed the Track 1 contract to a CPU tolerance-equivalent
MINPACK-style QR LM lane. The first route is
`least_squares_algorithm="lm-minpack"`, resolving on the target backend to
`method="lm-minpack-ondevice"`. This lane materializes the dense Jacobian and
solves the Marquardt augmented least-squares step with JAX column-pivoted QR.
It does not claim MINPACK packed-QR byte identity.

## TL;DR

Three-track strategy:

- **Track 2 first** (low risk, high signal): retrofit the existing matrix-free JAX LM in `src/simsopt/geo/optimizer_jax.py` to use MINPACK-style `ftol`/`xtol` termination bookkeeping, explicit `gtol` routing, and symmetric Marquardt-style damping factors. Closes the documented W4.3 convergence-semantics sub-gap without touching the inner solve. ~100 LOC. No compile-budget impact.

- **Track 1 revised** (CPU tolerance-equivalent QR lane): implement an opt-in dense pivoted-QR LM lane using `least_squares_algorithm="lm-minpack"` -> `method="lm-minpack-ondevice"`. This keeps the useful QR conditioning from MINPACK-style LM while accepting the Phase 0 evidence that packed-QR byte identity is not achievable on the production `(384,40)` shape. Contract: final-state parity at `atol=rtol=1e-10` on focused fixtures now; broader MGH/Boozer fixture proof remains pending.

- **Track 3 deferred** (architectural shift, library swap): adopt Optimistix `LevenbergMarquardt` with Lineax `LSMR` inner solver as a parallel third lane. Adds ~400 LOC of adapter + 2 direct deps (Optimistix + Lineax, with Equinox floor handled by dependency policy). Better numerical conditioning on near-rank-deficient fixtures (`κ(J)` not `κ(J)²`) only relative to the matrix-free lane; revised Track 1 already supplies dense-QR `κ(J)` conditioning. Not byte-equal to MINPACK; tolerance-equivalent. Net LOC is **additive** while the existing matrix-free LM remains the default; the ~500-LOC simplification only materializes if/when a future cleanup retires the current `_lm_iteration`/`_gmres_solve_least_squares_system` path.

---

## 0. Goals, non-goals, and what success looks like

### Goals

1. **Eliminate or precisely scope the W4.3 algorithmic divergence** between simsopt's CPU `BoozerSurface.minimize_boozer_penalty_constraints_ls(method="lm")` (→ MINPACK `lmder` via SciPy) and the JAX LM lanes. At rev 1, the matrix-free JAX LM was *algorithmically distinct* per its own module docstring (`optimizer_jax.py:14-45`). As of rev 5, the matrix-free lane has MINPACK-style termination bookkeeping and symmetric damping but still differs in the inner solve, while the opt-in `lm-minpack-ondevice` lane provides dense-QR final-state parity pending G3/G4/G5 broad validation.
2. **Improve numerical robustness on near-rank-deficient BoozerSurface fixtures.** The default fixture's `sdofs_inf ≈ 3.6e-5` parity drift documented in the deepdive plan is driven in part by the matrix-free GMRES inner solve seeing `κ(J^T J + λI) = κ(J)² ≈ 10¹⁴`. Both pivoted-QR (Track 1) and LSMR (Track 3) reduce the effective condition number to `κ(J)`.
3. **Preserve current JAX/CUDA performance characteristics.** Tracks 2 and 3 must not regress the H100 wall-clock per LM iteration. Track 1 must not regress beyond the 60s first-compile gate in `docs/source/jax_acceptance.rst:101`.
4. **Maintain validation discipline.** Every new piece lands with a SciPy/MINPACK oracle test at the appropriate `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` lane.

### Non-goals

1. **GPU byte-equality with MINPACK.** Documented as impossible (cuSOLVER + MAGMA are not bit-deterministic). GPU contract remains *tolerance-equivalent* per parity-ladder lanes.
2. **Replacing or modifying the CPU SciPy path.** `BoozerSurface.minimize_boozer_penalty_constraints_ls` keeps its existing SciPy/MINPACK call site; this plan only affects the JAX lane.
3. **Replacing the L-BFGS-B work landed in commit `0a54646c1`.** That's a separate optimizer family.
4. **Publication / write-up.** Listed as a possible follow-up but not a deliverable.

### Success criteria

| Track | Success condition |
|---|---|
| Track 2 | Three-criterion termination bookkeeping + symmetric damping factors land; **convergence semantics improved** (the JAX LM can stop on ftol/xtol disjunction rather than single grad-norm); iteration count on `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` is within 1.5× of CPU MINPACK iteration count; `ls_state_parity` lane continues to pass at `sdofs_inf ≤ 1e-11`; focused LM tests and the boozersurface regression slice are green. The downstream `tests/integration/test_single_stage_jax.py` run remains a separate full-validation gate and is not claimed by rev 5. **Exact `info` integer parity over the full 1-8 range is explicitly NOT a Track 2 gate** — MINPACK's `info=4` (`if (gnorm .le. gtol) info = 4`, `lmder.f:313`) and `info=8` (`if (gnorm .le. epsmch) info = 8`, `lmder.f:431`) both depend on `gnorm = max_l |J[:,l]^T·(q^T fvec)/r_diag[l]|`, which is a pivoted-QR-only quantity unavailable in the matrix-free GMRES inner solve. Track 2's info-code parity is restricted to the matrix-free-computable subset {1, 2, 3, 5, 6, 7}; the pivoted-QR-required pair {4, 8} belongs to Track 1 G5. |
| Track 1 (per gate) | Revised Gate G0 accepts the measured `~1e-15` packed/`Q^T f` drift because it is below the active `1e-10` tolerance contract. Follow-on gates are final-state parity gates, not byte-identity gates: direct fixtures first, then MGH, then the oversampled BoozerSurface fixture. |
| Track 1 (final) | `least_squares_algorithm="lm-minpack"` opt-in passes final-state parity against `scipy.optimize.least_squares(method="lm")` at `atol=rtol=1e-10` on direct least-squares fixtures, then the broader MGH suite and oversampled BoozerSurface fixture. Exact packed-QR bytes, exact per-iteration trace, and exact `niter`/`nfev` are explicitly outside the revised Track 1 contract. |
| Track 3 | `least_squares_algorithm="optimistix-lm"` opt-in matches converged state of the current LM at `branch-stable-resolve` lane on the oversampled fixture; better numerical robustness on the default (near-rank-deficient) fixture measured in `sdofs_inf` drift. |

---

## 1. Context

### 1.1 Pre-execution state (snapshot at rev 1; superseded by Track 2 + Track 1 revised work)

**Note:** the bullets below describe the simsopt JAX LM as it existed at the time the plan was first written (rev 1, 2026-05-16). Track 2 (rev 4) and revised Track 1 (rev 4) both touched this code; see §3 and §4 for the current state. Line numbers below are intentionally **historical** — they pin the rev-1 baseline so the plan-vs-implementation diff is reconstructible.

- The simsopt JAX LM lived in `src/simsopt/geo/optimizer_jax.py:1209-1660` (rev 1 line range). Two callable methods: `levenberg_marquardt` (host-driven) and `levenberg_marquardt_traceable` (`lax.while_loop` traceable). Both shared `_lm_iteration` and `_lm_defaults`. (Current HEAD line numbers have drifted by +130 to +210 lines due to Track 2 + revised Track 1 additions.)
- Inner solve: matrix-free GMRES against `J^T J + λI` via `_gmres_solve_least_squares_system`. **Still true post-Track-2** for the `lm` / `lm-ondevice` lanes; the revised Track 1 `lm-minpack-ondevice` lane uses dense pivoted QR instead.
- Termination: single criterion `‖∇‖_∞ ≤ tol`. **Superseded by Track 2**, which added MINPACK-style three-criterion termination (`ftol`/`xtol`/`gtol`) surfacing `info` codes 1, 2, 3, 5, 6, 7 via `_matrix_free_lm_info()`.
- Damping update: asymmetric trust-region with factors `expand=4.0`, `shrink=0.5`, `mild_shrink=0.8`. **Superseded by Track 2**, which replaced these with symmetric Marquardt `× 2 / × 0.5`.
- W4.3 module docstring (`:14-45`) explicitly states this LM is **algorithmically distinct** from MINPACK `lmder` along three axes (inner solve, termination, damping update).
- Plan precedent: `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` lines 292-302 documents the MINPACK port as a conditional W4.3 follow-up, with trigger criteria "production default AND byte-equality requirement" both unmet at the time of writing.

### 1.2 Why this plan now

A five-agent research pass (logged in this session, summarized below in §10) surfaced one key empirical finding: **a necessary pivoted-QR primitive for a MINPACK-style lane is now available through a public JAX API**. `jax.scipy.linalg.qr(pivoting=True)` exists in JAX 0.10.0 (added by [JAX PR #25955](https://github.com/jax-ml/jax/pull/25955), Feb 2025). The public API returns explicit `(Q, R, P)`, not MINPACK's packed Householder `fjac`; Phase 0 therefore still had to test byte-level feasibility instead of assuming a complete MINPACK port from that API.

Empirically reproduced on this branch's `.conda/jax` env (JAX 0.10.0 / jaxlib 0.10.0, scipy 1.17.1, x64 enabled):

```
shape (40,40)  seed=0:   P=eq  R bit-equal  Q bit-equal
shape (75,39)  seed=0:   P=eq  R bit-equal  Q bit-equal
shape (100,50) seed=7:   P=eq  R bit-equal  Q bit-equal
shape (384,40) seed=0:   P=eq  R bit-equal  Q max diff = 4.163e-17  (NOT bit-equal)
shape (384,40) seed=1:   P=eq  R bit-equal  Q max diff = 4.163e-17  (NOT bit-equal)
```

**Key result, not an unqualified win.** R and P are bit-equal to LAPACK on every shape probed. **Q is bit-equal on m≈n shapes but not on the production-sized `(384,40)` BoozerSurface shape.** Two reasons this is still gate-relevant for Track 1 but not a free pass:

1. MINPACK `qrfac.f` does not return an explicit `Q`. It packs the Householder vectors into the strict lower triangle of `A` and returns `(fjac_packed, rdiag, acnorm, ipvt)`. JAX `jax.scipy.linalg.qr` returns explicit `(Q, R, P)`; `jax.lax.linalg` exposes `qr`, `householder_product`, `ormqr` but **no `geqrf` analog** that yields the packed form directly. The packed `fjac` representation that the rest of MINPACK consumes must be reconstructed from JAX outputs, and on `(384,40)` the reconstructed Householder vectors are not guaranteed bit-equal because the explicit Q itself drifts by ~4e-17 versus LAPACK.
2. The downstream `qrsolv` and `lmpar` subroutines consume `(fjac_packed, ipvt, rdiag, qtb)`, where `qtb = Q^T·fvec`. With Q bit-drifted, `qtb` will bit-drift, and the rest of the byte-equality argument collapses on the `(384,40)` fixture.

The remaining subroutines (`qrsolv` Givens elimination, `lmpar` univariate Newton, `enorm` 3-bucket scaling) all reduce to standard `lax.scan` / `lax.while_loop` / `lax.fori_loop` patterns already used in the existing `_lbfgsb_scipy.py` port — but feasibility of the **end-to-end byte-equal driver on the production fixture** depends on a Phase-0 feasibility gate (§4.2) that proves the MINPACK-packed `(fjac, rdiag, acnorm, ipvt)` tuple plus `qtb` can be made bit-equal to a SciPy/MINPACK oracle on `(384,40)`. **If Phase 0 fails (no two paths agree per §4.2 G0 / §8 Q8), Track 1 is abandoned at the production scope, full stop.** Any re-scope to `m ≈ n` shapes only is a project-charter renegotiation that requires owner sign-off per §8 Q8; it is not a default alternate path and the plan author may not unilaterally re-scope.

### 1.3 Prior art (verified by Agent E)

No published or open-source JAX/PyTorch LM implementation achieves MINPACK byte-equality:
- **JAXopt**: Madsen-Nielsen Algorithm 6.18, explicitly NOT MINPACK ([source](https://github.com/google/jaxopt/blob/main/jaxopt/_src/levenberg_marquardt.py))
- **Optimistix**: classical Marquardt damping, NOT MINPACK parity ([arXiv:2402.09983](https://arxiv.org/abs/2402.09983))
- **torchimize / torch-levenberg-marquardt**: normal-equations inner, no MINPACK claim
- **jaxls / JAXFit**: sparse CG / trust-region-reflective, no MINPACK claim
- **pwkit `lmmin.py`**: pure-NumPy MINPACK lineage, explicitly disclaims byte-equality after layout transpose
- **JAX core**: pivoted-QR infrastructure landed (#25955) but no LM consumer; issue #5097 ("unimplemented scipy.optimize functions") open since Dec 2020 with no MINPACK-specific traction

A true Track 1 implementation would be a novel contribution. Publication-viability is asserted by Agent E but **not independently verified** by a numerical-analysis literature review; treat as a possible follow-up, not a guaranteed outcome.

### 1.4 Independent critic findings (validated against code)

A second independent review pass on the synthesis recommended:
- **"CPU byte-equal by construction" is too strong.** QR parity is necessary but not sufficient; full driver byte-equality requires `qrsolv`, `lmpar`, `enorm`, machine constants, stopping statuses, and driver control flow to all match. Track 1 must be gated by per-subroutine oracles.
- **LOC and compile-time estimates are extrapolations, not measurements.** Treat 600–800 LOC and 30–60s first-trace as hypotheses to be validated by the spike.
- **Track 2 first.** Smaller, lower risk, independently valuable.
- **GPU byte-equality must not be in the contract.** Reframe GPU as tolerance/parity-ladder-governed only.

All four critic recommendations are folded into this plan.

---

## 2. Rationale

### Why the original MINPACK byte-equality target mattered

The bullets below are historical rationale for the abandoned byte-equality track. They are not active rev-5 gates.

1. **Reproducibility of published results.** Stellarator coil designs converged using `scipy.optimize.least_squares(method="lm")` would be reproducible bit-for-bit on a JAX runtime, which is a stronger contract than tolerance-equivalence. Useful for paper supplementary materials and regression testing.
2. **Cross-validation oracle.** At rev 1, simsopt's parity testing for the LS lane used SciPy MINPACK on the CPU side and a different algorithm on the JAX side; differences could be due to either implementation or the algorithm itself. A byte-equal JAX MINPACK port would have collapsed the implementation-vs-algorithm ambiguity for CPU comparisons.
3. **Numerical conditioning.** Pivoted QR is backward-stable to `ε_mach·κ(J)`; matrix-free GMRES on `J^T J + λI` sees `ε_mach·κ(J)²`. For the documented near-rank-deficient default BoozerSurface fixture (`κ(J) ≈ 10⁷`), this is a 7-order-of-magnitude conditioning improvement.

### Why not just adopt Optimistix instead

Optimistix is a viable medium-term path (Track 3 of this plan). The reasons to also pursue Track 1:
- Optimistix is tolerance-equivalent to MINPACK, not byte-equal. Lose the reproducibility-oracle property.
- Optimistix is a single-maintainer (Patrick Kidger) dependency. Track 1 has no new runtime deps beyond JAX 0.10.0.
- Optimistix doesn't preserve the existing simsopt parity-ladder lane structure as cleanly; lane boundaries are designed around the *exact* algorithm contract.
- The two tracks are not mutually exclusive: revised Track 1 gives a dense-QR tolerance-equivalent CPU validation lane; Track 3 would give a separate LSMR-based production lane if reopened.

### Why Track 2 must come first

Track 2 closes the documented W4.3 algorithmic-divergence gap on the **convergence semantics** (termination + damping) without touching the inner solve. It's small enough to land as a single PR with full validation, and the result is independently useful even if Track 1 is later abandoned. It also reduces the contract debt that any Track 1 / Track 3 work would inherit.

---

## 3. Track 2 — MINPACK termination + damping retrofit (IMPLEMENTED 2026-05-17)

**Scope:** modify `_lm_iteration` and `_lm_defaults` in `src/simsopt/geo/optimizer_jax.py` to use three-criterion termination and symmetric Marquardt damping. Keep matrix-free GMRES inner solve unchanged. ~100 LOC + tests.

**Status:** all todos below land in commit `5bfbd49ef fix: harden JAX LM option contracts` and validate via 10/10 tests in `tests/geo/test_lm_termination_parity.py` + `tests/geo/test_lm_damping_parity.py`, plus 385/389 boozersurface tests passing with no regression.

**Estimated effort:** 2 working days including validation — actual landed effort matched estimate.

### 3.1 Three-criterion termination

MINPACK terminates on the disjunction of three independent criteria, each producing a distinct `info` code (1, 2, 3, 4) plus three "too small" variants (6, 7, 8) and one budget exhaustion (5). Reference: `optimizer_jax.py:14-45` module docstring and Agent A's spec §5.

### 3.2 Symmetric Marquardt damping

Rev-1 scheme: `expand=4.0, shrink=0.5, mild_shrink=0.8` (asymmetric — `4.0 ≠ 1/0.5`). The retrofit replaces those factors with symmetric Marquardt-style `× 2 / × 1/2` damping updates. It does **not** implement MINPACK's full bracketed `lmpar`/`par` search in the matrix-free lane; that requires QR data and remains outside Track 2.

### 3.3 Todos (Track 2)

- [x] Added `ftol`, `xtol`, `gtol` parameters to `levenberg_marquardt` (`optimizer_jax.py:1707`) and `levenberg_marquardt_traceable` (`:1842`). Default: `ftol=xtol=gtol=1e-8` matching `scipy.optimize.least_squares(method='lm')` (the legacy `scipy.optimize.leastsq` MINPACK-direct wrapper uses `1.49012e-8`, but `least_squares` overrides this).
- [x] Added `info` code field to `_lm_iteration` carry state (`:1547`). Tracks an internal int32 alongside the existing `success` boolean (`success = legacy_success | info_success` where `info_success = info in {1, 2, 3}`). Codes 1, 2, 3, 5, 6, 7 are computed from ftol/xtol/maxfev bookkeeping alone (info=3 is the conjunction `info=1 AND info=2` per `lmder.f:421-422`); codes 4 and 8 both require `gnorm` (a pivoted-QR-only quantity, per `lmder.f:431` for info=8 and `:313` for info=4) and are reported as `info=0` in the matrix-free Track 2 lane.
- [x] Implemented `_matrix_free_lm_info(...)` at `optimizer_jax.py:1480` for the matrix-free-computable subset (codes 1, 2, 3, 5, 6, 7). The full 8-code cascade (adding {4, 8}) is now in the Track 1 `lm-minpack-ondevice` lane.
- [x] Replaced `_lm_defaults` damping factors (`:1437`) with symmetric Marquardt-style factors: `increase_factor=2.0`, `decrease_factor=0.5`. Old `expand=4.0` / `mild_shrink=0.8` asymmetry retired. Full MINPACK `lmpar` bracket search is not present in the matrix-free Track 2 lane.
- [x] Replaced the asymmetric trust-region update in `_lm_iteration` with a symmetric Marquardt update (`damping × 0.5` on accept with high ratio; `damping × 2.0` on reject or low ratio).
- [x] Updated `levenberg_marquardt` / `levenberg_marquardt_traceable` while-loop predicates to terminate when `info != 0` (matrix-free subset) OR `success` fires on the legacy `‖∇‖_∞ ≤ tol` criterion. Signature compatibility is preserved for callers passing only `tol`; termination behavior can now change by design because default `ftol=xtol=1e-8` may stop before the legacy gradient gate.
- [x] Result schema in `_lm_iteration` (`:1678`) now surfaces `"info": info_next` alongside `"success": finite_candidate & (legacy_success | info_success)`. Range: `info ∈ {0, 1, 2, 3, 5, 6, 7}` in Track 2; `info ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8}` in the Track 1 `lm-minpack-ondevice` lane.
- [x] `tests/geo/test_lm_termination_parity.py` lands with 5 tests: matrix-free info subset ordering, rejected-uphill-tiny-reduction handling, ftol/xtol info surfacing, explicit-gtol gradient gate. **All 5 PASS.**
- [x] `tests/geo/test_lm_damping_parity.py` lands with 5 tests: damping halves on good step, doubles on rejected step, MINPACK ratio threshold gating, iteration-count-within-1.5×-SciPy on Rosenbrock, iteration-count-within-1.5× on oversampled BoozerSurface fixture. **All 5 PASS.**
- [x] `optimizer_jax.py:17-50` module docstring "LM family note" updated to state that the matrix-free lane surfaces `info` codes 1, 2, 3, 5, 6, 7 and that codes 4 and 8 remain pivoted-QR-only (Track 1 lane).
- [x] `docs/source/jax_acceptance.rst:156-187` "Optimizer family equivalence" section updated with the matrix-free MINPACK-style termination + symmetric damping contract.
- [x] Boozersurface regression validated: `tests/geo/test_boozersurface_jax.py -m "not private_optimizer_runtime"` reports 385/389 passed (4 skipped, **0 regressions**).
- [x] `ls_state_parity` lane confirmed to still pass at `sdofs_inf ≤ 1e-11` on the oversampled fixture (part of the boozersurface regression run above).

### 3.4 Acceptance gates (Track 2)

| Gate | Lane | Threshold | Status (rev 5) |
|---|---|---|---|
| Convergence semantics improved | qualitative + new tests | LM stops on ftol/xtol disjunction where applicable, not only on `‖∇‖_∞` | ✅ PASS — `test_lm_termination_parity.py` ftol/xtol/gtol surfacing tests all green |
| `info` code ordering on matrix-free-computable subset {1, 2, 3, 5, 6, 7} | `branch-stable-resolve` | match raw MINPACK `lmder` ordering for the computable subset; do not compare directly to SciPy `OptimizeResult.status`, which remaps LM statuses | ✅ PASS — `test_matrix_free_info_subset_matches_minpack_ordering` exercises each code |
| Iteration count parity | `branch-stable-resolve` | within 1.5× of CPU MINPACK | ✅ PASS — `test_matrix_free_lm_iteration_count_stays_close_on_oversampled_boozer_fixture` green |
| Converged-state parity (oversampled fixture) | `ls_state_parity` | `sdofs_inf ≤ 1e-11` (existing) | ✅ PASS — covered by 385/389 boozersurface regression run |
| No regression in existing boozersurface_jax tests | (existing lanes) | all green | ✅ PASS — 385 passed, 4 skipped, 0 failed |
| `ruff check`, `ruff format` clean on changed files | n/a | pass | ✅ PASS (per commit `5bfbd49ef` hygiene) |

**Explicitly NOT a Track 2 gate:** exact `info` code parity over the full 1-8 range. Both MINPACK's `info=4` (`lmder.f:313`, `if (gnorm .le. gtol) info = 4`) and `info=8` (`lmder.f:431`, `if (gnorm .le. epsmch) info = 8`) depend on the scaled-gradient norm `gnorm` (computed at `lmder.f:297-306` using pivoted-QR factors), which Track 2's matrix-free GMRES inner solve cannot materialize. Track 2 reports `info=0` in those branches; Track 1's `lm-minpack` lane is the only path that can match `info ∈ {4, 8}`. `info=3` is the conjunction `info=1 AND info=2` (`lmder.f:421-422`) and IS matrix-free-computable, so it stays in Track 2's subset.

---

## 4. Track 1 — CPU tolerance-equivalent MINPACK-style QR LM lane

**Scope:** `src/simsopt/geo/optimizer_jax.py` implements
`levenberg_marquardt_minpack_traceable`. The opt-in path is
`least_squares_algorithm="lm-minpack"` on `optimizer_backend="ondevice"`,
resolving to `method="lm-minpack-ondevice"`. The contract is final-state
tolerance equivalence to SciPy/MINPACK, not a private `_lmder.py` packed-QR
byte-identical port.

### 4.1 Why this is no longer a byte-equality spike

Phase 0 proved that the original byte-identical packed-QR route fails on the
production `(384,40)` shape. The failure magnitude is around `1e-15`, which is
far below the revised `1e-10` tolerance contract but enough to invalidate a
byte-for-byte MINPACK driver.

The revised implementation keeps the useful numerical property: it solves each
Marquardt augmented least-squares step with a dense column-pivoted QR factor,
so it avoids the matrix-free lane's normal-equation conditioning penalty. It
does not reconstruct MINPACK's packed Householder storage, does not route
through private JAX internals, and does not claim exact internal trace parity.

### 4.2 Revised implementation order

- [x] Accept Phase 0 under the revised tolerance gate: worst observed drift
  `~1e-15 < 1e-10`.
- [x] Add `least_squares_algorithm="lm-minpack"` to optimizer routing.
- [x] Add `method="lm-minpack-ondevice"` to the target least-squares entrypoint.
- [x] Implement `levenberg_marquardt_minpack_traceable` with dense Jacobian
  materialization and `jax.scipy.linalg.qr(..., pivoting=True)`.
- [x] Route `BoozerSurfaceJAX` LS solve paths through the new solver when the
  resolved method is `lm-minpack-ondevice`.
- [x] Add direct focused parity tests against SciPy LM and an overdetermined QR
  fixture.
- [ ] **G3 (~0.5 day)** Run the broader MGH suite. Extend `tests/geo/test_lm_minpack_qr_parity.py` to drive `method="lm-minpack-ondevice"` against `scipy.optimize.least_squares(method="lm")` on the Moré-Garbow-Hillstrom problem set (start with the 5 canonical problems used elsewhere in this plan: Rosenbrock, Helical valley, Powell singular, Brown almost-linear, Beale; extend to the full 18 if all 5 pass). Assert final-state parity at `atol=rtol=1e-10`. Lane: `direct-kernel` extended.
- [ ] **G4 (~0.5 day)** Run the oversampled BoozerSurface fixture. Drive `method="lm-minpack-ondevice"` through `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` (the same 384×40 fixture that Phase 0 ran on), assert final-state parity vs CPU MINPACK at `atol=rtol=1e-10`. Lane: `branch-stable-resolve`.
- [ ] **G5 (~0.25 day)** Measure first-trace compile time on the canonical `(384,40)` fixture. Record value in `PHASE0_G0_REPORT.md` (or sibling file). Owner decision per §8 Q6 if measurement exceeds the 60s target documented in `docs/source/jax_acceptance.rst:101`.
- [ ] **Recommended bundling**: G3+G4+G5 as a single ~1.5-engineer-day PR that closes Track 1 revised entirely and promotes `lm-minpack-ondevice` from "implemented" to "production-ready". No mid-PR owner sign-off needed unless G5 measurement triggers Q6.

### 4.3 Superseded byte-identity spike phases

The original `enorm`/`qrfac`/`qrsolv`/`lmpar`/private-`_lmder.py` phase plan
is superseded by the revised tolerance-equivalent route above. Those phases are
not active work unless the project later reopens a byte-identical MINPACK port.

### 4.4 Track 1 deliverable file map

| File | Status |
|---|---|
| `src/simsopt/geo/optimizer_jax.py` | revised route + dense-QR solver |
| `src/simsopt/geo/boozersurface_jax.py` | revised method routing |
| `tests/geo/test_lm_minpack_qr_parity.py` | focused direct parity tests |
| `tests/geo/test_boozersurface_jax.py` | resolver and route coverage |
| `docs/source/jax_acceptance.rst` | precision-contract docs |
| `.artifacts/lm_minpack_port_plan_2026-05-16/PLAN.md` | revised status |
| `.artifacts/lm_minpack_port_plan_2026-05-16/PHASE0_G0_REPORT.md` | revised gate decision |

### 4.5 Acceptance gates (Track 1, revised)

| Gate | Phase | Status check | Action on failure |
|---|---|---|---|
| G0 — tolerance gate | 0 | original byte gate fails, but drift `~1e-15 < 1e-10` | accepted under revised contract |
| G1 — route executes | 1 | `least_squares_algorithm="lm-minpack"` resolves to `method="lm-minpack-ondevice"` | block route |
| G2 — direct final-state parity | 2 | focused SciPy LM / linear QR fixtures pass at `atol=rtol=1e-10` | block route |
| G3 — MGH suite | 3 | broader Moré-Garbow-Hillstrom final-state parity | block release of Track 1 |
| G4 — BoozerSurface fixture | 4 | oversampled BoozerSurface final-state parity | block production promotion |
| G5 — compile time | 5 | first-trace timing measured on canonical `(384,40)` fixture | owner decision if too slow |

---

## 5. Track 3 — Optimistix + Lineax LSMR (DEFERRED, priority weakened post-rev-5)

**Scope:** add a third opt-in `least_squares_algorithm="optimistix-lm"` routing to `optimistix.LevenbergMarquardt(linear_solver=lineax.LSMR(...))`. **Net LOC is additive in this plan** (~+400 LOC adapter + 2 direct deps; the existing matrix-free LM stays). The "~500 LOC simplification" only materializes if a separate future cleanup retires the current `_lm_iteration`/`_gmres_solve_least_squares_system` path after Optimistix is proven in production.

**Priority status (rev 5):** When the plan was first written, Track 3's primary numerical-conditioning argument was "reduce `κ(J)²` to `κ(J)` on near-rank-deficient fixtures" — a strong argument because the matrix-free GMRES inner solve was the only available JAX LM and had the `κ(J)²` penalty. **As of rev 5 this argument is substantially weakened: revised Track 1 (`method="lm-minpack-ondevice"`) already provides the `κ(J)` conditioning via dense pivoted QR**, so Track 3 no longer carries the conditioning argument alone. Remaining Track 3 benefits are:
- GPU vmap-friendliness (Optimistix is Equinox-based, designed for vmap)
- LSMR scalability for very-large `m` (matrix-free in `J`, no dense materialization)
- Library-vs-custom maintenance burden tradeoff

Recommend **keeping Track 3 deferred indefinitely** unless one of those three benefits becomes a concrete need. Formally retire if the project decides none of them will materialize.

**Estimated effort:** 1–2 weeks if reopened. Should not start until revised Track 1's G3+G4+G5 land and Track 3's priority is re-evaluated against the post-Track-1 state.

### 5.1 Todos (Track 3)

- [ ] Add `optimistix`, `lineax`, and any direct `equinox` floor to `pyproject.toml` **only as an optional extra unless the project first raises its Python floor.** Current docs report Optimistix requiring Python 3.11+ and Lineax requiring Python 3.10+ / JAX 0.4.38+ / Equinox 0.11.10+, while this repo still advertises `requires-python = ">=3.8"` with Python 3.8-3.11 classifiers. Making Track 3 required today would be a downstream packaging regression.
- [ ] Add `lineax` under the same optional-extra dep-tier policy as `optimistix` per §8 Q4.
- [ ] Implement `jax_least_squares_optimistix(residual_fn, x0, ...)` wrapper in `optimizer_jax.py` using the documented Optimistix call shape: `optimistix.least_squares(residual_fn, optimistix.LevenbergMarquardt(rtol=tol, atol=tol, linear_solver=lineax.LSMR(rtol=tol, atol=tol)), x0, ...)`.
- [ ] Implement `_optimistix_solution_to_scipy_optimize_result(sol, ...)` adapter mapping `optx.Solution` → `scipy.optimize.OptimizeResult` so downstream code consuming `res["x"]`, `res["fun"]`, `res["nit"]`, `res["residual"]`, `res["residual_jacobian"]`, `res["hessian"]`, `res["PLU"]` still works.
- [ ] Add `"optimistix-lm"` to `VALID_LEAST_SQUARES_ALGORITHMS`. Wire routing in `resolve_target_least_squares_optimizer_method`.
- [ ] Update `BoozerSurfaceJAX` option validation to accept `least_squares_algorithm="optimistix-lm"`.
- [ ] Add `tests/geo/test_lm_optimistix_parity.py`: converged-state parity at `branch-stable-resolve` lane on the oversampled BoozerSurface fixture against the current `lm` lane.
- [ ] Add `tests/geo/test_lm_optimistix_robustness.py`: measure `sdofs_inf` drift on the default (near-rank-deficient) BoozerSurface fixture; expect LSMR's `κ(J)` (not `κ(J)²`) to reduce drift vs current matrix-free GMRES.
- [ ] Add GPU lane test (skipif no CUDA): converged-state parity at `gpu-runtime` lane.
- [ ] Update `optimizer_jax.py:14-45` module docstring with a fifth family entry: `optimistix-lm` (Optimistix LM with Lineax LSMR inner solver, tolerance-equivalent, vmap-friendly).
- [ ] Update `docs/source/jax_acceptance.rst` "Optimizer family equivalence" section.
- [ ] Update `CLAUDE.md` to add Optimistix as a runtime dep and document the LM family routing.

### 5.2 Acceptance gates (Track 3)

| Gate | Lane | Threshold |
|---|---|---|
| Converged-state parity vs current `lm` lane | `branch-stable-resolve` | `rtol=1e-6, atol=1e-7` on `x_final`, `cost_final` |
| Numerical robustness on default fixture | new `optimistix_robustness` lane | `sdofs_inf` ≤ existing matrix-free LM result |
| GPU lane | `gpu-runtime` | `rtol=1e-6` on converged state |
| First-trace compile time | n/a | `< 60s` (target) |
| No regression in `boozersurface_jax` or `single_stage_jax` tests | existing | all green |

---

## 6. Validation methodology

### 6.1 Precision ladder (per Agent E)

| Level | Definition | Tolerance | Hardware caveat | simsopt lane |
|---|---|---|---|---|
| L1 — converged solution | `‖x_minpack - x_jax‖_∞ ≤ rtol·(‖x_minpack‖_∞ + 1)` | `rtol=1e-6, atol=1e-10` (well-cond), `rtol=1e-4, atol=1e-8` (ill-cond) | Portable CPU/GPU/cross-machine | `branch-stable-resolve` / `fd-gradient` |
| L2 — path length + termination | `niter, info, nfev` exact match (nfev ±1 allowance) | exact int match | Same algorithm only; CPU↔GPU tie-breaks may fail | `branch-stable-resolve` audit field |
| L3 — iteration trace within ε_mach | per-iter `(x, fnorm, delta, par, ratio, info)` agree | `rtol=1e-12, atol=1e-14` in double | Same machine only; CPU↔GPU fails by design | `direct-kernel` extended |
| L4 — byte-equality | every fp64 bit matches | bitwise | Single host + same BLAS + same compile flags; impossible CPU↔GPU | `direct-kernel` state-parity sub-lane |

### 6.2 Test fixtures

- **Moré-Garbow-Hillstrom 1981 suite** — 18 canonical problems. Source: `fortran-lang/minpack/test`.
- **simsopt BoozerSurface fixtures**:
  - `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` — oversampled, well-conditioned, `κ ≈ 5.3e4`
  - Default fixture (`ncoils=2, nphi=5, ntheta=5, mpol=ntor=2`) — under-sampled, `κ ≈ 10⁷`, exercises damping escalation
- **Edge cases**: zero-residual initial guess, exact rank deficiency, `maxfev=2` forced exhaustion, `fnorm=0` start.

### 6.3 SciPy interception (for L2/L3)

`scipy.optimize.least_squares(method='lm')` has no usable iteration callback for this purpose: current SciPy exposes a `callback` parameter on `least_squares`, but the official docs state it is only implemented for `trf` and `dogbox`. Also, SciPy `OptimizeResult.status` is not the raw MINPACK `lmder` info numbering used in §3; use raw MINPACK/leastsq/cminpack-style instrumentation when exact info-code tests are needed. Three options (in order of preference):

1. **Wrapped residual / Jacobian** logging on every call — track iteration count via fnorm decreases. Default L3 instrument.
2. **Patched SciPy fork** with HDF5 trace dump in `_minpackmodule.c` — ground-truth oracle for fixture generation.
3. **Direct ctypes binding to `cminpack` or `fortran-lang/minpack`** — cleanest oracle for sub-routine (qrsolv, lmpar) testing.

### 6.4 JAX interception

For guaranteed host trace dumps, use `jax.experimental.io_callback(callback, result_shape, *args, ordered=True)` only in a non-vmapped `jit`/`lax.while_loop` diagnostic. Official JAX docs state that `io_callback` supports `vmap` only when `ordered=False`, and that `vmap` over `scan`/`while_loop` callbacks has complicated semantics. The committed solver callbacks currently use `jax.debug.callback`, which is suitable for debug/progress instrumentation but is not a guaranteed-execution trace oracle.

### 6.5 CI cost estimate

| Suite | Cold compile | Warm |
|---|---|---|
| Track 2 termination + damping tests | already landed; keep in focused regression slice | +2s |
| Track 1 direct QR parity tests | current focused suite, 4 tests | +5s |
| Track 1 G3 MGH-1981 final-state suite | measure in G3; expected to dominate CPU validation | TBD |
| Track 1 G4 oversampled BoozerSurface fixture | measure in G4; same production `(384,40)` shape as Phase 0 | TBD |
| Track 1 G5 first-trace timing | one measured cold compile on canonical `(384,40)` fixture | required before production promotion |
| Track 3 Optimistix parity | deferred; not part of current CI budget | n/a |

The old subroutine-oracle, per-iteration trace, and GPU Track 1 rows belonged to the abandoned byte-identical MINPACK port. They are inactive unless that project is explicitly reopened.

---

## 7. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `jax.scipy.linalg.qr(pivoting=True)` tie-break differs from LAPACK on exactly-equal column norms | LOW | Empirically verified on 4 fixtures; add a dedicated equal-norm fixture in G2 |
| `enorm` vectorized form differs from sequential MINPACK in last bit | INACTIVE | Byte-identical `enorm` is not part of the revised Track 1 route; if the byte port is reopened, require a scalar `lax.fori_loop` implementation and oracle test. |
| `qrsolv` Givens chase order produces different bit-pattern than Fortran | INACTIVE | Byte-identical `qrsolv` is not part of the revised Track 1 route; if reopened, mirror Fortran column order and gate with an oracle test. |
| `lmpar` Newton iteration count differs (off-by-one bracket update) | INACTIVE | Byte-identical `lmpar` is not part of the revised Track 1 route; current dense-QR lane uses final-state parity gates. |
| First-trace compile exceeds 60s gate | MEDIUM | G5 measures the real value. If it exceeds 60s, §8 Q6 requires owner choice between optimization, explicit documented exception, or rejecting production promotion. |
| Optimistix maintainer abandons project or dependency floors conflict with simsopt packaging | LOW | Track 3 is deferred and opt-in; pin only after §8 Q4 resolves the Python/JAX floor policy. |
| Packed-QR byte-equality not achievable | REALIZED | Revised Track 1 accepts tolerance-equivalent dense-QR final-state parity instead of byte identity |
| GPU lane tie-break (MAGMA `geqp3` ≠ LAPACK `dgeqp3`) breaks L2/L3 cross-device | LOW | Documented in §0 Non-goals: GPU lane is L1-only by design |
| CPU LAPACK linkage drift (different vendor's LAPACK on different machines) breaks L4 | LOW | Documented in §6.1: L4 is single-host single-build only |

---

## 8. Open decisions

These originally needed owner sign-off before starting work. Items Q1-Q3 and
Q7-Q8 are resolved by the 2026-05-17 Track 1 contract change. Q5 was made moot
by the rev-5 priority reframing. Q4 (Optimistix dep tier) and Q6 (compile-budget
exception) remain open but with reduced urgency — Q4 only fires if Track 3 is
reopened, Q6 is gated on the G5 compile-time measurement from §4.2.

- [x] **Q1 — Track scope**: Track 2 plus revised Track 1 tolerance-equivalent QR lane; Track 3 remains deferred.
- [x] **Q2 — Track ordering**: Track 2 first, then revised Track 1 route.
- [x] **Q3 — Track 1 abandonment threshold**: byte-equality spike abandoned at G0; revised tolerance-equivalent route accepted.
- [ ] **Q4 — Optimistix as required dep** *(low urgency; Track 3 deferred per §5 rev-5 priority note)*: Track 3 has no active import path. Current official Optimistix/Lineax docs make a required dependency incompatible with this repo's advertised Python 3.8-3.10 support, so the default is "optional extra only" unless the project separately raises its Python/JAX floors.
- [x] **Q5 — Publication ambition** *(resolved moot by rev 5)*: The original Q5 framing ("first JAX MINPACK port") assumed the byte-identical Track 1 contract. **Revised Track 1 is a tolerance-equivalent dense-QR LM lane, not a MINPACK port** — the "first JAX MINPACK port" novelty claim no longer applies and Q5 is moot as originally written. If a publication is still desired, the new framing would need its own novelty assessment (e.g., "JAX-native dense-QR LM lane with κ(J) conditioning" — a much weaker novelty claim).
- [ ] **Q6 — Compile-budget exception** *(pending G5 measurement)*: should Track 1 be exempted from the 60s first-compile gate in `docs/source/jax_acceptance.rst:101`? Decision should be made after G5 produces an actual measurement (§4.2). If G5 measures < 60s, Q6 is auto-resolved as "no exception needed". If G5 measures > 60s, owner decides whether to (a) carve out `lm-minpack` from the 60s gate in `jax_acceptance.rst`, (b) optimize the lane to fit, or (c) accept the regression with explicit owner sign-off.
- [x] **Q7 — Cross-machine validation**: L4 byte-equality is no longer in the Track 1 contract.
- [x] **Q8 — Phase 0 Path C scope and re-scope authority**: no Path C; no `m≈n` byte-identity re-scope; revised route is tolerance-equivalent dense QR.

---

## 9. References

### Source code

- `src/simsopt/geo/optimizer_jax.py` — current LM (`_lm_iteration`, `levenberg_marquardt`, `levenberg_marquardt_traceable`, `_gmres_solve_least_squares_system`)
- `src/simsopt/geo/optimizer_jax_reference.py` — host-driven LM dispatch
- `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py` — reference port pattern (NamedTuple carry + single `lax.while_loop` with phase dispatch)
- `src/simsopt/geo/boozersurface_jax.py` — `BoozerSurfaceJAX` consumer of the LM
- `benchmarks/validation_ladder_contract.py` — `PARITY_LADDER_TOLERANCES` SSOT
- `docs/source/jax_acceptance.rst` — acceptance gates (compile time, parity lanes)
- `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` — parent plan (W4.3 conditional)

### MINPACK Fortran reference (netlib)

- `https://www.netlib.org/minpack/lmder.f` — outer driver
- `https://www.netlib.org/minpack/lmpar.f` — damping search
- `https://www.netlib.org/minpack/qrfac.f` — pivoted Householder QR
- `https://www.netlib.org/minpack/qrsolv.f` — Givens damped solve
- `https://www.netlib.org/minpack/enorm.f` — 3-bucket Euclidean norm
- `https://www.netlib.org/minpack/dpmpar.f` — machine constants
- `https://github.com/fortran-lang/minpack` — modernized MINPACK + test suite

### Algorithmic references

- Moré, J.J. (1978), "The Levenberg-Marquardt algorithm: Implementation and theory," Lecture Notes in Mathematics 630, ed. G.A. Watson, pp. 105-116. Springer. DOI: 10.1007/BFb0067700
- Moré, Garbow, Hillstrom (1980), MINPACK User Guide, ANL-80-74
- Moré, Garbow, Hillstrom (1981), "Testing Unconstrained Optimization Software," ACM Trans. Math. Softw. 7(1):17-41 — the MGH test suite
- Madsen, Nielsen, Tingleff (2004), "Methods for Non-linear Least Squares Problems," 2nd ed., IMM-TR-2004
- Paige, Saunders (1982), LSQR algorithm — predecessor to LSMR
- Fong, Saunders (2011), LSMR algorithm — Lineax's iterative solver

### JAX / SciPy / library docs

- `https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.qr.html` — JAX pivoted QR API
- `https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.qr.html` — JAX low-level QR
- `https://docs.jax.dev/en/latest/external-callbacks.html` — `io_callback` for trace dumps
- `https://docs.jax.dev/en/latest/ffi.html` — JAX FFI (relevant for cuSOLVER FFI option if pursued)
- `https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html` — SciPy `method='lm'` wraps MINPACK
- `https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html` — SciPy QR via LAPACK `dgeqp3`
- `https://github.com/jax-ml/jax/pull/25955` — JAX PR adding pivoted QR on GPU via MAGMA
- `https://github.com/jax-ml/jax/issues/12897` — long-standing JAX pivoted QR issue
- `https://github.com/jax-ml/jax/issues/5097` — unimplemented `scipy.optimize` functions (open since 2020)
- `https://docs.kidger.site/optimistix/` — Optimistix docs (Track 3)
- `https://docs.kidger.site/lineax/api/solvers/` — Lineax solvers including LSMR
- `https://github.com/google/jaxopt/blob/main/jaxopt/_src/levenberg_marquardt.py` — JAXopt LM (Madsen-Nielsen, not MINPACK)
- `https://github.com/patrick-kidger/optimistix/blob/main/optimistix/_solver/levenberg_marquardt.py` — Optimistix LM
- `https://github.com/hahnec/torchimize` — PyTorch LM (no MINPACK claim)
- `https://github.com/fabiodimarco/torch-levenberg-marquardt` — PyTorch LM (no MINPACK claim)
- `https://github.com/pkgw/pwkit/blob/master/pwkit/lmmin.py` — pure-NumPy MINPACK lineage (explicitly non-byte-equivalent)

---

## 10. Research provenance

This plan consolidates findings from a 5-agent max-effort opus 4.7 research pass deployed in this session:

| Agent | Focus | Key deliverable |
|---|---|---|
| A | MINPACK lmder algorithmic spec | ~3,200 word specification with state variables, subroutine contracts (qrfac, qrsolv, lmpar, enorm, dpmpar), outer iteration logic, three-criterion math, 15 load-bearing byte-equality choices |
| B | JAX-traceable pivoted QR options | Discovery that `jax.scipy.linalg.qr(pivoting=True)` exists since JAX PR #25955 and calls the same `lapack_dgeqp3_ffi` symbol SciPy uses; empirically verified byte-equality on 40×40 fixture |
| C | Trust-region damping + 3-criterion termination in `lax.while_loop` | Concrete JAX pseudocode for `LmderState`, `lmpar`, `qrsolv` (`lax.fori_loop` × `lax.fori_loop`), driver (`lax.while_loop` + `lax.switch` phase dispatch), `enorm` (3-bucket); compile-time estimate 30–60s |
| D | GPU performance tradeoff analysis | Tradeoff matrix for 6 strategies; key finding that GPU byte-equality is impossible (cuSOLVER non-deterministic) and that full byte-exact port estimated 90–180s compile time (extrapolated, before Agent B's QR discovery) |
| E | Validation methodology + prior art survey | L1-L4 precision ladder; MGH-1981 fixture suite; SciPy interception strategies; survey of JAXopt/Optimistix/torchimize/torch-LM/jaxls/JAXFit/jax-core/arXiv finding zero existing MINPACK-byte-exact JAX port |

Plus an independent critic validation pass (logged in session) that flagged:
- "CPU byte-equal by construction" overclaim — QR parity necessary not sufficient
- Timing measurement reproducibility caveat (x64 must be enabled)
- LOC and compile estimates are extrapolations, not measurements
- "Track 2 first" with Track 1 as gated spike — folded into this plan structure

Plus a second independent critic validation pass (2026-05-16, logged in session) that flagged 4 blocking corrections, all reproduced empirically and folded into this revision:
- SciPy `least_squares(method='lm')` default is `ftol=xtol=gtol=1e-8`, not `1.49012e-8` / `0.0` (1.49012e-8 belongs to the legacy `leastsq` MINPACK-direct wrapper). Fixed in §3.3.
- Track 2 cannot achieve exact `info` parity because MINPACK's `info=4` requires pivoted-QR data (`gnorm = max_l |J[:,l]^T·(q^T fvec)/r_diag[l]|`) absent from the matrix-free GMRES inner solve. Reframed Track 2 acceptance gates in §3.4.
- `jax.scipy.linalg.qr(pivoting=True)` returns explicit Q+R+P, not MINPACK's packed `fjac` form. Empirically reproduced: `(384,40)` shape has `Q max diff = 4.163e-17` versus LAPACK while `R/P` remain bit-equal. JAX exposes `qr`, `householder_product`, `ormqr` but no `geqrf`-style API. Added **Phase 0 feasibility gate G0** in §4.2.
- "L1-only-byte-equal" mixes orthogonal axes (L1 = tolerance parity per §6.1; L4 = bit-equality). Fixed in Phase 5 re-scope wording.

Plus a third independent Crucible review pass (2026-05-16, 4 parallel discovery lenses) that flagged 1 critical + 9 major findings against rev 2; rev 3 incorporates all required fixes:
- **CRITICAL**: info=8 was misclassified as matrix-free-computable. `lmder.f:431` is `if (gnorm .le. epsmch) info = 8` — info=8 depends on `gnorm`, same as info=4. Conversely info=3 is `info=1 AND info=2` (`lmder.f:421-422`) and IS matrix-free. Corrected subset to **{1, 2, 3, 5, 6, 7}** matrix-free, **{4, 8}** pivoted-QR-required, in §3.3, §3.4, §12.
- **MAJOR**: `lmder_serial.f` doesn't exist on netlib; corrected to `lmder.f` (same line numbers correct) in §3.2 and §3.3.
- **MAJOR**: `lmpar.f:229-230` was wrong; the secondary-exit predicate is on `lmpar.f:220-222` (fused into the primary-exit compound `if`). Fixed in §4.2 Phase 4.
- **MAJOR**: Front-matter "Track 1 spike through G5: 1–3 days cumulative" contradicted the §4.2 phase-duration sum (~8.5 days). Replaced with a per-phase breakdown.
- **MAJOR**: §5.1 listed `optimistix` as a required dep, pre-empting §8 Q4 owner decision. Reframed as conditional on Q4 default-optional-until-decided.
- **MAJOR**: `dgeqpf` was cited as a G0-fail alternate target; `dgeqpf` is the deprecated Level-2 BLAS predecessor of `dgeqp3`. SciPy and JAX both call `dgeqp3`, so `dgeqpf` would guarantee losing byte-equality. Corrected to `dgeqp3` and incorporated as Path C (XLA custom call) in Phase 0.
- **MAJOR**: "Re-scope to m≈n only" was offered as a G0-fail default; m≈n shapes have no production payoff because the BoozerSurface fixture is 384×40 (m≫n). Removed the default; re-scope now requires owner sign-off via new §8 Q8.
- **MAJOR**: Phase 0 omitted Path C (direct LAPACK `dgeqp3` FFI). Added as the strongest-bit-equality candidate path with build-system tradeoff documented.
- **MAJOR**: L3 success criterion on (384,40) was hard-asserted in §0 but Phase 5 G6 allows L3 failure. Reconciled: L3 on (384,40) is now explicitly aspirational, L3 on MGH-5 m≈n subset is the targeted contract.
- **MAJOR**: G0 acceptance was disjunctive (Path A OR B passes). Tightened to conjunctive (≥ 2 paths must agree) for downstream-risk reduction.

---

## 11. Glossary

| Term | Definition |
|---|---|
| MINPACK | Fortran library from 1980 (Argonne, Moré/Garbow/Hillstrom) implementing the canonical nonlinear least-squares and root-finding routines. `lmder` is the LM-with-analytic-Jacobian variant. |
| LAPACK | Linear Algebra PACKage — Fortran library (1992, Anderson et al.) for dense numerical linear algebra. `dgeqp3` is the double-precision column-pivoted QR routine. |
| `qrfac` | MINPACK's pivoted Householder QR. Used inside `lmder` for the Jacobian factorization at each outer iteration. |
| `qrsolv` | MINPACK's Givens-rotation damped solve. Given the QR factorization plus `λ` and `D`, computes the LM step without ever forming `J^T J`. |
| `lmpar` | MINPACK's univariate Newton iteration on the secular equation `φ(par) = ‖D·s(par)‖ - Δ`. Finds the damping `par` such that the step length matches the trust-region radius. |
| `enorm` | MINPACK's overflow/underflow-safe Euclidean norm. Three-bucket (small/intermediate/large) scaling to avoid loss of precision at extreme magnitudes. |
| `dpmpar` | MINPACK's machine constants (`epsmch`, smallest normal, largest finite). Uses literal decimal values, not the IEEE-exact `np.finfo` values. |
| LSMR | Iterative least-squares solver by Fong/Saunders (2011). Operates on `J` directly via `J·v` / `J^T·v` matrix-vector products, never forms `J^T J`. Available in Lineax. |
| Optimistix | JAX nonlinear optimization library (Kidger, ~2023) — root finding, minimization, least squares, fixed points. Equinox-based, pluggable inner solvers. |
| Lineax | JAX linear algebra library (Kidger, ~2023) — linear solvers including LSMR, QR, LU, Cholesky, CG, GMRES. Pluggable solver protocol. |
| Equinox | JAX neural network / pytree framework (Kidger). Foundation that Optimistix and Lineax build on. |
| `lax.while_loop` | JAX primitive for traceable while-loops with carry-state. Body and condition functions must be pure and trace-compatible. |
| `lax.fori_loop` | JAX primitive for fixed-bound for-loops with carry-state. Cheaper to compile than `while_loop` when iteration count is statically known. |
| `lax.switch` | JAX primitive for branch dispatch on an integer index. Used for the `phase_jacobian` / `phase_inner` body dispatch. |
| L1/L2/L3/L4 | Precision-ladder levels for parity validation. L1 = converged state, L2 = path + info, L3 = per-iter trace, L4 = bitwise equality. |
| MGH-1981 | Moré-Garbow-Hillstrom 1981 test suite of 18 canonical unconstrained optimization problems, standard MINPACK validation battery. |
| Pivoted QR | Householder QR with column pivoting at each step (column with largest remaining 2-norm pivoted into next position). Rank-revealing, backward stable. |
| Givens rotation | 2×2 orthogonal rotation used to selectively zero one entry of a matrix. The MINPACK `qrsolv` and `lmpar` use Givens chains to update QR factors without re-factorization. |
| BLAS / cuBLAS / cuSOLVER | Basic Linear Algebra Subprograms (Fortran reference) / NVIDIA's GPU implementation / NVIDIA's LAPACK-style GPU library. |
| MAGMA | Hybrid CPU+GPU dense linear algebra library (UTK ICL). What jaxlib's `cu_hybrid_geqp3` uses for pivoted QR on GPU. |
| Parity ladder | `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`. SSOT for per-lane rtol/atol contracts in the simsopt-jax parity testing framework. |

---

## 12. Checkbox roll-up (master TODO list)

### Pre-work

- [x] Owner decisions Q1-Q3, Q5, and Q7-Q8 resolved by rev 5.
- [ ] Q4 remains open only if Track 3 is reopened; default is optional-extra only unless Python/JAX floors change.
- [ ] Q6 remains open pending G5 compile-time measurement.
- [ ] Allocate engineering time for G3+G4+G5 if revised Track 1 should be promoted to production-ready.

### Track 2 — termination + damping retrofit

- [x] Phase 2.1: extend `levenberg_marquardt` and `_traceable` signatures with `ftol`/`xtol`/`gtol`
- [x] Phase 2.2: add `info` int32 to `_lm_iteration` state while preserving `success`
- [x] Phase 2.3: implement `_matrix_free_lm_info` restricted to matrix-free-computable subset {1, 2, 3, 5, 6, 7} (info=4 and info=8 reserved for the dense-QR lane because both depend on `gnorm` per `lmder.f:431` and `:313`)
- [x] Phase 2.4: replace asymmetric damping factors with symmetric Marquardt-style damping update (`× 2` / `× 1/2`); full MINPACK `lmpar` bracket search is not in Track 2
- [x] Phase 2.5: update while-loop predicate so termination fires on `info != 0` (over the matrix-free subset) OR the legacy `success` boolean — whichever first; single-`tol` callers keep API compatibility but may see the intentional ftol/xtol termination behavior
- [x] Phase 2.6: update result schema with `info` field (range `{0, 1, 2, 3, 5, 6, 7}`), preserve `success = legacy_success | info in {1, 2, 3}`
- [x] Phase 2.7: `tests/geo/test_lm_termination_parity.py` — matrix-free info subset and gtol routing tests
- [x] Phase 2.8: `tests/geo/test_lm_damping_parity.py` — damping and iteration-count tests
- [x] Phase 2.9: update `optimizer_jax.py:14-45` module docstring
- [x] Phase 2.10: update `docs/source/jax_acceptance.rst:156-187`
- [x] Phase 2.11a: `tests/geo/test_boozersurface_jax.py -m "not private_optimizer_runtime"` regression run (385 passed, 4 skipped)
- [ ] Phase 2.11b: downstream `tests/integration/test_single_stage_jax.py` full regression run remains unclaimed by rev 5
- [x] Phase 2.12: `ls_state_parity` lane confirmation at `sdofs_inf ≤ 1e-11`
- [x] Phase 2.13: ruff check / format clean on the landed slice

### Track 1 — CPU tolerance-equivalent MINPACK-style QR LM lane

- [x] Accept revised G0 tolerance gate (`~1e-15 < 1e-10`)
- [x] Add `"lm-minpack"` to `VALID_LEAST_SQUARES_ALGORITHMS`
- [x] Route target backend to `method="lm-minpack-ondevice"`
- [x] Implement `levenberg_marquardt_minpack_traceable`
- [x] Route BoozerSurfaceJAX LS paths through the new method
- [x] Add focused direct SciPy/QR parity tests
- [x] Update optimizer and acceptance docs
- [ ] Run broader Moré-Garbow-Hillstrom final-state parity suite
- [ ] Run oversampled BoozerSurface fixture
- [ ] Measure first-trace compile time on `(384,40)`

### Track 3 — Optimistix + Lineax LSMR (deferred)

- [ ] Phase 3.1: add Optimistix/Lineax/Equinox dependency policy as an optional extra or raise project Python/JAX floors first
- [ ] Phase 3.2: implement `jax_least_squares_optimistix` wrapper
- [ ] Phase 3.3: implement `_optimistix_solution_to_scipy_optimize_result` adapter
- [ ] Phase 3.4: add `"optimistix-lm"` to `VALID_LEAST_SQUARES_ALGORITHMS` + routing
- [ ] Phase 3.5: `BoozerSurfaceJAX` option validation
- [ ] Phase 3.6: `tests/geo/test_lm_optimistix_parity.py`
- [ ] Phase 3.7: `tests/geo/test_lm_optimistix_robustness.py` — `sdofs_inf` reduction vs current LM on default fixture
- [ ] Phase 3.8: GPU lane test
- [ ] Phase 3.9: update LM family note + jax_acceptance.rst + CLAUDE.md

### Post-implementation

- [ ] Run full validation suite per `CLAUDE.md` "Validation" section
- [ ] Tag commits per existing convention (`feat(boozersurface): …`)
- [x] Update `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` status table for B3 row
- [ ] (Optional) Q5 publication write-up if Track 1 lands and novelty claim survives literature review

---

**End of plan.**
