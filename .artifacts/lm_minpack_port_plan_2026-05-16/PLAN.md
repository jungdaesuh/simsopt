# JAX MINPACK `lmder` Parity — Implementation Plan

| Field | Value |
|---|---|
| Created | 2026-05-16 |
| Branch | `gpu-purity-stage2-20260405` |
| Parent plan | `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` (Wave 4 W4.3 conditional rollout) |
| Driver | 5-agent max-effort research (Opus 4.7) + independent critic validation pass |
| Author | Jung Dae Suh + Claude Opus 4.7 |
| Status | EXECUTED PARTIAL (2026-05-16) — Track 2 implemented; Track 1 abandoned at Phase 0 G0 for production scope after packed-QR/`qtb` bit-equality failure on `(384, 40)`; Track 3 remains deferred |
| Estimated effort | Track 2: 2 days · Track 1 Phase 0 alone (cheapest abandon checkpoint): 1–3 days depending on implementer's `jax.lax.linalg` FFI familiarity · Track 1 spike through G5 cumulative: ~8.5 days (Phase 0: 1–3d, Phase 1: 1d, Phase 2: 1d, Phase 3: 2d, Phase 4: 1.5d, Phase 5: 2d) · full Track 1 through G8: ~10–12 days · Track 3 (Optimistix): 1–2 weeks. **All estimates are extrapolations from `_lbfgsb_scipy.py` (~3300 LOC), not measurements.** |

---

## Execution Status — 2026-05-16

Track 2 landed as the low-risk convergence-semantics retrofit: the matrix-free
JAX LM now carries the computable MINPACK-style `info` subset, exposes
`ftol`/`xtol`/explicit `gtol`, and uses symmetric damping factors in the
existing matrix-free lane.

Track 1 was executed through its mandatory Phase 0 gate and abandoned at
production scope. The G0 probe in
`PHASE0_G0_REPORT.md` compares JAX internal packed `geqp3` + `ormqr` against
SciPy LAPACK `dgeqp3` + `qr_multiply`; all 100 production-shape `(384, 40)`
seeds fail bit equality for packed factor and `Q^T f`. Per the G0 policy below,
no `lm-minpack` implementation should proceed without explicit owner sign-off
to re-scope the project.

## TL;DR

Two-track strategy:

- **Track 2 first** (low risk, high signal): retrofit the existing matrix-free JAX LM in `src/simsopt/geo/optimizer_jax.py` to use MINPACK-style three-criterion termination (`ftol` / `xtol` / `gtol` OR'd) and symmetric Marquardt damping. Closes the documented W4.3 algorithmic-divergence gap on convergence behavior and termination semantics without touching the inner solve. ~100 LOC. No compile-budget impact.

- **Track 1 as gated spike** (research-grade, novel): build a CPU-byte-exact JAX port of MINPACK `lmder` keyed off the empirically-verified `jax.scipy.linalg.qr(pivoting=True)` FFI to LAPACK `dgeqp3`. New module `_lmder.py`, new opt-in path `least_squares_algorithm="lm-minpack"`. Each subroutine validated against a SciPy/MINPACK oracle before integration. Commit-or-discard at the end of the spike. Estimated 600–800 LOC and 30–60s first-trace; both are **extrapolated estimates from the existing `_lbfgsb_scipy.py` port**, not measurements.

- **Track 3 deferred** (architectural shift, library swap): adopt Optimistix `LevenbergMarquardt` with Lineax `LSMR` inner solver as a parallel third lane. Adds ~400 LOC of adapter + 2 deps. Better numerical conditioning on near-rank-deficient fixtures (`κ(J)` not `κ(J)²`). Not byte-equal to MINPACK; tolerance-equivalent. Net LOC is **additive** while the existing matrix-free LM remains the default; the ~500-LOC simplification only materializes if/when a future cleanup retires the current `_lm_iteration`/`_gmres_solve_least_squares_system` path.

---

## 0. Goals, non-goals, and what success looks like

### Goals

1. **Eliminate the W4.3 algorithmic divergence** between simsopt's CPU `BoozerSurface.minimize_boozer_penalty_constraints_ls(method="lm")` (→ MINPACK `lmder` via SciPy) and the JAX LM lane. Today the JAX lane is *algorithmically distinct* per its own module docstring (`optimizer_jax.py:14-45`); this plan aims to either close the gap (Track 1) or document the tolerance contract precisely after closing the convergence-criteria sub-gap (Track 2).
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
| Track 2 | Three-criterion termination + symmetric damping land; **convergence semantics improved** (the JAX LM stops on ftol/xtol/gtol disjunction rather than single grad-norm); iteration count on `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` is within 1.5× of CPU MINPACK iteration count; `ls_state_parity` lane continues to pass at `sdofs_inf ≤ 1e-11`; no regression in `tests/geo/test_boozersurface_jax.py` or `tests/integration/test_single_stage_jax.py`. **Exact `info` integer parity over the full 1-8 range is explicitly NOT a Track 2 gate** — MINPACK's `info=4` (`if (gnorm .le. gtol) info = 4`, `lmder.f:313`) and `info=8` (`if (gnorm .le. epsmch) info = 8`, `lmder.f:431`) both depend on `gnorm = max_l |J[:,l]^T·(q^T fvec)/r_diag[l]|`, which is a pivoted-QR-only quantity unavailable in the matrix-free GMRES inner solve. Track 2's info-code parity is restricted to the matrix-free-computable subset {1, 2, 3, 5, 6, 7}; the pivoted-QR-required pair {4, 8} belongs to Track 1 G5. |
| Track 1 (per gate) | At each spike gate (Phase 0 feasibility, enorm, qrfac, qrsolv, lmpar, driver), the JAX kernel matches its SciPy/MINPACK oracle at the lane-appropriate tolerance. Gate failure → abandon spike. **Phase 0 (Gate G0) is the only existential gate**: G0 failure invalidates the byte-equal contract on production BoozerSurface shapes and ends the spike on day 1–3 with no project-charter re-scope by default (see §8 Q8). G1–G5 failures trigger per-gate retry-then-abandon, not project abandonment. |
| Track 1 (final) | `least_squares_algorithm="lm-minpack"` opt-in passes L1 (converged-state, `rtol=1e-6, atol=1e-10` per §6.1) and L2 (exact `niter`, `info`, `nfev` match per §6.1) parity against `scipy.optimize.least_squares(method="lm")` on at least 5 canonical Moré-Garbow-Hillstrom problems plus the oversampled BoozerSurface fixture, **CPU same-machine only**. L3 (per-iteration trace at `rtol=1e-12, atol=1e-14`) is **target-only on the MGH-5 m≈n subset** and is **aspirational on the (384,40) BoozerSurface fixture** because the §1.2 (384,40) Q-drift may propagate through `qtb` to per-iteration `(x, fnorm, delta, par, ratio)` divergence even after G0 passes via Path B or C. Per Phase 5 Gate G6, L3 failure on (384,40) drops the contract to L1+L2 without abandoning the spike. |
| Track 3 | `least_squares_algorithm="optimistix-lm"` opt-in matches converged state of the current LM at `branch-stable-resolve` lane on the oversampled fixture; better numerical robustness on the default (near-rank-deficient) fixture measured in `sdofs_inf` drift. |

---

## 1. Context

### 1.1 Current state

- The simsopt JAX LM lives in `src/simsopt/geo/optimizer_jax.py:1340-1660`. Two callable methods: `levenberg_marquardt` (`:1522`, host-driven) and `levenberg_marquardt_traceable` (`:1636`, `lax.while_loop` traceable). Both share `_lm_iteration` (`:1416`) and `_lm_defaults` (`:1403`).
- Inner solve: matrix-free GMRES against `J^T J + λI` via `_gmres_solve_least_squares_system` (`:1340`).
- Termination: single criterion `‖∇‖_∞ ≤ tol` set at `_lm_iteration` (`:1510-1511`).
- Damping update: asymmetric trust-region with factors `expand=4.0`, `shrink=0.5`, `mild_shrink=0.8` (`:1403-1410`).
- W4.3 module docstring (`:14-45`) explicitly states this LM is **algorithmically distinct** from MINPACK `lmder` along three axes (inner solve, termination, damping update).
- Plan precedent: `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` lines 292-302 documents the MINPACK port as a conditional W4.3 follow-up, with trigger criteria "production default AND byte-equality requirement" both unmet at the time of writing.

### 1.2 Why this plan now

A five-agent research pass (logged in this session, summarized below in §10) surfaced one key empirical finding: **the hardest piece of a MINPACK port — column-pivoted Householder QR with byte-equality to LAPACK `dgeqp3` — is already a one-line JAX call**. `jax.scipy.linalg.qr(pivoting=True)` exists in JAX 0.10.0 (added by [JAX PR #25955](https://github.com/jax-ml/jax/pull/25955), Feb 2025), dispatches to the same `lapack_<t>geqp3_ffi` symbol SciPy uses on CPU and to `cu_hybrid_geqp3` (MAGMA) on GPU.

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

The remaining subroutines (`qrsolv` Givens elimination, `lmpar` univariate Newton, `enorm` 3-bucket scaling) all reduce to standard `lax.scan` / `lax.while_loop` / `lax.fori_loop` patterns already used in the existing `_lbfgsb_scipy.py` port — but feasibility of the **end-to-end byte-equal driver on the production fixture** depends on a Phase-0 feasibility gate (§4.2) that proves the MINPACK-packed `(fjac, rdiag, acnorm, ipvt)` tuple plus `qtb` can be made bit-equal to a SciPy/MINPACK oracle on `(384,40)`. **If Phase 0 fails (no two paths agree per §4.2 G0 / §8 Q8), Track 1 is abandoned at the production scope, full stop.** Any re-scope to `m ≈ n` shapes only is a project-charter renegotiation that requires owner sign-off per §8 Q8; it is not a default fallback and the plan author may not unilaterally re-scope.

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

### Why MINPACK byte-equality matters

1. **Reproducibility of published results.** Stellarator coil designs that were converged using `scipy.optimize.least_squares(method="lm")` are reproducible bit-for-bit on a JAX runtime, which is a stronger contract than tolerance-equivalence. Useful for paper supplementary materials and regression testing.
2. **Cross-validation oracle.** Today, simsopt's parity testing for the LS lane uses SciPy MINPACK on the CPU side and a different algorithm on the JAX side; differences could be due to either implementation or the algorithm itself. A byte-equal JAX MINPACK port collapses the implementation-vs-algorithm ambiguity for CPU comparisons.
3. **Numerical conditioning.** Pivoted QR is backward-stable to `ε_mach·κ(J)`; matrix-free GMRES on `J^T J + λI` sees `ε_mach·κ(J)²`. For the documented near-rank-deficient default BoozerSurface fixture (`κ(J) ≈ 10⁷`), this is a 7-order-of-magnitude conditioning improvement.

### Why not just adopt Optimistix instead

Optimistix is a viable medium-term path (Track 3 of this plan). The reasons to also pursue Track 1:
- Optimistix is tolerance-equivalent to MINPACK, not byte-equal. Lose the reproducibility-oracle property.
- Optimistix is a single-maintainer (Patrick Kidger) dependency. Track 1 has no new runtime deps beyond JAX 0.10.0.
- Optimistix doesn't preserve the existing simsopt parity-ladder lane structure as cleanly; lane boundaries are designed around the *exact* algorithm contract.
- The two tracks are not mutually exclusive: Track 1 gives a CPU validation oracle; Track 3 gives the GPU production lane. Together they replace the current single-algorithm story with a CPU-byte-exact / GPU-fast separation.

### Why Track 2 must come first

Track 2 closes the documented W4.3 algorithmic-divergence gap on the **convergence semantics** (termination + damping) without touching the inner solve. It's small enough to land as a single PR with full validation, and the result is independently useful even if Track 1 is later abandoned. It also reduces the contract debt that any Track 1 / Track 3 work would inherit.

---

## 3. Track 2 — MINPACK termination + damping retrofit (DO FIRST)

**Scope:** modify `_lm_iteration` and `_lm_defaults` in `src/simsopt/geo/optimizer_jax.py` to use three-criterion termination and symmetric Marquardt damping. Keep matrix-free GMRES inner solve unchanged. ~100 LOC + tests.

**Estimated effort:** 2 working days including validation.

### 3.1 Three-criterion termination

MINPACK terminates on the disjunction of three independent criteria, each producing a distinct `info` code (1, 2, 3, 4) plus three "too small" variants (6, 7, 8) and one budget exhaustion (5). Reference: `optimizer_jax.py:14-45` module docstring and Agent A's spec §5.

### 3.2 Symmetric Marquardt damping

Current scheme: `expand=4.0, shrink=0.5, mild_shrink=0.8` (asymmetric — `4.0 ≠ 1/0.5`). MINPACK uses Marquardt's symmetric `× 2 / × 1/2` scheme with bracket-based escalation. The retrofit replaces the asymmetric factors and adds the `par` (Marquardt parameter) escalation logic from `lmder.f:381-396` (note: there is no `lmder_serial.f` on netlib — the par-escalation block lives in the standard `lmder.f`).

### 3.3 Todos (Track 2)

- [ ] Add `ftol`, `xtol`, `gtol` parameters to `levenberg_marquardt` (`optimizer_jax.py:1522`) and `levenberg_marquardt_traceable` (`:1636`). Default: `ftol=xtol=gtol=1e-8` matching the user-facing default of `scipy.optimize.least_squares(method='lm')` in scipy ≥1.5 (the legacy `scipy.optimize.leastsq` MINPACK-direct wrapper uses `1.49012e-8`, but `least_squares` overrides this — we follow `least_squares` because that's the API simsopt's CPU path calls).
- [ ] Add `info` code field to `_lm_iteration` carry state (`:1416+`). Track an internal int32 alongside the existing `success` boolean (keep `success = info in {1, 2, 3}` for backward compat — info=4 is Track 1 only). **Codes 1, 2, 3, 5, 6, 7 are computable from ftol/xtol/maxfev bookkeeping alone (info=3 is the conjunction `info=1 AND info=2` per `lmder.f:421-422`); codes 4 and 8 both require `gnorm` (a pivoted-QR-only quantity, per `lmder.f:431` for info=8 and `:313` for info=4) and are reported as `info=0` (not converged on that criterion) in the matrix-free Track 2 lane** — see §3.4 for the gate consequence.
- [ ] Implement `compute_info(...)` per Agent C's pseudocode §5 restricted to the **matrix-free-computable** subset (codes 1, 2, 3, 5, 6, 7). Place in a new private helper in `optimizer_jax.py`. The full 8-code cascade (adding {4, 8}) lands in Track 1 G5.
- [ ] Replace the `_lm_defaults` damping factors (`:1403-1410`) with MINPACK-style symmetric Marquardt scheme (`×2` / `×0.5`, no asymmetric `expand=4.0` / `shrink=0.5` mismatch) plus the `par` escalation logic from `lmder.f:381-396`. Keep matrix-free GMRES inner solve.
- [ ] Replace the asymmetric trust-region update around `_lm_iteration` accept/reject branch (the lax.cond block at `:1467-1490` that selects `expand_factor`, `shrink_factor`, `mild_shrink_factor`) with a symmetric Marquardt update.
- [ ] Update `levenberg_marquardt` / `levenberg_marquardt_traceable` while-loop predicate to terminate when `info != 0` (matrix-free subset) OR `success` fires on the legacy `‖∇‖_∞ ≤ tol` criterion, whichever comes first. Backward-compat: existing callers passing only `tol` get identical termination behavior.
- [ ] Update result schema in `_lm_iteration` (`:1492-1512`) to surface `info` alongside `success`. Document in the result dict docstring that `info ∈ {0, 1, 2, 3, 5, 6, 7}` in Track 2 — `info=4, 8` only set by the Track 1 `lm-minpack` lane.
- [ ] Add `tests/geo/test_lm_termination_parity.py`: drive new termination on a fixture; assert that for each matrix-free-computable `info` value (1, 2, 3, 5, 6, 7), the JAX LM stops with the same code as SciPy `least_squares(method='lm')` **when SciPy's run also exits on that same code**. Allow JAX to exit on `info=2` (xtol) while SciPy exits on `info=4` (gtol) — that's a known and expected algorithm divergence, not a bug.
- [ ] Add `tests/geo/test_lm_damping_parity.py`: assert iteration count is within 1.5× of CPU MINPACK on the oversampled BoozerSurface fixture (lane: `branch-stable-resolve`).
- [ ] Update `optimizer_jax.py:14-45` module docstring "LM family note" to reflect the new convergence semantics. State explicitly: termination is now MINPACK-style three-criterion (ftol/xtol/gtol) where computable, but the gtol criterion (info=4) requires pivoted-QR data and is therefore deferred to the `lm-minpack` lane if it lands.
- [ ] Update `docs/source/jax_acceptance.rst:156-187` "Optimizer family equivalence" section with the new contract.
- [ ] Run `tests/geo/test_boozersurface_jax.py -m "not private_optimizer_runtime"` and `tests/integration/test_single_stage_jax.py` to confirm no regression in existing parity tests.
- [ ] Validate `ls_state_parity` lane still passes at `sdofs_inf ≤ 1e-11` on the oversampled fixture.

### 3.4 Acceptance gates (Track 2)

| Gate | Lane | Threshold |
|---|---|---|
| Convergence semantics improved | qualitative + new tests | LM stops on ftol/xtol disjunction where applicable, not only on `‖∇‖_∞` |
| `info` code parity on matrix-free-computable subset {1, 2, 3, 5, 6, 7} | `branch-stable-resolve` | exact int match **only when SciPy also exits on a matrix-free-computable code** |
| Iteration count parity | `branch-stable-resolve` | within 1.5× of CPU MINPACK |
| Converged-state parity (oversampled fixture) | `ls_state_parity` | `sdofs_inf ≤ 1e-11` (existing) |
| No regression in existing boozersurface_jax tests | (existing lanes) | all green |
| `ruff check`, `ruff format` clean on changed files | n/a | pass |

**Explicitly NOT a Track 2 gate:** exact `info` code parity over the full 1-8 range. Both MINPACK's `info=4` (`lmder.f:313`, `if (gnorm .le. gtol) info = 4`) and `info=8` (`lmder.f:431`, `if (gnorm .le. epsmch) info = 8`) depend on the scaled-gradient norm `gnorm` (computed at `lmder.f:297-306` using pivoted-QR factors), which Track 2's matrix-free GMRES inner solve cannot materialize. Track 2 reports `info=0` in those branches; Track 1's `lm-minpack` lane is the only path that can match `info ∈ {4, 8}`. `info=3` is the conjunction `info=1 AND info=2` (`lmder.f:421-422`) and IS matrix-free-computable, so it stays in Track 2's subset.

---

## 4. Track 1 — CPU byte-exact MINPACK port (GATED SPIKE)

**Scope:** new module `src/simsopt/geo/optimizer_jax_private/_lmder.py` implementing `levenberg_marquardt_minpack_traceable`. New opt-in path `least_squares_algorithm="lm-minpack"` on the `ondevice` backend. Each subroutine validated against a SciPy/MINPACK oracle before integration. Commit-or-discard at the end of the spike per gate outcomes.

**Estimated effort:** ~10–12 working days for full implementation through G8 (Phase 0 + Phases 1–6 + Phase 7 docs). Spike-only validation through G0 alone (cheapest abandon checkpoint): 1–3 days. Spike-only validation through G0+G1: 2–4 days. Spike-only validation through G0+G1+G2: 3–5 days. **All effort estimates are extrapolations from `_lbfgsb_scipy.py`'s ~3300 LOC, not measurements of unbuilt code; the 1-day lower bound for Phase 0 assumes the implementer is fluent in `jax.lax.linalg` low-level FFI primitives (`householder_product`, `ormqr`), else 3 days is realistic.**

### 4.1 Why this is a spike, not a commitment

Per critic validation: QR byte-equality is necessary but not sufficient. The full driver byte-equality depends on 15 load-bearing choices (Agent A spec §8): pivot tie-breaks (`>` not `≥`), Householder sign convention, the `0.5/√(0.25+0.25x²)` Givens form, `prered = temp1² + temp2²/0.5` model evaluation, `enorm` 3-bucket scaling, `dpmpar` literal decimals (not `np.finfo`), and others. Each must be reproduced exactly. The spike validates each as a separately-passing gate before committing to the full driver.

Abandon criteria for the spike:
- Any gate fails after 2 attempts to align with MINPACK behavior
- Compile time at the integrated driver level exceeds 90s on the canonical fixture
- A critical-path subroutine (`enorm`, `qrsolv`, `lmpar`) cannot be byte-aligned with the MINPACK Fortran reference

If abandoned, document the gate that failed and the empirical evidence; keep Track 2 work.

### 4.2 Implementation order (each item is a separately-gated commit)

**Phase 0: Packed-`fjac` + `qtb` feasibility gate (1–3 days, depending on `jax.lax.linalg` FFI fluency) — MUST PASS BEFORE PHASES 1-5**

This phase exists because the §1.2 empirical probe showed `Q` is bit-equal to LAPACK only on `m ≈ n` shapes (40×40, 75×39, 100×50). On the production-sized `(384,40)` BoozerSurface shape, `Q max diff = 4.163e-17` even though `R` and `P` remain bit-equal. MINPACK's downstream subroutines (`qrsolv`, `lmpar`) consume the **packed Householder form** of `A` and `qtb = Q^T·fvec`, not the explicit `Q`. Phase 0 proves whether we can produce those quantities byte-equal to LAPACK on the production shape, without which the byte-equality story collapses at the first outer LM iteration on a real fixture.

- [ ] Reconstruct (or directly capture) MINPACK-packed `fjac` and `qtb = Q^T·fvec` from the underlying LAPACK `dgeqp3` call. **Three candidate paths**, evaluated in increasing order of bit-equality strength:
  - **A.** Extract Householder vectors directly from explicit `Q` (returned by `jax.scipy.linalg.qr(A, pivoting=True, mode='economic')`) via the standard reconstruction (Schreiber-Van Loan or block-Householder backward), then write them into the strict lower triangle of a packed array matching MINPACK's storage layout. Inherits any drift in JAX's post-processed `Q`; expected to fail on `(384,40)` since the §1.2 probe shows that `Q` itself is not bit-equal there.
  - **B.** Use `jax.lax.linalg.householder_product` / `ormqr` to compute `Q^T·fvec` directly without materializing `Q`, sidestepping the explicit-`Q` reconstruction. Avoids the §1.2 `Q` drift but still depends on JAX's `householder_product`/`ormqr` kernels matching LAPACK's `ormqr`/`larf` bit-for-bit (not empirically verified anywhere).
  - **C.** Direct FFI to LAPACK `dgeqp3` via an XLA custom call (or JAX's private `lapack_dgeqp3_ffi` primitive if accessible from outside JAX), returning the native packed `(A_packed, tau, jpvt)` form. **Bit-equal to LAPACK by construction — it IS the LAPACK call.** This is the strongest path; the only reason not to default to it is build-system cost: Path C either (i) depends on a JAX private API that may break across JAX upgrades (fragile), or (ii) adds a small C-extension XLA custom call (~50 LOC, well-supported pattern, but adds a build dep). Spike evaluates Path C unless explicitly out-of-scope per owner sign-off — see §8 Q8.
- [ ] Build a `qtb_via_jax(A, fvec)` helper that returns `Q^T·fvec` using the chosen path and compare bit-by-bit against `scipy.linalg.qr_multiply(A, fvec, mode='left', pivoting=True)` (or equivalently `Q^T @ fvec` from SciPy's `(Q, R, P)`).
- [ ] Run the comparison on the four canonical shapes (40×40, 75×39, 100×50, 384×40) plus a synthetic `(2000, 80)` shape that brackets above the production fixture. **Plus** at least 100 random seeds at the production `(384,40)` shape (the §1.2 probe established the drift is deterministic-per-shape, so seed-sweeping is for failure-mode-coverage on poorly-conditioned data, not for the headline drift number).
- [ ] **Gate G0**: `qtb` AND packed `fjac` byte-equal to LAPACK/MINPACK reference on `(384,40)` at all 100 random seeds AND on the 4 canonical shapes. **Conjunctive across BOTH paths attempted (A+B, or A+C, or B+C) — at least two paths must agree, to reduce the risk that a single passing path fails at G3/G4 due to subtle downstream mismatches.**
  - **G0 pass (≥2 paths agree)**: proceed to Phase 1 with the strongest agreeing path frozen as the spike's QR-frontend convention.
  - **G0 fail (no two paths agree)**: **ABANDON Track 1 spike at this checkpoint, full stop.** Re-scoping to `m ≈ n` only or "MGH-test-suite-only" is **not** a default fallback — the production BoozerSurface fixture is exclusively `m ≫ n` (§6.2: `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` is 384×40, `m=384`), so a Track 1 that works only on `m ≈ n` has no production payoff and re-scoping is a project-charter renegotiation, not an implementation decision. The plan author may not unilaterally re-scope without owner sign-off (see new §8 Q8). Track 2 work is unaffected and proceeds independently.
- [ ] Test file: `tests/geo/test_lmder_phase0_feasibility.py`. Lane: `direct-kernel` byte-parity.

**Phase 1: `enorm` port (1 day)**

- [ ] Implement `enorm(x)` per Agent C pseudocode §8: 3-bucket (small/intermediate/large) classification with `rdwarf = 3.834e-20`, `rgiant = 1.304e+19` literal constants. First pass: vectorized using `jnp.where` masks and `jnp.sum`. Verify byte-equality requirement on test fixtures.
- [ ] If vectorized form fails strict byte-equality on stress fixtures (entries spanning >150 orders of magnitude), fall back to `lax.fori_loop` scalar accumulator preserving Fortran iteration order.
- [ ] Oracle: ground-truth from `scipy.optimize._minpack._enorm` (if exposed) OR from a one-time compile of the netlib `enorm.f` source via f2py in a test-time fixture. Compare on:
  - Random vectors with entries in `[-1, 1]` (intermediate bucket)
  - Vectors with one entry near `1e-20` (small bucket boundary)
  - Vectors with one entry near `1e+19` (large bucket boundary)
  - Vectors with all three bucket ranges represented
- [ ] **Gate**: `enorm(x) == enorm_ref(x)` bit-exact on at least 100 randomized vectors per bucket configuration. If fails: investigate accumulator order, retry once, then abandon spike.
- [ ] Test file: `tests/geo/test_lmder_enorm_oracle.py`. Lane: `direct-kernel` extended for byte-parity.

**Phase 2: `qrfac` wrapper (1 day)**

- [ ] Wrap `jax.scipy.linalg.qr(A, pivoting=True, mode='economic')` in a `qrfac_byte_parity(A)` adapter that returns the MINPACK-style packed `(fjac, rdiag, acnorm, ipvt)` tuple.
- [ ] Compute `acnorm` (initial column 2-norms) via the ported `enorm` from Phase 1.
- [ ] Verify pivot tie-break: the MINPACK rule is strict `>` (`qrfac.f:109`). JAX's `jsl.qr(pivoting=True)` calls LAPACK `dgeqp3` which also uses strict `>` — verify empirically on fixtures with equal column norms.
- [ ] Oracle: `scipy.linalg.qr(A, mode='economic', pivoting=True)` plus separate `enorm` calls for `acnorm`. Compare byte-exact on:
  - 40×40 random float64 (BoozerSurface-scale)
  - 384×40 (oversampled BoozerSurface)
  - 10×10 well-conditioned
  - Rank-deficient (Powell singular function pattern)
- [ ] **Gate**: `(fjac, rdiag, acnorm, ipvt)` byte-equal to SciPy + ported `enorm` on all four fixtures. Already empirically verified for 40×40; extend to other shapes.
- [ ] Test file: `tests/geo/test_lmder_qrfac_oracle.py`. Lane: `direct-kernel` byte-parity.

**Phase 3: `qrsolv` Givens elimination (2 days)**

- [ ] Implement `qrsolv(r, ipvt, diag, qtb)` per Agent C pseudocode §2.1. Two-level `lax.fori_loop` (outer over n columns, inner over n Givens-elimination rows). Mirror Fortran ordering and `0.5/√(0.25+0.25x²)` Givens form.
- [ ] Return `(x, sdiag, S_lower)` as explicit outputs — break the Fortran aliasing convention (the strict lower triangle of `R` holds `S^T`) because JAX immutable arrays don't support it cleanly. Agent A spec §3.2 notes this is the only ergonomic concession and does not affect byte-equality of the numerical outputs.
- [ ] Oracle: instrumented `scipy.optimize._minpack` call OR direct ctypes binding to `cminpack` / `fortran-lang/minpack` `qrsolv` function. Compare:
  - Random R + random pivot + random diag + random qtb (~50 fixtures)
  - Edge case: zero entries in `diag` (skip-branch in MINPACK line 105-115)
  - Edge case: rank-deficient R (zero diagonal entries)
- [ ] **Gate**: `x` and `sdiag` byte-equal to MINPACK reference on all fixtures. If fails on Givens order / sign / formulation, retry once with explicit sequential `lax.fori_loop` accumulator. Then abandon spike.
- [ ] Test file: `tests/geo/test_lmder_qrsolv_oracle.py`. Lane: `direct-kernel` byte-parity.

**Phase 4: `lmpar` univariate Newton (1.5 days)**

- [ ] Implement `lmpar(r, rdiag, ipvt, diag, qtb, delta, par_init)` per Agent C pseudocode §2.2. Outer `lax.while_loop` with `LmparState` carry; capped at 10 iterations per `lmpar.f:222`.
- [ ] Implement the zero-par Gauss-Newton accept branch (Agent C §2.2 step 1-2): if `||D·s_gn|| ≤ 1.1·delta`, return par=0 with the Gauss-Newton step.
- [ ] Implement the bracket setup (`parl`, `paru`) per `lmpar.f:151-187`. Forward triangular solve for `parl`; permuted `R^T·qtb` for `paru`.
- [ ] Implement the secondary-exit condition (`lmpar.f:220-222`, fused into the same compound `if` as the primary-exit `dabs(fp) <= p1*delta` test and the `iter == 10` cap): `parl == 0 AND fp <= prev_fp AND prev_fp < 0`. The `prev_fp` is captured at `lmpar.f:213` as `temp = fp`. Encode via state carrying `prev_fp` field.
- [ ] Implement the Hebden/Reinsch Newton update `par ← max(parl, par + parc)` with bracket maintenance.
- [ ] Oracle: same instrumented MINPACK / cminpack binding as Phase 3, exposing `lmpar` directly. Compare:
  - Well-conditioned R, par_init=0 (should accept Gauss-Newton immediately)
  - Ill-conditioned R requiring 3-5 Newton iterations
  - Rank-deficient R (parl == 0 path)
  - Pathological: par_init at the upper bound
- [ ] **Gate**: final `par`, `x`, `sdiag` byte-equal to MINPACK reference; iteration count exact match. If fails: investigate Newton update formula and secondary-exit predicate ordering. Abandon if 2 attempts insufficient.
- [ ] Test file: `tests/geo/test_lmder_lmpar_oracle.py`. Lane: `direct-kernel` byte-parity.

**Phase 5: Driver integration + outer iteration logic (2 days)**

- [ ] Implement `LmderState` NamedTuple per Agent C pseudocode §1.
- [ ] Implement `phase_jacobian` and `phase_inner` body functions per Agent C pseudocode §6. Use `lax.switch(state.phase, [phase_jacobian, phase_inner], state)` for dispatch — avoid nested while_loops.
- [ ] Implement scaling vector `D` update via `update_diag` per Agent C §3 (pure `jnp.where`).
- [ ] Implement trust-region radius + damping update via `update_delta_par` per Agent C §4 (pure `jnp.where`).
- [ ] Wire termination via `compute_info` per Agent C §5 — same logic as Track 2 but parameterized for the byte-exact lane.
- [ ] Public entry point `levenberg_marquardt_minpack_traceable` with signature matching `levenberg_marquardt_traceable` per Agent C pseudocode §9.
- [ ] Add `"lm-minpack"` to `VALID_LEAST_SQUARES_ALGORITHMS` (`optimizer_jax.py:155`). Wire routing in `resolve_target_least_squares_optimizer_method` (`:632+`).
- [ ] Update `BoozerSurfaceJAX` option validation in `boozersurface_jax.py` to accept `least_squares_algorithm="lm-minpack"`.
- [ ] **Gate**: end-to-end parity vs `scipy.optimize.least_squares(method='lm', ftol=..., xtol=..., gtol=...)` on the oversampled BoozerSurface fixture and at least 5 MGH problems. L1 + L2 must pass; L3 is desired but may fail on accumulated rounding through `enorm`-vectorized form.
- [ ] If L3 fails: drop to `lax.fori_loop` scalar `enorm` accumulator (Phase 1 fallback), re-test. If still fails: keep `lm-minpack` at the **L1+L2 contract only** (converged-state parity per §6.1 tolerance ladder + exact `info`/`niter`/`nfev` match), document the L3 gap as a per-iteration-trace divergence that does not affect the final state. **Do not call this "L1-only-byte-equal"** — L1 in §6.1 is explicitly tolerance parity, not bit-equality; bit-equality is L4.
- [ ] First-trace compile-time measurement: must be `< 90s` on n=40, m=384 to retain. `< 60s` is the target.
- [ ] Test file: `tests/geo/test_lmder_scipy_parity.py`. Lane: `direct-kernel` state-parity sub-lane.

**Phase 6: MGH-1981 regression suite (1 day)**

- [ ] Implement (or vendor) the 18 canonical Moré-Garbow-Hillstrom test functions. Source: `fortran-lang/minpack/test` directory has reference implementations. Translate to JAX-traceable Python residual functions.
- [ ] L1 parity: `‖x_final - x_minpack‖_∞ ≤ 1e-6` and `|cost_final - cost_minpack| ≤ 1e-10` on all 18 problems.
- [ ] L2 parity: exact match on `info`, `niter`, `nfev` (nfev allowance ±1 for terminator).
- [ ] L3 parity (subset of 5 canonical problems): per-iteration `(x, fnorm, delta, par, ratio)` within `rtol=1e-12`. Use `jax.experimental.io_callback(ordered=True)` to dump JAX trace; intercept SciPy via wrapped residual logging.
- [ ] Test file: `tests/geo/test_lmder_mgh_parity.py`. Lane: `direct-kernel` extended.

**Phase 7: Documentation + plan reconciliation (0.5 day)**

- [ ] Update `optimizer_jax.py:14-45` "LM family note" to add a fourth family entry: `lm-minpack` (byte-equal CPU only, tolerance-equivalent on GPU).
- [ ] Update `docs/source/jax_acceptance.rst` "Optimizer family equivalence" with the third option and its precision contract.
- [ ] Update `CLAUDE.md` "BFGS device residency" entry or add a new "LM family routing" entry documenting the triply opt-in path (`optimizer_backend="ondevice"` + `least_squares_algorithm="lm-minpack"`) and its CPU-only byte-equality claim.
- [ ] Update `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` W4.3 status: flip from "implemented, Wave 4 (docs)" to "implemented, Wave 4 (docs) + Wave 5 (port)" if Track 1 lands.

### 4.3 Track 1 deliverable file map

| File | Status | LOC estimate |
|---|---|---|
| `src/simsopt/geo/optimizer_jax_private/_lmder.py` | new | ~600 |
| `src/simsopt/geo/optimizer_jax.py` | modify (route `"lm-minpack"`) | +30 |
| `src/simsopt/geo/boozersurface_jax.py` | modify (option validation) | +5 |
| `tests/geo/test_lmder_phase0_feasibility.py` | new (G0 gate) | ~120 |
| `tests/geo/test_lmder_enorm_oracle.py` | new | ~100 |
| `tests/geo/test_lmder_qrfac_oracle.py` | new | ~120 |
| `tests/geo/test_lmder_qrsolv_oracle.py` | new | ~150 |
| `tests/geo/test_lmder_lmpar_oracle.py` | new | ~150 |
| `tests/geo/test_lmder_scipy_parity.py` | new | ~200 |
| `tests/geo/test_lmder_mgh_parity.py` | new | ~250 |
| `docs/source/jax_acceptance.rst` | modify | +30 |
| `CLAUDE.md` | modify | +5 |
| `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` | modify (status flip) | +3 |

**Total LOC estimate (extrapolation, not measurement): ~1770 across 13 files.** Subject to halving or doubling based on Phase 4-5 complexity. **If G0 fails on day 1, only `tests/geo/test_lmder_phase0_feasibility.py` (~120 LOC) is written; the spike is then abandoned and Phases 1-7 are not funded.**

### 4.4 Acceptance gates (Track 1, hierarchical)

| Gate | Phase | Status check | Action on failure |
|---|---|---|---|
| G0 — Packed `fjac` + `qtb` byte-equal on (384,40) | 0 | `qtb` and reconstructed packed `fjac` bit-exact vs SciPy MINPACK reference on 100 random seeds at production shape, **conjunctive across ≥ 2 attempted paths** | **ABANDON spike immediately**; do not fund Phases 1-5. Re-scope to `m ≈ n` only requires owner sign-off per §8 Q8 — not a default fallback. |
| G1 — `enorm` byte-equal | 1 | bit-exact on 100+ randomized vectors per bucket | retry with scalar accumulator (once); else abandon spike |
| G2 — `qrfac` adapter byte-equal | 2 | bit-exact on the 4 canonical shapes **assuming G0 passed** | abandon (already empirically verified on m≈n; G0 covers production shape) |
| G3 — `qrsolv` byte-equal | 3 | bit-exact vs MINPACK reference on ~50 fixtures | investigate Givens order; retry once; else abandon |
| G4 — `lmpar` byte-equal | 4 | bit-exact final state on ~30 fixtures | investigate Newton update + secondary exit; retry once; else abandon |
| G5 — Driver L1 + L2 parity | 5 | ≥ 5 fixtures pass | retry with corrections; if 2 attempts fail, abandon |
| G6 — Driver L3 parity (CPU same-machine) | 5 | per-iter `(x, fnorm, delta, par, ratio)` at `rtol=1e-12` | acceptable to fail; drop to L1+L2 contract if fails |
| G7 — Compile time | 5 | first-trace `< 90s` (target `< 60s`) | acceptable to fail at 90s threshold but keep "research" tag; abandon if `> 180s` |
| G8 — MGH-1981 L1 on 18 problems | 6 | all 18 pass at `rtol=1e-6` | if subset fails, document failing problems; do not abandon |

**Spike commit/discard decision: after G0 (fast fail) and after G5 (final commit).**
- **G0 is a true gate**: G0 fail → spike abandoned at end of day 1, ~1 engineer-day spent, zero further work funded. This is the cheapest possible discard checkpoint.
- **G1-G5 are the full spike**: if G0 passes and G1-G5 all pass, the spike succeeded — commit Phases 0-5 and continue Phases 6-7 in a separate PR. If any of G1-G5 fails twice, discard the spike branch and document the failing gate.

---

## 5. Track 3 — Optimistix + Lineax LSMR (DEFERRED)

**Scope:** add a third opt-in `least_squares_algorithm="optimistix-lm"` routing to `optimistix.LevenbergMarquardt(linear_solver=lineax.LSMR(...))`. **Net LOC is additive in this plan** (~+400 LOC adapter + 2 deps; the existing matrix-free LM stays). The "~500 LOC simplification" only materializes if a separate future cleanup retires the current `_lm_iteration`/`_gmres_solve_least_squares_system` path after Optimistix is proven in production. Better numerical conditioning on near-rank-deficient fixtures via LSMR. Tolerance-equivalent to MINPACK; not byte-equal.

**Estimated effort:** 1–2 weeks. Should not start until Tracks 1 and 2 land.

### 5.1 Todos (Track 3)

- [ ] Add `optimistix>=0.0.10` + `equinox>=0.11.0` to `pyproject.toml` — **either as required deps OR as `extras_require[jax-optimistix]` optional deps depending on §8 Q4 owner decision.** Default until Q4 is decided: keep optional (current state per `pyproject.toml:92`) to preserve install-without-Optimistix path for downstream users.
- [ ] Add `lineax>=0.0.7` to `pyproject.toml` under the same dep-tier policy as `optimistix` per §8 Q4.
- [ ] Implement `jax_least_squares_optimistix(residual_fn, x0, ...)` wrapper in `optimizer_jax.py` calling `optimistix.least_squares(...)` with `solver=optimistix.LevenbergMarquardt(rtol=tol, atol=tol, linear_solver=lineax.LSMR(rtol=tol, atol=tol))`.
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
  - Default fixture (`ncoils=2, nphi=5, ntheta=5, mpol=ntor=2`) — under-sampled, `κ ≈ 10⁷`, exercises `par` escalation
- **Edge cases**: zero-residual initial guess, exact rank deficiency, `maxfev=2` forced exhaustion, `fnorm=0` start.

### 6.3 SciPy interception (for L2/L3)

`scipy.optimize.least_squares(method='lm')` exposes no callback. Three options (in order of preference):

1. **Wrapped residual / Jacobian** logging on every call — track iteration count via fnorm decreases. Default L3 instrument.
2. **Patched SciPy fork** with HDF5 trace dump in `_minpackmodule.c` — ground-truth oracle for fixture generation.
3. **Direct ctypes binding to `cminpack` or `fortran-lang/minpack`** — cleanest oracle for sub-routine (qrsolv, lmpar) testing.

### 6.4 JAX interception

`jax.experimental.io_callback(callback, result_shape, *args, ordered=True)` inside the `lax.while_loop` body. Compatible with `jit` and `vmap`. Used for both Track 1 L3 validation and Track 2 termination diagnostic.

### 6.5 CI cost estimate

| Suite | Cold compile | Warm |
|---|---|---|
| Track 2 termination + damping tests | +5–10s | +2s |
| Track 1 sub-routine oracles (enorm, qrfac, qrsolv, lmpar) | +30s | +5s |
| Track 1 MGH-1981 L1 suite (18 problems) | +60s | +20s |
| Track 1 L3 trace parity (5 fixtures) | +30s | +15s |
| Track 1 GPU lane (if implemented) | +90s | +30s |
| Track 3 Optimistix parity | +30s | +10s |

**Total budget impact: ~5-8 min CPU, ~10 min with GPU lane.** Fits existing `jax_smoke.yml` cadence.

---

## 7. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `jax.scipy.linalg.qr(pivoting=True)` tie-break differs from LAPACK on exactly-equal column norms | LOW | Empirically verified on 4 fixtures; add a dedicated equal-norm fixture in G2 |
| `enorm` vectorized form differs from sequential MINPACK in last bit | MEDIUM | Fallback to `lax.fori_loop` scalar accumulator documented in Phase 1 |
| `qrsolv` Givens chase order produces different bit-pattern than Fortran | MEDIUM | Sequential `lax.fori_loop` mirrors Fortran column-by-column order; oracle test catches deviation |
| `lmpar` Newton iteration count differs (off-by-one bracket update) | MEDIUM | L2 test asserts iteration count match; debuggable via `io_callback` trace dump |
| First-trace compile exceeds 60s gate | MEDIUM | G7 sets 90s soft limit; if breaks, document as "research lane" with explicit acceptance carve-out |
| Optimistix maintainer abandons project | LOW | Optimistix v0.0.10+ is stable; pin tightly; Track 3 is opt-in not default |
| Critic concerns prove valid mid-spike (byte-equality not achievable) | MEDIUM | Spike gate G5 is explicit commit/discard checkpoint; Track 2 work survives independently |
| GPU lane tie-break (MAGMA `geqp3` ≠ LAPACK `dgeqp3`) breaks L2/L3 cross-device | LOW | Documented in §0 Non-goals: GPU lane is L1-only by design |
| CPU LAPACK linkage drift (different vendor's LAPACK on different machines) breaks L4 | LOW | Documented in §6.1: L4 is single-host single-build only |

---

## 8. Open decisions

These need owner sign-off before starting work:

- [ ] **Q1 — Track scope**: pursue all three tracks, or only Track 2 + Track 3 (skip the byte-exact spike entirely)?
- [ ] **Q2 — Track ordering**: Track 2 first (recommended) or parallel Track 1 spike + Track 2 production work?
- [ ] **Q3 — Track 1 abandonment threshold**: at what gate failure are we OK discarding the spike branch? (Recommendation: any of G1, G3, G4, G5 failing twice triggers discard.)
- [ ] **Q4 — Optimistix as required dep**: Track 3 currently has `optimistix` as an optional dep. Move to required for the JAX lane, or keep optional?
- [ ] **Q5 — Publication ambition**: if Track 1 lands, write up as JOSS / workshop paper? Per Agent E this is potentially the first JAX MINPACK port. **This claim was not independently verified by a numerical-analysis literature review** — treat as a possible follow-up, not a guaranteed novelty.
- [ ] **Q6 — Compile-budget exception**: should Track 1 be exempted from the 60s first-compile gate in `docs/source/jax_acceptance.rst:101`, given that it's an opt-in research lane? (Recommendation: yes, with explicit `lm-minpack` carve-out in the docs.)
- [ ] **Q7 — Cross-machine validation**: should the Track 1 L4 byte-equality contract require validation on multiple machines (per the existing "Floating-point reproducibility across machines" caveat in CLAUDE.md), or single-machine only?
- [ ] **Q8 — Phase 0 Path C scope and re-scope authority**: (a) is the Phase 0 spike permitted to evaluate Path C (XLA custom call FFI to LAPACK `dgeqp3`), or is the JAX-pure constraint hard? (b) if G0 fails (no two paths agree), the plan now treats "re-scope to m≈n only" as out-of-bounds without owner sign-off; confirm this discipline, OR explicitly authorize the re-scope as a default fallback. **Default if Q8 is unanswered: Path C in scope, re-scope NOT a default fallback.**

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
- "L1-only-byte-equal" mixes orthogonal axes (L1 = tolerance parity per §6.1; L4 = bit-equality). Fixed in Phase 5 fallback wording.

Plus a third independent Crucible review pass (2026-05-16, 4 parallel discovery lenses) that flagged 1 critical + 9 major findings against rev 2; rev 3 incorporates all required fixes:
- **CRITICAL**: info=8 was misclassified as matrix-free-computable. `lmder.f:431` is `if (gnorm .le. epsmch) info = 8` — info=8 depends on `gnorm`, same as info=4. Conversely info=3 is `info=1 AND info=2` (`lmder.f:421-422`) and IS matrix-free. Corrected subset to **{1, 2, 3, 5, 6, 7}** matrix-free, **{4, 8}** pivoted-QR-required, in §3.3, §3.4, §12.
- **MAJOR**: `lmder_serial.f` doesn't exist on netlib; corrected to `lmder.f` (same line numbers correct) in §3.2 and §3.3.
- **MAJOR**: `lmpar.f:229-230` was wrong; the secondary-exit predicate is on `lmpar.f:220-222` (fused into the primary-exit compound `if`). Fixed in §4.2 Phase 4.
- **MAJOR**: Front-matter "Track 1 spike through G5: 1–3 days cumulative" contradicted the §4.2 phase-duration sum (~8.5 days). Replaced with a per-phase breakdown.
- **MAJOR**: §5.1 listed `optimistix` as a required dep, pre-empting §8 Q4 owner decision. Reframed as conditional on Q4 default-optional-until-decided.
- **MAJOR**: `dgeqpf` was cited as a G0-fail fallback target; `dgeqpf` is the deprecated Level-2 BLAS predecessor of `dgeqp3`. SciPy and JAX both call `dgeqp3`, so a `dgeqpf` fallback would guarantee losing byte-equality. Corrected to `dgeqp3` and incorporated as Path C (XLA custom call) in Phase 0.
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

- [ ] Owner decision on §8 Q1–Q7 (especially Q1 track scope and Q3 abandonment threshold)
- [ ] Allocate engineering time per agreed track scope

### Track 2 — termination + damping retrofit

- [ ] Phase 2.1: extend `levenberg_marquardt` and `_traceable` signatures with `ftol`/`xtol`/`gtol`
- [ ] Phase 2.2: add `info` int32 to `_lm_iteration` state, replace `success` boolean
- [ ] Phase 2.3: implement `compute_info` per Agent C §5 **restricted to matrix-free-computable subset {1, 2, 3, 5, 6, 7}** (info=4 and info=8 reserved for Track 1 G5 — both depend on `gnorm` which requires pivoted-QR data per `lmder.f:431` and `:313`)
- [ ] Phase 2.4: replace asymmetric damping with MINPACK symmetric Marquardt damping update (`× 2` / `× 1/2`) plus `par` escalation logic per Agent C §4 / `lmder.f:381-396`
- [ ] Phase 2.5: update while-loop predicate so termination fires on `info != 0` (over the matrix-free subset) OR the legacy `success` boolean — whichever first; existing single-`tol` callers see identical behavior
- [ ] Phase 2.6: update result schema with `info` field (range `{0, 1, 2, 3, 5, 6, 7}`), preserve `success = info in {1, 2, 3}` for backward-compat
- [ ] Phase 2.7: `tests/geo/test_lm_termination_parity.py` — 5 MGH + 1 BoozerSurface fixture; assertion is "JAX `info` matches SciPy `info` **only when SciPy's `info` is also in the matrix-free subset {1, 2, 3, 5, 6, 7}**"; JAX exit on info=2 while SciPy exits on info=4 is expected algorithm divergence, not a bug
- [ ] Phase 2.8: `tests/geo/test_lm_damping_parity.py` — iteration count within 1.5× CPU MINPACK
- [ ] Phase 2.9: update `optimizer_jax.py:14-45` module docstring
- [ ] Phase 2.10: update `docs/source/jax_acceptance.rst:156-187`
- [ ] Phase 2.11: full `tests/geo/test_boozersurface_jax.py` + `test_single_stage_jax.py` regression run
- [ ] Phase 2.12: `ls_state_parity` lane confirmation at `sdofs_inf ≤ 1e-11`
- [ ] Phase 2.13: ruff check / format clean

### Track 1 — CPU byte-exact MINPACK port (gated spike)

- [ ] **GATE G0 — Phase 0**: packed `fjac` + `qtb` feasibility on production shape (384,40)
  - [ ] Implement path A: reconstruct packed Householder from explicit Q
  - [ ] Implement path B: `householder_product`/`ormqr` to compute `Q^T·fvec` directly
  - [ ] Implement path C: direct FFI to LAPACK `dgeqp3` via XLA custom call (unless §8 Q8 owner sign-off marks Path C out-of-scope)
  - [ ] Oracle test vs `scipy.linalg.qr_multiply(..., mode='left', pivoting=True)` on 100 random seeds at (384,40) + (40,40) + (75,39) + (100,50) + (2000,80)
  - [ ] **Spike checkpoint G0**: bit-equal `qtb` and packed `fjac` on all shapes incl. (384,40), **conjunctive across ≥ 2 attempted paths**; on fail, **abandon spike immediately, full stop**. Re-scoping to `m ≈ n` only is **not a default fallback** — it is a project-charter renegotiation that requires owner sign-off per §4.2 Phase 0 G0-fail clause and §8 Q8. The plan author may not unilaterally re-scope.
- [ ] **GATE G1 — Phase 1**: `enorm` byte-equal port
  - [ ] Implement vectorized 3-bucket form
  - [ ] Implement scalar `lax.fori_loop` fallback
  - [ ] Oracle test against `_minpack._enorm` or f2py-built reference
  - [ ] **Spike checkpoint G1**: 100+ vectors byte-equal per bucket; if fails, abandon
- [ ] **GATE G2 — Phase 2**: `qrfac` wrapper around `jsl.qr(pivoting=True)`
  - [ ] Adapter to MINPACK-style packed output
  - [ ] Empirical pivot tie-break verification
  - [ ] Oracle test on 4 canonical shapes
  - [ ] **Spike checkpoint G2**: byte-equal on all shapes; low risk per empirical validation
- [ ] **GATE G3 — Phase 3**: `qrsolv` Givens elimination
  - [ ] Two-level `lax.fori_loop` per Agent C §2.1
  - [ ] Explicit `(x, sdiag, S_lower)` return
  - [ ] Oracle test via cminpack ctypes binding
  - [ ] **Spike checkpoint G3**: byte-equal on ~50 fixtures; if fails 2×, abandon
- [ ] **GATE G4 — Phase 4**: `lmpar` univariate Newton
  - [ ] `LmparState` carry + `lax.while_loop`
  - [ ] Zero-par GN accept branch
  - [ ] Bracket setup + secondary-exit predicate
  - [ ] Oracle test on 4 fixture types
  - [ ] **Spike checkpoint G4**: byte-equal final state + iteration count; if fails 2×, abandon
- [ ] **GATE G5 — Phase 5**: Driver integration
  - [ ] `LmderState` NamedTuple
  - [ ] `phase_jacobian` + `phase_inner` body
  - [ ] `lax.switch` dispatch (no nested while-loops)
  - [ ] Scaling-vector D, trust-region delta, par update
  - [ ] `compute_info` wired
  - [ ] `levenberg_marquardt_minpack_traceable` public entry
  - [ ] `"lm-minpack"` in `VALID_LEAST_SQUARES_ALGORITHMS` + routing
  - [ ] `BoozerSurfaceJAX` option validation
  - [ ] **Spike checkpoint G5**: L1 + L2 byte-parity on 6 fixtures; commit-or-discard decision
- [ ] **GATE G6 — Phase 5 (continued)**: L3 trace parity
  - [ ] `io_callback`-based trace dump from `lax.while_loop` body
  - [ ] SciPy-side per-call logging
  - [ ] Trace comparison harness with structured diff dump
  - [ ] **G6 result**: pass → L1+L2+L3 contract; fail → L1+L2-only contract with documented gap
- [ ] **GATE G7 — Phase 5**: compile-time measurement
  - [ ] First-trace timing on n=40, m=384
  - [ ] First-trace timing on n=39, m=75 (default BoozerSurface)
  - [ ] **G7 threshold**: `< 90s` retain, `< 60s` target hit, `> 180s` abandon
- [ ] **Phase 6**: MGH-1981 suite
  - [ ] Translate 18 problems to JAX-traceable residuals
  - [ ] L1 parity tests (all 18)
  - [ ] L2 parity tests (all 18)
  - [ ] L3 parity tests (5 canonical subset)
  - [ ] **GATE G8**: 18/18 L1 pass at `rtol=1e-6`
- [ ] **Phase 7**: documentation
  - [ ] Update `optimizer_jax.py:14-45` LM family note (add 4th entry: `lm-minpack`)
  - [ ] Update `docs/source/jax_acceptance.rst` Optimizer family equivalence
  - [ ] Update `CLAUDE.md` with `lm-minpack` routing + CPU-only byte-equality contract
  - [ ] Update `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` W4.3 status

### Track 3 — Optimistix + Lineax LSMR (deferred)

- [ ] Phase 3.1: add `optimistix>=0.0.10`, `equinox>=0.11.0`, `lineax>=0.0.7` to deps
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
- [ ] Update `.artifacts/boozersurface_ls_deepdive_2026-05-15/PLAN.md` status table for B3 row
- [ ] (Optional) Q5 publication write-up if Track 1 lands and novelty claim survives literature review

---

**End of plan.**
