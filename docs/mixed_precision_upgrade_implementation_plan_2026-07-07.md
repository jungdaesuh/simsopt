# Mixed-Precision Upgrade: fp32 Compute Under fp64 Certificates

**Status:** Draft (research complete; awaiting go)
**Last updated:** 2026-07-07

## Purpose

Make production single-stage runs (a) *possible at full speed* on FP64-weak
consumer GPUs — the RTX 5090 32 GB target (FP64 = 1/64 of FP32) — and
(b) 1.5–3× faster on datacenter GPUs (roadmap M4). The strategy is the
Carson–Higham mixed-precision iterative-refinement family: do the expensive
sweeps in fp32, refine against fp64 residuals, and keep every acceptance
certificate in fp64 so correctness contracts are unchanged.

## Goals

- RTX 5090: e2e single-stage run with per-eval walls limited by its 104.8
  TFLOPS fp32 rate, not its 1.64 TFLOPS fp64 rate (raw fp32 on a 5090 is
  ~10.8× an A100's fp64 — a correctly mixed pipeline can beat today's A100
  walls on a consumer card).
- A100/H100: accepted-eval K1 build + HVP-heavy phases 1.5–3× (M4).
- Memory: fp32 hot-path arrays halve footprints (mpol18 tuned peak 24.8 GiB
  → ~15 GiB class), doubling 32 GB headroom.
- Zero tolerance changes: the 1e-11 Boozer gate, parity-matrix thresholds,
  and all success certificates stay fp64 and byte-compatible in fp64 mode.

## Non-Goals

- fp16/bf16 anywhere (fp32 already buys the 5090 win; half precision buys
  little more for elementwise BiotSavart kernels and risks the κ budget).
- Changing the E-W loose-path algorithm, the dense-IR routing contract, or
  any gate/tolerance (guardrail: no tolerance loosening).
- Tensor-core matmul tricks (our kernels are broadcast-sum elementwise, not
  GEMM-shaped; `default_matmul_precision` is pinned only as a safety).

## Current Context (verified, 2026-07-07)

- Everything runs fp64 (`JAX_ENABLE_X64=1`); dtype is partially centralized
  (`get_backend_policy().runtime_dtype`, `optimizer.py:5096`) but ~740
  explicit `float64` pins exist across `src/simsopt_jax/core`.
- Precision-critical map (measured, from closed campaigns):
  - κ(J) ≈ 625 (well-conditioned); κ(H) = κ(J)² ≈ 3.9e5 — the squaring is
    what killed naive fp32 (dense-PLU NaNs, 2026-06 campaign). J-based
    LSMR/QR is the only *true-fp32* solve path.
  - `lineax.LSMR` is already wired (`solve/dispatch.py:517`,
    `contracts.py:27` default) — the J-based route is infrastructure, not
    greenfield.
  - dense-IR (`hybrid_final_dense_ir`, shipped `ad3cc28b7`) is precisely the
    classical mixed-precision enabler: factor once, refine against
    *current* residuals, accept only on a measured backward-error gate,
    fail loud. Its e2e certificates: ‖grad‖ 2.4e-14, IR residual 1.4e-15.
  - E-W loose phase runs at tolerances 1e-2…1e-10 with 5–23 matvecs/iter —
    naturally fp32-tolerant; the strict-cap predicate is a ready-made
    precision handoff boundary.
  - BiotSavart cpp↔jax kernel parity 3e-16 (fp64 anchor for sensitivity
    harnesses).
- Convergence theory (Carson–Higham IR3; five-precision GMRES-IR 2024;
  least-squares IR, Carson–Daužickaitė SIMAX 2025): IR with an fp32
  factorization converges when κ·u_f ≲ 1. Ours: κ(J)·2⁻²⁴ ≈ 3.7e-5
  (trivial) and even κ(H)·2⁻²⁴ ≈ 0.023 (comfortable) — i.e. **the shipped
  dense-IR can take fp32 factors with ~0.023 error contraction per pass**;
  2 refinement passes reach ~5e-4 of the presolve error, and the existing
  gate rejects any violation.
- Hardware: RTX 5090 = 104.8 TFLOPS fp32 / 1.637 TFLOPS fp64 (1:64), 32 GB
  GDDR7. A100-40GB = 19.5 fp32 / 9.7 fp64 (1:2).
- User directive 2026-07-06: memory first (delivered — 24.8 GiB fits), then
  mixed precision. This plan is the "then".

## Rationale

Every expensive object in the pipeline is a *solve input*, not a
*certificate*: the 663–2055-column dense builds, the HVP matvecs inside
GMRES/IR, the bfgs pre-stage, and the K2 adjoint sweeps. IR theory says all
of them may be computed at u_f = fp32 provided residuals and acceptance
are evaluated at u = fp64. That is exactly the structure dense-IR already
has — this plan generalizes "factors may be stale" (chord iteration) to
"factors may be stale *and low-precision*", which the same backward-error
gate already polices. The J-based LSMR route (κ≈625, un-squared) extends
the same trick to the adjoint.

## Assumptions

- κ estimates (625 / 3.9e5) hold across the seed family at production
  resolution (measured at 255×64 mpol10; mpol18 κ re-measured in P0).
- JAX per-array dtype control under `jax_enable_x64` (fp32 arrays remain
  fp32 through jit; explicit casts at phase boundaries) — standard, but
  P0 asserts no silent upcast/downcast in the hot graphs.
- RunPod (or similar) can supply an RTX 5090 instance for validation.

## Implementation Plan

0. **P0 — dtype plumbing + sensitivity scout (no behavior change)**
   - [ ] Thread an explicit `compute_dtype` through the polish runner and
         BiotSavart kernel entry points (default fp64 → all-green baseline
         byte-identical; env `SIMSOPT_MIXED_PRECISION` off by default).
   - [ ] Sensitivity harness: B/residual/HVP at fp32 vs fp64 on the
         production seed — record rel-error distributions (expect ~1e-7
         class) and κ(J)/κ(H) at mpol10 AND mpol18.
   - [ ] Pin `jax.default_matmul_precision("highest")` in fp32 graphs
         (block TF32 contamination); assert no silent dtype promotion via
         a trace-time dtype audit on the K1 graph.
1. **P1 — fp32 factors inside dense-IR (IR3 on the shipped machinery)**
   - [ ] Factor-once build (`entry_hessian_matvec` sweeps + `lu_factor`)
         in fp32; `lu_solve` presolve in fp32; residuals + corrections +
         gate in fp64 (u_f=fp32, u=u_r=fp64 — textbook IR3).
   - [ ] Raise `_DENSE_IR_NEWTON_REFINEMENT_STEPS` only if measured
         contraction demands it (theory: 2 passes suffice at 0.023).
   - [ ] Gate-fail path: one fp64 refactor retry (folds into the planned
         v2 lazy-carry), then existing fail-loud stall.
   - [ ] Tests: κ=1e5-class mixed-factor equivalence vs fp64 dense_lu
         (mirror `test_..._matches_dense_lu_ill_conditioned`); fp32-factor
         gate-rejection test; counter honesty unchanged.
2. **P2 — fp32 loose phase with E-W handoff (the big 5090 win)**
   - [ ] bfgs pre-stage + far-from-target E-W GMRES entirely in fp32
         (state, matvecs, line search); promote the carry to fp64 exactly
         at the strict-cap predicate (already the routing boundary).
   - [ ] Certificates unchanged: final ‖grad‖/acceptance in fp64.
   - [ ] Validate the handoff does not perturb accept/reject at the
         boundary (the ULP-flip fragility class from the stall-bug
         history) — probe with the init-probe A/B harness.
3. **P3 — K2 adjoint via LSMR-IR on J (κ≈625, true-fp32 inner)**
   - [ ] Wire the existing lineax LSMR into a least-squares IR loop
         (Carson–Daužickaitė 2025 shape): fp32 J-sweeps, fp64 residuals;
         keep the dense fp64 path as the comparator mode.
   - [ ] Byte-parity contract (`_traceable_solve_plu_linearization`) holds
         in fp64 mode; mixed mode gets its own tolerance-based parity
         gates (existing matrix thresholds, NOT loosened).
4. **P4 — fp32 BiotSavart sweep kernels + fp64 accumulation**
   - [ ] Segment sums accumulate in fp64 (or pairwise) with fp32
         gamma/gammadash arrays; measure end-to-end error budget vs the
         1e-11 gate chain.
   - [ ] Memory re-measure at mpol18/255×64 (expect ~15 GiB class).
5. **P5 — 5090 validation campaign**
   - [ ] RunPod RTX 5090: e2e walls + memory + parity vs A100-fp64
         reference artifacts (reuse the B-lane NDJSON instruments; note
         RunPod ops runbook + driver ≥R575 gotchas).

## Validation Plan

- [ ] Phase-gated: full private test file green per phase; fp64 mode
      byte-identical at P0 (hash the K1 NDJSON certificates).
- [ ] Parity matrix per phase: jax-mixed vs jax-fp64 within existing
      tolerances (cross-solver band decision pending from A5b applies
      here too — surface, don't loosen).
- [ ] e2e anchors: A100 accepted-eval ≤ 67.3 s baseline must improve or
      hold; 5090 target = accepted eval within ~2× of A100-fp64 walls
      (stretch: parity, given 10.8× raw fp32 advantage vs serial taxes).
- [ ] IR contraction measured per phase and compared to the κ·u_f theory
      number (0.023 / 3.7e-5) — a contraction violating theory = wrong κ
      or a silent dtype leak; stop and root-cause.

## Risks and Mitigations

- Risk: silent dtype promotion in traced graphs (one fp64 constant
  upcasts a whole chain) → the fp32 win silently evaporates.
  Mitigation: P0 trace-time dtype audit + `rhs = jnp.asarray(rhs,
  dtype=lu_piv[0].dtype)`-style explicit alignment (the review advisory's
  sibling pattern, now load-bearing).
- Risk: accept/reject flips at the fp32→fp64 handoff (ULP-fragility class
  that caused the historical stall bugs).
  Mitigation: handoff at the strict-cap boundary only; certificates fp64;
  the retry-at-strict-cap safeguard already covers marginal directions.
- Risk: transfer-guard/backend-policy collisions with fp32 arrays
  (`SIMSOPT_JAX_TRANSFER_GUARD=disallow` paths assume runtime_dtype).
  Mitigation: P0 threads dtype through the policy, not ad-hoc casts.
- Risk: TF32 contamination on Ampere+ makes "fp32" secretly ~10-bit
  mantissa in matmuls. Mitigation: pin matmul precision "highest" in
  mixed graphs; assert in the sensitivity harness.
- Risk: 5090 availability/driver (RunPod R575+ requirement, jaxlib CUDA
  wheels for Blackwell consumer). Mitigation: P5 is last; smoke jaxlib on
  the pod before committing the campaign.

## Completion Criteria

- [ ] fp64 mode byte-identical (P0 hash gate) — mixed mode strictly
      opt-in via env, mirroring the dense-IR rollout pattern.
- [ ] A100: measured accepted-eval improvement recorded in the roadmap
      (M4 line) with the same NDJSON instrument.
- [ ] RTX 5090: full e2e run, peak < 30 GiB, certificates identical
      quality (‖grad‖ ≤ 1e-11 gate chain), wall recorded vs A100.
- [ ] Crucible strict PASS per phase; no tolerance touched anywhere.

## Open Questions

- LSMR-IR loop placement: inside the traceable graph (lax.while) vs
  host-driven like the outer loop — decide from P0 compile diagnostics.
- Does the K2 dense build stay fp64 (it is the 27.5 s floor M3 targets) or
  join P1? Depends on P0's measured κ of the adjoint system at mpol10/18.
- Whether P2's fp32 bfgs changes iteration counts materially (it may
  *reduce* wall even with +10% iters given 32×/64× rate advantages on
  consumer cards; A100 fp32 is only 2× fp64, so P2 may be 5090-only).

## References

- Carson & Higham, three-precision IR (SIAM 2018) — convergence κ·u_f<1.
- Amestoy, Buttari, Higham, L'Excellent, Mary, Vieublé, "Five-Precision
  GMRES-based Iterative Refinement" (SIMAX 2024).
- Carson & Daužickaitė, "A Comparison of Mixed Precision Iterative
  Refinement Approaches for Least-Squares Problems" (SIMAX 2025) — the
  J-based/LSMR route's theory.
- NVIDIA RTX Blackwell GPU Architecture whitepaper — GB202: 2 FP64 cores/SM,
  FP64 = 1/64 FP32; RTX 5090 104.8 TFLOPS fp32 / 1.64 fp64 / 32 GB.
- Closed campaigns: fp32 adjoint NaN limit + J-based alternatives
  (2026-06 memories); dense-IR Phase A delivery (2026-07-05/06).
