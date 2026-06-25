# m18 Boozer Adjoint: Iterative-Refinement Fix for the Linear-Solve Tolerance Floor

> Created 2026-06-24 · Updated 2026-06-24 (lit-search solver family folded in; doc-review-fix pass:
> live code/artifacts verified; existing one-correction square-solve refinement and exact-vs-LS deliverable
> split corrected) · Status: PLANNED
> (Track A residual-status diagnostic pending; exact-adjoint failure and LS `optimistix-lm` gate failure
> captured in `.m18-adjoint-artifacts/`; H100 pod `ve7vibu5eur76s` PAUSED) · Owner: JAX-port

## Purpose

Unblock the m18 Boozer adjoint path without lowering success gates. The evidence now separates two
related but different targets:

- **Track A (exact, matched 37×37 grid):** the exact-jacobian adjoint fails the linear-solve status
  check after the forward exact surface converges. Current code already performs one conditional
  operator-GMRES correction in the shared square solve; Track A must measure why that existing
  correction still fails and then either generalize the refinement or change the exact-adjoint solver
  family.
- **Track B (LS, 96×96 grid):** the final "GPU is faster" speed deliverable is the heavy LS
  single-stage `J + dJ` path, where the GPU work is substantial. The current `optimistix-lm` probe
  still misses the LS Newton gate (`‖grad‖ = 1.431e-11 > 1e-11`), so Track B remains required before
  declaring the m18 speed demo complete.

The fix must be grounded in numerical linear algebra: residual refinement or a J/augmented-system
formulation, not a stronger optimizer label, not higher hardware precision, and not a magic-number
tolerance loosen.

## Goals

- Exact-form m18 Boozer adjoint (`IotasJAX.J()` / `.dJ()`) returns finite `J` and `dJ` on GPU and
  CPU, with the inner adjoint linear solve reporting `success=True`.
- Track A is **principled and measured**: residual-status telemetry shows whether the current
  one-correction square solve fails because the correction count is insufficient, the matvec/operator
  residual is inaccurate, or the solver family is wrong. Any change passes the existing `1e-14` gate
  without changing the gate or diverging from upstream native semantics.
- A GPU-vs-CPU timing table is produced for the exact-adjoint fix, but it is not treated as the final
  LS speed deliverable.
- Track B keeps a documented, literature-grounded path for the LS (penalty) form via LM/LSMR or an
  augmented-system formulation. If the m18 deliverable remains the 96×96 LS single-stage path, Track B
  is required, not optional.

## Non-Goals

- Changing the LS success gate (`newton_tol = 1e-11`, `‖∇f‖₂ ≤ tol`). It is **byte-identical to
  native** simsopt (`src/simsopt/geo/boozersurface.py:138-141,510-527`); diverging would break SSOT
  parity. The LS m18 plateau (1.4e-11) is a real float64 floor of the *squared* formulation, not a
  port bug.
- Introducing DL optimizers (Adam/Lion/Muon/Shampoo). They are first-order approximations to
  second-order info, do not address normal-equation κ² conditioning, and are the wrong category (see
  conversation lit search / GPD pattern).
- Treating exact-adjoint success as a proxy for the LS speed demo. The exact form is the cleaner
  Track A diagnostic target; the LS 96×96 path remains the final GPU-speed target unless the user
  explicitly narrows the deliverable.
- Using `SIMSOPT_ADJOINT_LINEAR_SOLVER=cg` as an exact-adjoint fix. Live code routes that env knob
  only through the LS Hessian adjoint path (`_solve_hessian_least_squares_system_with_status`), not
  the exact-jacobian path.

## Current Context (code-confirmed facts)

- **Exact adjoint solve path.** `IotasJAX.J/.dJ` → `_resolved_boozer_solved_runtime_state` →
  inner solve `_solve_boozer_adjoint` (`src/simsopt_jax_adapters/geo/surface_objectives.py:1997`)
  → `_checked_boozer_linear_solve` (`:2010`), which calls `adjoint_state.solve_transpose_with_status(rhs)`
  and **raises** `RuntimeError("Boozer adjoint linear solve failed on the JAX runtime-state path …")`
  at `:2024-2028` when `_linear_solve_status_success(status)` is False.
- **The status gate** (`src/simsopt_jax/geo/optimizers/optimizer.py:4260` `_linear_solve_status`):
  `success = finite & (residual_relative ≤ effective_tolerance)`, where
  `residual_relative = ‖rhs − A·x‖ / max(‖rhs‖, eps)` (`_linear_solve_residual_scale`, `:4210`) and
  `effective_tolerance = clamp(tol, [floor, cap])` (`_effective_linear_solve_tolerance`, `:4197`).
- **The exact-form tolerance** (`src/simsopt_jax_adapters/geo/boozer_surface.py:4325`
  `_linear_solve_tolerance`): for `boozer_type == "exact"`,
  `tol = min(cap, max(newton_tol × 0.1, floor))`. With exact `newton_tol = 1e-13`
  (`_DEFAULT_OPTIONS_EXACT`, `:3544`), float64 `floor = 1e-14`, `cap = 1e-10`
  (`src/simsopt_jax/backend/runtime.py:235-236`) ⇒ **effective tolerance = 1e-14**.
- **Measured conditioning (CPU, native, decisive).** At the converged m18 chomp surface
  (`.m18-adjoint-artifacts/log_cond_m18.txt`): exact Boozer Newton Jacobian is 2055×2055,
  σ_max = 1276, σ_min = 0.2267, **κ₂(J) = 5629.3**, κ·eps = 1.25e-12
  (κ·unit-roundoff ≈ 6.3e-13). The square system is not near singular.
- **Current square solve already refines once.** `_solve_square_vector_system_operator_only`
  (`optimizer.py:4467-4507`) runs operator GMRES, computes status, and if the result is finite but
  failed, solves one correction equation and recomputes the residual. The exact-adjoint failure in
  `.m18-adjoint-artifacts/log_exactadj_gpu.txt` therefore proves that **the existing one-correction
  path did not satisfy status**, but it does not yet prove the achieved `residual_relative` or why the
  remaining gap exists. A1 must measure that before coding.
- **Lit-search verdict (this session).** GPD pattern recorded; keystone refs:
  Björck (1967, *BIT* 7:257) LS iterative refinement on the (m+n) augmented system;
  Carson, Higham & Pranesh (2020, *SIAM J. Sci. Comput.* 42:A4063) GMRES-LSIR;
  Carson & Higham (2017, *SIAM* 39:A2834) GMRES-IR for square `Ax=b`;
  Carson & Oktay (2024, arXiv:2401.03755). Literature supports residual-driving IR for κ≪1/u, but
  the current code already has one correction, so the local question is whether more/better
  refinement or a different exact-adjoint solver is needed.
- **Existing infrastructure.** Solve callbacks expose `apply_transpose` (matvec) and
  `solve_transpose_with_status` (`_build_runtime_linear_solve_callbacks`,
  `src/simsopt_jax_adapters/geo/boozer_surface.py:4349`). Square operator-GMRES solve
  `_solve_square_vector_system_operator_only` at `optimizer.py:4467`; dense backward-error success at
  `:4289`; `_forward_error_success` at `:4330`. The square solve is shared by exact-jacobian solves
  and some LS Hessian fallback paths, so global changes need call-site coverage.
  `SIMSOPT_ADJOINT_LINEAR_SOLVER` env knob exists but governs the **LS** Gauss-Newton (JᵀJ) adjoint,
  not the exact path (`optimizer.py:3609-3625`).

## Rationale

The exact-adjoint forward solve is fine, but the adjoint status failure is not yet diagnosed deeply
enough. The working hypothesis is that the existing one-correction operator-GMRES solve cannot drive
the *relative residual* below the hardcoded `1e-14` gate at κ≈5.6e3 / n≈2053. Two ways to reconcile:

- **(A) Refine the solve to working precision (chosen first, but diagnostic-gated).** Generalize the
  existing one-correction residual refinement into a bounded, trace-safe loop or switch to an exact
  square solver family that directly achieves the status gate. Candidate correction step:
  `r = rhs − A·x; solve A·dx = r; x ← x + dx`. This is an accuracy improvement only if A1 shows the
  residual decreases as expected and the post-change status is finite and below `1e-14`.
- **(B) Make the gate conditioning-aware** (set tol ≈ κ·u). Rejected as primary: it is closer to
  loosening a threshold than fixing the solve, and the `1e-14` floor is shared policy.

Track A uses (A) on the **exact** (square, κ≈5.6e3) adjoint as the smallest isolated diagnostic fix.
Track B extends the same numerical idea to the **LS** augmented system (un-squares κ(JᵀJ)→κ(J)) for
the final 96×96 LS speed path.

## Solver Family & Selection (literature)

The Track A fix is a **residual-status solve fix** on the existing exact J-based system; the Track B
fix is a **change of LS formulation** to the J/augmented-system space. Neither is a stronger optimizer
label (Adam/Lion/Muon are first-order and not the right stationarity tool for κ²-conditioned LS
stationarity), and neither is
higher hardware precision. The gated floor scales with the conditioning of the *space you solve in* —
and the two solves are gated on different quantities:

- **Exact adjoint** (Track A; gate = `residual_relative ≤ 1e-14`). The base operator-GMRES solve
  plus the existing one correction still fails at m18. A1 must capture the pre/post-correction
  `residual_relative`, iteration count, and backend before assuming the next correction drives the
  residual to ≈u.
- **LS form** (Track B; gate = `‖∇f‖₂ ≤ 1e-11`). The current penalty solve is exposed to the
  *squared* normal-equation conditioning. Do **not** reuse the exact-track κ=5629 for LS without
  measuring the actual 96×96 LS residual Jacobian: the adjacent m18 handoff estimates a different LS
  scale (κ(J)≈625, κ(J)²≈3.9e5). B1 must report the LS conditioning it uses before claiming an
  augmented-system solve reaches the upstream gate.

For Track A, this is a residual-status problem on an existing square J-based solve. For Track B, the
core numerical risk is the **κ-squaring formulation** problem.

| Method | What it does | Maps to |
|---|---|---|
| **Björck LS-IR** (Björck 1967, *BIT* 7:257) | Transforms LS into the `(m+n)` augmented system `[[I,J],[Jᵀ,0]]`; refines to working precision; operates at κ(J), never κ(J)². | foundational |
| **GMRES-LSIR** (Carson, Higham & Pranesh 2020, *SIAM J. Sci. Comput.* 42:A4063) | Refinement step via GMRES preconditioned by J's QR factors; `M⁻¹Ã` well-conditioned even when κ(J) huge ⇒ extends solvable range. | **Track B workhorse** |
| **GMRES-LSIR → weighted LS** (Carson & Oktay 2024, arXiv:2401.03755) | Same for `min‖D½(b−Jx)‖`; matches our label-penalty weight `cw`. | Track B (our `cw`) |
| **IR-approach selection** (Carson & Daužickaitė 2024, arXiv:2405.18363) | Picks the IR variant by (κ, residual size). The observed LS residual is ≈1.6e-4; use B1 conditioning telemetry to decide whether standard LS-IR is enough or GMRES-LSIR is needed. | selection rule |
| **Preconditioned LSQR + incomplete-Cholesky** (Scott & Tůma 2025, arXiv:2504.07580) | Recovers fp64 accuracy on ill-conditioned LS even from single-precision factors; memory-lean. | Track B alt (memory) |
| **GMRES-IR for square `Ax=b`** (Carson & Higham 2017, *SIAM* 39:A2834) | Refines `Ax=b` via GMRES preconditioned by the LU factors. | **Track A (exact adjoint)** |

**Selection for our two solves.** Track A is the *square*, κ≈5.6e3 exact adjoint ⇒ start by
instrumenting the current one-correction GMRES path, then generalize refinement depth only if the
telemetry shows it helps. Track B is the LS penalty ⇒ Björck LS-IR on
`[J; √cw·I]` (standard variant, since residual is benign); escalate to GMRES-LSIR only if κ(J) grows
past the measured standard-IR comfort range at higher resolutions. **Building blocks already
in-stack:** `lineax` (LSMR, QR), `jax.scipy.sparse.linalg.gmres` — Track B changes formulation plus
refinement loop, not dependencies. Aligns with the existing GPD pattern (J-based inner solve); the lit
adds the IR layer that can certify working precision when residual telemetry confirms the assumptions.

## Assumptions

- The m18 exact adjoint's achieved `residual_relative` after the current one-correction path is above
  `1e-14`, finite, and reducible by additional refinement. **Confirm in A1 before coding.**
- The adjoint `apply_transpose` matvec is exact-enough that IR's residual computation is meaningful
  (it is the same operator used to define the system).
- The exact forward solve already converges at m18. Local artifacts show the chomp/clean lanes reach
  `‖b‖∞` of order 1e-15, while the baseline lane is looser (`~2e-10`) but still reports success; the
  Track A target surface is chomp unless the run matrix says otherwise.
- fp64 throughout; no fp32/mixed-precision path is needed here (κ ≪ 1/u_fp64).

## Implementation Plan

1. **Track A — Diagnose (confirm the mechanism before editing).**
   - [ ] **A1.** Instrument `_checked_boozer_linear_solve` (pod-local, non-committed) to print
     `status.residual_relative`, `status.residual`, `effective_tolerance`, and
     `linearization_kind` before the raise. Re-run the exact m18 adjoint
     (`exact_adjoint_demo.sh`) on the GPU and on CPU. Expected: `residual_relative ≈ 1e-13`
     (1–2 orders above `1e-14`) after the current one-correction path, `linearization_kind =
     "exact_jacobian"`, and finite residuals. If the residual is NaN/Inf, or already far below the
     gate, stop and diagnose a different failure.
   - [ ] **A2.** Record the exact call route:
     `_build_runtime_linear_solve_callbacks` exact branch →
     `_solve_jacobian_operator_with_status` → `_solve_square_array_system_operator_only`.
     Confirm no staged dense factors are used for Track A, and capture GMRES/correction iteration
     counts.

2. **Track A — Generalize the existing exact-adjoint residual refinement only if A1 supports it.**
   - [ ] **A3.** Replace the hard-coded single `refine` branch in `_solve_square_vector_system_operator_only`
     with a bounded trace-safe refinement loop, or add an exact-branch wrapper that calls the same
     square solver with a measured correction budget. Given base solution `x₀` and the available
     `apply_transpose` matvec, iterate `r = rhs − A·xₖ; dx = base_solve(r); xₖ₊₁ = xₖ + dx` until
     `residual_relative ≤ effective_tolerance` or the measured correction budget is exhausted. Prefer
     `lax.scan` with a fixed small bound for trace stability.
   - [ ] **A4.** Keep HEAD reproducibility explicit: one correction reproduces current behavior;
     zero corrections is a diagnostic no-refinement mode and is **not** byte-identical to HEAD. Avoid
     adding a permanent public env knob unless the run harness genuinely needs external control; a
     private option or test-only parameter is enough for bisecting.
   - [ ] **A5.** Preserve the byte-parity contract for dense PLU/factor paths. Track A exact-jacobian
     uses the operator callbacks, but the square solver is shared by LS fallback paths, so do **not**
     perturb `_traceable_solve_plu_linearization` (`surface_objectives_traceable.py` ~431-446) or dense
     Hessian status semantics while fixing exact-adjoint residuals.
   - [ ] **A6.** Ensure failure propagation is intact: if refinement does not reach tolerance within
     the measured correction budget, keep the all-NaN failure contract (`_solve_with_nan_on_failure`)
     so silent zero-gradient propagation cannot occur.
   - [ ] **A6b.** If extra correction passes do not monotonically reduce `residual_relative`, stop and
     switch to solver-family diagnosis (LSMR/GMRES settings, operator residual accuracy, or explicit
     materialization for a bounded small case). Do not lower `linear_solve_tolerance_floor`.

3. **Track A — Run the exact-adjoint benchmark.**
   - [ ] **A7.** Re-run the m18 exact adjoint on H100 (chomp converged surface). Confirm finite
     `J`, finite `dJ`, `success=True`, and capture compile wall, steady-state eval wall, peak GPU
     memory.
   - [ ] **A8.** Run the same on CPU and produce the exact-adjoint GPU-vs-CPU timing table
     (cold + steady-state). This is Track A evidence, not the final LS speed artifact unless the
     deliverable scope is explicitly narrowed to exact-adjoint timing.

4. **Track B — LS augmented-system / LM-LSMR path (required for the LS speed deliverable).**
   - [ ] **B1.** Prototype Björck/GMRES-LSIR on the LS penalty: solve/refine on the augmented
     `[[I, J],[Jᵀ, 0]]` (or `[J; √cw·I]`) system at the measured LS κ(J) instead of the κ(JᵀJ)
     penalty Hessian. Use `lineax` LSMR/QR + a refinement loop. Target: certify the LS solution to
     `‖∇f‖₂ ≤ 1e-11` (the upstream gate) *without* changing the gate. The current local artifact
     `.m18-adjoint-artifacts/log_lm_optx.txt` failed this gate (`1.431e-11`), so the next Track B run
     must close the final tolerance gap, not just report LM progress.
   - [ ] **B2.** Treat `SIMSOPT_ADJOINT_LINEAR_SOLVER=cg` as an LS-only bounded-memory experiment on
     the Hessian adjoint. It does not repair the exact-jacobian path and does not remove κ² by itself;
     if CG fails or is too slow, proceed to the J/LSMR or augmented-system plan.

## Validation Plan

- [ ] **V1 (mechanism).** A1 shows the current HEAD path (one correction) has finite
  `residual_relative > 1e-14` on the failing exact-adjoint baseline, plus telemetry for the base GMRES
  residual, correction residual, effective tolerance, and iteration counts. This proves
  tolerance-vs-residual-floor only if the residuals are finite and consistent with the κ·eps/n·eps
  scale.
- [ ] **V2 (fix, GPU).** Post-A3 m18 exact adjoint on H100: `success=True`,
  `residual_relative ≤ 1e-14`, `J`/`dJ` finite. Marker + `PROBE_RESULT_JSON`.
- [ ] **V3 (fix, CPU parity).** Same on CPU; `J` agrees with GPU to fp64 tolerance; `dJ` rel-L2
  agreement consistent with the documented CPU/JAX adjoint non-bit-parity caveat
  (`surface_objectives.py:1998-2005`).
- [ ] **V4 (no regression at low mpol).** Existing exact-adjoint tests (the resolutions that already
  passed) remain green with the default correction budget. A one-correction setting reproduces HEAD;
  a zero-correction setting is diagnostic only. Targets: `tests/geo/test_adjoint_cg_solver.py`,
  `tests/geo/test_surface_objectives_jax.py`,
  `tests/geo/test_boozersurface_jax.py`, `tests/geo/test_boozer_derivatives_jax.py`. Run on the pod's
  clean editable env: `PYTHONPATH=/workspace/simsopt /workspace/venv/bin/python -m pytest <files>`.
  ⚠️ Do **not** trust a bare local `python3 -m pytest`: in this working tree `import simsopt` resolves
  to the **simsopt-surrogate** repo via a `.pth` shadow (verified — the clean-repo-triad confound);
  even `PYTHONPATH=src` only redirects `simsopt_jax_adapters`, not native `simsopt`. Confirm
  `simsopt.__file__` points into this repo before trusting any local CPU run.
- [ ] **V5 (gradient correctness).** Finite-difference check of `dJ` vs `J` for a small DOF
  perturbation at m18, with at least a two-step perturbation sweep (`h`, `h/2`) to check slope
  stability and cancellation risk, per GPD numerical-convergence.
- [ ] **V6 (speed deliverable).** Exact-adjoint A8 timing table populated (GPU cold/steady, CPU,
  ratio). Do not use it as a proxy for the LS 96×96 speed deliverable.
- [ ] **V7 (Track B, if pursued).** LS m18 augmented-IR solve certifies `‖∇f‖₂ ≤ 1e-11` and the LS
  adjoint then runs; compare `J` to the exact-form `J`.
- [ ] **V8 (direct-vs-proxy guard).** Final m18 speed-demo completion requires the direct requested
  observable for the selected deliverable: exact-adjoint timing if the deliverable is narrowed to
  exact, or LS 96×96 forward+adjoint timing if the deliverable remains the single-stage LS path.
- [ ] **V9 (limit/recovery).** On a low-mpol exact case that already passes, the default correction
  budget must recover current HEAD `J`/`dJ` and status within tolerance; extra correction budget must
  not move `J`/`dJ` outside fp64 tolerance. For Track B, recover the existing low-resolution LS pass
  before promoting the m18 LS augmented-system run.

## Risks and Mitigations

- **Risk:** IR adds matvecs (BiotSavart JVP/VJP) per step → cost at m18.
  **Mitigation:** keep the correction budget measured and small; measure A7 wall. The exact path is
  operator-backed and does **not** have a reusable LU/QR preconditioner, so every extra correction is
  another GMRES solve, not a free triangular solve.
- **Risk:** IR loop breaks jit/trace-safety on the runtime-state path.
  **Mitigation:** prefer fixed-bound `lax.scan` for the small correction budget; use `lax.while_loop`
  only where reverse-mode through a dynamic loop is not required. Mirror existing traceable solver
  control flow (e.g. `_build_traceable_newton_polish_runner`).
- **Risk:** Altering the adjoint solve perturbs the byte-parity `(lu,piv)` contract or the forward
  solve.
  **Mitigation:** A5 — keep exact-adjoint changes at the exact-jacobian callback/square-solver seam
  and cover LS fallback callers. One correction is current HEAD; zero corrections are diagnostic, not
  byte-identical.
- **Risk:** Tolerance still not met if `residual_relative` floor is actually > 1e-14 even after IR
  (e.g. operator inexactness).
  **Mitigation:** if A1 shows the operator matvec itself limits accuracy, fall back to extended
  precision *residual only* (compute `r = rhs − A·x` in higher precision) per Björck, or accept a
  conditioning-aware tolerance (Option B) as documented secondary.
- **Risk:** Scope confusion between Track A and Track B.
  **Mitigation:** V8 gate — exact-adjoint success can close Track A, but the LS speed deliverable
  remains open until the direct LS 96×96 forward+adjoint run passes, unless the user explicitly
  narrows scope.

## Completion Criteria

- [ ] m18 exact Boozer adjoint returns finite `J` + `dJ` with `success=True` on GPU **and** CPU
  (V2, V3).
- [ ] The one-correction setting reproduces current HEAD; the default setting passes all
  previously-green adjoint tests (V4).
- [ ] FD gradient check passes at m18 (V5).
- [ ] GPU-vs-CPU exact-adjoint timing table produced (V6).
- [ ] If the requested deliverable remains the LS 96×96 single-stage path, Track B also passes
  forward+adjoint and produces the LS GPU-vs-CPU timing table (V7, V8).
- [ ] Change committed on the PR branch with the IR rationale + lit citations in the commit body;
  GPD pattern referenced.
- [ ] Plan updated: Track B either passes for the LS deliverable, or is explicitly deferred only after
  the deliverable scope is narrowed away from LS.

## Open Questions

- Should the exact-adjoint correction budget be hard-coded from A1 evidence or exposed privately for
  tests/run harnesses? (Lean: hard-code the minimal measured budget; avoid a permanent public env knob
  unless ops needs it.)
- Is the `1e-14` float64 `linear_solve_tolerance_floor` itself worth revisiting upstream (it is below
  the achievable relative-residual floor for κ·n at production resolutions), or is per-solve IR the
  better-scoped fix? (Lean: IR; do not touch shared policy.)
- Track B: does the user still require the **LS** 96×96 single-stage path as the speed demo? The
  current handoff says yes; only an explicit scope change makes exact-adjoint timing sufficient.
