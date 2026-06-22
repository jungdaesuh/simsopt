# Matrix-Free Inner-Solve Adjoint (Gate 4) — Implementation Plan

> Created 2026-06-22. Status: **DESIGN-NOTE DRAFT (not implementation signoff).** Motivated by the
> dense-linearization GPU OOM (chunk-bridged 2026-06-21/22), the jax-warm-vs-cpp slowness analysis,
> and the optax/torax architecture study. Scoped + corrected after an external review that (correctly)
> flagged the forward solve is already matrix-free.
> Doc-review pass 2026-06-22: all file:line refs verified against the working tree (codex edits had
> shifted several); the byte-parity enforcement site was located (it was NOT "refactored away").

## Purpose

Decide and (behind a flag) implement whether the **exact-Jacobian inner-Boozer lane** — the path that
materializes a dense final linearization, OOMs under XLA preallocation, and dominates the warm-eval
wall — should route its **adjoint** through the repo's already-existing matrix-free (operator-GMRES +
`custom_vjp` IFT) machinery instead of the dense factor. The decision hinges on the deliberate
forward/adjoint **byte-parity contract**, so this plan centers on an evidence-producing A/B, not a
blind swap.

## Goals

- A flag selecting **matrix-free vs dense** adjoint for the exact-Jacobian lane, default = dense
  (behavior-preserving), that reuses existing infra (no new solver written from scratch).
- A/B evidence on ≥1 production seed at `preallocate=true`: peak GPU memory, warm-eval wall,
  gradient parity (`grad_rel` vs cpp AND matrix-free-vs-dense delta), GMRES iteration count, and the
  byte-parity-contract delta.
- An explicit, evidence-backed decision: flip the lane's default to matrix-free (accepting the
  byte-parity tradeoff) OR keep dense + the chunk bridge (with rationale).

## Non-Goals

- The **forward** inner Newton hot loop — ALREADY matrix-free (JVP + GMRES), dense rebuilt only at the
  final iterate (`optimizer.py:5135-5136`, comment + `jvp_fn`). Not touched.
- The outer host-driven L-BFGS loop / per-step kernel (Gate 1; already in place via the
  `scipy-jax-decomposed` lane).
- Flipping the **repo-wide** `xla_gpu_preallocate` default (currently `False` for `jax_gpu_*` modes,
  `runtime.py`) — a separate policy decision, not proven safe by this plan.
- Removing the chunk bridge (`optimizer.py:3608`, `batch_size=8`) — it stays as the memory-safe
  fallback for the dense path.
- The second unbounded dense assembler `_apply_column_batched_operator` (`optimizer.py:4503/4507`) —
  tracked separately (mistake-book Pattern 81); only in scope if the chosen lane routes through it.

## Current Context (verified in this repo unless marked)

- **Forward solve already matrix-free:** `optimizer.py:5135-5136` — *"Jacobian-vector products,
  avoiding dense Jacobian materialization in the hot loop. The dense Jacobian is rebuilt once at the
  final iterate only for [the policy]."* Gated by `materialize_dense_linearization`
  (`optimizer.py:1893, 2024`). [verified]
- **The matrix-free machinery ALREADY EXISTS:**
  - Operator GMRES: `_run_operator_gmres` (`optimizer.py:4129`), `_gmres_solve_exact_newton_system`
    (`:4162`), `_gmres_solve_newton_system` (`:4151`). [verified]
  - IFT `@jax.custom_vjp` adjoints: `src/simsopt_jax/solve/minimize_runtime.py:62`,
    `src/simsopt_jax/core/_root.py:103` (the implicit-diff root boundary — this repo's analog of
    torax's `jax.lax.custom_root`), `src/simsopt_jax/geo/boozer_residual.py:146,166`. [verified]
  - A **matrix-free contract** already defaults off-dense: `optimistix/contracts.py:28`
    `materialize_dense_linearization: bool = False`. [verified]
- **The dense (byte-parity) lane:** `simsopt/contracts.py:65` `materialize_dense_linearization: bool =
  True`; set/threaded by `src/simsopt_jax/solve/dispatch.py:168-169,539`. [verified]
- **Dense linearization assembly:** `_materialize_dense_linear_operator` (`optimizer.py:3598`),
  chunked `batch_size=8` at `:3608` (2026-06-22 bridge; **uncommitted**). [verified]
- **GPU memory:** `runtime.py` defaults `xla_gpu_preallocate=False` for `jax_gpu_*` modes;
  `max_dense_jacobian_bytes` 256 MiB GPU. [verified]
- **Byte-parity contract (LOCATED):** the dense lane routes the forward AND adjoint solve through one
  shared packed `(lu, piv)` factor for byte-exact consistency — `_traceable_solve_plu_linearization`
  in `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py:431-446` (*"the forward and adjoint
  solves consume the same packed `(lu, piv)` factor bytes via `jsp_linalg.lu_solve`"*), per Phase-2
  contract `docs/parity_scientific_equivalence_contract_2026-05-09.md` §5.3 (punch-list
  `docs/lbfgs_ondevice_open_gates_punchlist.md` Gate 4). CORRECTION: a prior draft of this plan
  claimed the file was "refactored away into `simsopt_jax/`" — that is FALSE. The file exists (3762
  lines) in the `simsopt_jax_adapters` package; the punch-list's path was correct all along. [verified]
- **Reference (torax):** wraps the implicit Newton in `jax.lax.custom_root(f, initial_guess=x0,
  solve=…, tangent_solve=back)` (`torax/_src/solver/jax_root_finding.py:124-137`), implementing
  `tangent_solve` densely only because its root is ~100-dim; the boundary is where a matrix-free
  Krylov adjoint plugs in. [verified]
- **Measured/known:** exact-Jacobian κ≈625 (prior session, memory
  `project_exact_jacobian_conditioning_measured`); dense-path warm-eval at `preallocate=true` =
  jax 315s / cpp 236s, OOM-free post-chunk, `grad_rel` 1.47e-16, `value_abs_diff` 0.0 (this session,
  `m_fix`).

## Rationale

The forward is already matrix-free, so the dense final linearization is the only remaining dense
object — and it is the binding memory cost (OOMs `preallocate=true`) and (per the warm-vs-cpp
analysis) the dominant warm-eval term (~80 sequential per-column JVP batches after chunking). The
repo already contains a matrix-free adjoint path (operator GMRES + `custom_vjp` IFT) used by the
optimistix contract; the exact-Jacobian/simsopt lane simply defaults to the dense factor for the
byte-parity contract. A well-conditioned operator (κ≈625) should let GMRES converge in few iterations,
giving the adjoint without forming the ~600×600 factor — hypothesized strictly better on memory and
warm-time. The single real cost is byte-parity; the A/B must measure (a) that matrix-free gradients
stay within the parity-ladder tolerance and (b) the byte-parity delta, so the flip is a decision on
evidence, not assumption.

## Assumptions (explicit)

- ASSUMPTION: operator-GMRES on the residual JVP at the converged surface converges in few iterations
  for production seeds (inferred from κ≈625; NOT measured — validation item).
- ASSUMPTION: the existing matrix-free `custom_vjp`/operator-GMRES adjoint (used by the optimistix
  contract) can be selected for the exact-Jacobian/simsopt lane via a flag without rewriting it.
- ASSUMPTION: matrix-free adjoint gradient matches the dense-PLU adjoint within parity-ladder rtol
  1e-10 — to be measured.
- ASSUMPTION: the parity matrix + the exact-Jacobian production path use the dense (simsopt) contract
  (`materialize_dense_linearization=True`); consistent with the earlier ondevice-LM dense trace —
  re-confirm in Step 1.

## Implementation Plan

1. **Confirm wiring + reuse surface (read-only)**
   - [ ] Trace `dispatch.py:168/539` → which contract (simsopt dense vs optimistix matrix-free) the
         parity matrix and the production single-stage exact-Jacobian path resolve to.
   - [ ] Byte-parity enforcement is LOCATED at `surface_objectives_traceable.py:431-446`
         (`_traceable_solve_plu_linearization`, shared `(lu, piv)` factor for forward+adjoint). Trace
         whether the exact-Jacobian/`optimizer.py` dense linearization feeds this PLU solve or is a
         distinct dense object; record the relationship file:line.
   - [ ] Confirm the matrix-free adjoint (`minimize_runtime.py:62` / `core/_root.py:103` +
         `_gmres_solve_exact_newton_system` `optimizer.py:4162`) produces the inner-solve cotangent
         and can be selected independently of the forward.
2. **Add the flag (default = dense, behavior-preserving)**
   - [ ] Introduce one selector (a contract field or `SIMSOPT_*_ADJOINT=operator|dense` env) routing
         the exact-Jacobian lane's adjoint to operator-GMRES vs the dense factor; default `dense`.
   - [ ] When `operator`: set `materialize_dense_linearization=False` for that lane and route the
         adjoint through the existing `custom_vjp` + `_run_operator_gmres`; leave the forward GMRES
         untouched.
   - [ ] Keep the dense path + chunk bridge intact as the default/fallback.
3. **A/B harness**
   - [ ] Run ONE production seed (april285) BOTH ways at `preallocate=true`, recording: peak GPU,
         warm wall, `grad_rel` vs cpp, matrix-free-vs-dense gradient delta, GMRES iters, byte-parity
         delta. Extend `benchmarks/run_parity_matrix_pod.sh` or a focused probe.

## Validation Plan

- [ ] **Gradient parity:** matrix-free adjoint grad vs dense adjoint grad rel-diff ≤ parity-ladder
      rtol 1e-10, AND vs cpp `grad_rel` ~1e-16, on april285.
- [ ] **Memory:** matrix-free runs at `preallocate=true` (full 75% pool) with **no CUBIN OOM even
      UN-chunked**, peak GPU < dense-chunked peak.
- [ ] **Warm-time:** matrix-free warm wall vs dense-chunked warm wall (hypothesis: matrix-free ≪).
- [ ] **GMRES iters recorded** (validate the "few iters" assumption from κ≈625; check across ≥2
      seeds of differing conditioning).
- [ ] **Byte-parity behavior:** measure/record whether/how much the operator adjoint breaks the
      shared forward+adjoint factor byte-consistency (the decision input).
- [ ] **Tests green:** `test_materialize_dense_linear_operator_matches_linear_map` + the inner-solve
      adjoint tests under `./.conda-env/bin/python`.
- [ ] **Crucible PASS** on the diff.
- Commands: pod A/B via `run_parity_matrix_pod.sh cuda <out>.json jax-gpu --seeds .../april285`
  (both flag values, `preallocate=true`, with warm eval); local lint + adjoint tests.

## Risks and Mitigations

- Risk: operator-GMRES adjoint breaks the deliberate forward/adjoint **byte-parity** contract.
  Mitigation: dense stays default; matrix-free behind a flag; flip only on measured byte-delta +
  parity evidence.
- Risk: GMRES does not converge in few iters for ill-conditioned seeds.
  Mitigation: record iters across seeds; preconditioner or cap-and-fall-back to dense (chunked).
- Risk: concurrent (codex) edits to `optimizer.py` / the solve contracts.
  Mitigation: re-read before editing; scoped edits; coordinate (HEAD `223ddb37c` + in-flight WIP).
- Risk: the matrix-free gradient drifts beyond parity tol.
  Mitigation: the A/B gate (rtol 1e-10) blocks the flip; dense remains default.

## Completion Criteria

- [ ] Flagged matrix-free adjoint implemented + lint-clean; dense default unchanged.
- [ ] A/B evidence table (memory, warm-time, grad parity, GMRES iters, byte-parity delta) on ≥1
      production seed at `preallocate=true`.
- [ ] Crucible PASS on the diff.
- [ ] Explicit recorded decision: flip default to matrix-free (accept byte-parity tradeoff) OR keep
      dense + chunk (with rationale).

## Open Questions

- Is the forward/adjoint byte-parity contract still required by any live consumer, or can it be
  relaxed for the exact-Jacobian lane? (Owner: user / parity-contract decision.)
- Does the parity matrix's ondevice-LM lane share the same adjoint factor path as the Newton-Krylov
  lane, or need separate wiring? (Resolve in Step 1.)
- If matrix-free wins, does that also unblock revisiting the repo-wide `preallocate=false→true`
  policy, or keep them independent decisions?

## Provenance of cited numbers (per external-review request)

- κ≈625: measured **prior** session (memory `project_exact_jacobian_conditioning_measured`), not re-run.
- ≈79s jax-vs-cpp warm gap (jax 315.23s − cpp 235.83s), nphi255 CPU/GPU crossover: measured **this**
  session (`m_fix`, `corner_ms`) with ~25s pod load-variance caveat.
- "few GMRES iters", ~80 JVP batches: **inferences** (from κ and N/batch_size), NOT measured — they
  are validation items above, not facts.
