# Parity Scientific-Equivalence Contract — Plan

- Date: 2026-05-09
- Branch: `gpu-purity-stage2-20260405`
- Status: Phase 0 + Phase 1 reporting skeleton landed in `2f71d5afa`
  (schema-drift fix in `1f1adfc42`). Phase 1.5 reporting wiring,
  Phase 2 LS factor-once adjoint hybrid, and Phase 3
  `cpp_compatible_probe` harness all landed in the working tree on
  2026-05-10 (uncommitted; awaits final commit). Phase 1.5
  calibration sweep + tolerance lock remains deferred — it requires
  production runs against `.artifacts/parity/` corpus. Phase 5
  (Skeel/EW/Hager–Higham) remains deferred per §9.
- Companion docs:
  - `docs/parity_dual_mode_contract_2026-05-08.md`
    (existing dual-mode runtime contract; this plan adds a **second-axis
    contract** — scientific equivalence — that lives alongside the strict
    byte-identity gate, not in place of it)
  - `docs/boozer_bfgs_pre_newton_contract_impl_plan_2026-05-07.md`
    (open pre-Newton root-cause workstream; this plan does NOT replace it)
  - `docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`
    (Phase 4 strict-byte-identity workstream remains a release blocker)
  - `benchmarks/validation_ladder_contract.py` (existing
    `PARITY_LADDER_TOLERANCES` and `OPTIMIZER_DRIFT_TOLERANCES` SSOT;
    this plan adds two new ladder lanes, `ls-solve-quality` and
    `exact-solve-quality`)
  - `CLAUDE.md` (sections "Adjoint / warm-start operator solves" and
    "Exact Boozer scaling-limit contract" — amendment notes below)

## 0. Purpose

Move the C++ vs JAX Newton-polish + final-output parity contract from
"byte-identity to C++" to **scientific equivalence under a two-lane
architecture**. The reference lane supplies C++-oracle regression
evidence that JAX solves the same problem within condition-aware
tolerances; the production lane is free to use JAX-native
operator-backed solves wherever JAX is genuinely better.

This plan is **additive**:

- The strict release-blocker exit at
  `benchmarks/single_stage_init_parity.py:3047-3051` and the
  pre-Newton hard gate at `:2000-2030` remain unchanged
  (per the dual-mode contract walk-back of 2026-05-08).
- Phase 4 byte-identity work
  (`docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`)
  remains unchanged.
- The pre-Newton BFGS root-cause work
  (`docs/boozer_bfgs_pre_newton_contract_impl_plan_2026-05-07.md`)
  remains the upstream investigation; this plan does NOT close that
  gap.

What this plan adds is a **complementary scientific-equivalence
ladder** that the production-operator lane can target without claiming
byte-identity, so the JAX path can ship operator-backed Newton/Krylov
without giving up provable parity to the C++ oracle.

## 1. Two-Lane Architecture

The current dual-mode contract is **runtime-mode dual** (`*_parity`
verification vs `*_fast` speed-opt-out) under a **single byte-identity
gate**. This plan adds an orthogonal axis: **algorithmic-lane dual**.

The two lanes are scoped distinctly:

- **`production_operator`** is the existing default user-facing
  `BoozerSurfaceJAX` algorithm. No new constructor parameter is
  introduced. Live behavior today: residual-gated GMRES on
  HVP/JVP operator solves (`optimizer_jax.py:1866-1905, 2366`),
  conditional iterative refinement (`optimizer_jax.py:2497-2515`),
  no globalization. Forward-error-gated refinement (Skeel/FERR) and
  Eisenstat–Walker INB backtracking are **future work** (Phase 5),
  not properties of the current lane.

- **`cpp_compatible_probe`** is a **harness-only diagnostic mode**
  exposed through the parity benchmark, not a user-facing
  `BoozerSurfaceJAX(lane=...)` product path. It exists to materialize
  the dense reference Newton trajectory for comparison against the
  C++ oracle, and operates on host arrays so the linear-solve bytes
  can be compared to LAPACK-bytes from `boozersurface.py:1129, 1668`.

| Lane                    | Purpose                                            | Linear solver (today)                  | Iterative refinement (today)         | Globalization (today) |
|-------------------------|----------------------------------------------------|----------------------------------------|--------------------------------------|-----------------------|
| `production_operator`   | JAX-native at scale (existing default user lane)   | Operator GMRES on HVP/JVP              | Conditional, GMRES-residual gated   | None                  |
| `cpp_compatible_probe`  | Harness-only diagnostic vs C++; not a user lane    | Host `np.linalg.solve` on materialized H/J via `optimizer_backend="scipy"` skeleton + harness adapter | LS: conditional, gated on `norm < 1e-9` (matches `boozersurface.py:1130`); Exact: unconditional Wilkinson (matches `boozersurface.py:1669`) | None (matches C++)    |

Future-work properties (Phase 5) are tracked separately and are NOT
implied by the lane labels above.

The two algorithmic lanes are orthogonal to the runtime modes:

```
                  cpp_compatible_probe      production_operator
  jax_cpu_parity   strict byte gate          scientific-equivalence gate
  jax_cpu_fast     n/a (probe not run        scientific-equivalence gate
                   in *_fast)                (relaxed reporting)
  jax_gpu_parity   strict byte gate          scientific-equivalence gate
  jax_gpu_fast     n/a                       scientific-equivalence gate
                                             (relaxed reporting)
```

`native_cpu` (C++ reference) remains the oracle in both lanes.

## 2. Acceptance Gates

LS Newton polish and BoozerExact solve **different mathematical
systems**. LS minimizes the scalar objective `½‖r‖² + constraints`,
so its solve-quality fields are Hessian/gradient based. Exact solves
the residual equation `r(x) = 0`, so its solve-quality fields are
Jacobian/residual based. The contract therefore splits into two
gate sets, evaluated independently per fixture.

### 2.1 LS Newton polish gates

Path: `boozersurface.py:1127-1163` (CPU oracle); LS branch of
`BoozerSurfaceJAX` via `_make_traceable_newton_polish_runner`
(`optimizer_jax.py:2310-2351`).

| #  | Gate                                                            | Tolerance (`cpp_compatible_probe`)  | Tolerance (`production_operator`) |
|----|-----------------------------------------------------------------|-------------------------------------|------------------------------------|
| L1 | Same inputs (surface DOFs, iota, G, coils, quadrature, weights) | byte-identical                      | byte-identical                     |
| L2 | Same residual vector + scalar objective                         | rtol=1e-12, atol=1e-14              | rtol=1e-10, atol=1e-12             |
| L3 | Same gradient                                                   | rtol=1e-10, atol=1e-12              | rtol=1e-9, atol=1e-11              |
| L4 | Same Hessian *action* on a deterministic probe set              | rtol=1e-9, atol=1e-11               | rtol=1e-8, atol=1e-10              |
| L5 | Same Newton step solve residual ‖H·dx − g‖/‖g‖                  | rtol=1e-10                          | rtol=1e-8                          |
| L6 | Same final Boozer state (iota, G, surface DOFs)                 | rtol=1e-10, atol=1e-12              | rtol=1e-9, atol=1e-11              |
| L7 | Same outer objective + gradient                                 | rtol=1e-10                          | rtol=1e-9                          |
| L8 | Same outer-optimizer trajectory                                 | gates L1–L7 hold per-step           | gates L1–L6 hold per-step          |

### 2.2 BoozerExact gates

Path: `boozersurface.py:1640-1722` (CPU oracle); exact branches of
`BoozerSurfaceJAX` via `newton_exact` (non-traceable) and
`newton_exact_traceable` (production traceable) at
`optimizer_jax.py:2473-2672`.

| #  | Gate                                                            | Tolerance (`cpp_compatible_probe`)  | Tolerance (`production_operator`) |
|----|-----------------------------------------------------------------|-------------------------------------|------------------------------------|
| E1 | Same inputs (surface DOFs, iota, G, coils, quadrature, weights) | byte-identical                      | byte-identical                     |
| E2 | Same nonlinear residual vector r(x)                             | rtol=1e-12, atol=1e-14              | rtol=1e-10, atol=1e-12             |
| E3 | Same Jacobian *action* J·v on a deterministic probe set         | rtol=1e-9, atol=1e-11               | rtol=1e-8, atol=1e-10              |
| E4 | Same Newton step linear residual ‖J·dx − b‖/‖b‖ where `b` is the augmented residual vector and Newton step is `x ← x − dx` (matches CPU sign convention at `boozersurface.py:1645,1668-1670`) | rtol=1e-10 | rtol=1e-8 |
| E5 | Same iterative-refinement correction direction (when applied)   | rtol=1e-9                           | reporting-only (operator lane is conditional) |
| E6 | Same final Boozer state (iota, G, surface DOFs)                 | rtol=1e-10, atol=1e-12              | rtol=1e-9, atol=1e-11              |
| E7 | Same wrapper adjoint solve residual ‖J^T λ − u‖/‖u‖             | rtol=1e-10                          | rtol=1e-8                          |
| E8 | Same outer objective + gradient + outer trajectory              | gates E1–E7 hold per-step           | gates E1–E6 hold per-step          |

The load-bearing change vs the current contract is gates **L4 / E3**:
today the parity arbiter compares dense LU `(P, L, U)` bytes
(`benchmarks/single_stage_init_parity.py:215-252`); under this plan
the arbiter compares operator *action* on a deterministic probe set,
not dense bytes. This permits the `production_operator` lane to use
HVP-built or operator-only Hessians/Jacobians without breaking the
diagnostic. **L4 / E3 are smoke-regression diagnostics on `k+1 ≤ 9`
probe directions, not proofs of operator equivalence** — see §4 for
the proof framing and the cross-reference to the existing
`direct-hessian-oracle` lane.

**Initial enforcement status.** Phase 1 landed the new ladder
lane, result-schema fields, and the deterministic operator-action
probe helpers (commits `2f71d5afa` + `1f1adfc42`). Phase 1.5 added
the per-pair arbiter wiring that emits `ls_hessian_action_max_rel`
and `exact_jacobian_action_max_rel` into the parity artifact JSON
under `solve_quality_probes`. Gates L4 / L5 / E3 / E4 / E5 / E7
remain **reporting-only** until the calibration sweep populates the
corpus and the §2 tolerance schedule is locked. The existing strict
gates (`linear_solve_factors`, `final_hessian`, and the pre-Newton
blocker — see §8) remain authoritative until they are formally
retired by a follow-up amendment after the new ladder is populated
and calibrated.

## 3. Solve-Quality Field Sets

The current `linear_solve_factors` byte-parity probe at
`benchmarks/single_stage_init_parity.py:192-197, 246-252` is
**augmented**, not replaced, until §2's calibration corpus
materializes and the existing strict gate is formally retired.

`linear_solve_factors` itself is **not** debug-only in live code —
the SciPy reference runtime callbacks build
`H_host = P @ L @ U` and use it as `apply_forward`/`apply_transpose`
(`boozersurface_jax.py:3418-3475`), and the traceable adjoint
`_traceable_solve_plu_linearization` consumes the PLU factors for
triangular solves (`surfaceobjectives_jax.py:3017-3055`). The factor
storage is therefore load-bearing for those code paths and must
continue to be emitted faithfully. This plan adds new fields *next
to* `linear_solve_factors`; it does not remove or relabel it.

### 3.1 LS solve-quality fields (gates L4–L5)

| Field                           | Definition                                                          | Source path                                          |
|---------------------------------|----------------------------------------------------------------------|------------------------------------------------------|
| `ls_hessian_symmetry_rel`       | `‖H − H.T‖_F / ‖H‖_F`                                                | computed from the final materialized Hessian in `boozersurface_jax.py` after `optimizer_jax._materialize_dense_hessian(..., symmetrize=True)` |
| `ls_hessian_action_max_rel`     | max over deterministic probe set of `‖H_jax v − H_cpp v‖ / ‖H_cpp v‖` | parity arbiter, see §4                              |
| `ls_newton_linear_residual_rel` | `‖H·dx − g‖ / ‖g‖`                                                  | computed at the Newton step site post-solve         |
| `ls_newton_step_abs_diff_rel`   | `‖dx_jax − dx_ref‖ / max(‖dx_ref‖, ε)` against seeded reference dx  | parity arbiter, see §4                              |
| `ls_factorization_backend`      | string ∈ {`lapack-dgetrf`, `cusolver-getrf-ffi`, `dense-plu-shared`} (SciPy/JAX CPU LU is LAPACK-backed; the CUDA label denotes the intended cuSOLVER-backed JAX device path and must be hardware-proven before enforcement) | result-dict assignment in `boozersurface_jax.py` |
| `ls_condition_estimate`         | Hager–Higham 1-norm condition number of H (operator matvecs)         | new helper near `optimizer_jax.py:1899`              |

### 3.2 Exact solve-quality fields (gates E3–E5, E7)

| Field                                | Definition                                                         | Source path                                          |
|--------------------------------------|---------------------------------------------------------------------|------------------------------------------------------|
| `exact_jacobian_action_max_rel`      | max over probe set of `‖J_jax v − J_cpp v‖ / ‖J_cpp v‖`             | parity arbiter, see §4                              |
| `exact_newton_linear_residual_rel`   | `‖J·dx − b‖ / ‖b‖` where `b` is the augmented residual vector and Newton step is `x ← x − dx` (matches CPU sign convention at `boozersurface.py:1645,1668-1670`) | computed at the Newton step site post-solve |
| `exact_refinement_correction_rel`    | `‖dx_after_IR − dx_before_IR‖ / max(‖dx_before_IR‖, ε)` (per-iter)  | optimizer_jax.py around `:2497-2515, 2587-2623`      |
| `exact_adjoint_solve_residual_rel`   | `‖J^T λ − u‖ / ‖u‖` measured at adjoint solve completion            | `surfaceobjectives_jax.py` adjoint exit             |
| `exact_factorization_backend`        | string ∈ {`lapack-dgetrf`, `cusolver-getrf-ffi`, `operator-gmres`} (operator GMRES is the runtime path; the LAPACK / cuSOLVER aliases are reserved for the Phase 3 `cpp_compatible_probe` harness reference solver) | result-dict assignment |
| `exact_condition_estimate`           | Hager–Higham 1-norm condition number of J (operator matvecs)        | new helper near `optimizer_jax.py:1899`              |

The Hager–Higham implementation must be JAX-native (see §6); a
placeholder `None` is acceptable in Phase 1 with population in Phase
5.3.

## 4. Operator-Action Probe Specification

Gates L4 (Hessian action) and E3 (Jacobian action) require a fixed
probe set so the comparison is **process-stable across runs and
machines**.

**Probe construction.**
- Decision-vector dimension: `n = decision_size` (per `boozersurface_jax.py:2196` pattern).
- Probe count: `k = min(8, n)`.
- Seed: deterministic and **process-stable**. Python's builtin
  `hash()` is randomized by `PYTHONHASHSEED` and is unsafe here.
  Use sha256:
  ```python
  import hashlib
  seed = int.from_bytes(
      hashlib.sha256(artifact_name.encode("utf-8")).digest()[:4],
      byteorder="little",
  )
  rng = np.random.default_rng(seed)
  ```
  Draw `k` Gaussian probes, orthonormalize via QR.
- Also include `e_0` (first standard basis vector) as probe `k+1` —
  pins one diagonal-direction comparison.

**Comparison metric.** For each probe `v_i`, compute `Op_jax · v_i`
and `Op_cpp · v_i` (Op = H for LS gate L4, J for Exact gate E3).
Report the maximum relative error across the probe set.
Standard-basis-only probes are explicitly forbidden — they collapse
the operator-action gate to dense-bytes parity, defeating the point.

**Status.** This is a **smoke / regression test of operator equality**
on `k+1 ≤ 9` directions, **not a proof**. The rigorous proof method
remains the existing `direct-hessian-oracle` lane in
`benchmarks/validation_ladder_contract.py:86` (column-complete CPU/C++
basis sweep at `rtol=1e-8, atol=1e-10`); the LS gate L4 in this plan
is a complementary fast diagnostic that does NOT supersede that lane.
Per `docs/jax_parity_manifest.md:108` and the existing operator-path
HVP test, full operator parity is established through the existing
direct-hessian-oracle column-complete sweep — this plan's gate L4
adds a cheaper continuous-monitoring signal.

## 5. Lane Implementation Notes

### 5.1 `cpp_compatible_probe` (harness-only diagnostic)

This is **not** a `BoozerSurfaceJAX(lane=...)` constructor parameter
or a user-facing product surface. It is a parity-benchmark harness
mode that constructs a dense reference Newton trajectory for
comparison against the C++ oracle. The harness owns this code path;
the production `BoozerSurfaceJAX` is unchanged.

For LS Newton polish (harness skeleton):

- Use the existing `optimizer_backend="scipy"` pathway as the dense
  oracle skeleton (`boozersurface_jax.py:4882-4884`). This pathway
  already routes through host `np.linalg.solve` and `scipy.linalg.lu`,
  matching CPU bytes within LAPACK pivot tie-breaks.
- Apply mirror-upper symmetrization at `optimizer_jax.py:1632` so the
  LU input is bit-symmetric.
- Keep `dense_newton_steps=True` so Newton's per-iter solve uses
  host `np.linalg.solve` (`optimizer_jax.py:1729-1735`).

For BoozerExact (harness skeleton — requires new code, see caveats
below):

- The exact normalizer at `boozersurface_jax.py:3097` currently
  pops `optimizer_backend` for `boozer_type == "exact"`, so today the
  exact path has no `optimizer_backend="scipy"` channel. The harness
  cannot route through the existing scipy skeleton; it must materialize
  a dedicated host-resident exact reference solver.
- This solver lives **in the harness**, not in
  `BoozerSurfaceJAX`. Implementation:
  - Build the exact residual `r(x)` and Jacobian `J(x)` on JAX (reusing
    `_jacobian_linear_operator` and the residual closure from
    `optimizer_jax.py:2113-2138`).
  - Materialize `J` to a host `np.ndarray` via
    `np.asarray(jax.jit(jax.jacobian(r))(x))`.
  - Apply C++-equivalent Newton: assemble the augmented residual
    `b = [r[mask], label.J() − target_label, …]` (matching
    `boozersurface.py:1645-1648`); solve `dx = np.linalg.solve(J, b)`;
    apply **unconditional** Wilkinson refinement
    `dx += np.linalg.solve(J, b − J @ dx)` matching
    `boozersurface.py:1669`; step `x -= dx` matching `:1670`.
  - **No** monotone-norm guard.
  - Device `jnp.linalg.solve` is **forbidden** in this path — it does
    not match LAPACK bytes.
- This solver is for diagnostic comparison only; nothing in the
  production single-stage pipeline calls it.

### 5.2 `production_operator` (the existing default user lane)

Unchanged from current default. The existing
`newton_exact_traceable` (`optimizer_jax.py:2559-2672`) and the M5 IFT
adjoint operator-backed contract (CLAUDE.md "Adjoint / warm-start
operator solves") are preserved. Live algorithmic properties **today**:

- Linear solve: residual-gated GMRES on operator HVP/JVP
  (`optimizer_jax.py:1866-1905, 2366`).
- Iterative refinement: conditional, gated on GMRES linear residual
  (`optimizer_jax.py:2507-2515` non-traceable;
  `optimizer_jax.py:2610-2615` traceable).
- Globalization: none on the traceable production path; a strict
  monotone-norm guard exists on the legacy non-traceable
  `newton_exact` only (`optimizer_jax.py:2519-2524`).

Phase 5 may add a JAX-native Skeel forward-error gate (replacing the
GMRES-residual gate) and Eisenstat–Walker INB backtracking. Those are
production improvements deferred to a future workstream and **are
not properties of `production_operator` today**.

### 5.3 Adjoint factor-once hybrid

For the `production_operator` lane, when
`decision_size² × 8 ≤ max_dense_jacobian_bytes` (default 512 MB,
`boozersurface_jax.py:2734`), share `(lu, piv)` between LS forward
and adjoint solves via `jax.custom_vjp` with `lax.stop_gradient` on
the factors. The dense `(P, L, U)` triple is emitted from the same
factorization for the public `linear_solve_factors` reporting field
and continues to be load-bearing for the LS reference callbacks
(`boozersurface_jax.py:3418-3475`) and the traceable adjoint PLU
solve (`surfaceobjectives_jax.py:3017-3055`).

Above the byte-budget, fall back to the existing operator-only
adjoint, preserving the CLAUDE.md "exact JAX never falls back to
dense factors" guarantee for large-n exact problems.

In the LS path the `(P, L, U)` triple is **already load-bearing**
runtime data (consumed by the SciPy reference callbacks at
`boozersurface_jax.py:3418-3475` and the traceable adjoint at
`surfaceobjectives_jax.py:3017-3055`); Phase 2 unifies the forward and
adjoint solves onto the same factor bytes via `lu_factor`/`lu_solve`
without changing the load-bearing status of those bytes.

This closes the existing `iota_penalty.adjoint = 2.32e-10` drift slice
(DM-E #1) at typical SIMSOPT decision sizes (n ≤ ~5000) without
breaking the scaling-limit contract.

## 6. JAX-Native Implementation Discipline

All new code paths must use JAX-native primitives. Specifically:

- Loops: `lax.while_loop` / `lax.fori_loop` / `lax.scan`. The codebase
  already has 4 `lax.while_loop` sites (`optimizer_jax.py:1143, 1228,
  2405, 2625`); follow that pattern.
- IFT: `jax.custom_vjp` with residuals carrying factor state under
  `jax.lax.stop_gradient`.
- Transpose solves: `jax.scipy.linalg.lu_solve(lu_piv, b, trans=1)`,
  not manual `_piv_from(P)` reconstruction; or
  `jax.linear_transpose(matvec)` for operator transposes.
- Dynamic indexing: `jnp.where(jnp.arange(n) == j, 1.0, 0.0)` or
  `lax.dynamic_update_slice`, **never** `int(jnp.argmax(...))` inside
  a JIT-traced loop.
- Factor reuse: `jax.scipy.linalg.lu_factor` + `lu_solve` for runtime,
  `jax.scipy.linalg.lu` only for debug metadata.

Forbidden in hot paths: `int()`/`float()`/`bool()` casts (force
host roundtrips), Python `for` loops with traced bounds, `np.asarray`
on JAX arrays inside JIT regions. The existing
`int(np.asarray(jnp.asarray(leaf).size))` patterns at
`optimizer_jax.py:1230, 1232, 1302, 1537, 1539, 2005, 2062, 2116-2117,
2196, 2343` are at static-shape boundaries (outside JIT) and are fine.

No `lineax` / `optimistix` dependency is needed; all patches stay
within `jax` / `jax.scipy.linalg` plus the existing
`_run_operator_gmres` seam (`optimizer_jax.py:1899-1905`).

## 7. CLAUDE.md Amendments

Two sections require updating once Phase 2 lands:

### "Adjoint / warm-start operator solves"

Current: "JAX wrapper adjoints and traceable warm-start predictors
use operator-backed linear solves by contract."

Amended: "JAX wrapper adjoints and traceable warm-start predictors
use operator-backed linear solves by default, with the following
exception: when `decision_size² × 8 ≤ max_dense_jacobian_bytes`, the
LS forward and adjoint solves consume the same `(lu, piv)` factors
stored under `lax.stop_gradient` to ensure bit-equal forward/adjoint
Hessian action. The LS `(P, L, U)` field is load-bearing runtime data
(see `boozersurface_jax.py:3418-3475`,
`surfaceobjectives_jax.py:3017-3055`); the **exact** lane's `(P, L, U)`
remains debug metadata only, and `BoozerSurfaceJAX.get_adjoint_runtime_state()`
remains the runtime SSOT for the exact-lane adjoint."

### "Exact Boozer scaling-limit contract"

Current: "exact JAX never falls back to dense factors; batched exact
adjoints solve one RHS at a time through the same operator seam."

Amended: "the `production_operator` exact lane never falls back to
dense factors at runtime. The `cpp_compatible_probe` harness
materializes a dense host-resident reference exact solver for
diagnostic comparison only; it is not exposed through the
`BoozerSurfaceJAX` user API and the exact normalizer at
`boozersurface_jax.py:3097` continues to strip
`optimizer_backend` from the user-visible exact path. Batched exact
adjoints in `production_operator` solve one RHS at a time through the
same operator seam."

### Note on `linear_solve_factors`

The CLAUDE.md "Adjoint / warm-start operator solves" rule that
"dense PLU data in exact results is public/debug metadata" applies to
the **exact** lane. In the **LS** lane, the SciPy reference runtime
callbacks at `boozersurface_jax.py:3418-3475` build `H_host = P @ L @ U`
from `self.res["PLU"]` and use it as `apply_forward`/`apply_transpose`,
and the traceable adjoint `_traceable_solve_plu_linearization` at
`surfaceobjectives_jax.py:3017-3055` consumes the PLU factors for
triangular solves. In those LS paths, `linear_solve_factors` is
load-bearing runtime data, not metadata. This plan does not
change that — it only changes how the parity arbiter compares
the *bytes* of those factors across lanes.

## 8. Pre-Newton BFGS Root Cause (Hard Block on Production CI)

The headline 4.5e-9 `pre_newton_state` and 1.58e-10
`pre_newton_objective_gradient` divergences are upstream of every
Newton-polish artifact. **This plan does not close them and does not
relax the live blocker on them.**

The strict pre-Newton gate is implemented at
`benchmarks/single_stage_init_parity.py:2000-2030`
(`_pre_newton_census_gate_failures`) and produces a hard
`SystemExit(1)` failure at `:3047-3051` for any divergent
`boozer_solve.pre_newton_*` layer. That gate is per the existing
`docs/parity_dual_mode_contract_2026-05-08.md` §2.4 / §11.5 contract
and per `docs/boozer_bfgs_pre_newton_contract_impl_plan_2026-05-07.md`.
**Production CI cannot turn green until that gate goes green**,
regardless of any solve-quality reporting added by this plan.

What this plan adds is **complementary downstream diagnostic
coverage**: even while the pre-Newton blocker is unresolved, the
LS gates L4–L7 and Exact gates E3–E7 can be *measured and reported*
on candidates whose pre-Newton split is still failing. This gives
engineers visibility into whether downstream Newton-polish behavior
has independent regressions. It does **not** convert pre-Newton
into a passable layer.

When the pre-Newton work
(`docs/boozer_bfgs_pre_newton_contract_impl_plan_2026-05-07.md`)
lands, gate L3 / E2 tolerances in this plan can tighten to match the
post-fix empirical baseline; until then the L3 / E2 thresholds
above are upper bounds, not floors.

## 9. Phase Plan and Dependencies

```
Phase 0 (this doc + ladder + CLAUDE.md note)
  ├── Phase 1 (Class B symm + reporting fields)
  │     └── Phase 2 (factor-once adjoint hybrid)
  ├── Phase 3 (cpp_compatible_probe harness solver)  ── independent
  └── Phase 5.x (production-only enhancements)        ── after 1+2+3 stable

Phase 4 (BFGS root cause) ── independent investigation, blocks gate L3 / E2 final tolerance only
```

### Phase 0 deliverables (landed in `2f71d5afa`)
- `docs/parity_scientific_equivalence_contract_2026-05-09.md` (this file)
- Two new ladder lanes in `benchmarks/validation_ladder_contract.py`
  (slots into the existing `PARITY_LADDER_TOLERANCES` SSOT; the
  unknown-lane rejection test accepts both keys):
  ```python
  PARITY_LADDER_TOLERANCES["ls-solve-quality"] = {
      "ls_hessian_symmetry_rel_tol": 1e-10,
      "ls_hessian_action_max_rel_tol": 1e-8,
      "ls_newton_linear_residual_rel_tol": 1e-8,
      "ls_newton_step_abs_diff_rel_tol": 1e-8,
      "ls_condition_estimate_present": False,  # placeholder; flip True in Phase 5.3
  }
  PARITY_LADDER_TOLERANCES["exact-solve-quality"] = {
      "exact_jacobian_action_max_rel_tol": 1e-8,
      "exact_newton_linear_residual_rel_tol": 1e-8,
      "exact_adjoint_solve_residual_rel_tol": 1e-8,
      "exact_condition_estimate_present": False,  # placeholder; flip True in Phase 5.3
  }
  ```
- CLAUDE.md amendment notes (do not land amendments until Phase 2 ships).
- Both lanes are **reporting-only** at Phase 0; no arbiter wiring,
  no enforcement, no removal of the existing `linear_solve_factors`
  byte-parity probe.

### Phase 1 deliverables (landed in `2f71d5afa`)
- `optimizer_jax.py` — mirror-upper symmetrization with `symmetrize=True`
  default for materialized Hessian reporting. Dense Newton-step
  application remains operator-policy controlled and is not a
  byte-identity claim by itself.
- `boozersurface_jax.py` — emit the LS and Exact solve-quality fields
  into public and traceable result schemas alongside the existing
  `linear_solve_factors`.
- `single_stage_banana_example.py` — propagate the reporting keys into
  parity summaries where result metadata is already collected.
- `benchmarks/parity_solve_quality.py` — provide deterministic
  operator-action probe helpers. Full `single_stage_init_parity.py`
  arbiter rows for solve-quality probes remain Phase 1.5 / Phase 2
  calibration work and are not enforced in Phase 1.
- `tests/test_benchmark_helpers.py` — add the two new lane keys to the
  accepted set.
- `*_condition_estimate` field emits `None` placeholder; the
  matching `*_condition_estimate_present` ladder key stays `False`
  until Phase 5.3 lands the Hager–Higham implementation, at which
  point both flip together.

### Phase 1.5 deliverables (after calibration)
- Calibration sweep per §10 risk register #4. **Status:** deferred —
  requires production CPU/GPU runs against the
  `.artifacts/parity/` corpus. The Phase 1.5 reporting wiring below
  emits the metrics needed to seed the calibration corpus once
  production runs are available.
- Lock the §2 tolerance values in `PARITY_LADDER_TOLERANCES`. **Status:**
  pending calibration. Today the §2 thresholds remain the documented
  upper bounds; the `ls_solve_quality` and `exact_solve_quality` lanes
  retain `reporting_only=True` until the corpus completes.
- Promote LS gates L4 / L5 and Exact gates E3 / E4 / E5 / E7 from
  reporting-only to enforcing. **Status:** L4 / E3 reporting hooks
  landed (see below); promotion to enforcing waits on calibration.
- File a follow-up amendment to retire the existing `(P, L, U)`
  byte-parity probe in favor of the operator-action probes; this
  amendment is gated on the `cpp_compatible_probe` harness landing
  and producing reproducible reference trajectories.

**Phase 1.5 reporting hooks landed (this commit set).** The arbiter
in `benchmarks/single_stage_init_parity.py` now consumes paired
CPU/JAX `final_hessian` and `final_jacobian` summaries through:

- `benchmarks/parity_solve_quality.py::compute_dense_operator_action_max_rel_error`
  — SSOT composition of `construct_operator_action_probes` +
  `operator_action_max_relative_error` for gates L4 (LS Hessian
  action) and E3 (exact Jacobian action).
- `benchmarks/single_stage_init_parity.py::_summary_matrix` —
  reshape a flattened `_summarize_host_array` payload back to its 2D
  ``shape`` so the parity arbiter can apply operator probes against
  recovered Hessians / Jacobians.
- `benchmarks/single_stage_init_parity.py::_compute_solve_quality_probe_pair`
  — per-pair probe computation; returns ``None`` when either lane is
  missing the dense operator capture (no NaN injection).
- `benchmarks/single_stage_init_parity.py::_aggregate_solve_quality_probes`
  — fixture-level max aggregator with pair-index annotation.
- ``compare_same_candidate_objective_replay`` now returns a
  ``"solve_quality_probes"`` field carrying the aggregated values.
  Reporting-only: failures are NOT extended on these values.
- Exact-mode capture: `summarize_single_stage_boozer_solve_decomposition`
  now emits ``"final_jacobian"`` only for ``boozer_type == "exact"``,
  preserving LS gradient semantics on the existing ``"final_gradient"``
  slot.

### Phase 2 deliverables
- `boozersurface_jax.py:3356-3416` — adjoint factor-once dispatch
  (non-scipy LS lane; the SciPy reference runtime callbacks at
  `:3418-3475` already share the PLU bytes — Phase 2 wires the
  packed `(lu, piv)` channel for the JAX-on-device LS lane and
  preserves the SciPy host-LAPACK lane for `cpp_compatible_probe`
  byte parity).
- `surfaceobjectives_jax.py:2949-3378` — IFT wrappers consume shared
  factors. The traceable adjoint at `:3017-3055` already does this;
  Phase 2 propagates the same single-source-of-truth to the LS
  forward path.
- New test `tests/integration/test_factor_once_adjoint_phase2.py`
  proving forward and adjoint Hessian action are bit-equal
  (`np.array_equal`) under `dense-plu-shared`.
- CLAUDE.md amendments land here.

### Phase 3 deliverables
- New harness-only reference solver in
  `benchmarks/_cpp_compatible_probe.py` (or equivalent location)
  implementing the dense host-resident exact Newton per §5.1.
- **No** `lane=` parameter on `BoozerSurfaceJAX`. The exact
  normalizer at `boozersurface_jax.py:3097` is unchanged.
- Parity benchmark harness invokes the probe via direct import,
  not through the user constructor.
- Test: probe reproduces C++ Newton iterates within Exact gates
  E1–E6 across all fixtures.

**Status:** module landed. `benchmarks/_cpp_compatible_probe.py`
exposes `cpp_compatible_ls_newton_polish` (LS skeleton: validates
`optimizer_backend="scipy"` and `materialize_dense_linearization=True`,
then forwards explicit `iota_initial` / `G_initial` to
`BoozerSurfaceJAX.run_code(iota, G=...)`) and
`cpp_compatible_exact_newton` (host-resident dense exact Newton with
unconditional Wilkinson refinement, no monotone-norm guard, host
``np.linalg.solve`` only, sign convention ``x ← x − dx`` matching
`boozersurface.py:1670`). Public-API contract tests in
`tests/test_cpp_compatible_probe_phase3.py` cover host-LAPACK
discipline, refinement unconditionality, monotone-guard absence, sign
convention, augmented-residual assembly, smoke E3 probe shape, and
constructor pre-conditions. End-to-end E1–E6 reproducibility across
the `.artifacts/parity/` corpus is a follow-up calibration item that
joins the Phase 1.5 calibration sweep deferred above.

### Phase 5 deliverables (deferred)
- 5.1 Skeel/FERR forward-error gate (JAX-native rewrite required; see §6).
- 5.2 Eisenstat-Walker INB backtracking with Choice-2 forcing.
- 5.3 Hager–Higham `condition_estimate` field implementation.

## 10. Risk Register

1. **Pre-Newton root not closed.** Phases 0–3 narrow the gap; the
   headline 4.5e-9 number persists until the BFGS investigation lands.
   Mitigation: gate 3 tolerance schedule allows current values; tighten
   when BFGS root is fixed.

2. **`cpp_compatible_probe` harness materializes dense factors for
   exact, departing from the CLAUDE.md "exact JAX never falls back to
   dense factors" guarantee.** Deliberate amendment scoped to the
   parity-benchmark harness only; production user-facing
   `BoozerSurfaceJAX` exact path is unchanged.

3. **Factor-once adjoint memory budget.** At `n ≥ 8000`, dense
   factor exceeds 512 MB and auto-reverts to operator-only.
   Mitigation: budget already enforced by
   `_DEFAULT_MAX_DENSE_JACOBIAN_BYTES` (`boozersurface_jax.py:2734`);
   no new ceiling required.

4. **Solve-quality tolerances need calibration.** Initial values in
   §2 are estimates. Run a calibration sweep against existing
   fixtures in `.artifacts/parity/` before locking values in
   `PARITY_LADDER_TOLERANCES`. Calibration protocol:
   - Pick 10 representative fixtures (Banana coil, HBT, mixed-quadrature).
   - Compute per-fixture solve-quality fields under
     `production_operator` lane against C++.
   - Tolerances = max(95th percentile across fixtures, theoretical floor).

5. **JAX-native discipline regressions.** All new code paths must
   pass JIT trace inspection. Mitigation: add a CI check that
   `jax.make_jaxpr(...)` succeeds on the new helpers (no host
   roundtrips inside traced regions).

## 11. What This Plan Does NOT Claim

- Does **not** unblock production research before the strict release
  gate at `single_stage_init_parity.py:3047-3051` and the pre-Newton
  hard gate at `:2000-2030` go green.
- Does **not** loosen any release-blocker tolerance, including the
  pre-Newton hard gate at `single_stage_init_parity.py:2000-2030`.
- Does **not** introduce a "production gate" separate from the
  current strict gate.
- Does **not** invalidate Phase 4 plan §2 "Hard Constraints."
- Does **not** address the pre-Newton BFGS root split (separate
  workstream); the LS L3 and Exact E2 tolerances in §2 remain
  upper-bound reporting thresholds, not floors.
- Does **not** require any `lineax` / `optimistix` dependency.
- Does **not** change `native_cpu` (C++ reference) behavior.
- Does **not** introduce a `lane=` parameter on the user-facing
  `BoozerSurfaceJAX` constructor; the `cpp_compatible_probe` is a
  benchmark-harness diagnostic, not a product API.
- Does **not** classify `linear_solve_factors` as debug-only;
  it remains load-bearing runtime data for the LS reference SciPy
  callbacks (`boozersurface_jax.py:3418-3475`) and the traceable
  adjoint PLU solve (`surfaceobjectives_jax.py:3017-3055`).
- Does **not** enforce gates L4 / E3 as proofs of operator
  equivalence — they are smoke-regression diagnostics on `k+1 ≤ 9`
  probe directions; rigorous operator parity remains under the
  existing `direct-hessian-oracle` lane.

## 12. References

- C++ Newton polish + final outputs:
  `src/simsopt/geo/boozersurface.py:640, 836, 1129, 1130, 1155, 1668-1669`
- C++ Hessian assembly: `src/simsoptpp/boozerresidual_impl.h:203-217, 283-298, 336-341`
- JAX Hessian materialization: `src/simsopt/geo/optimizer_jax.py:1615-1632`
- JAX Newton dense step gate: `src/simsopt/geo/optimizer_jax.py:2284`
- JAX dense Newton solve: `src/simsopt/geo/optimizer_jax.py:1729-1735`
- JAX exact Newton (non-traceable, has monotone guard):
  `src/simsopt/geo/optimizer_jax.py:2473-2556` (guard at 2519-2524)
- JAX exact Newton (traceable, no monotone guard):
  `src/simsopt/geo/optimizer_jax.py:2559-2672`
- JAX operator GMRES seam: `src/simsopt/geo/optimizer_jax.py:1866-1905`
- JAX adjoint runtime state:
  `src/simsopt/geo/boozersurface_jax.py:3124-3412`
- JAX dense PLU metadata:
  `src/simsopt/geo/boozersurface_jax.py:4882-4887, 5308`
- M5 IFT adjoint consumers:
  `src/simsopt/geo/surfaceobjectives_jax.py:2949-3378`
- Existing parity ladder SSOT:
  `benchmarks/validation_ladder_contract.py`
- Existing parity arbiter probes:
  `benchmarks/single_stage_init_parity.py:192-197, 246-252`
- Existing strict release-gate exit and pre-Newton hard gate
  (release blockers, unchanged by this plan):
  `benchmarks/single_stage_init_parity.py:2000-2030, 3047-3051`
- LS reference runtime callbacks (load-bearing PLU usage):
  `src/simsopt/geo/boozersurface_jax.py:3418-3475`
- Traceable adjoint PLU solve (load-bearing PLU usage):
  `src/simsopt/geo/surfaceobjectives_jax.py:3017-3055`
- Exact normalizer that strips `optimizer_backend`:
  `src/simsopt/geo/boozersurface_jax.py:3097`
- Unknown-lane rejection in ladder contract:
  `benchmarks/validation_ladder_contract.py:254-259`
- Unknown-lane test guard:
  `tests/test_benchmark_helpers.py:3165`

### External literature

- Higham, *Accuracy and Stability of Numerical Algorithms* 2nd ed.,
  SIAM 2002, §3 (summation), §12 (iterative refinement).
- Skeel, "Iterative refinement implies numerical stability for
  Gaussian elimination," *Math. Comp.* 35(151), 1980, pp. 817–832.
- Carson & Higham, "Accelerating the solution of linear systems by
  iterative refinement in three precisions," *SIAM J. Sci. Comput.*
  40(2), 2018, A817–A847.
- Eisenstat & Walker, "Globally Convergent Inexact Newton Methods,"
  *SIAM J. Optim.* 4(2), 1994, 393–422.
- Eisenstat & Walker, "Choosing the Forcing Terms in an Inexact
  Newton Method," *SIAM J. Sci. Comput.* 17(1), 1996, 16–32.
- Knoll & Keyes, "Jacobian-free Newton-Krylov methods: a survey,"
  *J. Comput. Phys.* 193, 2004, 357–397.
- Blondel et al., "Efficient and modular implicit differentiation,"
  *NeurIPS* 2022 (jaxopt design).
- NVIDIA cuSOLVER documentation:
  https://docs.nvidia.com/cuda/cusolver/index.html
- NVIDIA cuBLAS documentation:
  https://docs.nvidia.com/cuda/cublas/index.html
- JAX precision config:
  https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html
