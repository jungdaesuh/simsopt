# Parity Audit 02 — `integral_BdotN` (Stage-II flux objective)

**Audit timestamp:** 2026-05-16
**JAX version (declared SSOT, `CLAUDE.md`):** `jax==0.10.0`, `jaxlib==0.10.0`
**Branch:** `gpu-purity-stage2-20260405`

## Files Audited

| File | LOC | Role |
|------|-----|------|
| `src/simsopt/objectives/integral_bdotn_jax.py` | 128 | JAX kernel — three flux definitions, residual, signed flux |
| `src/simsoptpp/integral_BdotN.cpp` | 123 | C++ reference reducer (single function) |
| `src/simsoptpp/integral_BdotN.h` | 4 | declaration |
| `src/simsopt/objectives/fluxobjective_jax.py` | 433 | `SquaredFluxJAX` adapter (Optimizable wrapper, value/grad orchestration) |
| `src/simsopt/objectives/fluxobjective.py` | 134 | CPU `SquaredFlux` reference (the legitimate Python caller of `sopp.integral_BdotN`) |
| `src/simsopt/jax_core/objectives_flux.py` | 138 | Spec-based JAX helpers that wrap `integral_BdotN_jax` for the fixed-surface fast path |
| `src/simsopt/jax_core/reductions.py` | 121 | Pairwise / Kahan / `vdot` reduction primitives consumed by the integral |
| `tests/objectives/test_integral_bdotn_jax.py` | 454 | Definition, NumPy reference, and C++ parity tests for the kernel |
| `tests/objectives/test_integral_bdotn_item10_closeout.py` | 247 | Production-scale chained `BiotSavartJAX → integral_BdotN` vs `sopp.integral_BdotN` parity |
| `tests/objectives/test_fluxobjective_jax_parity.py` | 730 | Adapter-level `SquaredFlux` / `SquaredFluxJAX` value + gradient parity (used as a downstream sanity oracle) |

## Executive Summary

**Top three findings:**

1. **MEDIUM (M-1) — Denominator reduction-order divergence for the `"normalized"` definition.**
The JAX residual builds the per-point weight `n / Σ |B|² · |n|` using `pairwise_sum_flat(B² · |n|)`, but the C++ oracle computes `Σ |B|² · |n|` via an OpenMP `reduction(+:denominator_sum)` (linear sequential reduction with implementation-defined merge order). The C++ result also uses the *same* `Σ` value to *divide* the integral whereas the JAX residual divides each point first, then sums the squared-residuals via either `vdot` (default lane) or compensated Kahan (strict-oracle lane). On the `_normalized_reduction_stress_data` 257-point ±10^120-magnitude fixture the kernel is already validated to recover `J=0.5` (see `tests/objectives/test_integral_bdotn_jax.py:244-253`), and the production-scale chained test (`test_integral_bdotn_item10_closeout.py`) passes at `rtol=1e-10`. Still, the dual division pathway means CPU↔JAX exact byte identity is **not** guaranteed in the `"normalized"` lane and the audit could not find a same-state byte-identity gate for it. **Severity: MEDIUM** (parity-ladder fragility, not a math bug).

2. **LOW (L-1) — Different per-point weight algebra in `"normalized"` vs `"quadratic flux"` / `"local"`.**
JAX uses an asymmetric algebraic form: `weight = norm_n / denominator`, `residual = BdotN · sqrt(weight)`, then `J = 0.5 · Σ residual²`. C++ uses the symmetric form: `numerator = Σ BdotN² · normN`, `J = 0.5 · numerator / denominator`. Mathematically identical (`0.5 · Σ BdotN² · norm_n / D ≡ 0.5 · numerator / D`), but the JAX form multiplies `1/sqrt(D)` into the residual *before* squaring and summing — this enables AD through the residual but expands the per-point work and (combined with M-1) makes byte identity to C++ very unlikely. Verified algebraic equivalence below.

3. **INFO (I-1) — Two JAX-only public symbols have no C++ counterpart at all.**
`residual_BdotN(...)` (integral_bdotn_jax.py:38) and `signed_BdotN_flux(...)` (integral_bdotn_jax.py:86) exist only in JAX. They are exercised internally (residual is the LHS of `integral_BdotN`, signed flux backs `_pre_newton_census_gate` reporting) but cannot be directly oracle-checked against C++. The closed-form NumPy reference in `tests/objectives/test_integral_bdotn_jax.py:55-69` only covers the reduced scalar `integral_BdotN`. Audit confirms there is no `sopp.residual_BdotN` or `sopp.signed_BdotN_flux` symbol.

The audit found **no CRITICAL or HIGH** divergence: the three definitions compute the same mathematical object as the C++ reference, sign conventions match, target subtraction matches, the unit-normal recovery is bit-identical in spec, and all three definitions' degenerate cases (`|n|=0`, `|B|=0`) return the same finite/inf values as the C++ oracle.

## Function-by-Function Parity Matrix

| JAX symbol (`integral_bdotn_jax.py`) | C++ symbol (`integral_BdotN.cpp`) | MATH | PHYSICS | ALGO | COMPUTE | Severity |
|---|---|---|---|---|---|---|
| `integral_BdotN(Bcoil, target, normal, definition, reduction_mode)` line 93 | `integral_BdotN(Bcoil, Btarget, n, definition)` line 12 | OK | OK | divergent reformulation for `"normalized"` (see L-1) | tree/vdot/Kahan vs sequential OMP reduction | LOW + MEDIUM (M-1, L-1) |
| `residual_BdotN(...)` line 38 — quadratic flux branch | implicit (C++ never returns a residual) | OK | OK | OK | pairwise / vdot | INFO (I-1) |
| `residual_BdotN(...)` — normalized branch | implicit | OK (algebraic equivalent) | OK | divergent (point-weight using global `Σ`) | tree | MEDIUM (M-1) |
| `residual_BdotN(...)` — local branch | implicit | OK | OK | OK | pairwise / vdot | INFO (I-1) |
| `signed_BdotN_flux(Bcoil, normal)` line 86 | — none — | N/A | N/A | N/A | tree (`pairwise_sum_flat`) | INFO (I-1) |

For the adapter:

| JAX symbol (`fluxobjective_jax.py`) | CPU/C++ counterpart (`fluxobjective.py`) | Notes |
|---|---|---|
| `SquaredFluxJAX.__init__` (line 200) | `SquaredFlux.__init__` (line 66) | JAX captures the surface DOFs fingerprint at construction (line 232); CPU re-reads `surface.normal()` per call (line 81). Drift detection contract diverges — see I-2. |
| `SquaredFluxJAX.J()` (line 393) | `SquaredFlux.J()` (line 80) | Both ultimately call the matching definition path; JAX caches scalar under `self._cached_value`. |
| `SquaredFluxJAX.dJ()` (line 411) | `SquaredFlux.dJ()` (line 86) | JAX uses end-to-end `value_and_grad` through `_jit_val_grad_dofs`. CPU uses a hand-coded `dJdB` and then `field.B_vjp(dJdB)`. Different chain rules; analytical equivalence verified by tests at `derivative_heavy` lane (`first_derivative_rtol=1e-8`). |
| `SquaredFluxJAX._raise_if_*_drifted` (lines 358–391) | none | JAX-only safety scaffolding. |
| `coil_current_fixed_geometry_flux_jax(...)` (line 106) | none | JAX-only helper for current-only kernels, oracle is `SquaredFlux.J()` (test_fluxobjective_jax_parity.py:440). |
| `coil_current_fixed_geometry_value_and_grad_jax(...)` (line 123) | none | JAX-only; FD oracle (`test_fluxobjective_jax_parity.py:471-511`). |

## Canonical Mathematical Expressions

For all three definitions the discrete integrand is over `i = 0 .. nphi·ntheta − 1` indexing the surface quadrature points. Let:

- `B_i ∈ ℝ³` = `Bcoil[i, :]`
- `n_i ∈ ℝ³` = `normal[i, :]` (unnormalized surface normal)
- `|n_i| = sqrt(n_i · n_i)`
- `n̂_i = n_i / |n_i|` if `|n_i| > 0` else `0`
- `T_i` = `target[i]` (defaulting to 0)
- `r_i = B_i · n̂_i − T_i`
- `N = nphi · ntheta`

The three definitions:

1. **`"quadratic flux"`**:
   `J = (0.5 / N) · Σ_i r_i² · |n_i|`

2. **`"normalized"`**:
   `J = 0.5 · ( Σ_i r_i² · |n_i| ) / ( Σ_i |B_i|² · |n_i| )` if denominator > 0, else `+inf`

3. **`"local"`**:
   `J = (0.5 / N) · Σ_i r_i² · |n_i| / |B_i|²`, with `J = +inf` if any `i` has `|n_i| > 0` and `|B_i|² ≤ 0`

These match the docstring of CPU `SquaredFlux` (`fluxobjective.py:27-49`) and the docstring of JAX `integral_BdotN` (`integral_bdotn_jax.py:1-22`) verbatim.

## Detailed Findings

### Finding M-1: Normalized denominator reduction order

**Files:** `src/simsopt/objectives/integral_bdotn_jax.py:55-64`; `src/simsoptpp/integral_BdotN.cpp:63-114`.

**JAX form (lines 55-64):**

```python
elif definition == "normalized":
    B2 = jnp.sum(Bcoil * Bcoil, axis=-1)
    denominator = pairwise_sum_flat(B2 * norm_n)
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    point_weight = jnp.where(has_normal, norm_n / safe_denominator, 0.0)
    residual = jnp.where(
        denominator > 0.0,
        jnp.where(has_normal, BdotN * jnp.sqrt(point_weight), 0.0),
        jnp.full_like(BdotN, jnp.inf),
    )
```

…then in `integral_BdotN` (line 124):

```python
return 0.5 * scalar_square_sum(
    residual,
    reduction_mode=reduction_mode,
    default="vdot",
)
```

The residual carries `sqrt(|n_i| / D)`; after `scalar_square_sum` the kernel returns
`0.5 · Σ_i [B_i·n̂_i − T_i]² · |n_i| / D`. Algebraically equal to the C++ symmetric form.

**C++ form (lines 87-114):**

```cpp
if (definition_int != DEFINITION_QUADRATIC_FLUX)
    mod_B_squared =
        Bcoil_ptr[3 * i + 0] * Bcoil_ptr[3 * i + 0]
        + Bcoil_ptr[3 * i + 1] * Bcoil_ptr[3 * i + 1]
        + Bcoil_ptr[3 * i + 2] * Bcoil_ptr[3 * i + 2];
...
} else if (definition_int == DEFINITION_NORMALIZED){
    numerator_sum += (BcoildotN * BcoildotN) * normN;
    denominator_sum += mod_B_squared * normN;
}
...
if (definition_int == DEFINITION_NORMALIZED) {
    if (denominator_sum <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    result = 0.5 * numerator_sum / denominator_sum;
}
```

**Divergences:**

- **Denominator summation**: JAX uses a deterministic binary addition tree
(`pairwise_sum_flat`, `reductions.py:64-72`, padded to next power of two with zero entries
from `_pad_axis` and reduced via `_pairwise_reduce_axis0`). C++ uses an OpenMP
parallel `reduction(+:denominator_sum)` whose merge order is implementation-defined
and (per the spec) only requires *associativity* — not bit identity.
- **Numerator summation**: JAX uses `scalar_square_sum` with `default="vdot"`
(`reductions.py:99-120`); this calls `jnp.vdot(flat, flat).real` whose contraction
order is XLA-dependent. C++ likewise sums via OpenMP reduction.
- **Per-point algebra**: JAX divides at the per-point level (point_weight =
`|n_i|/D`); C++ divides only the final numerator. Algebraically the same, but the
order of `(× |n_i|) → (÷ D) → (×r_i²) → (Σ)` vs `(× |n_i| ×r_i²) → (Σ) → (÷ D)`
admits different float rounding.

**Why this is MEDIUM, not CRITICAL or LOW:**
The codebase's parity ladder declares `rtol=1e-10` for `direct_kernel` (`benchmarks/validation_ladder_contract.py:53-59`) and the closeout test
(`tests/objectives/test_integral_bdotn_item10_closeout.py:236-246`) passes at that
tolerance for all three definitions on a production-scale grid. So there is no
current parity break. However, byte identity to C++ is not provable for
`"normalized"`, and the divergence is structural: even moving JAX's
`pairwise_sum_flat` to `compensated_sum_flat` would only refine accuracy in JAX,
not align it to C++'s OpenMP reduction order. If a future parity gate ratchets to
`rtol=1e-13` (or to *bit*-identity), the `"normalized"` lane will need either a
per-point JAX rewrite that matches the C++ algebraic split or a documented
relaxation specific to this definition.

**Recommended action:** No code change. Document in CLAUDE.md / parity ladder
notes that `"normalized"` is structurally not byte-identical to the C++ oracle.
Consider an optional `strict_oracle` `"normalized"` path that mirrors C++'s
symmetric `numerator / denominator` algebra, sums both with `compensated_sum_flat`
(or just `jnp.sum` with `jnp.float64`), and divides at the end — purely for
diagnostic comparison.

### Finding L-1: Per-point divide in `"normalized"` is algebraic-only

Subsumed by M-1; documented separately because the reader of the JAX file may be
surprised by the `sqrt(point_weight)` device. The reformulation is required by
AD: by routing the `1/sqrt(D)` factor through the per-point residual, the
`integral_BdotN` kernel keeps the linear-residual / Jacobian-of-squared-norm
structure that `scalar_square_sum` consumes uniformly across all three
definitions. The C++ kernel has no AD requirement and therefore uses the
mathematically clearer symmetric form.

**Recommended action:** Add a 2-line comment on `integral_bdotn_jax.py:55-64`
explaining that the per-point `sqrt(|n|/D)` form is intentional for AD
uniformity — this prevents future readers from "simplifying" it back to the C++
form and breaking the residual contract used by `scalar_square_sum`.

### Finding I-1: JAX-only `residual_BdotN` and `signed_BdotN_flux` have no C++ oracle

- `residual_BdotN(Bcoil, target, normal, definition)` (integral_bdotn_jax.py:38)
  returns a `(nphi·ntheta,)`-shaped raveled residual vector. It is consumed by
  `integral_BdotN` (line 117), `fixed_surface_flux_residual_from_B`
  (objectives_flux.py:72-79), and `stage2_target_objective_jax.py:863`. It has
  **no C++ analog** because the C++ reducer only exposes the scalar.
- `signed_BdotN_flux(Bcoil, normal)` (integral_bdotn_jax.py:86) returns
  `(1/N) · Σ_i B_i · n_i`. Used by reporting / gate code; no C++ analog.

The closed-form NumPy oracle in `_numpy_integral_BdotN`
(`tests/objectives/test_integral_bdotn_jax.py:55-69`) only validates the *reduced
scalar* `integral_BdotN`; the residual vector itself is never compared to an
independent oracle. The closed-torus parity test for `signed_BdotN_flux`
(`tests/objectives/test_integral_bdotn_jax.py:152-168`) uses *analytical*
closure (`signed flux ≈ 0` on a closed torus) — that is a structural oracle, not
a C++ parity test.

**Recommended action:** Add a per-point oracle test (NumPy closed-form, not the
JAX function itself) for `residual_BdotN` covering all three definitions. Use
`np.testing.assert_allclose(jax_residual.reshape(nphi, ntheta),
np_reference, rtol=1e-13)`. Without this, a subtle bug in the
per-point algebra of `residual_BdotN` could pass the scalar test if the bug
cancels under squaring + summation.

### Finding I-2: SquaredFluxJAX vs SquaredFlux drift-detection contract divergence

**Files:** `src/simsopt/objectives/fluxobjective_jax.py:218-232, 358-391` vs
`src/simsopt/objectives/fluxobjective.py:66-83`.

`SquaredFluxJAX` captures the surface DOFs fingerprint, the field points version,
and the field DOF layout version at construction (`fluxobjective_jax.py:225-232`)
and raises `RuntimeError` if any drifts (`_raise_if_*_drifted`,
`fluxobjective_jax.py:358-391`).

CPU `SquaredFlux` does no such check: `self.field.set_points(xyz.reshape(...))`
runs once in `__init__` (`fluxobjective.py:74`), but if the caller subsequently
mutates `surface.normal()` or rebinds points, the CPU object will silently use
the stale state on the next `J()` (it re-reads `self.surface.normal()` per call,
line 81 — so for *surface* mutation the CPU object actually picks up the change,
whereas the JAX object raises).

**Impact:** Not a numerical parity bug. It is a contract divergence: the CPU
object is "live" w.r.t. surface DOF changes, while the JAX object is "frozen"
at construction. The JAX docstring is explicit (`fluxobjective_jax.py:184-186`,
"The plasma surface must be fixed during the optimization."), so the divergence
is documented. Users who relied on CPU `SquaredFlux`'s implicit surface mutation
support will get `RuntimeError` from `SquaredFluxJAX` rather than silent staleness.

**Recommended action:** None — the JAX contract is stricter and safer, and
documented. Possibly mention this in upgrade notes if there is a CPU→JAX
migration guide.

### Finding I-3: Different cached state lifecycle

`SquaredFluxJAX._clear_cached_results` (line 351) is called from `recompute_bell`
(line 355) and from `__init__`. `SquaredFlux` has no such per-object cache; it
recomputes on every `J()`/`dJ()` call. Both contracts are fine; the JAX cache is
correct because the surface and field-points are frozen, so any new value
requires `new_x = True`.

### Sign / convention / unit verification

Cross-checked against both implementations:

- **Sign convention for the residual `B·n̂ − T`.** Both JAX (`integral_bdotn_jax.py:50`)
  and C++ (`integral_BdotN.cpp:79-85`) compute `B·n̂` first, then subtract `T_i`.
  Matching.
- **Outward normal.** Neither implementation enforces outward orientation;
  both use whatever sign convention the surface provides. The squared form
  in `"quadratic flux"`, `"local"`, and `"normalized"` is sign-invariant in
  `n̂` (because of the square on `BdotN` and the absolute `|n_i|`).
- **`|n|` weighting (surface element factor).** Both implementations weight
  by the unnormalized magnitude `|n_i|`, which is the discrete surface element.
  The `1/N` factor in `"quadratic flux"` and `"local"` corresponds to the
  quadrature step `dphi · dtheta / (4π²) × 4π² = 1/N` on the unit-`(phi, theta)`
  parametric square — the nfp factor cancels as documented in `CLAUDE.md`.
  Matching.
- **`mu_0` factor.** Neither implementation includes `mu_0`. The Biot-Savart
  caller provides `B` in Tesla (multiplied by `mu_0/(4π)`). `integral_BdotN`
  is a pure quadratic functional of `B`, so the absence of `mu_0` here is
  correct.
- **Target subtraction order.** JAX (`integral_bdotn_jax.py:50`):
  `BdotN = jnp.sum(Bcoil * unit_n, axis=-1) - target`. C++
  (`integral_BdotN.cpp:84-85`): `if(Btarget_ptr != NULL) BcoildotN -= Btarget_ptr[i];`.
  Identical to within a possible NULL-check (JAX always subtracts `target`, which
  defaults to a zero array via `_fixed_surface_target_array` in
  `objectives_flux.py:38-42`; therefore zero subtraction is a no-op).

### Degenerate cases (boundary contract)

| Case | JAX behavior (`integral_bdotn_jax.py`) | C++ behavior (`integral_BdotN.cpp`) | Same? |
|---|---|---|---|
| `\|n_i\| = 0` (zero-area quadrature point) | weight set to `0.0` via `has_normal` mask, residual `0.0` (line 43-49, 53-54, 59, 69-72) | `Nx = Ny = Nz = 0.0`, so `BcoildotN = 0` and `numerator_sum += 0 · normN = 0` (line 74-78, 79-83, 94-103) | YES |
| `\|n_i\| > 0`, `\|B_i\|² = 0` (`"local"`) | per-point `singular = True`, residual set to `inf` (line 67, 74-78) → overall `J = inf` | `has_local_singularity = 1` → return `inf` (line 99-100, 116-117) | YES |
| Global `Σ \|B\|² \|n\| ≤ 0` (`"normalized"`) | denominator-guard sets entire residual to `inf` (line 60-64) → `J = inf` | `denominator_sum <= 0.0` → return `inf` (line 110-112) | YES |
| `\|n_i\| = 0` AND `target ≠ 0` (`"local"`) | `has_normal[i] = False` → residual `0.0` even though `B·n̂ - T = -T ≠ 0` (line 43-49 zero-out `unit_n`, line 69-72 zero-out weight) | `Nx=Ny=Nz=0`, so `BcoildotN = -Btarget_ptr[i]`, but the inner `if (normN > 0.0 && mod_B_squared <= 0.0)` is false and `mod_B_squared > 0.0` adds `T² · 0 / |B|² = 0` to numerator | YES (both give 0 for this contribution; verified by test `test_zero_normal_local_returns_zero` and `test_cpp_boundary_contract_matches_jax`) |

All four degenerate-case behaviors are explicitly oracle-tested in
`TestIntegralBdotNCppParity.test_cpp_boundary_contract_matches_jax`
(test_integral_bdotn_jax.py:331-386) for the cases `(quadratic flux, zero
normal, T=1) → 0`, `(normalized, zero B) → inf`, `(local, zero B, T=1) → inf`,
and `(local, zero normal, T=1) → 0`.

### Reduction-order specifics (`scalar_square_sum`)

`scalar_square_sum` (`reductions.py:99-120`) escalates by tier:

- `reduction_mode="strict_oracle"` → `compensated_sum_flat` (Kahan, deterministic
  but slow). Used in the `test_strict_oracle_scalar_reduction_matches_high_precision_reference`
  test against `math.fsum` (test_integral_bdotn_jax.py:255-289).
- `reduction_mode="default"` AND `default="pairwise"` → `pairwise_sum_flat`
  (tree).
- `reduction_mode="default"` AND `default="vdot"` (the `integral_BdotN` site at
  line 127) → `jnp.vdot(flat, flat).real`.

The hot-path entrypoint thus uses `vdot`. The C++ kernel uses an OpenMP
`reduction(+:numerator_sum)`. **These are different reduction orders.**

The codebase explicitly tolerates this divergence in the production
`*_parity` lanes via `rtol=1e-10`. For `direct_kernel` (the strictest lane,
`benchmarks/validation_ladder_contract.py:53-59`), the closeout test
(test_integral_bdotn_item10_closeout.py) holds at `rtol=1e-10` on a
production-scale fixture for all three definitions and both stellsym variants.

The kernel offers `reduction_mode="strict_oracle"` as a documented opt-in for
investigations (CLAUDE.md and integral_bdotn_jax.py:109-112). However, the
hot path stays on `vdot` because Kahan inside `jax.jit` materializes a
serial `lax.fori_loop` with explicit `subtract → add → subtract` state
(reductions.py:86-94), which is much slower than the XLA `vdot` lowering.

**Recommended action:** None. Properly documented and gated. The audit notes
this for completeness because it is the single largest source of CPU↔JAX
numerical drift at high quadrature counts.

### Compute / dtype audit

- **JAX `Bcoil`, `target`, `normal`** are all `float64` once routed through
  `_as_runtime_float64` (`fluxobjective_jax.py` → `objectives_flux.py:65-69`
  → `_math_utils.py:68-69`).
- **C++** uses `double` throughout (`integral_BdotN.cpp:1-4`,
  `PyArray = xt::pyarray<double>`).
- **No `float32` ingress** anywhere in this kernel.
- **`jnp.where` masking** for degenerate normals correctly avoids `nan` by
  using `safe_norm_n = 1.0` and then masking back to `0` after division
  (`integral_bdotn_jax.py:42-49`).

### Algorithm audit (branching, vectorization)

- **JAX** has three branches selected at JIT-trace time via
  `static_argnames=("definition",)` (line 37, 92). Each definition is a
  fully-vectorized JAX expression; no Python loops.
- **C++** has a single hot loop with a `definition_int` branch (lines 64-107)
  parallelized via `#pragma omp parallel for reduction(+:numerator_sum,
  denominator_sum) reduction(max:has_local_singularity)` (line 63). The
  `reduction(max:...)` clause is the correct way to communicate the
  `"local"`-singularity flag across threads.
- **Both** check the same invalid configurations and return the same finite /
  inf values.
- **Both** use the *unnormalized* magnitude `|n_i|` as the surface-element
  weight and the unit normal `n̂_i` only to project `B`.

## Test Coverage Map

| Symbol | Oracle | Tolerance | Notes |
|---|---|---|---|
| `integral_BdotN, "quadratic flux"` | NumPy reference (`_numpy_integral_BdotN`); C++ `sopp.integral_BdotN` | `rtol=1e-13` (NumPy) / `rtol=1e-13` (C++, kernel-only) / `rtol=1e-10` (closeout, B + reducer chained) | Fully covered |
| `integral_BdotN, "normalized"` | NumPy reference; C++ `sopp.integral_BdotN`; geomspace stress fixture | `rtol=1e-13` (NumPy/C++); contract-driven `atol/rtol` for stress | Fully covered |
| `integral_BdotN, "local"` | NumPy reference; C++ `sopp.integral_BdotN`; zero-normal & zero-B contract tests | `rtol=1e-13` (NumPy/C++) | Fully covered |
| `integral_BdotN, "quadratic flux"` strict-oracle path | `math.fsum` of `amplitude²` | `rtol=1e-15` | Confirms Kahan accuracy |
| `residual_BdotN` (all three definitions) | **NONE — no per-point oracle** | n/a | Gap (see I-1) |
| `signed_BdotN_flux` | analytical closure on closed torus | `atol = directional_derivative_floor` | Structural, not numeric |
| `SquaredFluxJAX.J()` | CPU `SquaredFlux.J()` (which calls `sopp.integral_BdotN`) | `direct_kernel` lane | Adapter-level, multiple variants |
| `SquaredFluxJAX.dJ()` | CPU `SquaredFlux.dJ()`; central-difference FD | `derivative_heavy.first_derivative_*` | Both gradient parity and FD |
| `coil_current_fixed_geometry_flux_jax` | CPU `SquaredFlux.J()` (which calls `sopp.integral_BdotN`) | `direct_kernel` | Covered |
| `coil_current_fixed_geometry_value_and_grad_jax` | FD on the JAX flux | `fd-gradient` | Self-consistency only — gradient not compared to CPU `SquaredFlux.dJ()` here |
| `integral_BdotN` `reduction_mode` validation | string assertion | n/a | covers invalid-mode raise |

### Gaps

1. **No per-point oracle for `residual_BdotN`.** Pin this for all three
   definitions with a closed-form NumPy reference at `rtol=1e-13`
   (recommended).
2. **No bit-identity gate for any definition.** The closeout test uses
   `rtol=1e-10`, not exact equality. If the team intends to claim
   `*_parity` byte identity for the flux kernel in a publication, the
   `"normalized"` lane needs a dedicated documentation note (per M-1) and a
   pinned numerical regression fixture rather than a relative-tolerance gate.
3. **`coil_current_fixed_geometry_value_and_grad_jax` gradient is only
   FD-validated.** Add a direct comparison against `SquaredFlux.dJ()` on
   currents — the existing FD test catches only one error mode
   (sign-of-direction errors), not coil-current chain-rule errors.

## Recommended Actions, Ordered by Severity

1. **MEDIUM (M-1):** Add a short note in `integral_bdotn_jax.py` (right above
   the `"normalized"` branch at line 55) and in `CLAUDE.md`'s "Parity ladder
   SSOT" section explaining that the `"normalized"` definition is *not*
   structurally byte-identical to `sopp.integral_BdotN` due to (i) the per-point
   vs symmetric algebra split and (ii) the pairwise-vs-OpenMP reduction
   divergence. Cite `tests/objectives/test_integral_bdotn_item10_closeout.py` as
   the binding tolerance contract (`rtol=1e-10`). No code change required.

2. **LOW (L-1):** Add a 2-line comment to `integral_bdotn_jax.py:55-64`
   explaining the AD-uniformity rationale for the per-point
   `sqrt(|n|/D)` form, to prevent future "simplification" PRs from breaking
   the residual contract.

3. **INFO (I-1):** Add a per-point oracle test for `residual_BdotN` covering all
   three definitions. Use the closed-form expressions in `_numpy_integral_BdotN`
   adapted to return the per-point residual rather than the reduced scalar; gate
   at `rtol=1e-13`. Add to `tests/objectives/test_integral_bdotn_jax.py`.

4. **INFO (I-3):** Add a direct gradient parity test between
   `coil_current_fixed_geometry_value_and_grad_jax` and `SquaredFlux.dJ()`
   restricted to coil-current DOFs — the current test (FD only) is
   self-consistent but not oracle-anchored to the C++ chain.

5. **INFO:** Consider documenting in
   `tests/objectives/test_integral_bdotn_jax.py` that the
   `TestIntegralBdotNBoundaryContracts` test class (lines 389-449) duplicates
   four tests already present in `TestIntegralBdotN` (the same `test_zero_*`
   names appear in both classes). The duplication is not harmful but is dead
   weight. Either delete `TestIntegralBdotNBoundaryContracts` or convert it to
   pure C++-parity-only variants (currently it tests JAX without C++).

## Open Questions

- **OQ-1.** Is there an implicit assumption that nfp surface points are
  *already expanded* into `nphi · ntheta` in both the CPU and JAX paths?
  Inspection shows that `SurfaceRZFourier.normal()` and `surface.gamma()` return
  arrays of shape `(nphi, ntheta, 3)` regardless of nfp, and that the
  `1/N = 1/(nphi · ntheta)` factor in both implementations does not multiply
  by `nfp`. The `CLAUDE.md` note "nfp cancels with quadrature step
  1/(nfp·nphi)" presumably means that the surface's `quadpoints_phi` array
  is the per-fp grid and that the actual full-torus surface integral is
  `nfp` times the result here. **The kernel itself is unaware of `nfp`.**
  This is consistent across JAX and C++ — both implementations would give
  identical answers for any `nfp` because `nfp` does not appear in either
  source — but it means the value returned is a per-`nfp`-section quantity,
  not a true full-torus integral. Confirming this is consistent with the
  expectations of downstream callers (`SquaredFlux`, the Stage-2 outer
  optimizer) is out of scope for this audit; the kernels themselves match.

- **OQ-2.** The `"normalized"` JAX form uses `safe_denominator = where(D>0, D, 1.0)`
  to avoid `nan`, then later sets the residual to `inf` when `D <= 0` via
  `where(D > 0, finite_branch, inf)`. The C++ form returns `inf` *before*
  multiplying or dividing. Is there any path where the JAX form could produce
  a *finite* value when the C++ form produces `inf`, or vice versa? Analysis
  says no, because `inf` is propagated through the final `0.5 * sum(residual²)`
  unchanged. Verified by `test_cpp_boundary_contract_matches_jax` for the
  zero-`B` normalized case.

- **OQ-3.** Could the `vdot`-based `scalar_square_sum` default produce a
  *finite negative* value due to catastrophic cancellation on extreme-range
  inputs? `(conj(x) * x).real` is non-negative per element so the sum is
  monotone non-decreasing — `vdot` cannot underflow to negative. The `0.5 *`
  multiplier likewise preserves non-negativity. The only "negative" outcome
  is `nan` if `inf - inf` ever occurs, which the `where` masking on
  `has_normal` and the `singular` branch prevents.

## Conclusion

The JAX `integral_BdotN` kernel computes the same three quadratic-flux integrals
as `sopp.integral_BdotN` to the documented `direct_kernel` parity-ladder
tolerance (`rtol=1e-10`). All sign conventions, target-subtraction order, surface
weighting, and degenerate-case behavior match the C++ reference; the four
oracle-tested boundary contracts are explicit. The two structural divergences
that are real but not numerically harmful at current tolerances are (i) the
per-point algebraic split in the `"normalized"` definition and (ii) the
hot-path use of `jnp.vdot` versus an OpenMP linear reduction. Both are
documented and gated.

The JAX-only symbols `residual_BdotN` and `signed_BdotN_flux` have no C++
analog and would benefit from per-point closed-form NumPy oracles in the test
suite.

The `SquaredFluxJAX` adapter (`fluxobjective_jax.py`) is straightforward
wrapping of the kernel; the only contract divergence with respect to CPU
`SquaredFlux` is its stricter (and documented) drift-detection scaffolding,
which is a feature, not a bug.

No CRITICAL or HIGH severity findings.
