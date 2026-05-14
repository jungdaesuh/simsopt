# Crucible Adversarial Review — 2026-05-13 closure of N02 / N04b / N03 §D

**Reviewer**: Crucible adversarial code review.
**Branch**: `gpu-purity-stage2-20260405`.
**Repo head at review start**: `ca276abbd`.
**Runtime**: `jax==0.10.0`, `jaxlib==0.10.0`, Python 3.11 (local `.conda/jax-0.9.2`).

## Verdict

**PASS WITH MINOR FINDINGS** — 0 CRITICAL, 0 MAJOR (numerical/contract), 1 MAJOR (test-coverage gap), 5 MINOR, 1 LOW.

All three deliverables match their cited oracles. All 134 new tests pass. `ruff check` is clean. The two pre-existing item-14 tracing failures predate this changeset and are documented as part of the silent-fallback removal (`ef946054b`). The one MAJOR finding is a test-coverage gap (no oracle-backed numerical test of 3-vector symmetry rules), not a code defect — the implementation itself is provably correct against the C++ oracle by line-for-line comparison.

## Findings (ordered by severity)

### CRITICAL

None.

### MAJOR

#### MAJOR-1 — Test-coverage gap: 3-vector symmetry rules (`apply_odd_vector_first_only`, `apply_even`) have no numerical oracle test

**File**: `tests/field/test_interpolated_boozer_field_jax.py`.
**Affected paths**: `interpolated_boozer_field.py:583-606` (`_apply_symmetry` for `value_size==3` branches).

The 14 scalars exercised by `BoozerAnalytic` (per `_BOOZER_ANALYTIC_SCALARS` at lines 67-82 of the test file) are all `value_size==1`. The `value_size==3` branches in `_apply_symmetry` — used by `Z_derivs`, `nu_derivs` (apply_odd_vector_first_only), `R_derivs`, `modB_derivs` (apply_even) — are **never exercised** by a parity test against the C++ oracle. Only `test_freeze_state_rejects_unknown_scalar_name` and the `KeyError`-on-unbuilt routing tests exercise these scalars, and the routing tests never reach `_apply_symmetry`.

A subtle sign error on any of those 3-vector branches would not be detected by the current test suite. The implementation **is** correct by my line-for-line comparison against `boozermagneticfield_interpolated.h:786-807`, but the test gate is weaker than the math guarantees.

State.json already acknowledges this: "Routing tests cover the remaining 19 scalars via KeyError contract on un-built specs ... The remaining 19 scalars are non-trivial to cover without a VMEC-derived fixture." Given the bounded scope (re-fit on host-resident grid via `regular_grid_interp`, no VMEC dependency), promoting this to "documented limitation" rather than "test bug" is justified. Filed as MAJOR to ensure the limitation is owned in residual-risk reporting.

**Recommendation**: Add a fixture-driven parity test with synthetic 3-vector samples (e.g., construct a `RegularGridInterpolant3DSpec` for `Z_derivs` over a known closed-form `Z(s, θ, ζ) = sin(θ)·cos(ζ)·s` analytic field, fold at points with `θ>π`, verify `_apply_symmetry` negates only column 0). This is a single test, no VMEC fixture required.

### MINOR

#### MINOR-1 — `InterpolatedBoozerFieldFrozenState` is not actually frozen

**File**: `src/simsopt/jax_core/interpolated_boozer_field.py:164-219`.

The dataclass uses `@dataclass(frozen=True)`, but the `specs: dict` field's contents are mutated by `InterpolatedBoozerFieldJAX._ensure_spec` (`boozermagneticfield_jax.py:1462`). The comment at line 197 says "immutable in spirit"; the comment at 1459-1461 acknowledges the mutation. The "frozen" naming is misleading versus sibling dataclasses (`BoozerAnalyticFrozenState`, `BoozerRadialInterpolantFrozenState`) which are truly immutable.

Aliasing surface: if a user re-uses the same frozen-state across `from_frozen_state` and a wrapper that triggers lazy build, the lazy-build mutation is observable through the aliased reference. The current `from_frozen_state` path sets `_field = None` and raises `KeyError` instead of mutating, but the contract is not enforced — only the convention.

**Recommendation**: Either rename to `InterpolatedBoozerFieldState` (drop the "Frozen" claim) or wrap `specs` in a copy-on-write container. The current behavior is intentional and documented, so this is MINOR.

#### MINOR-2 — Dead-code branch in `_apply_symmetry` for `(apply_odd=True, value_size==3)`

**File**: `src/simsopt/jax_core/interpolated_boozer_field.py:591-597`.

No entry in `SYMMETRY_EXPLOIT_SCALARS` has `apply_odd=True` with `value_size==3`. All 3-vector scalars use `apply_odd_vector_first_only` instead. The `if rule.value_size == 3:` branch at lines 591-597 mirrors the C++ `apply_odd_symmetry` dual dispatch but is unreachable under the current rule table. YAGNI violation.

**Recommendation**: Delete the dead branch; rely on `apply_odd_vector_first_only` for the size-3 case. Alternatively, collapse the two rule flags into a single one and use `value_size` to dispatch.

#### MINOR-3 — Silent fall-through for `(apply_odd=True, value_size==2)`

**File**: `src/simsopt/jax_core/interpolated_boozer_field.py:586-607`.

If a future scalar registered `apply_odd=True` with `value_size==2`, the dispatch would silently return `raw` unchanged (line 607) instead of applying the symmetry. The current rule table has no such entry, so this is a latent bug, not a current bug.

**Recommendation**: Replace the final `return raw` at line 607 with a `raise ValueError(f"unhandled symmetry rule combination: {rule!r}")` so typos in future scalar registrations surface immediately.

#### MINOR-4 — `_boozer_field_evaluators` lazy-import justification is imprecise

**File**: `src/simsopt/jax_core/tracing.py:1940-1947`.

The docstring claims a top-level import of `simsopt.field.boozermagneticfield_jax` would cause an import cycle through `simsopt._core.optimizable` and `simsopt.field.tracing`. In practice:
- `simsopt.field.__init__.py` uses lazy exports, so importing `boozermagneticfield_jax` does NOT trigger `simsopt.field.tracing`.
- `simsopt.field.tracing` imports `simsopt.jax_core.tracing` lazily inside a function (`tracing.py:1544`).

The lazy import is therefore defensive but probably not strictly necessary. The docstring's "in some import orderings" hedge is the only acknowledgment.

**Recommendation**: Either verify with a clean-import sequence test that the top-level import path is safe (and promote the imports out of the function), or tighten the docstring to cite the specific orderings that fail.

#### MINOR-5 — `evaluate_batch` is not transfer-guard-safe (N02 hot path triggers transfers on every call)

**File**: `src/simsopt/jax_core/regular_grid_interp.py:570-615`.

Every call to `evaluate_batch` calls `jnp.asarray(np_array)` on the spec's host-resident NumPy arrays (`cell_table`, `cell_to_row`, `xmesh`, `ymesh`, `zmesh`, `xmin/xmax/...`). Each call therefore lifts ~10-20 host arrays to device. The N02 test docstring (line 540-546) documents this as "intentional design trade-off".

This is a **performance issue**, not a correctness issue. For 12-scalar query patterns on large point sets (e.g. trace-particles inner loop), this could be a 10-100× slowdown versus a single up-front transfer. Acceptable for the current scope (construction-time use case), problematic if `InterpolatedBoozerFieldJAX` is wired into the tracing hot path later.

**Recommendation**: Move the `jnp.asarray` calls inside `RegularGridInterpolant3DSpec.__post_init__` (or a `to_device()` factory method) so each spec retains device-resident copies of its tables.

### LOW

#### LOW-1 — `_BOOZER_RHS_EVAL_KEYS` overcommits for the vacuum and no_K factories

**File**: `src/simsopt/jax_core/tracing.py:1906-1919`.

The SSOT advertises 12 keys (`modB`, ..., `dIds`), but `guiding_center_vacuum_boozer_rhs` only reads 6 (modB, dmodBds, dmodBdtheta, dmodBdzeta, G, iota) and `guiding_center_no_k_boozer_rhs` reads 9. The full RHS uses all 12. Forcing every frozen-state branch to register all 12 evaluators is mildly over-coupled — a state could support vacuum tracing without implementing `K`/`dKdtheta`/`dKdzeta`.

In practice, both `BoozerAnalyticFrozenState` and `BoozerRadialInterpolantFrozenState` happily implement all 12 (K-related evaluators return zero or analytic values). No current consumer is broken, so LOW.

## Test quality audit

### `tests/field/test_interpolated_boozer_field_jax.py` (35 tests)

| Test | Oracle | Verdict |
|------|--------|---------|
| `test_scalar_parity_to_boozer_analytic_oracle` (14 params) | Type 1 — CPU `BoozerAnalytic` closed-form | **acceptable** — CPU oracle is independent of the re-fit grid |
| `test_modB_parity_across_nfp_values` (4 params) | Type 1 — CPU `BoozerAnalytic` | **acceptable** |
| `test_fold_points_modular_theta_into_0_2pi` | Type 2 — modular arithmetic identity; cites C++ `boozermagneticfield_interpolated.h:765-768` | **acceptable** |
| `test_fold_points_stellsym_reflection_above_pi` | Type 2 — C++ algebra at h:769-779 | **acceptable** |
| `test_modB_invariant_under_theta_2pi_shift` | Type 2 — closed-form 2π periodicity | **acceptable** |
| `test_simsopt_jax_native_field_marker_is_set` | Type 3 — explicit class attribute contract | **acceptable** (routing) |
| `test_set_points_shape_validation` | Type 4 — error-contract test | **acceptable** (routing) |
| `test_set_points_invalidates_cache` | Type 1 — CPU `BoozerAnalytic` + anti-tautology `assert not allclose` | **acceptable** |
| `test_unbuilt_scalar_raises_keyerror_from_frozen_state_wrapper` | Type 3 — explicit `KeyError` contract | **acceptable** (routing) |
| `test_from_frozen_state_round_trip` | Type 3 — round-trip identity with `rtol=0, atol=0` | **acceptable** as routing (plumbing) test |
| `test_frozen_state_inventory_matches_documented_split` | Type 3 — explicit C++ inventory at h:736-784 | **acceptable** |
| `test_freeze_state_metadata_round_trip` | Type 2 — closed-form `period=2π/nfp` | **acceptable** |
| `test_freeze_state_rejects_unknown_scalar_name` / `_invalid_degree` / `_invalid_nfp` | Type 3 — error contract | **acceptable** |
| `test_non_stellsym_wrapper_does_not_flip` | Type 2 — algebraic identity (no flip path) | **acceptable** |
| `test_non_stellsym_parity_for_qa_field` | Type 1 — CPU `BoozerAnalytic` | **acceptable** |
| `test_wrapper_modB_with_lazy_built_specs_returns_array` | Type 3 — shape contract; transfer-guard skip is documented | **acceptable** |
| `test_qh_helicity_field_parity` | Type 1 — CPU `BoozerAnalytic` (N=2 fixture) | **acceptable** |

### `tests/geo/test_surface_henneberg_jax.py` (74 tests)

| Test | Oracle | Verdict |
|------|--------|---------|
| `test_to_spec_round_trips_fields` (16 params) | Type 3 — pinned-state snapshot at construction | **acceptable** (plumbing) |
| `test_gamma_matches_cpu_oracle` (12 params) | Type 1 — CPU `SurfaceHenneberg.gamma_impl` at sh.py:626-640 | **acceptable** |
| `test_gamma_matches_cpu_oracle_across_nfp` (4 params) | Type 1 — CPU | **acceptable** |
| `test_gammadash1_matches_cpu_oracle` (12 params) | Type 1 — CPU `gammadash1_impl` at sh.py:642-704 | **acceptable** |
| `test_gammadash2_matches_cpu_oracle` (12 params) | Type 1 — CPU `gammadash2_impl` at sh.py:706-739 | **acceptable** |
| `test_normal_matches_cpu_oracle` (3 params) | Type 1 — CPU `Surface.normal` | **acceptable** |
| `test_unitnormal_matches_cpu_oracle` | Type 1 — CPU `Surface.unitnormal` | **acceptable** |
| `test_area_matches_cpu_oracle` | Type 1 — CPU `Surface.area` | **acceptable** |
| `test_volume_matches_cpu_oracle` | Type 1 — CPU `Surface.volume` | **acceptable** |
| `test_axisymmetric_default_torus_matches_analytic` | Type 2 — closed-form `R=1+0.1·cos(θ)` | **acceptable** |
| `test_axisymmetric_gammadash2_matches_analytic` | Type 2 — closed-form differentiated torus | **acceptable** |
| `test_make_spec_rejects_invalid_alpha_fac` / `_shape_mismatch` | Type 3 — error contract | **acceptable** |
| `test_spec_is_frozen_dataclass` | Type 3 — `@dataclass(frozen=True)` contract | **acceptable** |
| `test_spec_register_dataclass_keys_match_host_class` | Type 3 — explicit registration field-list | **acceptable** |
| `test_spec_jit_round_trip_preserves_meta_fields` | Type 3 — JAX pytree contract | **acceptable** |
| `test_jit_cache_discriminates_alpha_fac` | Type 1 — CPU `SurfaceHenneberg.gamma()` + anti-tautology `> 1e-6` | **acceptable** |
| `test_kernels_run_under_strict_transfer_guard` | Type 3 — transfer-guard contract + shape sanity | **acceptable** (routing) |
| `test_spec_can_be_jitted_through_pytree_registration` | Type 3 — pytree round-trip identity | **acceptable** (routing) |
| `test_gamma_parity_with_custom_quadpoints` | Type 1 — CPU `SurfaceHenneberg.gamma()` | **acceptable** |
| `test_gammadash1_parity_with_custom_quadpoints` | Type 1 — CPU | **acceptable** |

### `tests/field/test_trace_boozer_analytic_jax.py` (25 tests)

| Test | Oracle | Verdict |
|------|--------|---------|
| `test_dispatch_returns_analytic_evaluators_for_boozer_analytic_state` | Type 3 — routing identity check (NOT a re-export check; the identity verifies dispatch selection from a multi-branch function) | **acceptable** |
| `test_dispatch_returns_radial_evaluators_for_radial_state` | Type 3 — same routing identity | **acceptable** |
| `test_dispatch_raises_typeerror_on_unknown_state` | Type 3 — error contract | **acceptable** |
| `test_dispatch_exposes_complete_key_set` | Type 3 — explicit contract | **acceptable** |
| `test_vacuum_rhs_accepts_boozer_analytic_jax` (3 params) | Type 3 — finiteness contract | **acceptable** (smoke) |
| `test_no_k_rhs_accepts_boozer_analytic_jax` (3 params) | Type 3 — finiteness contract | **acceptable** (smoke) |
| `test_full_rhs_accepts_boozer_analytic_jax` (3 params) | Type 3 — finiteness contract | **acceptable** (smoke) |
| `test_factory_rejects_unknown_field_type` | Type 3 — error contract | **acceptable** (routing) |
| `test_vacuum_rhs_matches_cpu_oracle_closed_form` (3 params) | Type 1+2 — CPU `BoozerAnalytic` scalars + Type 2 closed-form RHS algebra from tracing.cpp | **acceptable** — strongest gate in the file |
| `test_no_k_rhs_matches_cpu_oracle_closed_form` (3 params) | Type 1+2 | **acceptable** |
| `test_full_rhs_matches_cpu_oracle_closed_form` (3 params) | Type 1+2 | **acceptable** |
| `test_full_rhs_with_K1_terms_diverges_from_no_k_oracle` | Type 2 — anti-tautology cross-mode comparison | **acceptable** |
| `test_dispatch_is_pure_python_not_jit_traced` | Type 3 — JIT-trace contract | **acceptable** (routing) |

## Math/contract verification

### N02 — InterpolatedBoozerFieldJAX

| C++ oracle | JAX implementation | Verdict |
|------------|-------------------|---------|
| `boozermagneticfield_interpolated.h:724-734` `exploit_fluxfunction_points` (zero theta/zeta) | `interpolated_boozer_field.py:556-566` `_zeroed_flux_points` | **matches** |
| `boozermagneticfield_interpolated.h:736-784` `exploit_symmetries_points` (theta mod 2π, zeta mod 2π/nfp, stellsym reflection above π) | `interpolated_boozer_field.py:230-284` `fold_points_for_symmetry` | **matches** — including `jnp.trunc` matching C++ `int()`-cast |
| `boozermagneticfield_interpolated.h:786-797` `apply_odd_symmetry` (single column 0 negation for both shape=1 and shape=3) | `interpolated_boozer_field.py:586-601` — split into `apply_odd` (size-1 / size-3 dispatch) + `apply_odd_vector_first_only` | **matches semantically** (size-3 branch is dead but redundant with `apply_odd_vector_first_only`; see MINOR-2) |
| `boozermagneticfield_interpolated.h:799-807` `apply_even_symmetry` (columns 1,2 of shape=3) | `interpolated_boozer_field.py:602-606` | **matches** |
| `boozermagneticfield_interpolated.h:190` `apply_odd_symmetry(K)` | `SYMMETRY_EXPLOIT_SCALARS["K"] = _SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:269` `apply_odd_symmetry(nu)` | `SYMMETRY_EXPLOIT_SCALARS["nu"] = _SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:329` `apply_odd_symmetry(dnuds)` | `_SymmetryRule(1, True, False, False)` | **matches** |
| `boozermagneticfield_interpolated.h:351` `apply_odd_symmetry(nu_derivs)` (shape=3, column 0 only) | `_SymmetryRule(3, False, True, False)` with `apply_odd_vector_first_only=True` | **matches** |
| `boozermagneticfield_interpolated.h:392` `apply_odd_symmetry(dRdtheta)` | `_SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:414` `apply_odd_symmetry(dRdzeta)` | `_SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:455` `apply_even_symmetry(R_derivs)` | `_SymmetryRule(3, False, False, True)` with `apply_even=True` | **matches** |
| `boozermagneticfield_interpolated.h:477` `apply_odd_symmetry(Z)` | `_SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:537` `apply_odd_symmetry(dZds)` | `_SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:559` `apply_odd_symmetry(Z_derivs)` | `_SymmetryRule(3, False, True, False)` with `apply_odd_vector_first_only=True` | **matches** |
| `boozermagneticfield_interpolated.h:600` `apply_odd_symmetry(dmodBdtheta)` | `_SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:622` `apply_odd_symmetry(dmodBdzeta)` | `_SymmetryRule(1, True, ...)` | **matches** |
| `boozermagneticfield_interpolated.h:663` `apply_even_symmetry(modB_derivs)` | `_SymmetryRule(3, False, False, True)` | **matches** |
| Scalar inventory (no apply_*): `modB, dmodBds, dKdtheta, dKdzeta, K_derivs, dnudtheta, dnudzeta, R, dRds, dZdtheta, dZdzeta, d2modBdtheta2, d2modBdzeta2, d2modBdthetadzeta` | `_SymmetryRule(..., False, False, False)` for each | **matches** |

**N02 verdict: matches the C++ symmetry-fold semantic line-for-line.**

### N04b — SurfaceHenneberg

| CPU oracle (`src/simsopt/geo/surfacehenneberg.py`) | JAX implementation (`src/simsopt/jax_core/surface_henneberg.py`) | Verdict |
|------|-------|---------|
| Lines 597-602 `R0H[n], Z0nH[n], bn[n]` with `Z0nH[0]=0` (loop starts at `n=1`) | Lines 141-145 `z_mask` zeroes `Z0nH[0]`; `R0H = cos_ang @ R0nH`; `b = cos_ang @ bn`; `Z0H = sin_ang @ z_modes` | **matches** |
| Lines 605-616 `rho_realsp` sum: `Σ_{m,n} rhomn(m,n) cos(m·θ + (nfp·n - α)·φ)` with `(m=0, n<=0)` skip via `nmin = 1 if m == 0 else -nmax` | Lines 82-91 `valid` mask zeroes `(m=0, n<=nmax)` rows; lines 196-200 einsum with `masked_rhomn` | **matches** — mask `valid[0, :nmax+1] = 0.0` zeros the `n ∈ {-nmax..0}` entries in `m=0` row |
| Lines 660-697 `gamma_lin` / `gamma_impl`: `R = R0H + ρ·cos(αφ) - ζ·sin(αφ)`, `Z = Z0H + ρ·sin(αφ) + ζ·cos(αφ)`, `ζ = b·sin(θ - α·φ)`, `data[xyz] = R cos φ / R sin φ / Z` then `.T` to `(nphi, ntheta)` | Lines 213-241 `surface_henneberg_gamma_from_spec` with `indexing="ij"` meshgrid (no `.T` needed) | **matches** |
| Lines 716-789 `gammadash1_impl`: derivatives with `-sin·nfp·n` for R/b modes, `+cos·nfp·n` for Z mode; `d_ζ_dφ = (db/dφ)·sin(θ̄) - b·cos(θ̄)·α`; product rule for `R cos φ` / `R sin φ`; trailing `2π` multiplier | Lines 244-307 `surface_henneberg_gammadash1_from_spec` — `phi_modes` provides `dR0H_dphi/dZ0H_dphi/db_dphi`; `dzeta_dphi = db_2d * sin_tbar - b_2d * cos_tbar * alpha`; `dx_dphi = dR_dphi cos_phi - R sin_phi`; final `two_pi * stack(...)` | **matches** |
| Lines 791-824 `gammadash2_impl`: `d_ρ_dθ = -ρ_{m,n}·m·sin(...)`; `d_ζ_dθ = b·cos(θ-α·φ)`; `d_R_dθ = d_ρ_dθ cos(αφ) - d_ζ_dθ sin(αφ)`; trailing `2π` | Lines 310-348 `surface_henneberg_gammadash2_from_spec` — same formulas via `drho_dtheta`, `dzeta_dtheta`; `two_pi * stack(...)` | **matches** |
| `Σ_n` start: CPU has `for n in range(1, nmax+1)` for Z0H (line 597, 745) | JAX: `z_mask[0] = 0.0` (line 97), so `Z0nH * z_mask` zeros n=0 in `Σ cos_ang @ z_modes` | **matches** |
| Final 2π multiplier scope: CPU multiplies the per-component derivative (lines 787-789, 822-824); JAX multiplies the stacked output | Both yield identical numerical result | **matches** |

**N04b verdict: matches CPU `gamma_impl`/`gammadash1_impl`/`gammadash2_impl` line-for-line, including the `(m=0, n<=0)` skip rule, the `Z0nH[0]=0` convention, and the `2π` rescaling for derivatives.**

### N03 §D — BoozerAnalyticJAX tracing dispatch

| C++ oracle (`src/simsoptpp/tracing.cpp`) | JAX implementation (`src/simsopt/jax_core/tracing.py`) | Verdict |
|------|-------|---------|
| Lines 106-128 `GuidingCenterVacuumBoozerRHS::operator()`: `fak1 = m v² / |B| + m μ`; `ds = -|B|_θ · fak1 / (q ψ₀)`; `dθ = |B|_s · fak1 / (q ψ₀) + ι v |B|/G`; `dζ = v |B|/G`; `dv_par = -(ι |B|_θ + |B|_ζ) μ |B|/G` | Lines 2065-2082 `guiding_center_vacuum_boozer_rhs.rhs` | **matches** |
| Lines 160-186 `GuidingCenterNoKBoozerRHS::operator()`: `D = ((q + m v dI'/|B|) G - (-qι + m v dG'/|B|) I)/ι`; `ds = (I |B|_ζ - G |B|_θ) fak1 / (D ι ψ₀)`; `dθ = (G |B|_ψ fak1 - (-qι + m v dG'/|B|) v |B|) / (D ι)`; `dζ = ((q + m v dI'/|B|) v |B| - |B|_ψ fak1 I) / (D ι)`; `dv_par = -(μ/v) · (|B|_ψ · ds · ψ₀ + |B|_θ · dθ + |B|_ζ · dζ)` | Lines 2121-2159 `guiding_center_no_k_boozer_rhs.rhs` | **matches** |
| Lines 218-252 `GuidingCenterBoozerRHS::operator()`: `C = -m v (K_ζ - G')/|B| - qι`; `F = -m v (K_θ - I')/|B| + q`; `D = (F G - C I)/ι`; `ds = (I |B|_ζ - G |B|_θ) fak1 / (D ι ψ₀)`; `dθ = (G |B|_ψ fak1 - C v |B| - K fak1 |B|_ζ)/(D ι)`; `dζ = (F v |B| - |B|_ψ fak1 I + K fak1 |B|_θ)/(D ι)`; same `dv_par` formula | Lines 2198-2236 `guiding_center_boozer_rhs.rhs` | **matches** |
| Lazy import contract for evaluator dispatch | `_boozer_field_evaluators` at lines 1922-2017 dispatches `isinstance(state, BoozerAnalyticFrozenState)` vs `isinstance(state, BoozerRadialInterpolantFrozenState)`; raises `TypeError` on unknown | **matches** the documented contract (see MINOR-4 for lazy-import nit) |
| `_linear_state_at` deletion | Verified no references in `src/` or `tests/`. Dead code safely removed. | **safe deletion** |

**N03 §D verdict: all three RHS factories preserve the C++ algebra line-for-line.**

## Toolchain verification

- **`ruff check`**: PASSED. `ruff check src/simsopt/jax_core/interpolated_boozer_field.py src/simsopt/jax_core/surface_henneberg.py src/simsopt/jax_core/tracing.py src/simsopt/jax_core/specs.py src/simsopt/field/boozermagneticfield_jax.py src/simsopt/geo/surfacehenneberg.py src/simsopt/jax_core/__init__.py tests/field/test_interpolated_boozer_field_jax.py tests/geo/test_surface_henneberg_jax.py tests/field/test_trace_boozer_analytic_jax.py` → `All checks passed!`
- **N02 test count**: `35 passed, 2 warnings in 4.84s` (matches state.json).
- **N04b test count**: `74 passed in 3.13s` (matches state.json: "84 passed (10 N04a + 74 N04b)").
- **N03 §D test count**: `25 passed, 2 warnings in 2.50s` (matches state.json).
- **Regression checks**:
  - **item-33** (`tests/field/test_boozermagneticfield_jax_item33.py`): PASSED.
  - **N01** (`tests/field/test_boozer_analytic_jax.py`): PASSED.
  - **N04a** (`tests/geo/test_surface_garabedian_jax.py`): 10 passed.
  - **item-14** (`tests/jax_core/test_tracing_jax_item14.py`): 2 PRE-EXISTING failures unrelated to this changeset. Confirmed by `git stash`/`pytest` round-trip on the parent commit `ca276abbd`. These failures are part of the silent-fallback removal closed by `ef946054b`.
- **Composite tests run together**: `45 passed` across item-33, N01, N04a in one command.

## Mistake book entries

1. **C++ apply_odd_symmetry has two arms (shape=1 vs shape=3) but both do the same thing (negate column 0).** The JAX port chose to encode this as two boolean flags (`apply_odd` and `apply_odd_vector_first_only`) instead of one. Future ports should consider whether to mirror the C++ dual-arm dispatch (one flag + value_size lookup) or split into clearer roles.

2. **`@dataclass(frozen=True)` does NOT freeze container contents (lists, dicts).** When a container is the load-bearing payload (like `specs: dict` in `InterpolatedBoozerFieldFrozenState`), the "frozen" name is misleading. Either rename or wrap the container.

3. **Routing tests can use `is` identity checks legitimately when the function under test is a dispatcher.** `assert evals["modB"] is _analytic_modB` is acceptable for `_boozer_field_evaluators` because the dispatcher selects between multiple callables; it would be tautological for a single-branch re-export.

4. **Coverage of multi-shape kernels requires fixture diversity.** The N02 test suite covers only the 14 scalars that `BoozerAnalytic` happens to implement (all value_size=1). The 3-vector symmetry branches (`Z_derivs`, `R_derivs`, `nu_derivs`, `modB_derivs`) are wired but not exercised against an oracle. Synthetic fixtures (closed-form `Z(s, θ, ζ)`) can cover this without a VMEC dependency.

5. **Lazy imports inside dispatch helpers need explicit justification.** Comments like "in some import orderings" are signals that the actual cycle has not been characterized. If the cycle is real, name the orderings; if not, lift the imports.

6. **The `evaluate_batch` call site lifts host arrays to device on every call.** When a spec is queried in a hot loop, this becomes O(N) transfers. Bake `jnp.asarray` once during spec construction, not at evaluation time.

## Iteration 2 verdict

**PASS** — the iteration-1 MAJOR finding (3-vector symmetry-branch coverage gap) is closed.

**Audit of the 4 new tests in `tests/field/test_interpolated_boozer_field_jax.py`**:

1. `test_modB_derivs_3vector_apply_even_symmetry_parity` (lines 159-222): integration test against `BoozerAnalytic.modB_derivs()` QH N=2 fixture (type-1 CPU oracle, independent of the JAX evaluator). Confirmed via direct `fold_points_for_symmetry` execution that the 12-point fixture splits cleanly into 6 no-flip and 6 flipped samples — both `apply_even` branches are exercised. Non-triviality assertions (`max(abs(flipped[:,1])) > 1e-3` and same for col 2) guard against the QA-N=0 trap where `dmodBdzeta == 0` would render the test vacuous. Tolerance `rtol=1e-4, atol=1e-6` is justified by the degree-6 Lagrange truncation budget on the 8×8×8 grid.
2-4. The three direct unit tests on `_apply_symmetry` use synthetic `(N, 3)` or `(N, 1)` arrays with explicit `flipped` masks and closed-form expected outputs (type-2 oracle). Bit-exact tolerance `rtol=0, atol=0` is correct — these are pure sign flips with no floating-point arithmetic beyond multiplication by ±1.0.

All 4 tests are oracle-clean per `REVIEWER_ORACLE_LINT.md`: no `is`-identity tautologies, no JAX-vs-JAX comparisons, no self-driver passes. The 39-test N02 suite passes; ruff clean. No new CRITICAL/MAJOR introduced. MINOR-1 through MINOR-5 and LOW-1 from iteration 1 remain unaddressed but are non-blocking by their original classification.
