# Test Tautology / Oracle-Quality Adversarial Audit
Date: 2026-05-13

Auditor: Claude (Opus 4.7, adversarial-reviewer role)
Branch: `gpu-purity-stage2-20260405`
Lint spec: `tests/REVIEWER_ORACLE_LINT.md`

Test files audited:
1. `tests/field/test_interpolated_boozer_field_jax.py` — 39 parametrize-expanded tests (24 unique functions)
2. `tests/geo/test_surface_henneberg_jax.py` — 74 parametrize-expanded tests (21 unique functions)
3. `tests/field/test_trace_boozer_analytic_jax.py` — 25 parametrize-expanded tests (13 unique functions)

Total: 138 tests collected (verified via `pytest --collect-only`).

## Verdict

**PASS** — 0 REJECT findings.

Every test has a documented oracle from one of the four allowed types
(CPU C++/Python reference symbol, closed-form analytic expression,
shape/dtype contract, or `pytest.raises` error contract). Verified
oracle independence:

- `BoozerAnalytic` (CPU, `src/simsopt/field/boozermagneticfield.py:110`)
  is a pure-NumPy class. `grep` confirms zero `import jax` / `jax_core`
  references in the module. Its `_modB_impl` uses `np.cos` directly.
  Type-1 oracle.
- `SurfaceHenneberg` (CPU, `src/simsopt/geo/surfacehenneberg.py:21`)
  inherits `sopp.Surface` (C++ binding) and its `gamma_impl` /
  `gammadash1_impl` / `gammadash2_impl` use only NumPy. The
  `area()` / `volume()` / `normal()` accessors come from the C++
  `sopp.Surface` base. Type-1 oracle.
- The JAX kernels under test (`simsopt.jax_core.boozer_analytic._eval_*`,
  `simsopt.field.boozermagneticfield_jax._eval_*`,
  `simsopt.jax_core.surface_henneberg.surface_henneberg_*_from_spec`)
  are entirely disjoint from the oracle implementations.
- The closed-form RHS algebra in `_vacuum_rhs_cpu_oracle` /
  `_no_k_rhs_cpu_oracle` / `_full_rhs_cpu_oracle` operates on Python
  `float` values returned by `BoozerAnalytic`; it contains zero
  `jax.numpy` / `jnp` / `simsopt.jax_core` references. Type-2 oracle.

## Audit table: `tests/field/test_interpolated_boozer_field_jax.py`

| Test name | Oracle type | Specific oracle reference | Tautology check | Verdict |
|-----------|-------------|---------------------------|------------------|---------|
| test_scalar_parity_to_boozer_analytic_oracle[modB] | 1 | `BoozerAnalytic.modB()` — NumPy closed-form `B0*(1+etabar*r*cos(theta-N*zeta))` | Independent CPU Python (no JAX import) | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[dmodBdtheta] | 1 | `BoozerAnalytic.dmodBdtheta()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[dmodBdzeta] | 1 | `BoozerAnalytic.dmodBdzeta()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[dmodBds] | 1 | `BoozerAnalytic.dmodBds()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[K] | 1 | `BoozerAnalytic.K()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[dKdtheta] | 1 | `BoozerAnalytic.dKdtheta()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[dKdzeta] | 1 | `BoozerAnalytic.dKdzeta()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[G] | 1 | `BoozerAnalytic.G()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[I] | 1 | `BoozerAnalytic.I()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[iota] | 1 | `BoozerAnalytic.iota()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[dGds] | 1 | `BoozerAnalytic.dGds()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[dIds] | 1 | `BoozerAnalytic.dIds()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[diotads] | 1 | `BoozerAnalytic.diotads()` | Independent CPU Python | ACCEPT |
| test_scalar_parity_to_boozer_analytic_oracle[psip] | 1 | `BoozerAnalytic.psip()` | Independent CPU Python | ACCEPT |
| test_modB_derivs_3vector_apply_even_symmetry_parity | 1 | `BoozerAnalytic.modB_derivs()` plus QH (N=2) seed to actually exercise component-2 negation; bounded-magnitude sanity assertions prevent vacuous parity | Independent CPU Python; the apply_even branch is exercised by the points_will_flip half (theta>pi) and the sanity check confirms non-trivial component-1/2 magnitudes | ACCEPT |
| test_apply_symmetry_odd_vector_first_only_negates_component_zero | 2 | Hand-written expected `[[-1,2,3],[4,5,6],[7,8,9]]` — closed-form algebra, not derived from the kernel | Inputs/outputs constructed by hand; rtol=atol=0 (exact equality) | ACCEPT |
| test_apply_symmetry_apply_even_negates_components_one_two | 2 | Hand-written expected `[[1,-2,-3],[4,5,6]]` | Inputs/outputs by hand; exact equality | ACCEPT |
| test_apply_symmetry_apply_odd_scalar_negates_scalar_for_flipped | 2 | Hand-written expected `[[-1.5],[-2.5],[-3.5]]` | Inputs/outputs by hand; exact equality | ACCEPT |
| test_modB_parity_across_nfp_values[1] | 1 | `BoozerAnalytic.modB()` (independent of nfp scan) | Independent CPU Python | ACCEPT |
| test_modB_parity_across_nfp_values[2] | 1 | same | same | ACCEPT |
| test_modB_parity_across_nfp_values[3] | 1 | same | same | ACCEPT |
| test_modB_parity_across_nfp_values[5] | 1 | same | same | ACCEPT |
| test_fold_points_modular_theta_into_0_2pi | 2 | Closed-form modular invariant `theta mod 2pi ∈ [0,2pi]`; cited C++ source line `boozermagneticfield_interpolated.h:765-768` as algebra reference | Bounded-region test, not a re-derivation of the fold | ACCEPT |
| test_fold_points_stellsym_reflection_above_pi | 2 | Hand-written closed-form reflection `theta=2*pi-theta`, `zeta=period-zeta`; cited C++ `:769-779` | Expected values written by hand using closed-form rule | ACCEPT |
| test_modB_invariant_under_theta_2pi_shift | 2 | Closed-form periodicity identity `modB(theta) == modB(theta + 2*pi)` | The identity is a theorem about the kernel, not a re-export of the kernel. The kernel computes both legs but the test asserts they coincide — a non-trivial property | ACCEPT |
| test_simsopt_jax_native_field_marker_is_set | contract | `_simsopt_jax_native_field == True` class-attribute contract | Routing/contract test (no numerical claim) | ACCEPT (contract) |
| test_set_points_shape_validation | contract | `pytest.raises(ValueError)` | Error-contract test, no oracle required | ACCEPT (contract) |
| test_set_points_invalidates_cache | 1 + 2 | `BoozerAnalytic.modB()` after two distinct `set_points()` calls; closed-form non-triviality via `assert not np.allclose(modB1, modB2)` | Independent CPU oracle re-checked for both inputs; non-triviality assertion guards against vacuous cache-hit | ACCEPT |
| test_unbuilt_scalar_raises_keyerror_from_frozen_state_wrapper | contract | `pytest.raises(KeyError, match='R')` | Error-contract test (no parity claim) | ACCEPT (contract) |
| test_from_frozen_state_round_trip | 1 | The original wrapper's `modB()` output (which was itself CPU-parity-validated by `test_scalar_parity_to_boozer_analytic_oracle`) | Lateral check on the alternate constructor; rtol=atol=0 means bit-identity, which is appropriate for a from_frozen_state round-trip | ACCEPT (routing test on the alternate constructor) |
| test_frozen_state_inventory_matches_documented_split | contract | `ALL_SCALARS` vs explicit literal set | Inventory check, no parity claim | ACCEPT (contract) |
| test_freeze_state_metadata_round_trip | 2 + contract | Arithmetic identity `period = 2*pi/3` plus metadata-roundtrip contract | Hand-written expected values | ACCEPT |
| test_freeze_state_rejects_unknown_scalar_name | contract | `pytest.raises(ValueError, match='not_a_real_scalar')` | Error-contract test | ACCEPT (contract) |
| test_freeze_state_rejects_invalid_degree | contract | `pytest.raises(ValueError)` | Error-contract test | ACCEPT (contract) |
| test_freeze_state_rejects_invalid_nfp | contract | `pytest.raises(ValueError)` | Error-contract test | ACCEPT (contract) |
| test_non_stellsym_wrapper_does_not_flip | 2 | Closed-form algebra: `flipped == False` when `stellsym=False`; theta stays at 4.5 (no reflection) | Hand-written expected values; cites algebra contract at fold_points_for_symmetry:282 | ACCEPT |
| test_non_stellsym_parity_for_qa_field | 1 | `BoozerAnalytic.modB()` for stellsym=False wrapper | Independent CPU oracle | ACCEPT |
| test_wrapper_modB_with_lazy_built_specs_returns_array | contract | shape `(2,1)`, dtype `float64` | Shape/dtype contract, no numerical oracle needed | ACCEPT (contract) |
| test_qh_helicity_field_parity | 1 | `BoozerAnalytic` with N=2 helicity, finer grid | Independent CPU oracle | ACCEPT |

**Subtotal**: 39 ACCEPT / 0 REJECT.

## Audit table: `tests/geo/test_surface_henneberg_jax.py`

| Test name (parametrize expansion) | Oracle type | Specific oracle reference | Tautology check | Verdict |
|-----------------------------------|-------------|---------------------------|------------------|---------|
| test_to_spec_round_trips_fields[*] (16 variants) | contract | `surface.X` (host class state) for plumbing only — docstring explicitly says "round-trip plumbing, not the geometry math" | Honest plumbing test; not a parity claim about kernel | ACCEPT (contract, plumbing) |
| test_gamma_matches_cpu_oracle[*] (12 variants) | 1 | `SurfaceHenneberg.gamma_impl` at surfacehenneberg.py:698-714 (NumPy, sopp.Surface C++ binding base) | Independent CPU implementation: gamma_impl uses np.meshgrid/np.reshape and self.gamma_lin (C++); no `import jax` in the module | ACCEPT |
| test_gamma_matches_cpu_oracle_across_nfp[1/2/3/5] (4 variants) | 1 | same as above for varied nfp | Independent CPU | ACCEPT |
| test_gammadash1_matches_cpu_oracle[*] (12 variants) | 1 | `SurfaceHenneberg.gammadash1_impl` at surfacehenneberg.py:716-789 — NumPy with np.cos/np.sin loops | Independent CPU Python | ACCEPT |
| test_gammadash2_matches_cpu_oracle[*] (12 variants) | 1 | `SurfaceHenneberg.gammadash2_impl` at surfacehenneberg.py:791-866 — NumPy | Independent CPU Python | ACCEPT |
| test_normal_matches_cpu_oracle[-1/0/1] (3 variants) | 1 | `sopp.Surface.normal` (C++ base class) composing CPU gammadash1 x gammadash2 | C++/Python independent of JAX | ACCEPT |
| test_unitnormal_matches_cpu_oracle | 1 | `sopp.Surface.unitnormal` | C++/Python independent | ACCEPT |
| test_area_matches_cpu_oracle | 1 | `sopp.Surface.area` (riemann sum over |normal|) | C++/Python independent | ACCEPT |
| test_volume_matches_cpu_oracle | 1 | `sopp.Surface.volume` (riemann sum over gamma·n) | C++/Python independent | ACCEPT |
| test_axisymmetric_default_torus_matches_analytic | 2 | Closed-form torus `R=1+0.1*cos(theta)`, `Z=0.1*sin(theta)` — hand-written algebra | Hand-derived analytic expression; not a re-implementation of the kernel | ACCEPT |
| test_axisymmetric_gammadash2_matches_analytic | 2 | Hand-derived `2*pi*(-0.1*sin(theta)*cos(phi), -0.1*sin(theta)*sin(phi), 0.1*cos(theta))` | Hand-written analytic differentiation of closed-form torus | ACCEPT |
| test_make_spec_rejects_invalid_alpha_fac | contract | `pytest.raises(ValueError, match='alpha_fac must be one of')` | Error-contract test | ACCEPT (contract) |
| test_make_spec_rejects_shape_mismatch | contract | `pytest.raises(ValueError, match='rhomn shape mismatch')` | Error-contract test | ACCEPT (contract) |
| test_spec_is_frozen_dataclass | contract | `pytest.raises(dataclasses.FrozenInstanceError)` | Error-contract test | ACCEPT (contract) |
| test_spec_register_dataclass_keys_match_host_class | contract | Tree-util registration: 6 leaves and `meta_tuple == (2,1,1,1)` — explicit literal expectations | Contract check on JAX pytree registration | ACCEPT (contract) |
| test_spec_jit_round_trip_preserves_meta_fields | contract | Hand-written expectations on meta_fields after JIT | Contract check, R0nH path-through is identity (`s.R0nH + 0.0`) | ACCEPT (contract) |
| test_jit_cache_discriminates_alpha_fac | 1 + non-triviality | Both `gamma_pos` and `gamma_neg` independently checked against their own CPU `surface.gamma()` oracle, then asserts `max|pos-neg| > 1e-6` to confirm meta_field actually influences output | The non-triviality assertion has a meaningful claim: changing alpha_fac changes every mode of rho/zeta, so gamma must differ | ACCEPT |
| test_kernels_run_under_strict_transfer_guard | contract | Shape contract under `jax.transfer_guard("disallow")` — verifies kernels stay on-device | Routing/contract test (no numerical parity claim) | ACCEPT (contract) |
| test_spec_can_be_jitted_through_pytree_registration | contract | `np.testing.assert_array_equal` on rhomn/R0nH after `tree_map(negate_then_negate)` where `-(-x)==x` is the closed-form identity | Pytree registration contract test using an identity function | ACCEPT (contract) |
| test_gamma_parity_with_custom_quadpoints | 1 | `SurfaceHenneberg.gamma()` on custom quadpoint grid | Independent CPU Python | ACCEPT |
| test_gammadash1_parity_with_custom_quadpoints | 1 | `SurfaceHenneberg.gammadash1()` on custom quadpoint grid | Independent CPU Python | ACCEPT |

**Subtotal**: 74 ACCEPT / 0 REJECT.

## Audit table: `tests/field/test_trace_boozer_analytic_jax.py`

| Test name (parametrize expansion) | Oracle type | Specific oracle reference | Tautology check | Verdict |
|-----------------------------------|-------------|---------------------------|------------------|---------|
| test_dispatch_returns_analytic_evaluators_for_boozer_analytic_state | contract | `is`-identity routing check (`evals['modB'] is _analytic_modB`) | Routing test — dispatch correctness | ACCEPT (routing) |
| test_dispatch_returns_radial_evaluators_for_radial_state | contract | `is`-identity routing check on radial-state path | Routing test | ACCEPT (routing) |
| test_dispatch_raises_typeerror_on_unknown_state | contract | `pytest.raises(TypeError)` with message-content asserts | Error-contract test | ACCEPT (contract) |
| test_dispatch_exposes_complete_key_set | contract | Explicit literal key-set assertion | Inventory contract | ACCEPT (contract) |
| test_vacuum_rhs_accepts_boozer_analytic_jax[qa_standard/qh_symmetric/finite_beta_full] (3 variants) | contract | Shape `(4,)` and `jnp.all(jnp.isfinite(out))` | Smoke/acceptance test, parity covered separately | ACCEPT (contract) |
| test_no_k_rhs_accepts_boozer_analytic_jax[*] (3 variants) | contract | Same shape+finiteness | Smoke/acceptance | ACCEPT (contract) |
| test_full_rhs_accepts_boozer_analytic_jax[*] (3 variants) | contract | Same shape+finiteness | Smoke/acceptance | ACCEPT (contract) |
| test_factory_rejects_unknown_field_type | contract | `pytest.raises(TypeError)` | Error-contract | ACCEPT (contract) |
| test_vacuum_rhs_matches_cpu_oracle_closed_form[qa_standard/qh_symmetric/finite_beta_full] (3 variants) | 1+2 | CPU `BoozerAnalytic` scalars (type 1) composed with closed-form `_vacuum_rhs_cpu_oracle` (type 2). Verified: `_vacuum_rhs_cpu_oracle` operates on Python floats only — no `jnp`, no `jax`, no `simsopt.jax_core` references in the function body | Two-stage independent oracle: CPU scalars are pure NumPy from `BoozerAnalytic`; algebra is hand-written Python arithmetic. Neither touches the JAX kernel under test | ACCEPT |
| test_no_k_rhs_matches_cpu_oracle_closed_form[*] (3 variants) | 1+2 | Same as above with `_no_k_rhs_cpu_oracle` algebra | Same; closed-form `dGdpsi`, `dIdpsi`, D coefficient arithmetic in pure Python | ACCEPT |
| test_full_rhs_matches_cpu_oracle_closed_form[*] (3 variants) | 1+2 | Same as above with `_full_rhs_cpu_oracle` algebra (extends to K, dK/dtheta, dK/dzeta) | Same composition | ACCEPT |
| test_full_rhs_with_K1_terms_diverges_from_no_k_oracle | 2 + non-triviality | Closed-form algebra confirms full vs no-K differ by C and F coefficient terms in `_full_rhs_cpu_oracle` vs `_no_k_rhs_cpu_oracle`; `assert not np.allclose(no_k, full, rtol=1e-6, atol=1e-9)` | Both oracles built from independent CPU scalars + closed-form algebra (no JAX); the divergence claim is meaningful (K1!=0 introduces additional algebraic terms) | ACCEPT |
| test_dispatch_is_pure_python_not_jit_traced | contract | `callable(rhs)` and successful factory construction | Static-dispatch contract test | ACCEPT (contract) |

**Subtotal**: 25 ACCEPT / 0 REJECT.

## Findings by severity

### REJECT — tautologies
None.

### REJECT — missing oracle
None.

### REJECT — hacky/meaningless
None.

### ACCEPT — routing / contract tests (no oracle needed)
- `test_simsopt_jax_native_field_marker_is_set` — class-attribute contract
- `test_set_points_shape_validation` — error contract
- `test_unbuilt_scalar_raises_keyerror_from_frozen_state_wrapper` — error contract
- `test_frozen_state_inventory_matches_documented_split` — inventory contract
- `test_freeze_state_rejects_unknown_scalar_name` — error contract
- `test_freeze_state_rejects_invalid_degree` — error contract
- `test_freeze_state_rejects_invalid_nfp` — error contract
- `test_wrapper_modB_with_lazy_built_specs_returns_array` — shape/dtype contract
- `test_to_spec_round_trips_fields` (16 expansions) — plumbing/round-trip contract (honest docstring states this is not a parity claim)
- `test_make_spec_rejects_invalid_alpha_fac` — error contract
- `test_make_spec_rejects_shape_mismatch` — error contract
- `test_spec_is_frozen_dataclass` — error contract
- `test_spec_register_dataclass_keys_match_host_class` — pytree-registration contract
- `test_spec_jit_round_trip_preserves_meta_fields` — pytree-registration contract
- `test_kernels_run_under_strict_transfer_guard` — JAX-transfer-guard contract
- `test_spec_can_be_jitted_through_pytree_registration` — pytree-registration contract using `-(-x)` identity (closed-form)
- `test_dispatch_returns_analytic_evaluators_for_boozer_analytic_state` — dispatch routing (`is`-identity per the lint's explicit allowance)
- `test_dispatch_returns_radial_evaluators_for_radial_state` — dispatch routing
- `test_dispatch_raises_typeerror_on_unknown_state` — error contract
- `test_dispatch_exposes_complete_key_set` — inventory contract
- `test_vacuum_rhs_accepts_boozer_analytic_jax` (3 variants) — shape/finiteness smoke
- `test_no_k_rhs_accepts_boozer_analytic_jax` (3 variants) — shape/finiteness smoke
- `test_full_rhs_accepts_boozer_analytic_jax` (3 variants) — shape/finiteness smoke
- `test_factory_rejects_unknown_field_type` — error contract
- `test_dispatch_is_pure_python_not_jit_traced` — static-dispatch contract

### ACCEPT — oracle-clean parity / closed-form analytic
- `test_scalar_parity_to_boozer_analytic_oracle` (14 scalars) — type-1 `BoozerAnalytic` (verified pure-NumPy, no JAX imports)
- `test_modB_derivs_3vector_apply_even_symmetry_parity` — type-1 with apply_even branch exercised and a bounded-magnitude sanity guard
- `test_apply_symmetry_odd_vector_first_only_negates_component_zero` — type-2 hand-written expected values
- `test_apply_symmetry_apply_even_negates_components_one_two` — type-2 hand-written
- `test_apply_symmetry_apply_odd_scalar_negates_scalar_for_flipped` — type-2 hand-written
- `test_modB_parity_across_nfp_values` (4 variants) — type-1 `BoozerAnalytic.modB`
- `test_fold_points_modular_theta_into_0_2pi` — type-2 modular invariant
- `test_fold_points_stellsym_reflection_above_pi` — type-2 hand-written reflection
- `test_modB_invariant_under_theta_2pi_shift` — type-2 periodicity identity (non-trivial property of the kernel under test)
- `test_set_points_invalidates_cache` — type-1 + non-triviality
- `test_from_frozen_state_round_trip` — lateral check on alternate constructor against bit-identity oracle
- `test_freeze_state_metadata_round_trip` — type-2 arithmetic identity
- `test_non_stellsym_wrapper_does_not_flip` — type-2 hand-written
- `test_non_stellsym_parity_for_qa_field` — type-1 `BoozerAnalytic`
- `test_qh_helicity_field_parity` — type-1 `BoozerAnalytic` (N=2 helicity case)
- `test_gamma_matches_cpu_oracle` (12 variants) — type-1 `SurfaceHenneberg.gamma_impl` (verified pure-NumPy)
- `test_gamma_matches_cpu_oracle_across_nfp` (4 variants) — same
- `test_gammadash1_matches_cpu_oracle` (12 variants) — type-1
- `test_gammadash2_matches_cpu_oracle` (12 variants) — type-1
- `test_normal_matches_cpu_oracle` (3 variants) — type-1 `sopp.Surface.normal`
- `test_unitnormal_matches_cpu_oracle` — type-1
- `test_area_matches_cpu_oracle` — type-1
- `test_volume_matches_cpu_oracle` — type-1
- `test_axisymmetric_default_torus_matches_analytic` — type-2 closed-form torus
- `test_axisymmetric_gammadash2_matches_analytic` — type-2 closed-form differentiation
- `test_jit_cache_discriminates_alpha_fac` — type-1 (two CPU oracle checks) + non-triviality
- `test_gamma_parity_with_custom_quadpoints` — type-1 on custom grid
- `test_gammadash1_parity_with_custom_quadpoints` — type-1 on custom grid
- `test_vacuum_rhs_matches_cpu_oracle_closed_form` (3 variants) — type-1 (CPU `BoozerAnalytic` scalars) + type-2 (hand-written `_vacuum_rhs_cpu_oracle` algebra). Verified: oracle function operates on Python floats only (no `jnp`, no `jax`, no `simsopt.jax_core` references)
- `test_no_k_rhs_matches_cpu_oracle_closed_form` (3 variants) — same composition with no_K algebra
- `test_full_rhs_matches_cpu_oracle_closed_form` (3 variants) — same composition with full-K algebra
- `test_full_rhs_with_K1_terms_diverges_from_no_k_oracle` — type-2 non-triviality

## Oracle independence verification

Per the lint, the key risk is that a "CPU oracle" silently routes through the
JAX kernel under test. We verified independence by direct file inspection:

```
$ grep -n "import jax\|jax_core\|jax.numpy\|jnp\." src/simsopt/field/boozermagneticfield.py
(no matches)

$ grep -n "import jax\|jax_core\|jax.numpy\|jnp\." src/simsopt/geo/surfacehenneberg.py
328:        from ..jax_core.specs import make_surface_henneberg_spec
```

The single match in `surfacehenneberg.py:328` is inside `to_spec()` and
imports the spec **constructor** for plumbing (Python copy of host arrays
into a dataclass). It does **not** invoke any JAX kernel and is unrelated
to `gamma_impl` / `gammadash1_impl` / `gammadash2_impl`, which only use
NumPy and `self.gamma_lin` (C++ via `sopp.Surface`).

`BoozerAnalytic._modB_impl` uses `np.cos` directly
(`boozermagneticfield.py:243`). The class is implemented in pure NumPy
with zero JAX dependence.

The closed-form RHS oracle functions in `test_trace_boozer_analytic_jax.py`
(`_vacuum_rhs_cpu_oracle` at lines 339-361, `_no_k_rhs_cpu_oracle` at
364-401, `_full_rhs_cpu_oracle` at 404-443) operate on Python `float`
values returned by `_cpu_scalars` (which only calls `BoozerAnalytic`).
No `jnp`, `jax.numpy`, or `simsopt.jax_core` imports appear in those
function bodies — they are pure Python arithmetic.

## Recommendation

PASS — accept all 138 tests as written. The three test files satisfy the
oracle-lint contract in `tests/REVIEWER_ORACLE_LINT.md`:

1. Every numerical-parity assertion cites an independent oracle (type 1
   CPU `BoozerAnalytic` / `SurfaceHenneberg` / `sopp.Surface`, or type 2
   closed-form analytic expression).
2. Contract tests (error behavior, shape, dtype, dispatch routing,
   inventory) are well-scoped and do not make spurious numerical claims.
3. The `is`-identity routing checks in `test_trace_boozer_analytic_jax.py`
   (`_dispatch_returns_*_evaluators_*`) are explicitly allowed by the
   lint as "ACCEPT — routing test, not a numerical parity test" — the
   goal is to verify the dispatch bound the correct function objects.
4. The "round-trip plumbing" test `test_to_spec_round_trips_fields` is
   honest about its scope ("certifies the round-trip plumbing, not the
   geometry math").
5. Non-triviality guards (`assert not np.allclose(...)`,
   `assert max|gamma_pos - gamma_neg| > 1e-6`,
   `assert max(abs(flipped_block[:,1])) > 1e-3`) prevent vacuous parity
   wins where two equal-but-trivial outputs would pass the parity
   assertion.

No remediation required.
