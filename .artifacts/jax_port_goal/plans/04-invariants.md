# Item 04 Math And Physics Invariants

Status: comprehensive for the in-scope item-04 modules.

## Boozer Residual Physical Basis

Decision vector `x = (s_dofs, iota, G)` (or `(s_dofs, iota)` when
`optimize_G=False`); `s_dofs` are the active surface Fourier coefficients
(`m`/`n`/component triples consistent with the surface type and
stellsym branch).

Boozer residual (vacuum field, `I = 0`):

```
G * B(γ) − ι B(γ)·∂γ/∂φ − B(γ)·∂γ/∂θ + (mod_B / mod_B_target − 1) * z_constraint_terms = 0
```

The kernel reports the residual in the **original physical basis** at the
quadrature points; no implicit basis transform or rescaling is applied at
the reporting boundary. A `weight_inv_modB` flag toggles the standard
`1/mod_B` weighting (per upstream `boozer_surface_residual`).

CPU oracle: `simsopt.geo.surfaceobjectives.boozer_surface_residual`
(line 340 in `src/simsopt/geo/surfaceobjectives.py`). The non-banana
parity test uses `weight_inv_modB=False` (the
`examples/2_Intermediate/boozer.py` example default); upstream pinned-input
parity uses both branches.

## Units And Scales

- Coil currents: amps (A); G has units of T·m, consistent with
  `G = μ₀ · sum |I_k|` (`compute_G_from_currents` at
  `label_constraints_jax.py:49-61`).
- B field: tesla (T); `mod_B = |B|`.
- Surface position γ: meters; surface tangents ∂γ/∂φ, ∂γ/∂θ in meters
  per dimensionless coordinate.
- Toroidal flux: Wb (`Φ = ∮ A·t dl ≈ (1/nθ) Σ A·γ_θ`,
  `label_constraints_jax.py:25-46`).
- Volume: m³; Area: m²; aspect ratio: dimensionless; iota:
  dimensionless rotational transform per toroidal turn.

## Current Sign Convention

- `G ≥ 0` is enforced indirectly via
  `_boozersurface_current_guard.require_fixed_currents_for_none_G`
  (CLAUDE.md: vacuum-field rationale; `I = 0` plasma current; G from
  `μ₀ · sum |I_k|` so sign of `G` is positive in the standard convention).
- When the user passes `G=None`, free coil currents are rejected so the
  derived `G` is well-defined
  (`tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_run_code_rejects_G_none_with_free_currents`).
- Residual is even under simultaneous sign flip of `(G, B)`; under sign
  flip of `iota` alone the residual changes (rotational direction).

## Orientation

- `gammadash1 = ∂γ/∂φ`, `gammadash2 = ∂γ/∂θ`; outward normal
  `n = γ_φ × γ_θ` (`surfaceobjectives_jax.py:_surface_normal_from_tangents`
  line 346). This is the upstream SIMSOPT convention; flipping
  `flip_theta` reverses the θ direction in the surface fit (used by the
  non-banana example fixture).
- `unitnormal` is normalized `n / |n|`; `surface_unit_normal` is unsigned
  on the surface and oriented outward.
- B·n test in the non-banana parity test
  (`tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py::test_boozer_surface_basic_passes_residual_and_label_parity`)
  compares scalar B·n at every quadrature point against the CPU oracle.

## Stellsym Coverage

- **`stellsym=True`** is the primary path; uses
  `stellsym_scatter_indices(mpol, ntor)` (cos-cos+sin-sin for x;
  cos-sin+sin-cos for y, z), matching the CPU
  `SurfaceXYZTensorFourier` DOF ordering exactly (verified by
  `TestStellsymScatterIndices`). The decision vector has the reduced free
  DOF count `n_active`; the kernel reconstructs the full surface basis.
- **`stellsym=False`** uses the full DOF set; the residual / Jacobian
  paths are equivalent up to the absence of the scatter step.
- Both branches are exercised by:
  - `tests/geo/test_label_constraints_jax.py::TestLabelConstraintsParity`
    (toroidal_flux / volume / area gradient FD over both branches).
  - `tests/geo/test_boozersurface_jax.py::TestParametrizedPenaltyGradientTaylor::test_gradient_taylor[stellsym=True/False-optimize_G=True/False]`
    (Taylor convergence matrix).
  - `tests/geo/test_boozersurface_jax.py::TestBoozerExactConstraintsJacobianTaylor::test_exact_jacobian_taylor_nonstellsym`
    / `::test_exact_jacobian_taylor_stellsym` (exact lane).
  - `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_exact_constraints_newton_nonstellsym_*`
    (non-stellsym exact public API).
  - `tests/geo/test_boozer_derivatives_jax.py::TestComposedStellsym::test_gradient_stellsym_fd`
    / `::test_decision_vector_shorter` (reduced decision vector under
    stellsym).
  - `tests/geo/test_boozersurface_jax.py::TestStellsymMaskCPUJAXParity`
    (mask CPU/JAX parity at multiple `(mpol, ntor, nfp)` combinations).

## Derivative Shape Contract

- Residual vector shape: `(3 * nphi * ntheta + n_constraints,)`; the
  number of constraints depends on `optimize_G` and the label type.
- `boozer_residual_grad`: shape `(n_active + n_dof_extras,)` where the
  extras are `iota` and optionally `G`.
- `boozer_residual_hessian`: shape `(n_active + extras, n_active + extras)`.
- `boozer_residual_jacobian_composed`: shape `(n_residual, n_decision)`.
- `boozer_residual_coil_vjp`: returns coil-DOF derivative dict with
  per-group entries (`gammas`, `gammadash`, `currents`).
- M5 wrapper `dJ()`: returns a `Derivative` projection from JAX
  cotangents through `Coil.vjp`, `Current.vjp`, and surface DOF vjp.
- Failed adjoint solve surfaces non-finite gradient at the public
  boundary (per CLAUDE.md "Adjoint / warm-start operator solves"); never a
  silent fallback.

## LS vs Exact Lane Distinction

- **LS lane** (`optimizer_backend="scipy"`,
  `"ondevice"`, `"hybrid"`): solves the least-squares penalty form
  `J(x) = 1/2 ||r(x)||²` plus constraint terms. Residual evidence is
  reported in the original physical basis. The forward AND adjoint
  solves share the same `(lu, piv)` factors stored under
  `lax.stop_gradient` when
  `decision_size² × 8 ≤ max_dense_jacobian_bytes`, ensuring bit-equal
  forward/adjoint Hessian action.
  - `linear_solve_factors` is **load-bearing runtime data** in the LS
    lane (`boozersurface_jax.py:3418-3475`,
    `surfaceobjectives_jax.py:3017-3055`).
  - Tolerance lane: `ls_wrapper_gradient` / `derivative_heavy` /
    `direct_kernel` per the kernel being compared.
- **Exact lane** (`optimizer_backend` stripped from public API at
  `boozersurface_jax.py:3097`): solves the exact-Newton system using the
  operator-backed GMRES seam in `optimizer_jax._run_operator_gmres`.
  - Dense PLU is **public/debug metadata only** (per CLAUDE.md).
  - `linear_solve_backend="operator"`,
    `dense_linear_solve_factors_available` may be reported.
  - Batched exact adjoints solve one RHS at a time through the same
    operator seam (CLAUDE.md "Exact Boozer scaling-limit contract").
  - Tolerance lane:
    `exact_well_conditioned_adjoint` (`rtol=1e-6`, `atol=1e-8`, residual
    `<=1e-10`) for well-conditioned fixtures only;
    `exact_ill_conditioned_adjoint` is residual/failure-only and must
    not assert vector parity.
  - Scaling-limit failures: `failure_category="scaling_limit"`,
    `failure_stage="dense_jacobian_finalization"`; these are predictable
    reporting limits, not adjoint availability failures.

## Excluded / Singular Regimes

- `weight_inv_modB=True` with `B ≈ 0`: the kernel returns non-finite
  values; covered by
  `tests/geo/test_boozer_residual_jax.py::TestBoozerResidualScalar::test_weighted_zero_field_is_nonfinite`.
- Equality at `abs(I) == threshold` for current penalty is not item 04
  scope (item 01).
- Surface in the immediate near-coil regime is not exercised by the
  item-04 NCSX fixtures; that regime is documented as outside the
  smoke-resolution contract.
- Ill-conditioned exact Newton Jacobians: scipy and JAX `lu()` can
  produce adjoint vectors differing by ~3x in norm, both satisfying
  `J^T adj = rhs` to machine precision (mathematical limitation, not a
  code bug — known issue per `project_known_issues.md` line 54). Tests
  use FD validation rather than direct vector parity at
  `exact_ill_conditioned_adjoint`.

## Parity Tolerances (Source Of Truth)

All assertions in NEW item-04 tests use
`benchmarks.validation_ladder_contract.parity_ladder_tolerances` or
`PARITY_LADDER_TOLERANCES[<lane>]`:

- `direct_kernel`: forward residual / label values
  (`PARITY_LADDER_TOLERANCES["direct_kernel"]`).
- `ls_wrapper_gradient`: LS adapter dJ at solved state.
- `derivative_heavy`: composed first derivatives.
- `direct_hessian_oracle`: Hessian column-complete CPU parity.
- `exact_well_conditioned_adjoint`: exact lane operator vs dense-reference
  adjoint vector parity.
- `boozer_residual_floor_{vector,scalar}`: lane-floor stress.

No NEW item-04 test inlines numeric `atol`/`rtol` literals.

## Red-Step Evidence

Item-04 NEW tests
(`tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py::test_boozer_surface_basic_passes_residual_and_label_parity`
and `::test_boozer_qa_wrappers_passes_native_supported_parity`) were
confirmed FAILING at the pre-impl tree (`e94464801~1` = `3d5b51731`); see
`.artifacts/jax_port_goal/red/04.txt`.

## Negative Controls

- Wrong-sign residual: caught by the `direct_kernel`-bucketed
  `boozer_residual` comparison in
  `test_boozer_surface_basic_passes_residual_and_label_parity` (CPU
  oracle returns the SIMSOPT-convention residual; sign drift fails the
  tolerance).
- Wrong-scale residual: same comparison plus
  `tests/geo/test_boozer_residual_jax.py::TestBoozerResidualParityStress`
  which exercises the lane floor.
- Wrong-state dependency:
  `tests/geo/test_boozersurface_jax.py::TestEnsureSolvedGuard::test_dirty_unsolved_surface_without_cached_result_is_rejected`
  catches stale-state reads; the adjoint runtime state SSOT contract is
  validated by `TestBuildBoozerSurfaceRuntimeState`.
