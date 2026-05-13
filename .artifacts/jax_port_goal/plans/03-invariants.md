# Item 03 Math And Physics Invariants

## SquaredFlux family

### Units and sign convention

- Magnetic field `B` is in Tesla (T) at SI units.
- Surface position `gamma` is in meters (m).
- Unnormalized surface normal `n = ∂γ/∂φ × ∂γ/∂θ` has units of
  `m²/rad²` (since `γ` is differentiated against the quadrature
  parameters `φ, θ ∈ [0, 1]` with the `2π` scaling absorbed into
  `n` by the SIMSOPT convention; see `simsopt.geo.surface`).
- The scalar objective `J` has units of `T² · m²` for the
  `quadratic flux` definition (per surface integral element); the
  `normalized` and `local` definitions absorb a `1/|B|²` and are
  dimensionless / weighted by `m²`.

### Three definition formulas

The CPU C++ oracle is `simsoptpp/integral_BdotN.cpp:12-123`. For
`B_dot_N = B · (n / |n|) - B_T` (target subtracted from unit-normal
projection) and `|n|` the unnormalized normal magnitude:

1. **`quadratic flux`** (oracle at `integral_BdotN.cpp:93-94, 119`):
   ```
   J = 0.5 * Σ_{i,j} (B·n̂ − B_T)² · |n| / (nphi · ntheta)
   ```
   The CPU pure-Python NumPy oracle (mirror of CPU C++) is at
   `tests/objectives/test_fluxobjective.py:91-95`:
   ```python
   should_be = (
       0.5
       * sum((B_dot_n - target.reshape((-1,))) ** 2 * norm_normal)
       / (ntheta * nphi)
   )
   ```

2. **`normalized`** (oracle at `integral_BdotN.cpp:95-97, 110-114`):
   ```
   J = 0.5 · Σ_{i,j} (B·n̂ − B_T)² · |n|
            / Σ_{i,j} |B|² · |n|
   ```
   When `Σ |B|² · |n| ≤ 0`, the C++ oracle returns `+inf` (see
   `integral_BdotN.cpp:111`).
   The NumPy oracle mirror is at
   `tests/objectives/test_fluxobjective.py:99-107`:
   ```python
   numerator = 0.5 * sum((B_dot_n - target.reshape((-1,)))**2 * norm_normal) / (ntheta * nphi)
   denominator = sum(mod_B_squared * norm_normal) / (ntheta * nphi)
   J2 = numerator / denominator
   ```

3. **`local`** (oracle at `integral_BdotN.cpp:98-103, 116-119`):
   ```
   J = 0.5 · Σ_{i,j: |B|²>0 and |n|>0} (B·n̂ − B_T)² · |n| / |B|² / (nphi · ntheta)
   ```
   When any positive-area quadrature point has `|B|² = 0`, the C++
   oracle sets `has_local_singularity` and returns `+inf` (see
   `integral_BdotN.cpp:99-101, 116-118`).
   The NumPy oracle mirror is at
   `tests/objectives/test_fluxobjective.py:111-115`:
   ```python
   J3 = 0.5 * sum((B_dot_n - target.reshape((-1,)))**2 / mod_B_squared * norm_normal) / (ntheta * nphi)
   ```

### JAX kernel formulas (must match CPU oracle bit-for-bit at lane tolerance)

The JAX implementation at `src/simsopt/objectives/integral_bdotn_jax.py`
expresses the same quantities as a squared-sum of weighted residuals
(`residual_BdotN`) followed by `0.5 * scalar_square_sum`:

- For `quadratic flux` (lines 52-54):
  ```
  weight_i = |n|_i / (nphi · ntheta) if |n|_i > 0 else 0
  residual_i = (B·n̂ − B_T)_i · sqrt(weight_i)
  J = 0.5 · Σ residual_i²
  ```
- For `normalized` (lines 55-64):
  ```
  D = Σ |B|² · |n|
  if D > 0: weight_i = |n|_i / D ; residual_i = (B·n̂ − B_T)_i · sqrt(weight_i)
  else: J = inf
  J = 0.5 · Σ residual_i²
  ```
- For `local` (lines 65-78):
  ```
  if any (|n|_i > 0 and |B|²_i ≤ 0): J = inf
  else: weight_i = |n|_i / (|B|²_i · nphi · ntheta) if |n|_i > 0 else 0
  J = 0.5 · Σ ((B·n̂ − B_T)_i · sqrt(weight_i))²
  ```

The JAX `Σ residual_i²` form is algebraically identical to the C++
oracle's direct numerator sum but provides a stable autodiff path and
allows the kernel to fan out as a single `vdot` reduction. Parity at
the `direct_kernel` lane (`rtol=1e-10, atol=1e-12`) has been validated
in existing tests (see coverage matrix).

### Symmetry coverage

- `stellsym=False` is covered by existing parity fixtures in
  `tests/objectives/test_fluxobjective_jax_parity.py` (2 coils, nfp=1).
- `stellsym=True` is covered by Stage 2 production fixtures
  (`tests/integration/test_stage2_jax.py`, nfp=5, banana production fixture).
- The item 03 closeout fixture explicitly exercises
  `stellsym=True, nfp=2` to add a single-file production-scale
  witness combining all three definitions under strict transfer
  guard discipline.

### Derivative shape

- `SquaredFlux.dJ()` returns a `Derivative` keyed on the field
  dependencies (one block per `Coil`'s `Current`, `Curve`, and
  `ScaledCurrent` participation as discovered via `unique_dof_lineage`).
- `SquaredFluxJAX.dJ()` returns the same `Derivative` shape; the JAX
  flat coil-DOF gradient is projected back via
  `_field_dofs_gradient_to_derivative` at
  `src/simsopt/objectives/fluxobjective_jax.py:69-91`. The symmetry
  share between duplicate dependencies (`dep_opts = tuple(lineage_opt.dofs.dep_opts())`)
  is divided by `len(dep_opts)`, matching the CPU `Derivative` accumulation
  convention.

### Excluded singular regimes

- `local` with `|B|² = 0` on any positive-area quadrature point ⇒
  `J = +inf` (both CPU and JAX).
- `normalized` with `Σ |B|² · |n| ≤ 0` ⇒ `J = +inf` (both CPU and JAX).
- `local` gradient with `|B|² = 0` on any positive-area quadrature
  point ⇒ CPU raises `ObjectiveFailure` (`fluxobjective.py:102-106`);
  JAX raises `ObjectiveFailure` via
  `_raise_if_nonfinite_squared_flux_gradient` at
  `fluxobjective_jax.py:47-56`.
- `normalized` gradient with zero denominator ⇒ CPU raises
  `ObjectiveFailure` (`fluxobjective.py:120-124`); JAX raises the same.
- Zero-area quadrature points (`|n| = 0`) contribute zero to all
  three definitions; both CPU and JAX use a safe-division mask
  (`has_normal` in JAX kernel; `normN > 0.0` guard in C++ oracle).

### Negative controls / red evidence

- Existing tests
  `test_quadratic_flux_zero_normals_contract`,
  `test_singular_zero_field_contract`,
  `test_squaredfluxjax_zero_current_gradient_raises_objective_failure`,
  and `test_degenerate_normals_do_not_perturb_valid_flux_contracts`
  in `tests/objectives/test_fluxobjective_jax_parity.py` are the
  existing negative controls.
- Item 03 itself does not change kernel logic and therefore does not
  introduce a new wrong-sign / wrong-scale negative control. The new
  closeout fixture's `stellsym=True` + production scale would
  catch a missing surface-symmetry factor or a missing
  `dgamma_by_dcoeff` chain rule in the rotation-aware fast path, if
  such a regression were introduced.

### Parity contract

The closeout fixture asserts `fixed_state_scalar` parity for
`SquaredFluxJAX.J()` vs `SquaredFlux.J()` (C++ oracle) at the
`direct_kernel` lane:
- `rtol = 1e-10`
- `atol = 1e-12`
imported via `parity_ladder_tolerances("direct_kernel")`. Gradient
parity remains covered by the existing fixtures (see coverage matrix);
this item is `cpu_oracle_complete` for the scalar contract layer.
