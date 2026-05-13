# Item 10 Math And Physics Invariants

## Units and constants

- B in Tesla. Coil geometry in meters. Currents in Amperes.
- `_MU0_OVER_4PI = 1e-7` (T m / A) at `src/simsopt/jax_core/biotsavart.py:52`.
- Biot-Savart integrand (in SI):
  `B(x) = (mu_0 / 4 pi) * sum_coils I_c * integral (dl_c(s) x (x - gamma_c(s)))
  / |x - gamma_c(s)|^3 ds`,
  evaluated by trapezoidal quadrature on `gammadash(s)` and `gamma(s)`.

## Biot-Savart kernel contracts

- Forward `B`: `(npoints, 3)` array, contraction order `coil ->
  quadpoint -> point` (chunked along the same axis order).
- `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` — axis 1 is derivative direction,
  axis 2 is the B component. Matches `fields.rst` in SIMSOPT docs.
- Divergence-free constraint: `trace(dB_by_dX[p, :, :]) = 0` for every p
  (verified by `test_dB_dX_symmetric_and_divergence_free` upstream-test
  parity in `tests/field/test_biotsavart_jax_parity.py`).
- Current linearity: `B(I_c)` is linear in each coil current `I_c`.
- Superposition: `B(coils_1 + coils_2) = B(coils_1) + B(coils_2)`
  (verified by `test_multiple_coils` in
  `tests/field/test_biotsavart_jax.py`).

## integral_BdotN definition variants (all three exercised)

For unnormalized surface normal `n` with `|n| = sqrt(sum(n^2, axis=-1))`,
unit normal `n_hat = n / |n|`, and `r = B . n_hat - target`:

- `quadratic flux`:
  `J = 0.5 / (nphi * ntheta) * sum_{phi,theta} r^2 * |n|`
- `normalized`:
  `J = 0.5 * sum_{phi,theta} r^2 * |n| / sum_{phi,theta} |B|^2 * |n|`
- `local`:
  `J = 0.5 / (nphi * ntheta) * sum_{phi,theta} r^2 / |B|^2 * |n|`

Boundary contracts:

- Zero-area quadrature points (where `|n| = 0`) contribute zero.
- For `normalized`, the global denominator `sum |B|^2 |n|` non-positive
  is treated as invalid and returns `inf`.
- For `local`, any positive-area quadrature point with `|B|^2 = 0` is
  treated as invalid and returns `inf`.

## Stellarator symmetry

- `coils_via_symmetries(base_coils, base_currents, nfp, stellsym)`
  applies the discrete `Z_nfp` rotational symmetry around the z-axis
  and (when `stellsym=True`) the stellarator z-reflection symmetry. The
  expanded coil list has `len(base_coils) * nfp * (2 if stellsym else 1)`
  entries.
- `BiotSavartJAX` and `BiotSavart` consume the expanded coil list
  identically; the symmetry transform is upstream. The new closeout
  test exercises BOTH `stellsym=False` and `stellsym=True` at
  `nfp=1` (`stellsym=True` doubles the coil list to 8 base-equivalent
  entries with reflected geometry and sign-flipped currents).

## Stage-2 surface fixture

- `SurfaceRZFourier(nfp=nfp_lookup, stellsym=stellsym, mpol=1, ntor=1)`
  with `nphi=16`, `ntheta=8` parametric grid; production-scale floor
  from section 4c is `nphi >= 16, ntheta >= 8, ncoils >= 4`.
- `surface.gamma()` returns `(nphi, ntheta, 3)`; flattened to
  `(npoints, 3)` for `BiotSavart.set_points` / `BiotSavartJAX.set_points`.
- `surface.normal()` returns `(nphi, ntheta, 3)` unnormalized surface
  normal `dgamma/dphi x dgamma/dtheta`.

## Oracle contract

- `fixed_scalar` chained parity at production-scale: chain
  `BiotSavartJAX.B(points)` -> `integral_BdotN(B_jax_jnp, target, normal,
  definition)` against the C++ oracle chain
  `BiotSavart.B(points)` -> `sopp.integral_BdotN(B_cpu_np, target,
  normal, definition)`.
- Tolerances come from
  `benchmarks.validation_ladder_contract.parity_ladder_tolerances("direct_kernel")`
  (`rtol=1e-10`, `atol=1e-12`, same-state requirement).

## Excluded regimes

- Near-coil singularities (`|x - gamma| -> 0`) are physically singular
  and outside the production fixture (the surface in the new test is at
  `R1=0.2`, well separated from the coil major radius `R0=1.0` and
  minor radius `R1_coil=0.5`).
- The `inf` return path for normalized/local-degenerate inputs is
  covered by `tests/objectives/test_integral_bdotn_jax.py`; the new
  closeout test deliberately stays in the well-conditioned regime so
  the C++ chain produces finite, byte-tight parity.

## Negative controls

- The C++ -> JAX chain is asserted at byte-tight `direct-kernel`
  tolerance. If the JAX path silently used a wrong sign in the
  cross product, the integrand contraction `B . n_hat` would flip on
  half the surface and the parity would fail.
- All three definition variants are exercised; a mistake in one
  variant's denominator wiring would only fail that variant's
  parametrization.
- Both `stellsym=False` and `stellsym=True` are exercised; a mistake
  in the coil-graph unwrap (`_coil_graph.py`) under reflected
  geometry / sign-flipped currents would only fail the stellsym=True
  case.
