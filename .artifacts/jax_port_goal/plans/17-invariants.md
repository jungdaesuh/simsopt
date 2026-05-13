# Item 17 Math / Physics Invariants

## Units and Scales

- `Vns`, `Vnc`: SPEC-convention Fourier coefficients of the unnormalized
  `B . (∂r/∂θ × ∂r/∂ζ)` integrand on the toroidal surface. SI base unit
  Tesla * meter² (T * m²), but the SPEC-convention `1 / (2π)²`
  normalization absorbs the surface-quadrature factor so the numerical
  coefficients carry the modulus of an underlying field times a length²
  divided by `(2π)²`. The new closeout test uses non-dimensional Vns /
  Vnc values in `[-1e-3, +1e-3]` so no overflow / underflow regime is
  exercised.
- `real_space_field = NormalField.get_real_space_field()`: SI Tesla
  scalar field on the surface quadrature grid `(nphi, ntheta)`. Computed
  via
  `-1 * inverse_fourier_transform_scalar(Vns, Vnc) / |surface.normal()|`
  (line 517-519).
- `bs.B()`: SI Tesla, returned by `BiotSavart` over the surface grid
  reshaped to `(nphi * ntheta, 3)`. Re-reshaped to `(nphi, ntheta, 3)`
  for the `B . n` reduction inside `CoilNormalField.vns` / `.vnc`.

## Sign Convention

- The `CoilNormalField` reduction at lines 576-577 and 589-590 uses
  `np.sum(B * surface.normal() * -1, axis=2)`. The `-1` flips the
  conventional `B . n` outward-normal flux to match SPEC's convention
  for the SPEC `vns` / `vnc` Fourier harmonics (positive `vns` denotes
  a missing inward flux that the coils must supply).
- `NormalField.get_real_space_field` (line 519) divides by
  `|surface.normal()|` and pre-multiplies by `-1` to invert the
  unnormalized SPEC convention back to a real-space `B . unit_normal`
  scalar with the same sign convention as `B . n` on a SciPy-style
  exterior-normal surface.
- The closeout's `test_coil_normal_field_negative_control_wrong_sign_breaks_parity`
  explicitly verifies that dropping the `-1` produces a tolerance-
  busting Vns deviation. Without that check, a sign flip in either
  reduction would silently pass the same-state parity claim.

## Orientation

- `SurfaceRZFourier` quadrature grids use the `quadpoints_phi` and
  `quadpoints_theta` arrays in the closed-half-open interval `[0, 1)`.
  The closeout test uses `np.linspace(0, 1, 32, endpoint=False)` for
  `phi` and `np.linspace(0, 1, 16, endpoint=False)` for `theta` —
  matches the production-scale floor `(nphi >= 16, ntheta >= 8)` and
  exceeds it on the toroidal axis.
- `surface.normal()` is the outward unnormalized normal to the
  toroidal surface. The `CoilNormalField` reduction relies on the
  outward orientation; flipping the orientation flips the Vns
  reduction sign and is caught by the negative control.

## Stellsym Coverage

- `test_fourier_pair_identity_at_production_scale` is parameterized
  across `(stellsym=True, mpol=4, ntor=3)` and
  `(stellsym=False, mpol=3, ntor=2)`. Both branches are exercised in a
  single fixture. Non-stellsym additionally validates the cosine
  series `Vnc` round trip; stellsym validates the sine-only `Vns`
  round trip with `Vnc = 0` and `stellsym=True` flag set.
- `test_coil_normal_field_vns_vnc_match_direct_cpu_oracle` is also
  parameterized across both branches. Stellsym branch asserts `cnf.vnc`
  matches the all-zero oracle (the property returns the cached zeros
  built by `fourier_transform_scalar(..., stellsym=True)`); non-stellsym
  asserts both `cnf.vns` and `cnf.vnc` match direct oracle bit-tight.

## Derivative Shape

- `NormalField.J()` / `dJ()` is not directly exposed for free-boundary
  optimization in item 17's scope. The downstream consumer
  `CoilNormalField.optimize_coils` uses `JF.J()` / `JF.dJ()` from
  `coilset.flux_penalty + length_penalty`, owned by items 03 and 04.
  Item 17's closeout test exercises value parity only and does not
  assert a gradient invariant — there is no JAX-native derivative
  path through `normal_field.py` to validate.

## Excluded Singular / Near-Coil Regimes

- The closeout fixture's coil radii from
  `CoilSet._circlecurves_around_surface(surface, coils_per_period=4,
  order=4)` are computed by the helper and are bounded away from the
  surface; no near-coil singularity is exercised. This matches the
  `test_real_space_field` fixture pattern at
  `tests/field/test_normal_field.py:491-513`.
- Surface modes with `m = 0, n <= 0` are excluded from the sine series
  per the upstream constructor convention (line 178-185 in
  `normal_field.py`). The closeout test honors this by populating only
  `(m >= 1, |n| <= ntor)` and `(m = 0, n > 0)` modes through the
  Optimizable API and skipping the invalid modes from the
  hand-rolled `Vns_in` arrays.

## Oracle Contract

`fixed_scalar`. Same-state value parity for all three checked
invariants:

1. Fourier-pair identity: `Vns / Vnc` arrays passed through `IFT . FT`
   reproduce themselves bit-tight at `direct_kernel` lane tolerance.
2. `CoilNormalField.vns` / `.vnc`: reproduce a hand-rolled CPU oracle
   bit-tight at `direct_kernel` lane tolerance.
3. Cache invalidation: post-DOF-change `cnf.vns` reproduces the
   re-evaluated oracle bit-tight at `direct_kernel` lane tolerance.

No gradient parity check is required; item 17 does not introduce a
differentiable hot path.

## Residual-Basis Invariants

N/A. Item 17 is not Boozer or linear-solve work; no residual or
preconditioned-basis evidence applies.

## Sign / Scale / State Negative Control

- `test_coil_normal_field_negative_control_wrong_sign_breaks_parity`
  asserts that dropping the `-1` reduction sign produces a Vns
  deviation strictly above the `direct_kernel` tolerance. This proves
  the closeout fixture catches a wrong-sign regression rather than
  coincidentally matching the oracle.
- Implicit scale negative control: the fixture uses circular base
  curves around the toroidal surface with `current=1.0e5` A,
  which produces non-zero Vns at the chosen mpol / ntor. A
  zero-current fixture would coincidentally pass the parity claim
  while making the test vacuous; the chosen current and coil layout
  ensures the oracle Vns norm is strictly above floor.
