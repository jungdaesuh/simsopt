# Item 12 — Math / Physics Invariants

## Units and scales

- All inputs and outputs are in SI units. Cartesian `points` are in
  metres (m). `R0`, `B0`, `gamma`, `Z_m` are in metres / Tesla as
  documented on the upstream `ToroidalField` / `PoloidalField` /
  `MirrorModel` classes.
- `B` and `A` outputs are in Tesla and Tesla·metre respectively.
- `dB`, `dA` outputs are in `T/m` / `(T·m)/m`. `d2B` is in `T/m^2`.

## Coordinate conventions

- Cartesian inputs `(x, y, z)`; cylindrical `(R, phi, Z)` defined by
  `R = sqrt(x**2 + y**2)`, `phi = atan2(y, x)`, `Z = z` (right-handed).
- `e_phi = (-sin phi, cos phi, 0)` (positive about +z).
- `theta = atan2(z, R - R0)` for `PoloidalField` and the local minor
  radial coordinate `r = sqrt((R - R0)**2 + z**2)`.

## Sign conventions and orientation

- `ToroidalField.B = (B0 R0 / R) * e_phi`. Positive `B0` ⇒ field along
  `+phi`.
- `PoloidalField.B = (B0 / (R0 q)) * r * e_theta`, where `e_theta` is the
  poloidal unit vector pointing along increasing `theta` in the
  `(R, Z)` plane. Sign of `q` flips the field direction.
- `MirrorModel.B = B_R * e_R + B_Z * e_Z`. Positive `B0` ⇒ axial field at
  `R = 0` is positive on the `+z` half-plane.

## Stellsym coverage

- These fields do not consume Fourier surfaces or coil DOFs, so the
  `stellsym=True` / `stellsym=False` axis is not applicable. Coverage
  is unconditional in `(x, y, z)`.

## Singular regimes (matched but not guarded)

- `ToroidalField`: `R = 0`. CPU class returns `inf` / `NaN`. JAX kernel
  matches. No defensive guard is added.
- `PoloidalField`: `R = R0` (the magnetic axis). The CPU `_dB_by_dX_impl`
  divides by `(R - R0)**2 + z**2` and `(-R0 + R) * (1 + z^2/(R0-R)^2)`,
  so the derivative is singular on the axis. JAX matches.
- `MirrorModel`: `R = 0`. CPU class divides by `R**2`. JAX matches.
- Parity tests filter samples to keep > 0.2 m away from these surfaces
  so `assert_allclose` does not compare `NaN` against `NaN`.

## Derivative shape contract

- `dB / dX` per point has shape `(3, 3)` for all three fields. The
  storage layout differs between classes (upstream inconsistency
  documented below).
- `d2B / dXdX` per point has shape `(3, 3, 3)` for `ToroidalField`.

### Upstream layout deviation

The upstream Python analytic classes do not all use the same
`dB[p, j, l]` layout that `CLAUDE.md` documents for the C++ kernels:

- `ToroidalField._dB_by_dX_impl` and `_dA_by_dX_impl` use
  `np.array([dB_by_dX1, dB_by_dX2, dB_by_dX3]).T`. The pre-`.T` axes
  are `(deriv, component, point)`, so the post-`.T` storage is
  `dB[p, l_component, j_deriv]`.
- `PoloidalField._dB_by_dX_impl` follows the same pattern, so its
  storage is also `dB[p, l_component, j_deriv]`.
- `MirrorModel._dB_by_dX_impl` assigns directly: `dB[:, j, l] = ...`,
  storing `dB[p, j_deriv, l_component]`.

The JAX kernels match each class's actual storage exactly. This is
captured in the kernel docstrings and confirmed by `direct_kernel`-lane
parity tests in `tests/jax_core/test_analytic_pure_fields_item12.py`.

### `ToroidalField._d2B_by_dXdX_impl` typo

The CPU expression for `d2B[p, 0, 0, 0]` evaluates to
`2 B0 R0 (3 x^2 + y^3) / R^6`. The analytic third derivative of
`Bx = -B0 R0 y / R^2` is `(2 B0 R0 y (R^2 - 4 x^2)) / R^6`, which is
not the same expression. We treat this as a known upstream typo (or a
deliberate undocumented expression) and reproduce it literally in the
JAX kernel because the parity oracle in the prompt is the CPU class,
not the textbook analytic third derivative. A future upstream fix
would need a coordinated update in `_toroidal_d2B_pointwise` and a
ratchet of the parity tolerance.

## Curl / divergence sanity (informational)

- `ToroidalField` is a vacuum field and the CPU oracle test asserts
  divergence and `grad B` symmetry. The JAX kernel inherits these
  properties because it matches CPU element-by-element.
- `PoloidalField` and `MirrorModel` are not vacuum fields and curl is
  non-zero; no curl invariant is asserted.

## Tolerance ratchet

- All parity assertions use `parity_ladder_tolerances("direct_kernel")`
  (`rtol=1e-10`, `atol=1e-12`). The observed max errors on the 50-point
  fixture are `~1e-15` (machine precision).
