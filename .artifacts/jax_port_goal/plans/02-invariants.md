# Item 02 Math And Physics Invariants

## Units and scales

- `mu_0` is the vacuum permeability in SI units (T m / A), obtained
  from `scipy.constants.mu_0`.
- `Biot_savart_prefactor = mu_0 / (4 * pi)` at
  `src/simsopt/field/selffield.py:21` has units T m / A.
- `current` is in A; `gamma`, `gammadash`, `gammadashdash` are in m, m,
  m respectively. `quadpoints` are dimensionless on `[0, 1)`.
- `regularization` has units of length-squared (m^2): for the circular
  cross-section, `regularization_circ(a) = a^2 / sqrt(e)`; for the
  rectangular cross-section, `regularization_rect(a, b) = a * b *
  exp(-25/6 + K(a, b))`.
- `B_regularized_pure` returns Tesla.

## Curve parameter convention

`selffield.py:135-138`:

```
phi = quadpoints * 2 * pi
rc = gamma
rc_prime = gammadash / (2 * pi)
rc_prime_prime = gammadashdash / (4 * pi^2)
dphi = 2 * pi / phi.shape[0]
```

SIMSOPT curves use parameter `t in [0, 1)`; the conversion to a `phi in
[0, 2 pi)` curve parameter is local to this file and yields the
`rc_prime` factor of `1 / (2 pi)`.

## Closed-form oracle (Hurwitz / Landreman / Antonsen)

For a circular coil of radius `R0` in the `x-y` plane with circular
cross-section radius `a`:

```
B_z(centroid) = mu_0 * I / (4 * pi * R0) * (log(8 * R0 / a) - 3/4)
```

with `B_x = B_y = 0` at machine precision. For a rectangular cross-
section (`a x b`) the same paper Eq. (98) gives a `13/12 - K(a, b)/2`
correction.

Verified numerically at HEAD:

- `ncoils=4`, `nquadpoints=128`, `R0 in {1.70, 1.85, 2.00, 2.15} m`,
  `I = 1e5 A`, `a = 0.01 m`: relative deviation from the closed-form
  oracle is in the range `|rel| < 1.1e-15`. `|B_x|`, `|B_y|` are exactly
  zero (machine precision; `max |B_xy| = 0.0`).

## Symmetry / orientation invariants

- The regularized self-field has `B_x = B_y = 0` for a circular coil in
  the `x-y` plane (only `B_z` survives by axisymmetry).
- The kernel is linear in `current`: `B(c * I) = c * B(I)`. This is
  exploited by `tests/field/test_selffieldforces.py::CoilForcesTest::test_b_regularized_pure_jit_vmap_strict_transfer_guard_matches_wrapper`
  (`test_selffieldforces.py:191`) which checks `batched[1] = expected /
  2` with the same regularization.
- The rectangular regularization is symmetric under swapping `a <-> b`
  (covered by `test_selffieldforces.py::test_symmetry`).

## Stellsym coverage

`B_regularized_pure` is a per-coil kernel that does not directly depend
on `stellsym`. The downstream `RegularizedCoil` wrappers and the force
objectives consume coils produced by `coils_via_symmetries`, which is
exercised at multiple `stellsym` values by
`tests/field/test_selffieldforces.py::test_force_objectives` and
related production fixtures.

## Excluded singular regimes

The kernel includes an analytic singularity-subtraction term
`B_regularized_singularity_term` at `selffield.py:97` precisely to
handle `r = 0`. The kernel is not defined for points off the coil
filament (this kernel is `B_self`, the value on the filament itself);
mutual-field evaluations live in `src/simsopt/field/force.py` and are
out of scope for this item.

## Derivative shape and tracing

- `B_regularized_pure` returns shape `(n, 3)`.
- Reverse-mode gradients of `B_regularized_pure` propagate native JAX
  cotangents through the `jnp.cross`, `jnp.linalg.norm`, `jnp.log`
  primitives. Public-wrapper gradient projection through `Derivative`
  is handled by the consuming force objectives, not by this kernel.
- `regularization_circ(a)` is `C^infty` for `a > 0`.
- `regularization_rect(a, b)` is `C^infty` for `a > 0, b > 0`.

## Negative controls / red evidence

The closeout's negative control (`tests/field/test_selffield_item02_closeout.py::test_b_regularized_pure_wrong_regularization_breaks_closed_form_parity`)
substitutes `a' = 1.5 * a` into `regularization_circ` and asserts that
the resulting `B_z` deviates by more than `direct_kernel.rtol * 1e6`
from the closed-form oracle. Empirically `|rel| ~= 6.3e-2` at
`a = 0.01` m, `R0 = 1.7` m. This catches a silent mis-wiring of the
cross-section parameter on the public path.

The `x_y_components_vanish` parametrized test catches accidental basis
swaps in the `jnp.cross` analytic singular term at `selffield.py:111`
and the `jnp.cross(rc_prime[None, :], dr)` integral term at
`selffield.py:142`.
