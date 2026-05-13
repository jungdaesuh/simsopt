# Item 02 Math And Physics Invariants

- The circular regularization remains `a**2 / sqrt(e)`.
- The rectangular regularization remains `a * b * exp(-25 / 6 + K(a, b))`.
- Rectangular regularization remains symmetric under swapping `a` and `b`.
- The self-field kernel keeps the upstream Landreman/Hurwitz split into the
  analytic singularity term and the finite quadrature integral term.
- The returned field scales linearly with current. The `vmap` and two-device
  CPU proxy checks verify the half-current output is exactly half the
  full-current output within float64 tolerance.
- `RegularizedCoil.self_force()` keeps the Lorentz-force contract
  `I * tangent x B`; only the transfer boundary around the JAX cross product
  changed.
- The public wrapper fix does not change array shapes:
  `B_regularized()` and `self_force()` both return `float64[n, 3]`.
- No new tolerances were introduced outside the existing focused self-field
  tests.
- CUDA behavior is not claimed.
