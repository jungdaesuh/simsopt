# Item 13 Math And Physics Invariants

- Units: the interpolant carries the units of the user function `f`;
  it is dimension-agnostic. For `InterpolatedField` downstream
  consumers, the units are Tesla (B) and Tm (A); the item 13 kernel
  treats those as opaque float64 outputs.
- 1D nodes live in `[0, 1]` for each axis. The cell-local coordinate
  `(x - xmesh[i]) / hx` lies in `[0, 1]` for in-domain queries; the
  C++ `_EPS_ = 1e-13` clamp protects against floating-point drift at
  the boundary. The JAX kernel uses the same epsilon.
- Tensor-product basis: `p(x, y, z) = Σ_{i,j,k} p_i(x) p_j(y) p_k(z)
  v_{ijk}` with `v` taken from the per-cell DOF table. Each 1D
  Lagrange basis polynomial satisfies `p_i(x_i) = 1` and
  `p_i(x_j) = 0` for `j ≠ i`, so `Σ_i p_i(x) = 1` for any `x`
  (partition-of-unity). The JAX kernel preserves this property by
  building the same `scalings = Π_{i≠idx} 1/(nodes[idx] - nodes[i])`
  denominator as the C++ header.
- Lagrange exactness: a polynomial of degree `≤ d` in each variable
  is reproduced exactly by a degree-`d` rule. The JAX kernel passes
  this within `direct_kernel` rtol/atol against the closed-form
  separable polynomial oracle for `dim ∈ {1, 3, 6}` and
  `degree ∈ {1, 2, 3, 4}`.
- Skip semantics: cells whose 8 mesh corners all return `True` for the
  user-supplied `skip` predicate are excluded. Their DOFs are zeroed
  and their `cell_to_row` entry points at the sentinel row. The C++
  binding leaves the caller's output buffer unchanged for skipped
  cells; the JAX kernel returns zero instead (the pure-functional
  equivalent of "no write").
- OOB semantics: in-domain queries with `xidx < 0`, `xidx >= nx`,
  ..., are routed to the sentinel row. With
  `out_of_bounds_ok=False`, the JAX kernel surfaces `NaN` so the
  caller can detect the error post-hoc; the C++ binding raises a
  runtime error.
- Output dtype: `float64` end-to-end. `JAX_ENABLE_X64=True` is enforced
  by `tests/conftest.py`.
- Vector output: the kernel is vector-valued with `value_size >= 1`.
  The benchmark uses `value_size = 3` (cartesian B). The JAX kernel
  supports arbitrary `value_size`.
- `stellsym=True` vs `stellsym=False`: this kernel does not consume
  the surface symmetry flags. The downstream `InterpolatedField`
  wrapper handles symmetry by reflecting the query before evaluation.
- Excluded singular regimes: the C++ kernel does not handle queries
  whose floating-point cell index is exactly `nx` (the upper-bound
  corner); the soft-clamp at `_EPS_` keeps such queries inside the
  domain. The JAX kernel applies the same clamp.
- `nfp`: this kernel is `nfp`-agnostic; the downstream
  `InterpolatedField` handles the modular `phi` mapping.
- Tolerance: the new tests assert against the `direct_kernel` lane
  (`rtol=1e-10, atol=1e-12`). No new tolerances were introduced.
- The `estimate_error` helper returns a `(mean - std, mean + std)`
  bracket. For a degree-`d` polynomial input and a degree-`d` rule,
  both ends sit near machine zero; the test bound is the
  `derivative_heavy` `first_derivative_atol = 1e-10` as a comfortable
  upper bound.
- CUDA behavior is not claimed.
