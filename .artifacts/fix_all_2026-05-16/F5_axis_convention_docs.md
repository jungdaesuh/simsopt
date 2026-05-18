# F5 — Axis-convention docstrings for `dB`/`dA` kernels

Scope: docstring-only clarification of axis layouts on every public
function in `src/simsopt/jax_core/` that returns or consumes a
`dB`/`dA` tensor of shape `(n_points, 3, 3)`. No code changes; no
asserts; no refactors.

## Form taxonomy

- **Form A** (CLAUDE.md default): `dB[p, j, l] = ∂_j B_l(x_p)`.
  Axis 1 = spatial derivative direction, axis 2 = B-field component.
- **Form B** (component-first / CPU-oracle order):
  `dB[p, l, j] = ∂_j B_l(x_p)`. Axis 1 = B-field component, axis 2 =
  spatial derivative direction. Matches simsoptpp C++ storage.

## Per-function results

| function | file:line | form | status |
|---|---|---|---|
| `toroidal_dB` | `analytic_pure_fields.py:331` | B | updated to Returns-block wording (existing prose retained as parity rationale) |
| `toroidal_dA` | `analytic_pure_fields.py:363` | B | updated to Returns-block wording |
| `poloidal_dB` | `analytic_pure_fields.py:520` | B | updated to Returns-block wording |
| `mirror_dB`   | `analytic_pure_fields.py:660` | A | updated to Returns-block wording (was a one-liner) |
| `dipole_field_dB` | `dipole_field.py:234` | B | updated; previous docstring said component-first informally |
| `dipole_field_dB_from_spec` | `dipole_field.py:257` | B | added docstring (was bare) |
| `dipole_field_dA` | `dipole_field.py:306` | B | updated; previous docstring said component-first informally |
| `dipole_field_dA_from_spec` | `dipole_field.py:328` | B | added docstring (was bare) |
| `wireframe_segment_dB_by_dX` | `wireframe.py:149` | B | added Returns block (was a one-liner; module head already noted Form B) |
| `wireframe_segment_B_and_dB_by_dX` | `wireframe.py:169` | B | added Returns block |
| `wireframe_segment_dB_by_dX_contributions` | `wireframe.py:294` | B | updated; prior text used `[k, m]` notation |
| `wireframe_dB_by_dX` | `wireframe.py:517` | B | added Returns block (was a one-liner) |
| `wireframe_B_and_dB_by_dX` | `wireframe.py:539` | B | added Returns block (was a one-liner) |
| `circular_coil_dB` | `circular_coil.py:550` | A | updated wording (was already Form A informally) |
| `dommaschk_dB` | `analytic_fields.py:689` | A | updated wording (variable letters changed `i,j` → `j,l` for consistency with CLAUDE.md) |
| `reiman_dB` | `analytic_fields.py:920` | A | updated wording (variable letters changed `i,j` → `j,l`) |
| `biot_savart_dB_by_dX` | `biotsavart.py:597` | A | added Returns block (was bare) |
| `biot_savart_B_and_dB` | `biotsavart.py:627` | A | added docstring (was bare) |
| `biot_savart_B_and_dB_with_point_axis` | `biotsavart.py:644` | A | added docstring (was bare) |
| `biot_savart_dA_by_dX` | `biotsavart.py:675` | A | added Returns block (was bare) |
| `_tangent_map_A_matrix` (input `dB_by_dX`) | `magnetic_axis_helpers.py:222` | A | input convention pinned in `Parameters` block (was inline prose) |
| `tangent_map_rhs_from_field` (input `dB_by_dX`) | `magnetic_axis_helpers.py:295` | A | input convention added in `Parameters` block |

## Convention verification (read from code, not prior docstrings)

- `_toroidal_dB_pointwise` ends with `.T` over a `(deriv, comp, ...)` stack, so
  the per-point slice is `[comp, deriv]` → Form B.
- `_toroidal_dA_pointwise` follows the same `.T` pattern → Form B.
- `_poloidal_dB_pointwise` also ends with `.T` over a deriv-first stack →
  Form B.
- `_mirror_dB_pointwise` stacks `row_j0/row_j1/row_j2` on axis 0 directly
  with no transpose, where each `row_jk = (dBxd<k>, dByd<k>, dBzd<k>)`.
  Axis 0 = derivative direction `k`, axis 1 = component → Form A.
- `_dipole_field_dB_jit`: `contribution[p, j, k]` builds `mj_rk + mk_rj`
  with `mj_rk = m[None, :, None] * r[None, None, :]` (i.e. axis 1 = `m`
  component = B component, axis 2 = `r` = derivative direction) → Form B.
- `_dipole_field_dA_jit`: `skew[j, k]` is the antisymmetric of `m`, with
  `(m × r)_j = skew[j, k] r_k`, so axis 1 = `j` = A component → Form B.
- `_wireframe_segment_B_and_dB_by_dX_from_arrays`: `dB = stack((dBdx,
  dBdy, dBdz), axis=-2)` where each `dBdx` is the row-vector gradient
  of `B_<component>` (verified by expanding `∂B_0/∂x_l`). Axis -2 =
  B component, axis -1 = derivative direction → Form B. Matches the
  module head and `wireframe_field_impl.h` `dB_by_dX(p, k, m)`.
- `_dB_local_pointwise` (circular coil): stacks the deriv-first slabs
  `(dBxdx, dBydx, dBzdx)` etc. on axis 0 → Form A. `_dB_pointwise`
  rotates with `rot @ local_dB @ rot.T`, which preserves the
  index ordering on both axes.
- `biot_savart_dB_by_dX`: the JACOBIAN path computes
  `jacfwd(one_point, argnums=0)` (output `[comp, deriv]`) and then
  `swapaxes(-1, -2)` to land on `[deriv, comp]` → Form A. The
  VALUE_AND_JACOBIAN path uses `linearize` + `vmap` over an identity
  basis with `in_axes=0`, producing leading-axis = derivative direction
  → Form A.
- `_cylindrical_to_cartesian_dB` (shared by `dommaschk_dB`,
  `reiman_dB`): the `row<i>` stacks build axis -2 = row index = the
  C++ first inner index `i` of `dB(j, i, *, *)`. Cross-checking against
  `simsoptpp/dommaschk.cpp` line 515 (`dB(j, i, 0, 0) = dRBR*c²-…`),
  axis -2 carries the **derivative direction** and axis -1 the
  **component**. The simsoptpp public output is therefore Form A,
  consistent with `docs/source/fields.rst` line 234.
- `_tangent_map_A_matrix` reads `dB_by_dX[0, 1]` as `∂_x B_y`
  (verified by reconstructing `dB_R/dR` via cylindrical chain rule;
  the math closes only under Form A). The upstream CPU
  `simsopt/field/magnetic_axis_helpers.py::tangent_map` uses the same
  indexing, and `magnetic_field.dB_by_dX()` (Form A by C++ contract)
  is its source, so the JAX implementation consumes Form A.

## Verification commands

```
.conda/jax/bin/ruff check src/simsopt/jax_core/analytic_pure_fields.py \
  src/simsopt/jax_core/dipole_field.py \
  src/simsopt/jax_core/wireframe.py \
  src/simsopt/jax_core/circular_coil.py \
  src/simsopt/jax_core/analytic_fields.py \
  src/simsopt/jax_core/biotsavart.py \
  src/simsopt/jax_core/magnetic_axis_helpers.py
# All checks passed!

.conda/jax/bin/ruff format <same files>
# 7 files left unchanged.

.conda/jax/bin/python -c "import simsopt.jax_core; print(simsopt.jax_core.__name__)"
# simsopt.jax_core
```
