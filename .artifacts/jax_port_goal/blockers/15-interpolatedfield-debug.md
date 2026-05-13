# Item 15-sub -- InterpolatedField JAX wrapper blocker

Status: `blocked`.

Closure level: `blocked_architecture`.

## Scope

`src/simsopt/field/magneticfieldclasses.py::InterpolatedField` is part
of prompt item 15. It depends on the item-13
`RegularGridInterpolant3D` JAX kernel, but the public wrapper contract
is wider than the rectangular kernel item 13 closed.

## Why Item 13 Is Not Enough

The C++/Python public wrapper:

- receives ranges in cylindrical coordinates `(r, phi, z)`,
- folds points through `nfp` rotational symmetry,
- folds `z` through stellarator symmetry when `stellsym=True`,
- supports skip masks during table construction,
- samples an arbitrary source `MagneticField` through its public CPU
  evaluation methods while building the interpolation tables,
- exposes public `MagneticField` cache semantics for `B`, derivatives,
  and error estimates.

Item 13 only provides the immutable rectangular-grid JAX interpolation
kernel and direct parity against `simsoptpp.RegularGridInterpolant3D`.
It does not define the cylindrical/folding wrapper layer.

## Required Unblock

Implement a wrapper-level JAX spec builder that:

- samples the source field explicitly at construction time,
- owns the cylindrical-to-Cartesian and symmetry-folding contract,
- documents the strict out-of-bounds semantics difference between the
  pure JAX kernel and the mutable C++ output-buffer contract,
- adds public wrapper parity against `InterpolatedField.B()` for
  in-domain, folded, skipped, and out-of-domain points.

CUDA proof remains `not_claimed` unless the user approves GPU work.
