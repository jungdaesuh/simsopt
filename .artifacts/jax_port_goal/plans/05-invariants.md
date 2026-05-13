# Item 05 Math And Physics Invariants

## Units, scales, and sign conventions

- All curve geometry is expressed in meters (length unit of the coil set);
  curve coefficients (Fourier `xc`, `xs`, `yc`, `ys`, `zc`, `zs`, RZ `rc`,
  `zs`, planar `q*` quaternion, helical `dofs[0]`, perturbed-sample arrays)
  use the corresponding length / angle units.
- Quadpoint parameter `\theta` is unit-spaced on `[0, 1)`: `nquadpoints`
  points equally distributed by default (`np.linspace(0, 1, n, endpoint=False)`)
  for XYZ, Planar, Helical, and Symmetries; `CurveRZFourier` quadpoints are
  rescaled by `1/nfp` inside `_make_spec_rzfourier` to match the upstream
  convention (`test_biotsavart_jax_parity.py:670`).
- `\gamma(\theta)` and `\gamma'(\theta)` parity uses the convention that
  `gammadash = d\gamma / d\theta` with `\theta` on `[0, 1)` (the `jax.jvp`
  tangents are `_ones_like_float64(quadpoints)` per
  `curve_geometry.py:193-218`).
- Symmetry: curve geometry is a per-coil object; `stellsym=True/False` is a
  surface attribute and a CurveRZFourier attribute. `CurveXYZFourierSymmetries`
  itself carries `nfp`, `stellsym`, and `ntor`, but the parity contract is
  evaluated at a single instance (no stellarator symmetrization is applied
  inside the spec / pullback).

## Orientation and derivative shape

- `curve.gamma()` and `curve_geometry_from_dofs(spec, dofs)["gamma"]` both
  return `(nquadpoints, 3)` arrays with column order `[x, y, z]`.
- Counter-clockwise quadpoint ordering is preserved; the `CurveXYZFourier`
  direct geometry path uses
  `jaxfouriercurve_geometry_pure(curve_dofs, quadpoints, order)` and the
  remaining specs use `gamma_kernel(qp)` with the same quadpoint vector, so
  ordering is exactly the legacy convention.

## Negative controls / red evidence

The red step at parent commit `a9da18fac` shows:

- The seven `to_spec`-bearing classes return identical CPU/JAX geometry
  before and after the new file lands, so a forward port without the new
  tests is harmless.
- The `CurveXYZFourierSymmetries` row's documented skip is recorded as the
  `architecture` blocker — it cannot pass on parent because parent does not
  expose a JAX spec for that class; the new test communicates the blocker
  rather than silently passing.
- Production-scale parametrization at `ncoils=4, nquadpoints=64` runs the
  same kernels at a meaningfully larger working set than the existing
  `_CURVE_SPEC_FACTORIES` ncoils=1 fixture, and at a larger `nquadpoints`
  than the existing `_build_*_curve(16-32)` integration fixtures.

## Excluded singular regimes

- Trefoil-type degeneracies and `gcd(nfp, ntor) != 1` configurations are
  rejected at `CurveXYZFourierSymmetries.__init__` (`curvexyzfouriersymmetries.py:111`).
  The item 05 test fixture uses `nfp=3, ntor=1` or skips before any
  instantiation that would trip the gcd guard.
- Near-zero coil radii are avoided by selecting `R0 \\geq 1.0` and `R1, r
  \\in [0.18, 0.5]` in the new production-scale fixture; this matches the
  existing `_CURVE_SPEC_FACTORIES` and integration-test conventions.
