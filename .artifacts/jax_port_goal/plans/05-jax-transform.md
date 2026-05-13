# Item 05 JAX Transform And Memory Plan

## Compiled boundaries

Curve geometry passes through three spec-driven pure-function entry points in
`src/simsopt/jax_core/curve_geometry.py`:

- `curve_spec_from_curve(curve)` — host-side constructor that returns an
  immutable spec pytree (`CurveXYZFourierSpec`, `CurveRZFourierSpec`,
  `CurvePlanarFourierSpec`, `CurveHelicalSpec`, `CurveCWSFourierRZSpec`,
  `CurvePerturbedSpec`, or `CurveFilamentSpec`).
- `curve_geometry_from_dofs(spec, dofs)` -> `(gamma, gammadash, gammadashdash)`
  through `_curve_gamma_kernel(spec)` plus chained `jax.jvp` for higher
  derivatives.
- `curve_pullback_from_dofs(spec, dofs, dg, dgd)` -> cotangents through
  `jax.vjp` against the same kernel.

Item 05 does not introduce any new compiled hot path. It documents the
existing seven-class adapter coverage and adds production-scale parity
fixtures on the unchanged kernel.

## Transforms

- `jit`: not applied at the item-05 test boundary. The underlying `JaxCurve`
  `gamma_jax` and `gamma_pure` paths use `jit` inside the live curve object,
  and `curve_geometry_from_dofs` builds a Python-level closure that is
  traceable but not externally jitted.
- `vmap` over coils: handled at the caller (e.g. Biot-Savart grouped
  geometry); per-curve geometry is computed independently per coil. Item-05
  tests iterate over `ncoils=4` per curve class to mimic the realistic
  caller pattern without staging an extra `vmap` boundary.
- `jvp`: used inside `_curve_geometry_terms_from_kernel` to roll one
  `gamma_kernel` build into `gamma`, `gammadash`, `gammadashdash`, and
  (when requested) `gammadashdashdash`.
- `vjp`: used inside `curve_pullback_from_dofs` for coefficient cotangents.
- `scan`, `fori_loop`, `checkpoint` / `remat`, `shard_map`, `pmap`,
  collectives: N/A. Per-curve geometry is a small dense array (`nquadpoints x 3`)
  with no internal loop or partitioned body.

## Math/physics invariants for the transform

- Direct geometry path for `CurveXYZFourierSpec`
  (`_direct_curve_geometry_terms` in `curve_geometry.py:221-231`) constructs
  derivatives algebraically through `jaxfouriercurve_geometry_pure`; this is
  the SSOT for first/second/third derivatives of the `xc/ys/zs` Fourier
  curve.
- Other spec kinds use `gamma_kernel = lambda qp: ...(curve_dofs, qp, ...)`
  and chain `jax.jvp` with tangents equal to `_ones_like_float64(quadpoints)`
  (`curve_geometry.py:193-218`); this respects the parameter convention
  `\partial / \partial \theta` for unit-spaced quadrature points.

## Memory and donation

- Largest array in item 05 parity tests:
  `(ncoils=4, nquadpoints=64, 3)` float64 ≈ 6 KiB per stack and four classes
  in turn; total transient working set well under 0.5 MiB.
- `donate_argnums` / `donate_argnames`: N/A. Spec arrays are read-only inputs
  to `curve_geometry_from_dofs` and `curve_pullback_from_dofs`, and item 05
  tests reuse the same spec across `curve.gamma()` and the JAX path.
- Buffer donation is unsafe for these kernels because the same spec instance
  is consumed by both forward and pullback paths in downstream consumers
  (e.g. `BiotSavartJAX.grouped_coil_arrays_from_dofs`).

## Sharding / collectives

N/A for item 05. Curve geometry per coil is a small dense computation and
does not cross devices on the CPU lane. CPU proxy evidence for collective
paths is owned by item 01 (`tests/test_jax_import_smoke.py` row-sharding
tests). No CUDA artifact is claimed for item 05.

## HLO / benchmark artifact

- `.artifacts/jax_port_goal/bench/05.json` is N/A for hot-path timing because
  item 05 is documentation/test-coverage closure for already-JAX-native
  adapters. The artifact still records the production-scale floor and cites
  the existing curve-spec fixtures.
