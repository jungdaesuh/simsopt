# Item 13 JAX Transform And Memory Plan

## Kernel Boundary

`evaluate_batch(spec, xyz)` is the compiled evaluator. The inner JIT
boundary is `_evaluate_batch_jit` in
`src/simsopt/jax_core/regular_grid_interp.py`. It consumes:

- `xyz`: `float64[N, 3]` evaluation points.
- `cell_table`: `float64[cells_to_keep + 1, degree+1, degree+1,
  degree+1, value_size]` per-cell DOF values, padded with a zero
  sentinel row that absorbs skipped / OOB queries.
- `cell_to_row`: `int32[nx*ny*nz]` flat 3D cell index -> row in
  `cell_table`.
- `nodes`, `scalings`: `float64[degree+1]` 1D Lagrange rule data.
- `xmesh`, `ymesh`, `zmesh`: per-axis mesh node positions.
- `xmin`/`xmax`/`ymin`/`ymax`/`zmin`/`zmax`, `hx`/`hy`/`hz`, `nx`/`ny`/`nz`:
  scalar mesh metadata.
- `sentinel_row`: int32 scalar pointing at the zero sentinel row.
- `degree`, `value_size`, `out_of_bounds_ok`: static arguments
  (Python ints/bools).

The build phase (`build_regular_grid_interpolant_3d`) is pure-Python
NumPy. It evaluates the user function on the retained DOF set, packs
them into the per-cell table, and assembles the spec. No JAX transforms
fire during build.

## Transforms

- `jit`: required. The inner kernel is wrapped with
  `partial(jax.jit, static_argnames=("degree", "value_size",
  "out_of_bounds_ok"))` so that the static degree and value-size
  constants compile into the basis-evaluation and `einsum`
  contractions.
- `vmap`: the per-sample evaluator is `vmap`ped over the leading axis
  of `xyz`. Each sample independently looks up its cell row and
  contracts against the local DOF tensor.
- `einsum`: the tensor-product contraction is
  `jnp.einsum("i,j,k,ijkl->l", pkx, pky, pkz, local_vals,
  optimize=True)`. This is the O(degree^4) hot kernel from
  `regular_grid_interpolant_3d_impl.h:150-200`.
- `scan`, `fori_loop`, `checkpoint`/`remat`, `shard_map`,
  `pmap`/collectives: `N/A`. The item is a per-sample kernel; the
  expected production scale (16-64 cells per axis, value size 3-8)
  fits comfortably in a single device. No new collective path is
  introduced; the `git grep` over the diff confirms zero
  `shard_map`/`psum`/`all_reduce`/`pjit` introductions.
- `grad`: not exercised by item 13. The downstream tracing /
  interpolated-field consumers are read-only on the interpolant; they
  do not differentiate through `evaluate_batch`. Adding `grad` support
  is straightforward (the kernel is already pure-JAX with no host
  control flow over traced values) but is not in the item 13 scope.

## Memory Strategy And Dense Materialization Budget

The dense `cell_table` is the largest object. Worst-case shape for a
representative `InterpolatedField` setup
(nr = nphi = nz = 32, degree = 5, value_size = 3) is `(32^3 + 1, 6, 6,
6, 3) * 8 bytes ≈ 170 MB`. Typical Item 13 production use is
`nr = nphi = nz = 16-64` and `degree = 2-4`, which falls in the 1-40 MB
range. This is below the existing `max_dense_jacobian_bytes` ceiling
used elsewhere in the repo for dense per-batch buffers.

The benchmark fixture uses 16 × 16 × 16 cells, degree 4, value size 3
to keep the table compact and to leave headroom for the C++ oracle
comparison: `(16^3 + 1) * 5^3 * 3 * 8 = ~12 MB`.

The kernel never recomputes the table inside the jit boundary; it is
captured by the spec dataclass and staged to the device at
`evaluate_batch` call time. No buffer donation is used because the
caller may reuse the same spec across many evaluation batches. A
donate-argnums proof is therefore N/A for this item.

## Bench / HLO Artifact

See `.artifacts/jax_port_goal/bench/13.json`. CPU-only timings; CUDA
proof is not claimed.
