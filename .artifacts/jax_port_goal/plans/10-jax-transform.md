# Item 10 JAX Transform And Memory Plan

## Biot-Savart kernels

Compiled boundaries:

- `simsopt.jax_core.biotsavart.biot_savart_B(points, gammas, gammadashs,
  currents)` at `src/simsopt/jax_core/biotsavart.py:573` — pure JAX kernel.
  Returns `(npoints, 3)`.
- `simsopt.jax_core.biotsavart.biot_savart_B_vjp` at line 636 —
  reverse-mode VJP of `biot_savart_B`.
- `simsopt.jax_core.biotsavart.biot_savart_dB_by_dX` — spatial Jacobian.
- `simsopt.jax_core.field.grouped_biot_savart_B_from_spec(points,
  GroupedCoilSetSpec)` at `src/simsopt/jax_core/field.py:445` — groups
  coils by quadrature point count and dispatches into the dense JAX
  kernels (mixed-quadrature contract).
- `BiotSavartJAX.B()` and `BiotSavartJAX.B_vjp(v)` —
  `simsopt.field.biotsavart_jax_backend.BiotSavartJAX` wrappers that
  consume immutable grouped specs via `coil_set_spec()` and stage host
  points outside the jit boundary.

Transforms used by the hot path:

- `jit`: every chunked low-level kernel is jitted by an `lru_cache`-keyed
  factory at `src/simsopt/jax_core/biotsavart.py:409` (forward) and
  `:518` (VJP). The kernel factory keys on the tuning tuple `(coil_chunk_size,
  quadrature_block_size, point_chunk_size)` resolved by
  `_read_tuning_config()`.
- `vmap`: NOT used at the public boundary. Coil and point loops are
  expressed as `lax.scan` over chunk indices to keep dense
  materialization bounded.
- `scan` / `fori_loop`: `lax.scan` is used to walk coil chunks,
  quadrature blocks, and point chunks. `fori_loop` is not used here;
  carrier shapes are stable through the scan boundary.
- `checkpoint` / `remat`: applied to chunked kernel bodies where
  recomputation is cheaper than retaining the per-chunk pairwise
  matrices (consistent with the pairwise distance kernels in item 01).
- `shard_map` / `pmap` / collectives: explicit point/coil sharding is
  available through `simsopt.jax_core.sharding`; multi-device CPU
  proxies (`test_grouped_biot_savart_coil_collective_parity_and_lowering`)
  cover the collective lane. No new collective is introduced by item 10.

## integral_BdotN

Compiled boundary:

- `simsopt.objectives.integral_bdotn_jax.integral_BdotN(Bcoil, target,
  normal, definition, reduction_mode)` at
  `src/simsopt/objectives/integral_bdotn_jax.py:93`. Jit-decorated with
  `definition` and `reduction_mode` as static argnames.
- `residual_BdotN` at line 38 produces the residual vector; the
  scalar reduction uses `scalar_square_sum` and `pairwise_sum_flat`
  from `simsopt.jax_core.reductions`.
- `fixed_surface_flux_integral_from_B` at
  `src/simsopt/jax_core/objectives_flux.py:62` is the spec-driven
  wrapper that reshapes a flat B to `(nphi, ntheta, 3)` and dispatches
  into `integral_BdotN`.

Transforms used:

- `jit`: forward integral and residual are jitted with static
  `definition` / `reduction_mode`.
- `vmap`: NOT required at this boundary.
- `scan`: NOT directly used inside `integral_BdotN`. The reducers in
  `simsopt.jax_core.reductions` use a `pairwise_sum_flat` reduction that
  is `lax.scan`-based for numerical stability when `reduction_mode` is
  `strict_oracle`.
- `checkpoint` / `remat`: NOT used inside `integral_BdotN` (cheap
  reduction; no carry-heavy state).
- `shard_map` / `pmap` / collectives: N/A for this boundary. The
  reduction kernels are dense over a `(nphi, ntheta)` grid; the
  multi-device collective work is upstream in the BS evaluation, which
  is owned by the BS kernels above.

## SquaredFluxJAX adapter (consumer of this item, not modified here)

`SquaredFluxJAX.J()` captures fixed surface arrays in JIT closures at
construction time and consumes
`fixed_surface_flux_integral_from_B(B, flux_spec)` for the value path.
Gradient path consumes `BiotSavartJAX.B_vjp(v)`. This adapter is read,
not modified, by item 10. The new test exercises the bare
`integral_BdotN(BiotSavartJAX.B())` chain to anchor the kernel-level
contract that the wrapper rides on.

## Dense materialization budget

- Largest BS array in the new test: `dB_by_dX` would be
  `(npoints, 3, 3) = (16 * 8, 3, 3) = (128, 3, 3)` — 9216 float64 bytes
  per array. The new test does NOT use dB/dX; only B is needed.
  `B` shape `(npoints, 3) = (128, 3)` — 3072 float64 bytes.
- `normal` and `target` arrays at `(nphi, ntheta, 3)` and `(nphi,
  ntheta)` — 384 + 128 float64 bytes.
- The chunked path's largest intermediate is bounded by
  `O(coil_chunk_size * quadrature_block_size * point_chunk_size)` per
  the tuning policy. With ncoils=4 (8 after stellsym), 64 quadpoints,
  and 128 points, the dense fallback fits easily in any CPU/GPU
  ladder budget.
- No buffer donation. `donate_argnums` and `donate_argnames` are N/A
  because both `BiotSavartJAX` and `integral_BdotN` reuse their input
  spec arrays.

## HLO / benchmark artifact

- `.artifacts/jax_port_goal/bench/10.json` records that item 10
  introduces no new hot-path kernel; the closeout adds a parity gate
  only. No timing benchmark is required. See section 4c carve-out:
  "no perf change expected because <one-line justification>" applies
  because no implementation change is made; the existing kernels
  already have benchmark coverage in
  `tests/field/test_biotsavart_jax_parity.py` and
  `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedParity`.

## Static-shape strategy

- `definition` and `reduction_mode` are static argnames at the
  `integral_BdotN` boundary so the three definition variants compile to
  three distinct kernels.
- BS kernels key compilation on the chunk-size tuning tuple resolved
  by `_read_tuning_config()`; per-call kernels see fixed shapes after
  spec construction.
- The new closeout test only exercises one tuning tier; mixed-quadrature
  and chunked tier coverage already lives in
  `tests/field/test_biotsavart_jax_parity.py`.
