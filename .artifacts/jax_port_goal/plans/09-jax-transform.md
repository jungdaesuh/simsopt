# Item 09 JAX Transform Plan

Status: complete for CPU/JAX oracle closure. No transform-structure change is
introduced by item 09. This document is the audit record for the existing
transforms inside `src/simsopt/field/force.py`.

## Compiled boundary

Module-level lifted `jit + grad` entrypoints at
`src/simsopt/field/force.py:953-976` are the public compiled boundary used by
every wrapper:

| Entrypoint | Definition line | Static args | Notes |
| --- | --- | --- | --- |
| `_B2ENERGY_JAX` | `:953` | `static_argnums=(3,)` (downsample) | shared across all `B2Energy` instances |
| `_B2ENERGY_GRAD` | `:954` | same as above, `grad(..., argnums=(0, 1, 2))` | gammas, gammadashs, currents grads |
| `_NET_EXT_FLUX_JAX` | `:955` | `static_argnums=(2,)` (downsample) | shared across all `NetFluxes` instances |
| `_NET_EXT_FLUX_GRAD` | `:956` | same, `argnums=(0, 1)` | gammadash, A_ext grads |
| `_SQUARED_MEAN_FORCE_JAX` | `:957` | `static_argnums=(9,)` (downsample) | 9-input flat positional API |
| `_SQUARED_MEAN_FORCE_GRAD` | `:958` | same, `argnums=tuple(range(9))` | all nine inputs differentiated |
| `_LP_FORCE_JAX` | `:962` | `static_argnums=(14,)` (downsample) | 15-input flat positional API |
| `_LP_FORCE_GRAD` | `:963` | same, `argnums=tuple(range(10))` | only the 10 array inputs differentiated; `quadpoints`, `regularizations`, `p`, `threshold` are not grad targets |
| `_LP_TORQUE_JAX` | `:967` | `static_argnums=(14,)` (downsample) | same shape as LpCurveForce |
| `_LP_TORQUE_GRAD` | `:968` | same | same |
| `_SQUARED_MEAN_TORQUE_JAX` | `:972` | `static_argnums=(9,)` (downsample) | 9-input flat positional API |
| `_SQUARED_MEAN_TORQUE_GRAD` | `:973` | same | same |

Each wrapper class binds `self.J_jax` and `self.dJ_jax` to these module-level
compiled functions so the XLA compile cache hits across class instances. The
fork keeps `downsample` static for shape stability under JIT.

## Static metadata vs runtime arrays

Static metadata held inside the wrapper but passed as positional runtime
arrays (not static argnums) for `LpCurveForce` and `LpCurveTorque`:

- `quadpoints`: `_as_jax_float64([c.curve.quadpoints for c in target_coils])`
  at `force.py:2159` and `force.py:2511`.
- `regularizations`: `_as_jax_float64([c.regularization for c in target_coils])`
  at `force.py:2134` and `force.py:2486`.
- `p`, `threshold`: cast to float64 at `_J_args` time so their JIT traces are
  shape-stable.

`downsample` is the only Python static argument in every compiled call. This
is intentional because `downsample` slices array axes via Python slicing
`[:, ::downsample, :]` inside the kernels at `force.py:1026-1027`,
`force.py:1075-1076`, and elsewhere; slicing strides cannot be made traceable
without re-shape semantics that the upstream port already chose not to take.

## Transform inventory (hot path)

- `jit`: applied at module level at lines `953-976`. No nested `jit` inside the
  pure kernels.
- `grad`: applied at module level at lines `954, 956, 959, 964, 969, 974` and
  immediately wrapped in `jit`. Native JAX cotangents; no `value_and_grad`,
  no `vjp` construction at the public boundary.
- `vmap`: applied inside the pure kernels at the following locations:
  - `force.py:144` — `vmap(from_j)(jnp.arange(n))` inside
    `_B_at_point_from_coil_set_pure`.
  - `force.py:1685` — `vmap(B_at_pt)(gamma_i)` over target quadrature points
    inside `squared_mean_force_pure.mean_force_group1`.
  - `force.py:1689` — `vmap(mean_force_group1, in_axes=(0, 0, 0, 0, 0))` over
    target coils inside `squared_mean_force_pure`.
  - `force.py:2016` — `vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0))`
    over target coils inside `lp_force_pure` (regularized B self).
  - `force.py:2026` — nested `vmap` inside `lp_force_pure.per_coil_obj_group1`
    over quadrature points.
  - `force.py:2046` — `vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0))`
    over target coils inside `lp_force_pure`.
  - `force.py:2391` — `vmap(centroid_pure, in_axes=(0, 0))` over target coils
    inside `lp_torque_pure`.
  - `force.py:2394` — `vmap(B_regularized_pure, ...)` over target coils inside
    `lp_torque_pure`.
  - `force.py:2430` — `vmap(torque_at_point)(jnp.arange(npts1))` inside
    `lp_torque_pure.per_coil_obj_group1`.
  - `force.py:2432` — `vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0, 0))`
    over target coils inside `lp_torque_pure`.
  - `force.py:2764` — `vmap(centroid_pure, in_axes=(0, 0))` over target coils
    inside `squared_mean_torque`.
  - `force.py:2769` — nested `vmap` inside `squared_mean_torque` over source
    points.
  - `force.py:2789` — `vmap(mean_torque_group1, in_axes=(0, 0, 0, 0, 0))` over
    target coils inside `squared_mean_torque`.

- `scan` / `fori_loop` / `checkpoint`/`remat` / `shard_map` / `pmap` /
  collectives: `N/A: no scan, no fori_loop, no remat, no shard_map, no pmap, no
  collectives inside force.py`. Item 09 keeps the upstream all-vmap structure.

## Why the transform structure matches the SIMSOPT math contract

- Inner `vmap` over quadrature points and outer `vmap` over target coils mirror
  the upstream `\int d\ell` and `\sum_i` structure in the math docstrings on
  each wrapper. The compiled scalar output for each kernel is the same scalar
  the upstream `J()` returns.
- `grad` at the module level differentiates the scalar through both `vmap`
  axes. The resulting derivative blocks have shape:
  - `gammas`/`gammadashs`/`gammadashdashs`: `(ncoils, nquadpoints, 3)`.
  - `currents`: `(ncoils,)`.
  These shapes match what `_assemble_curve_current_derivative` projects
  through `curve.dgamma_by_dcoeff_vjp`, `curve.dgammadash_by_dcoeff_vjp`,
  `curve.dgammadashdash_by_dcoeff_vjp`, and `current.vjp` to produce the
  `Derivative` shape the legacy `Optimizable.dJ()` contract expects.
- No transform reshapes the scalar objective or changes the solve residual
  being compared. The objective remains a CPU/JAX-portable real-valued
  scalar.

## Dense materialization budget

Largest dense arrays under the new closeout fixture (`ncoils=4`,
`numquadpoints=64`, `coils_via_symmetries(ncoils=4, nfp=3, stellsym=True)` =>
24 expanded coils):

- Per-coil `(numquadpoints, 3)` curve states: `24 * 64 * 3 * 8 = 36864 B`.
- Stacked target `(ncoils_target, numquadpoints, 3)` = `(1, 64, 3)` in the
  closeout fixture (1 target coil) = 1536 B.
- Stacked source `(ncoils_source, numquadpoints, 3)` = `(24, 64, 3)` =
  36864 B.
- `gammas_targets` cotangent block from `grad`: same shape as `gammas_targets`
  = 1536 B.
- `gammas_sources` cotangent block: same shape as `gammas_sources` =
  36864 B.
- Inner `_mutual_B_field_at_point_pure` produces per-pair `(nsource_pts, 3)`
  intermediates inside `vmap` over target points. With 64 source quadrature
  points and 24 source coils, the peak intermediate during the inner sum is
  `(24, 64, 3) * float64 = 36864 B` plus the broadcasted `r_ij`/`rij_norm`
  block of shape `(24, 64) * float64 = 12288 B`.

Peak intermediate well under any practical CPU/GPU host or device memory
ceiling; no donation is required.

Buffer donation: **not used**. No positional `donate_argnums` and no named
`donate_argnames` are donated by any of the `_*_JAX` or `_*_GRAD`
entrypoints. The fork preserves all input arrays so the caller can re-enter
the cached `_J_args` payload on the next `J()` / `dJ()` call without
recomputing the host-side coil state. This matches the upstream non-donating
behavior.

HLO or benchmark evidence path: `.artifacts/jax_port_goal/bench/09.json`
records `hot_path_change: false` and the production-scale fixture sizes; no
new HLO is needed because the transform structure is unchanged from the
parent commit.

## Sharding / collective considerations

`N/A: no sharding or collective in force.py.` Item 09 does not touch any
sharded array, named axis, `Mesh`, `NamedSharding`, `shard_map`, `pmap`, or
collective. CPU proxy and CUDA artifact follow-up paths are not required.

## CUDA performance follow-up

`cuda_smoke="not_claimed"`. CPU JAX 0.10.0 / jaxlib 0.10.0 only. The
runtime's `policy.transfer_guard == "disallow"` path in
`src/simsopt/backend/runtime.py:913` already forces dense-audit chunking when
strict transfer guard is active, but no GPU run was approved for this item.
