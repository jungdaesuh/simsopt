# Item 08 JAX Transform And Memory Strategy

## Compiled Boundary

`strain_optimization.py` exposes six module-level jit-cached entrypoints
(no per-instance `jit` closures) and two bare `@jit` math primitives:

- `_lp_strain_penalty_value` (`src/simsopt/geo/strain_optimization.py:18`)
  — `partial(jit, static_argnames=("p", "threshold"))` wrapper around
  `Lp_torsion_pure`. Public boundary for `LPTorsionalStrainPenalty.J` and
  `LPBinormalCurvatureStrainPenalty.J`.
- `_lp_strain_penalty_grad` (`strain_optimization.py:23`) —
  `grad(_lp_strain_penalty_value, argnums=(0, 1))` with the same static
  argnames. Public boundary for `dJ`.
- `_torstrain_eval` (`strain_optimization.py:33`) —
  `partial(jit, static_argnames=("width",))` pointwise strain.
- `_binormstrain_eval` (`strain_optimization.py:38`) — same family.
- `_torstrain_vjp` (`strain_optimization.py:43`) and
  `_binormstrain_vjp` (`strain_optimization.py:48`) — `vjp`-based
  pullback of the two strain primitives, cached with `width` static.
- `torstrain_pure` (`strain_optimization.py:231`) and
  `binormstrain_pure` (`strain_optimization.py:239`) — bare `@jit`
  math SSOTs reused by the helpers above.

Item 08 does not change any of these boundaries. The change is the
explicit `jax.transfer_guard("allow")` block at each wrapper's `J()` /
`dJ()` body that stages the host `gammadash` array through
`jax_core._math_utils.as_jax_float64` before calling
`_lp_strain_penalty_value` / `_lp_strain_penalty_grad`.

## Static Shape Strategy

- `p` and `threshold` are Python scalars marked static through
  `static_argnames`. They never enter the compiled traced state.
- `width` is a Python scalar marked static.
- `strain_like` / `torsion` / `binorm` are `(nquadpoints,)` arrays
  coming from `framedcurve.frame_torsion()` or
  `framedcurve.frame_binormal_curvature()`. Those upstream functions
  already produce JAX `float64` arrays through the existing framedcurve
  Frenet/centroid pure paths.
- `gammadash` is `(nquadpoints, 3)` float64 staged from the host
  `framedcurve.curve.gammadash()` NumPy array. At the production-scale
  NCSX `coil_order=6, points_per_period=120` fixture, `nquadpoints = 720`.

## Transform Inventory

- `jit`: lines 17, 22, 32, 37, 42, 47, 230, 238 in
  `strain_optimization.py`. All `static_argnames` use Python scalars.
- `grad(_lp_strain_penalty_value, argnums=(0, 1))` at line 24 — single
  reverse-mode autodiff over the integrand wrt `(strain_like, gammadash)`.
- `vjp(lambda g: torstrain_pure(g, width), torsion)[1](v)[0]` at line 44
  (and the binormal equivalent at line 49) — explicit `vjp` pullback that
  composes with `framedcurve.dframe_torsion_by_dcoeff_vjp` /
  `dframe_binormal_curvature_by_dcoeff_vjp` and
  `curve.dgammadash_by_dcoeff_vjp` at the `Derivative` projection step.
- `vmap`: not used. The pointwise strain definitions broadcast natively
  over the `nquadpoints` axis through `jnp.abs` and elementwise products.
- `scan` / `fori_loop`: not used. Integrand is a `jnp.mean` reduction.
- `checkpoint` / `remat`: not used.
- `shard_map`, `pmap`, collectives: not used. The strain wrappers are
  per-curve operators; no inter-curve collective lives in this module.

## Dense Materialization And Donation

- Largest array in the new transfer-boundary work is the
  `(nquadpoints, 3)` host `gammadash` staged onto the JAX runtime. At
  the production-scale NCSX fixture, that is `(720, 3) * 8 bytes =
  17280 bytes` per call — well below any dense Jacobian budget. The
  jit-internal arrays produced by `_lp_strain_penalty_value` /
  `_lp_strain_penalty_grad` keep the same shape; no Jacobian matrix is
  materialized.
- No buffer donation is used. The wrapper consumes mutable Optimizable
  state and may be called repeatedly inside a scipy line search, so
  donating the `gammadash` staging buffer would be incorrect.
- No `donate_argnums` / `donate_argnames` evidence is required by this
  item.

## Performance / Memory Budget

`N/A: no hot-path change`. The integrand and pointwise strain kernels
already lived at module scope; the item-08 edit only adds an explicit
`as_jax_float64` host-staging boundary inside an `allow` transfer-guard
context. Bench artifact at `.artifacts/jax_port_goal/bench/08.json`
records the justification.

## CUDA Status

CPU-only. JAX 0.10.0 CPU runtime. Transfer-guard documentation notes
that fetching CPU buffers via `device_put` is always permitted; this
edit explicitly stages the host array through `as_jax_float64` inside a
single `jax.transfer_guard("allow")` block, which matches item 01's
`CurrentPenalty` pattern. No CUDA proof is claimed.
