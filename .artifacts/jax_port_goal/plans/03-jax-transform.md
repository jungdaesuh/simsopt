# Item 03 JAX Transform And Memory Plan

## SquaredFluxJAX

### Compiled boundary

- `_init_native_program` at
  `src/simsopt/objectives/fluxobjective_jax.py:164-169` selects the
  uniform-`CurveXYZFourier` fast path or the general spec-native path.
- The compiled boundary closure is bound in `_bind_native_forward` at
  `src/simsopt/objectives/fluxobjective_jax.py:171-182`:
  - `jit_forward = jax.jit(forward)` for value-only evaluation
  - `jit_val_grad = jax.jit(jax.value_and_grad(forward, argnums=0))` for
    combined value + gradient over flat coil DOFs
- The traced forward closure is
  `forward(flat_dofs, flux_spec) -> scalar` so the only dynamic argument
  is the flat coil-DOF vector. `flux_spec` is a pytree of immutable
  arrays captured at construction.
- Largest array on the hot path: the per-quadrature
  `(nphi, ntheta, 3)` B array. For the new closeout test fixture
  (`nphi=16, ntheta=8, ncoils=8` after symmetries, basis order=6) the
  dense B array is `16*8*3*8 = 3072` float64 = 24 KiB.

### Transforms

- `jax.jit` — used on `forward` and on the `value_and_grad(forward)`
  combination. Static spec metadata flows through the `FixedSurfaceFluxSpec`
  pytree (`definition`, `nphi`, `ntheta` are meta_fields).
- `jax.value_and_grad(forward, argnums=0)` over the flat coil DOF vector —
  this is the only autodiff transform on the SquaredFluxJAX hot path. No
  `vmap`, no `pmap`, no `shard_map` is applied at the SquaredFluxJAX
  level.
- Inside `forward`, the uniform-`CurveXYZFourier` fast path uses two
  matmuls per coil:
  `basis @ coeffs.T` and `dbasis @ coeffs.T` at
  `src/simsopt/objectives/fluxobjective_jax.py:222-223`. The Fourier
  basis tensors `basis` and `dbasis` are precomputed at construction
  (`build_fourier_basis` at
  `src/simsopt/jax_core/objectives_flux.py:45-59`).
- The general spec path applies
  `coil_specs_from_dof_extraction_spec` and
  `grouped_coil_set_spec_from_coil_specs` (no `vmap`/`scan` at the
  SquaredFluxJAX boundary — internal grouping is owned by item 10).
- `static_argnames=("definition",)` is used at
  `src/simsopt/objectives/integral_bdotn_jax.py:37` for `residual_BdotN`
  and at `src/simsopt/objectives/integral_bdotn_jax.py:92` for
  `integral_BdotN`. `definition` is therefore a compile-time constant;
  one compiled program is built per `definition` per fixture.
- `jax.checkpoint` / `remat`: N/A. SquaredFluxJAX itself does not
  rematerialize; the surface arrays are captured immutably.
- `jax.scan`, `jax.fori_loop`: N/A at the SquaredFluxJAX boundary. The
  inner grouped Biot-Savart kernel may scan internally but that is
  owned by item 10.
- `shard_map`, `pmap`, collectives: N/A for the SquaredFluxJAX hot path.
  Multi-device CPU subprocess proxy for Biot-Savart and pairwise
  reductions is owned by other items.

### Memory and donation

- Dense materialization budget on item 03's added fixture (production
  scale):
  - Surface gamma/normal arrays: `nphi*ntheta*3 = 384` float64 each =
    3 KiB.
  - Target Bn array: `nphi*ntheta = 128` float64 = 1 KiB.
  - Coil B output: `nphi*ntheta*3*ncoils = 3072` float64 (the inner
    grouped kernel sums across coils, so the live aggregate is
    `(nphi*ntheta, 3)` = 384 float64 = 3 KiB; the per-coil block is
    short-lived inside `grouped_biot_savart_B_from_spec`).
  - Flat coil DOFs: `2*ncoils_base*order = O(40)` float64.
  - All totals are bounded by `< 100 KiB` for the closeout fixture.
- Buffer donation: `donate_argnums` and `donate_argnames` are N/A.
  `SquaredFluxJAX` reuses captured surface arrays for the lifetime of
  the objective; CurveXYZFourier DOFs are also reused between calls.
  Donating either would break the explicit cache-reuse contract in
  `_clear_cached_results` and `J()`/`dJ()` at
  `src/simsopt/objectives/fluxobjective_jax.py:272-330`.

### Why this matches the SIMSOPT math contract

- `SquaredFlux.J()` is a scalar; the JAX value path returns the same
  scalar.
- `SquaredFlux.dJ()` returns a `Derivative` keyed on coil dependencies.
  The JAX path projects the flat DOF gradient back to a `Derivative`
  via `_field_dofs_gradient_to_derivative` at
  `src/simsopt/objectives/fluxobjective_jax.py:69-91`, using the
  field's `unique_dof_lineage` and `local_dofs_free_status` to share
  blocks across symmetry-induced duplicates.
- Item 03 introduces no transform change; it adds coverage only.

## HLO / benchmark artifact

- `.artifacts/jax_port_goal/bench/03.json` records that item 03 is not a
  new production hot path. No micro-bench is required because no kernel
  change was made; the closeout test exists to bundle production-scale +
  3 definitions + strict transfer + stellsym=True parity in a single
  fixture.
