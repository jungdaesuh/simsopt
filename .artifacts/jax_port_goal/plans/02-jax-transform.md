# Item 02 JAX Transform And Memory Plan

## Compiled Boundary

Two `@jit`-decorated entrypoints in `src/simsopt/field/selffield.py`:

- `@jit B_regularized_singularity_term(rc_prime, rc_prime_prime,
  regularization)` at `selffield.py:97`. Consumes:
  - `rc_prime`: `float64[n, 3]`
  - `rc_prime_prime`: `float64[n, 3]`
  - `regularization`: scalar
- `@jit B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints,
  current, regularization)` at `selffield.py:116`. Consumes:
  - `gamma`: `float64[n, 3]`
  - `gammadash`: `float64[n, 3]`
  - `gammadashdash`: `float64[n, 3]`
  - `quadpoints`: `float64[n]`
  - `current`: scalar
  - `regularization`: scalar
  - returns `float64[n, 3]`

No static argument specs are required — shape stability is provided
naturally by the caller, which is either a `RegularizedCoil` wrapper
that exposes consistent `gamma`/`gammadash`/`gammadashdash` arrays for
the lifetime of the coil, or a `vmap`-over-coils caller in
`src/simsopt/field/force.py` that groups by quadrature-point count.

## Transforms

- `jit`: applied directly to both `B_regularized_pure` and the inner
  `B_regularized_singularity_term` via `@jit` at the function
  definitions.
- `vmap`: the vmap-over-coils transform is performed by callers (not by
  the kernel module itself), e.g. `vmap(B_regularized_pure, in_axes=(0,
  0, 0, None, 0, 0))` at `src/simsopt/field/force.py:2016` and
  `src/simsopt/field/force.py:2394`. The `None` axis on `quadpoints`
  reflects that all coils in a group share the same quadrature grid.
- `grad` / `jacrev` / `jacfwd`: supported transparently because the
  kernel is pure JAX; downstream gradient consumers go through
  `RegularizedCoil` value paths and the surrounding force objective
  gradients.
- `scan` / `fori_loop`: N/A. The kernel is a single batched matmul-like
  reduction over `(n, n)` pairwise displacements; no chunked path is
  introduced by selffield.py.
- `checkpoint` / `remat`: N/A. The kernel does not introduce a
  gradient-rematerialization layer at this item; remat decisions live
  in the consuming force objective.
- `shard_map`: N/A. No sharding is introduced; `git diff a9da18fac..HEAD
  -- src/simsopt/field/selffield.py` is empty.
- `pmap` / collectives: N/A for the kernel module. Multi-device tests
  live in `tests/test_jax_import_smoke.py` for shared force-objective
  paths only.

## Memory and donation

- Largest array per coil: `dr = rc[:, None] - rc[None, :]` at
  `selffield.py:141` materializes `float64[n, n, 3]`. For
  `nquadpoints=128` per coil this is `128 * 128 * 3 * 8 = 393_216`
  bytes = `0.375` MiB per coil. The 4-coil production-scale fixture
  evaluated via `vmap` totals `1.5` MiB of peak `dr`-state plus
  comparable factors. This is well within any production budget and
  does not approach `max_dense_jacobian_bytes` because there is no
  Jacobian materialization at this layer.
- No buffer donation. `donate_argnums` / `donate_argnames` are N/A:
  callers (`RegularizedCoil.B_regularized()` and the force objective
  vmap) consume the geometry arrays multiple times across distinct
  objectives.

## Linkage to bench / HLO

`.artifacts/jax_port_goal/bench/02.json` is N/A: this item is a
documentation closeout and does not change `selffield.py`. The cited
production-scale parity test
`tests/field/test_selffield_item02_closeout.py::test_b_regularized_pure_matches_circular_closed_form_oracle_at_production_scale`
provides the witness that the existing JAX kernel meets the direct-
kernel parity-ladder lane at `ncoils=4, nquadpoints=128`.
