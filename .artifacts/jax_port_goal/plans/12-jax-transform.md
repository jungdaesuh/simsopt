# Item 12 — JAX Transform and Memory Strategy

## Compiled boundary

- One `@jax.jit` wrapper per public kernel (9 in total). Each jit closure
  receives the spec scalars and the `(N, 3)` cartesian points as
  positional arguments. No static_argnames beyond the implicit shape
  signature.
- Per-point pure functions are vmapped over the leading `N` axis with
  `in_axes=(0, None, None[, None])` so the spec scalars broadcast over
  all points. This keeps the inner functional form readable and the
  XLA fusion straightforward.

## Transform inventory

| Transform                | Used? | Rationale                                                                                                   |
| ------------------------ | ----- | ----------------------------------------------------------------------------------------------------------- |
| `jit`                    | yes   | One module-level jit per public kernel. Caching is shape-keyed on `points.shape`.                            |
| `vmap`                   | yes   | Per-point closed-form evaluator; vmap over `N` is the canonical batched evaluation for pointwise kernels.    |
| `scan`                   | N/A   | No sequential reduction; each point is independent.                                                          |
| `fori_loop`              | N/A   | Same as above.                                                                                                |
| `checkpoint` / `remat`   | N/A   | Largest tensor per point is `(3, 3, 3)`; no memory pressure that would justify rematerialisation.            |
| `shard_map`              | N/A   | No collective; single-device CPU validation.                                                                  |
| `pmap` / collectives     | N/A   | Same as above.                                                                                                |
| `custom_vjp`             | N/A   | Forward only; no autodiff path is exposed.                                                                    |

## Static-shape strategy

- The only varying argument is `points` (shape `(N, 3)`); jit caches one
  compile per `N`. There is no per-call branching, so no
  `static_argnames` is needed.
- Spec dataclasses carry only Python floats in `meta_fields`, so they
  enter the cache key as static. This means a different `R0` / `B0` / `q`
  / `gamma` / `Z_m` reuses the same compile (Python floats become traced
  scalars routed through `_as_jax_float64`). Because the meta floats
  themselves do not appear inside `jit` traces (they are converted to
  `jax.Array` outside `jit`), the cache is not invalidated by spec value
  changes.

## Dense materialisation budget

- Per-point tensor sizes:
  - `(N, 3)` for B / A: `24 * N` bytes float64.
  - `(N, 3, 3)` for dB / dA: `72 * N` bytes float64.
  - `(N, 3, 3, 3)` for d2B: `216 * N` bytes float64.
- At the `N = 1000` benchmark fixture, the largest array is 216 kB —
  well within any reasonable host or device budget. No allocation
  approaches `max_dense_jacobian_bytes`.

## Donation strategy

- **No buffer donation.** These kernels are pointwise evaluators; the
  output arrays are distinct from inputs (different shape) and the
  caller commonly re-evaluates with the same `points`. Donation does
  not buy any meaningful memory win at the scales used by analytic
  fields, and `donate_argnums` would silently invalidate input arrays
  in violation of the spec contract.

## HLO / bench evidence

Bench artifact: `.artifacts/jax_port_goal/bench/12.json`. The bench runs
`toroidal_B` at `N = 1000` against the CPU `ToroidalField._B_impl` over
≥ 100 timed calls after a ≥ 5-call warmup; the JAX entrypoint goes through
`block_until_ready()` to gate async dispatch.

## CUDA reality

`cuda_smoke: not_claimed` (port_closure profile). The kernels themselves
are JAX-native and trivially portable to CUDA when an approved GPU run is
available, but no GPU artifact is produced in this run.
