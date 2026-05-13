# Item 02 JAX Transform And Memory Plan

## Kernel Boundary

`B_regularized_pure(gamma, gammadash, gammadashdash, quadpoints, current,
regularization)` is the compiled self-field kernel. It consumes explicit
arrays:

- `gamma`: `float64[n, 3]`
- `gammadash`: `float64[n, 3]`
- `gammadashdash`: `float64[n, 3]`
- `quadpoints`: `float64[n]`
- `current`: scalar
- `regularization`: scalar

The public wrappers are `RegularizedCoil.B_regularized()` and
`RegularizedCoil.self_force()`. They read legacy mutable curve/current state at
the Python boundary and now enter explicit transfer-guard `allow` regions before
calling JAX kernels.

## Transforms

- `jit`: supported and tested for `B_regularized_pure` and both
  regularization helpers.
- `vmap`: supported and tested for batched current/regularization leaves.
- `grad`: supported and tested for scalar regularization helper derivatives.
- `pmap`: CPU proxy tested with two forced host devices and
  `B_regularized_pure`; no CUDA or performance claim is made.
- `scan`, `fori_loop`, `checkpoint`, `shard_map`: N/A for item 02. The kernel
  is dense over the quadrature grid and does not introduce a new chunking or
  rematerialization path in this item.

## Memory

The dense kernel forms `dr = rc[:, None] - rc[None, :]`, so the largest
temporary is `float64[n, n, 3]` plus `float64[n, n]` scalar factors. The item 02
production code change does not increase that memory order; it only makes the
wrapper transfer boundary explicit.

No buffer donation is used. Public wrappers may be called repeatedly on mutable
legacy curve/current objects, so donated inputs would be unsafe for this
boundary.

## Bench / HLO Artifact

`.artifacts/jax_port_goal/bench/02.json` records that no new hot-path kernel was
introduced. Item 02 validation focuses on strict transfer boundaries,
existing analytic self-field oracle tests, and a two-device CPU transform proxy.
