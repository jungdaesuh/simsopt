# Item 11 JAX Transform And Memory Strategy

## Compiled Boundary

- ``_dommaschk_B_kernel(m, n)`` and ``_dommaschk_dB_kernel(m, n)``
  return ``jax.jit``-wrapped functions. They are cached via
  ``functools.lru_cache(maxsize=None)`` keyed on the integer pair
  ``(m, n)`` so each unique mode index recompiles exactly once.
- ``_reiman_B_kernel(k_theta_tuple, m0_symmetry)`` and
  ``_reiman_dB_kernel(k_theta_tuple, m0_symmetry)`` are similarly
  cached on the tuple of integer indices plus the toroidal-symmetry
  Python integer.
- The public ``dommaschk_B`` / ``dommaschk_dB`` entrypoints iterate
  over the ``K`` coefficient pairs on the Python side (each loop
  iteration calls one already-compiled per-mode kernel). ``reiman_B``
  / ``reiman_dB`` make a single compiled call per evaluation since
  the entire ``k_theta`` loop is unrolled inside the JIT trace.

## Static Shape Strategy

- Dommaschk ``m`` and ``n`` are Python integer tuples of length
  ``K``; each ``(m, n)`` is part of the JIT cache key. ``K`` itself
  is the Python-side loop bound.
- Reiman ``k_theta`` is a Python integer tuple of length ``M`` and
  ``m0_symmetry`` is a Python integer; both are part of the JIT
  cache key. ``M`` is the Python loop bound inside the JIT trace.
- The point axis ``N`` is dynamic JAX shape data; broadcasted
  arithmetic produces shape ``[N]`` intermediates and final
  outputs of shape ``[K, N, 3]`` / ``[K, N, 3, 3]`` for Dommaschk
  and ``[N, 3]`` / ``[N, 3, 3]`` for Reiman.

## Transform Inventory

- ``jit``: yes, one ``@jax.jit`` boundary per compiled per-mode
  kernel.
- ``vmap``: no -- the point axis is handled by broadcasting.
- ``scan`` / ``fori_loop``: no -- the ``(m, n)`` / ``(k_theta)``
  loops are unrolled in Python at trace time.
- ``checkpoint`` / ``remat``: no.
- ``shard_map``, ``pmap``, collectives: no.
- Buffer donation (``donate_argnums``): no.

## Dense Materialization And Memory

- Per-kernel-call dense temporaries are dominated by intermediate
  scalar arrays of shape ``[N]`` produced by ``jnp.power(R, ...)``
  and ``jnp.power(Z, ...)`` evaluations of each polynomial term.
  For the longest Dommaschk ``(m, n) = (5, 10)`` mode used in the
  paper fixtures the term count is roughly ``O(50)`` per derived
  quantity (D, dR D, dZ D, dRR D, dRZ D, dZZ D, and the N
  analogues) so the kernel materializes on the order of ``600``
  ``[N]``-shaped temporaries per JIT call.
- Final outputs:
  - Dommaschk ``B``: ``float64[K, N, 3]`` = ``8 * K * N * 3``
    bytes.
  - Dommaschk ``dB``: ``float64[K, N, 3, 3]`` = ``8 * K * N * 9``
    bytes.
  - Reiman ``B``: ``float64[N, 3]`` = ``8 * N * 3`` bytes.
  - Reiman ``dB``: ``float64[N, 3, 3]`` = ``8 * N * 9`` bytes.
- For the benchmark scale ``K = 4``, ``N = 200`` the largest array
  is ``8 * 4 * 200 * 9 = 57.6 kB``, well under any production
  memory budget.

## CUDA Status

CPU-only. No CUDA proof is claimed under the active ``port_closure``
scope profile. JAX 0.10.0 on CPU is the current runtime per
``state.json::jax_runtime``.
