# Item 15 — JAX Transform And Memory Strategy

## Compiled boundary

The compiled-JIT boundary for item 15 lives in the upstream JAX kernel
modules already shipped by items 11 and 12. Item 15 itself adds a thin
Python wrapper layer that:

1. Pre-stages user-facing scalar attributes (``R0``, ``B0``, ``q``,
   ``iota0``, ``iota1``, ``coeffs``, ``epsilonk``, etc.) into device
   arrays at construction time via
   :func:`simsopt.jax_core._math_utils.as_jax_float64` (which calls
   :func:`jax.device_put` under the hood).
2. Stages host-resident ``points`` (from the ``sopp.MagneticField``
   cache) into device arrays through the same strict-safe helper
   ``_points_device`` at every ``_*_impl`` callback entry.
3. Calls the appropriate JIT-compiled JAX kernel
   (``toroidal_B`` / ``toroidal_dB`` / ``toroidal_A`` / ``toroidal_dA``
   / ``toroidal_d2B`` for ``ToroidalFieldJAX``; ``poloidal_B`` /
   ``poloidal_dB`` for ``PoloidalFieldJAX``; ``mirror_B`` /
   ``mirror_dB`` for ``MirrorModelJAX``; ``dommaschk_B`` /
   ``dommaschk_dB`` for ``DommaschkJAX``; ``reiman_B`` /
   ``reiman_dB`` for ``ReimanJAX``).
4. Materialises the device output back to a host NumPy array at the
   wrapper boundary so the ``sopp.MagneticField`` cache contract
   (contiguous host buffers ``B``, ``dB``, ``A``, ...) is satisfied.

The wrapper layer itself is not jitted. The kernels it calls are
jitted at the spec layer per item 11 / item 12.

## Static argument metadata

The kernels treat the following pieces as JIT-static metadata:

- ``ToroidalFieldSpec``: the dataclass is a JAX-registered pytree with
  ``R0`` and ``B0`` as ``meta_fields`` (so a value change forces a
  recompile — acceptable here because ``R0`` / ``B0`` are not DOFs
  in the upstream contract).
- ``PoloidalFieldSpec``: same pattern with ``R0`` / ``B0`` / ``q``.
- ``MirrorModelSpec``: same pattern with ``B0`` / ``gamma`` / ``Z_m``.
- ``DommaschkSpec``: ``m`` and ``n`` are Python integer tuples
  (JIT-static); ``coeffs`` is a runtime device array. The per-mode
  polynomial term lists are computed in pure Python and cached via
  :func:`functools.lru_cache`.
- ``ReimanSpec``: ``k_theta`` and ``m0_symmetry`` are JIT-static
  integers; ``iota0``, ``iota1``, ``epsilon`` are runtime device
  arrays.

## Transform inventory

| Transform | Used by | Reason |
| --- | --- | --- |
| ``jax.jit`` | Yes (item 11 / 12 kernels) | All kernel hot paths are pre-compiled. Wrapper boundary is non-jit. |
| ``jax.vmap`` | Yes (``toroidal_*``, ``poloidal_*``, ``mirror_*``) | Per-point pointwise kernels vectorised across the ``N``-point axis. |
| ``jax.lax.scan`` | N/A: no carry state | The Dommaschk per-mode loop is unrolled in Python at trace time because ``(m, n)`` are static. Reiman is similar. |
| ``jax.lax.fori_loop`` | N/A: no traced loop | Same reason as ``scan``. |
| ``jax.checkpoint`` / ``remat`` | N/A: no autodiff hot path | Wrapper is CPU-output-buffer compatibility layer; no reverse-mode autodiff flows through it. |
| ``jax.experimental.shard_map`` | N/A: single device | Item 15 covers per-class CPU/JAX parity for analytic fields. No sharding. |
| ``jax.pmap`` / collectives | N/A | Same. |

## Static-shape strategy

- ``points``: shape ``(N, 3)``. ``N`` is not a JIT-static constant
  but the JAX kernels handle it via ``jax.vmap`` (Toroidal /
  Poloidal / Mirror) or implicit broadcasting (Dommaschk / Reiman),
  so a change in ``N`` does not force a recompile.
- Spec metadata (``m``, ``n``, ``k_theta``) is JIT-static; changing
  the mode-index tuples does force a recompile. This matches the
  upstream CPU class's expectation that mode tuples are construction
  parameters, not runtime DOFs.

## Dense materialisation budget

- ``B``, ``dB``, ``A``, ``dA``: shape ``(N, 3)`` and ``(N, 3, 3)``;
  bytes are ``N * 3 * 8`` and ``N * 9 * 8`` respectively.
- ``d2B``: shape ``(N, 3, 3, 3)``; bytes ``N * 27 * 8``.
- ``DommaschkSpec.coeffs``: shape ``(K, 2)``; ``K = len(m)``.
- Dommaschk kernel returns ``(K, N, 3)`` per-mode arrays internally;
  the wrapper sums over the ``K`` axis on the host side at the
  ``np.add.reduce`` step.

At the production fixture used in the new parity test (``N = 60``,
``K <= 3``), the largest single array materialised is the Dommaschk
``(K, N, 3, 3)`` per-mode dB tensor at ``3 * 60 * 9 * 8 = 12960`` bytes
(~13 KB). This is far below the ``max_dense_jacobian_bytes`` ceiling.

## Buffer donation

Buffer donation is not used in the wrapper layer. The wrapper's
``_*_impl`` callbacks consume the same device ``points`` array across
all hot-path getters until ``set_points_cart`` invalidates the cache,
so donating that buffer to one kernel would invalidate the others.
Buffer donation may be revisited if a future single-stage objective
adapter consumes one of these wrappers and provides a fresh
device ``points`` buffer per call.

## Sharding / collective evidence

No sharding or collective ops are introduced by item 15. A
``git diff d79a869fd..HEAD -- src/simsopt/field/magneticfieldclasses_jax.py``
returns zero hits for ``shard_map``, ``psum``, ``all_reduce``, or
``pjit``. The HLO collective inspection checkbox in section 4c is
therefore N/A for this item.

## HLO and bench evidence

- Bench artifact: ``.artifacts/jax_port_goal/bench/15.json`` records
  the validation commands and their results.
- The hot-path kernels are already covered by the item 11 / item 12
  benches and the existing benchmarking infrastructure. The wrapper
  layer adds at most one host-to-device put plus one device-to-host
  fetch per call; both are CPU-backend operations bounded by NumPy
  memcpy and well below the kernel evaluation cost on the production
  fixture sizes used here.

## CPU-only validation

CPU validation only. ``cuda_smoke`` is ``not_claimed`` per the user
directive recorded in ``.artifacts/jax_port_goal/state.json``.
