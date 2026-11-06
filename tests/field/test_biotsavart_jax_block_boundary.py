"""Quadrature-block and coil-chunk tiling keep the Biot-Savart kernels' results.

``_quadrature_block_sum`` sums the quadrature axis along one of three
paths: a single block, an exact two-block split, or a loop over full blocks
plus a tail. ``_coil_chunk_reduce`` has the same three paths over coils. The
loops in each kernel's trace show which path each tiling row takes. Each tiled
kernel here is compared with the untiled one (quadrature_block_size=0,
coil_chunk_size=0); only the floating-point summation order may differ.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax.extend.core import jaxprs_in_params
    from simsopt_jax.core import biotsavart as biotsavart_core


import numpy as np

_COIL_COUNT = 5
_QUADRATURE_COUNT = 24
_POINT_COUNT = 11

# (name, coil_chunk_size, quadrature_block_size, loop trip counts) for 5 coils
# of 24 nodes. One or two blocks or chunks take the single or exact two-way
# path, which has no loop; three or more loop over the full blocks or chunks
# and add the tail. coil_chunk_tail loops over two full 2-coil chunks, and the
# quadrature sum of each chunk, traced once in the loop body and once for the
# 1-coil tail, loops over three full 7-node blocks.
_TILINGS = (
    ("two_quadrature_blocks", 0, 16, ()),
    ("three_quadrature_blocks_and_tail", 0, 7, (3,)),
    ("two_coil_chunks", 3, 16, ()),
    ("coil_chunk_tail", 2, 7, (2, 3, 3)),
)

_FORWARD_KERNELS = (
    ("B", "VALUE"),
    ("B", "JACOBIAN"),
    ("A", "VALUE"),
    ("A", "JACOBIAN"),
    ("B", "HESSIAN"),
    ("A", "HESSIAN"),
    ("B", "VALUE_AND_JACOBIAN"),
)

# Measured over every row below (JAX 0.12.0.1), |tiled - untiled| is at most
# 3.7e-16 * (|untiled| + max|untiled|) on CPU and 4.8e-16 * (...) on an RTX
# 5090, about two float64 epsilons (2.2e-16). Without the max|untiled| floor the
# elementwise relative difference reaches 6.6e-14, on d2A entries that cancel
# to near zero. 1e-14 (45 epsilons) leaves room for other summation orders and
# FMA contraction; a dropped or repeated node is off by at least 1/24.
_BLOCK_PARITY_RTOL = 1e-14


def _coil_fixture():
    """Five tilted, displaced, non-planar loops and points inside them."""
    rng = np.random.default_rng(20261105)
    s = np.arange(_QUADRATURE_COUNT) / _QUADRATURE_COUNT
    phi = 2.0 * np.pi * s
    gammas = []
    gammadashs = []
    for _ in range(_COIL_COUNT):
        radius = rng.uniform(0.8, 1.2)
        center = rng.uniform(-0.1, 0.1, size=3)
        wobble = rng.uniform(-0.2, 0.2)
        gamma = np.stack(
            (
                radius * np.cos(phi),
                radius * np.sin(phi),
                wobble * np.sin(2.0 * phi),
            ),
            axis=-1,
        )
        # d/ds of gamma, with phi = 2 pi s.
        gammadash = np.stack(
            (
                -2.0 * np.pi * radius * np.sin(phi),
                2.0 * np.pi * radius * np.cos(phi),
                4.0 * np.pi * wobble * np.cos(2.0 * phi),
            ),
            axis=-1,
        )
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        gammas.append(gamma @ rotation.T + center)
        gammadashs.append(gammadash @ rotation.T)
    currents = rng.uniform(0.5, 1.5, size=_COIL_COUNT) * 1e4
    points = rng.uniform(-0.3, 0.3, size=(_POINT_COUNT, 3))
    cotangent = rng.standard_normal((_POINT_COUNT, 3))
    return (
        jnp.asarray(points),
        jnp.asarray(cotangent),
        jnp.asarray(np.stack(gammas)),
        jnp.asarray(np.stack(gammadashs)),
        jnp.asarray(currents),
    )


def _forward_kernel(integrand, diff_mode, coil_chunk_size, quadrature_block_size):
    return biotsavart_core._make_kernel(
        biotsavart_core._Integrand[integrand],
        biotsavart_core._DiffMode[diff_mode],
        coil_chunk_size,
        quadrature_block_size,
        0,
    )


def _forward(integrand, diff_mode, coil_chunk_size, quadrature_block_size, inputs):
    points, _, gammas, gammadashs, currents = inputs
    kernel = _forward_kernel(
        integrand, diff_mode, coil_chunk_size, quadrature_block_size
    )
    return jax.tree.leaves(kernel(points, gammas, gammadashs, currents))


def _loop_trip_counts(jaxpr):
    """Sorted trip counts of the scans in ``jaxpr`` and the jaxprs nested in it.

    ``lax.fori_loop`` with static bounds traces to a scan of that length.
    """
    return tuple(
        sorted(
            [eqn.params["length"] for eqn in jaxpr.eqns if eqn.primitive.name == "scan"]
            + [
                count
                for eqn in jaxpr.eqns
                for sub_jaxpr in jaxprs_in_params(eqn.params)
                for count in _loop_trip_counts(sub_jaxpr)
            ]
        )
    )


def _B_vjp(coil_chunk_size, quadrature_block_size, inputs):
    kernel = biotsavart_core._make_B_vjp_kernel(
        coil_chunk_size, quadrature_block_size, 0
    )
    return jax.tree.leaves(kernel(*inputs))


def _assert_leaves_match(tiled, untiled):
    """Elementwise rtol with an absolute floor at rtol times the largest value.

    The floor covers entries that cancel to near zero, whose relative error is
    unbounded under any change of summation order.
    """
    for tiled_leaf, untiled_leaf in zip(tiled, untiled, strict=True):
        expected = np.asarray(untiled_leaf)
        np.testing.assert_allclose(
            np.asarray(tiled_leaf),
            expected,
            rtol=_BLOCK_PARITY_RTOL,
            atol=_BLOCK_PARITY_RTOL * np.max(np.abs(expected)),
        )


class TestBiotSavartBlockBoundaryParity(JaxTestCase):
    def test_tilings_take_the_block_and_chunk_paths_they_name(self):
        """Each forward kernel, traced for each tiling row, runs the loops the row
        states, and the untiled kernel runs none."""
        rows = (("untiled", 0, 0, ()),) + _TILINGS
        for integrand, diff_mode in _FORWARD_KERNELS:
            for name, coil_chunk_size, quadrature_block_size, trip_counts in rows:
                row = {"integrand": integrand, "diff_mode": diff_mode, "tiling": name}
                with self.subTest(**row), self.case():
                    self._case_tiling_takes_the_path_it_names(
                        integrand,
                        diff_mode,
                        coil_chunk_size,
                        quadrature_block_size,
                        trip_counts,
                    )

    def _case_tiling_takes_the_path_it_names(
        self,
        integrand,
        diff_mode,
        coil_chunk_size,
        quadrature_block_size,
        trip_counts,
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        points, _, gammas, gammadashs, currents = _coil_fixture()
        kernel = _forward_kernel(
            integrand, diff_mode, coil_chunk_size, quadrature_block_size
        )
        traced = jax.make_jaxpr(kernel)(points, gammas, gammadashs, currents)
        self.assertEqual(_loop_trip_counts(traced.jaxpr), trip_counts)

    def test_forward_kernels_match_untiled_across_block_boundaries(self):
        """B, A, their point Jacobians and Hessians and the fused B and dB/dX
        kernel match the untiled kernel for two blocks, three blocks plus a tail
        and coil chunk tails."""
        for integrand, diff_mode in _FORWARD_KERNELS:
            for name, coil_chunk_size, quadrature_block_size, _ in _TILINGS:
                row = {"integrand": integrand, "diff_mode": diff_mode, "tiling": name}
                with self.subTest(**row), self.case():
                    self._case_forward_kernel_matches_untiled(
                        integrand, diff_mode, coil_chunk_size, quadrature_block_size
                    )

    def _case_forward_kernel_matches_untiled(
        self, integrand, diff_mode, coil_chunk_size, quadrature_block_size
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        inputs = _coil_fixture()
        _assert_leaves_match(
            _forward(
                integrand, diff_mode, coil_chunk_size, quadrature_block_size, inputs
            ),
            _forward(integrand, diff_mode, 0, 0, inputs),
        )

    def test_B_vjp_matches_untiled_across_block_boundaries(self):
        """The B cotangents for positions, tangents and currents match the
        untiled VJP kernel for every tiling row."""
        for name, coil_chunk_size, quadrature_block_size, _ in _TILINGS:
            with self.subTest(tiling=name), self.case():
                self._case_B_vjp_matches_untiled(coil_chunk_size, quadrature_block_size)

    def _case_B_vjp_matches_untiled(self, coil_chunk_size, quadrature_block_size):
        """Run one row with case-local objects released before runtime cleanup."""
        inputs = _coil_fixture()
        _assert_leaves_match(
            _B_vjp(coil_chunk_size, quadrature_block_size, inputs),
            _B_vjp(0, 0, inputs),
        )
