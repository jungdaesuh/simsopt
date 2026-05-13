"""Item 29 parity tests for ``simsopt.jax_core.wireframe``.

Exercises the new pure-JAX wireframe magnetic-field kernels against the
upstream C++ oracle ``simsoptpp.WireframeField`` at the
``direct_kernel`` parity-ladder lane. All tolerances are imported from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances`` -- no
``rtol`` / ``atol`` literals are inlined.

Coverage
--------

* ``test_closed_loop_B_dB_parity``: a 10-segment closed polygon in 3D
  evaluated at 64 off-axis points; bit-identity against
  ``sopp.WireframeField`` for both ``B`` and ``dB_by_dX``.
* ``test_single_segment_closed_form_parity``: a single straight segment
  along the z-axis, evaluated against a closed-form NumPy reference
  computed directly from
  ``B = mu_0 I / (4 pi rho) * (cos theta_1 - cos theta_2) * dl_hat x rho_hat``.
* ``test_multi_halfperiod_seg_signs_parity``: the C++ kernel folds in
  per-half-period symmetry weights via ``seg_signs``. This test mirrors
  that behaviour with an nfp=2 stellarator-symmetric configuration.
* ``test_combined_B_and_dB_matches_separate``: ``wireframe_B_and_dB_by_dX``
  is consistent with the standalone ``wireframe_B`` / ``wireframe_dB_by_dX``.
* ``test_wireframe_runs_under_strict_transfer_guard``: the kernels do
  not trigger implicit host transfers when fed device-resident arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.jax_core.wireframe import (
    wireframe_B,
    wireframe_B_and_dB_by_dX,
    wireframe_dB_by_dX,
)
from .jaxpr_utils import count_jaxpr_primitives


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


# ── Fixtures ─────────────────────────────────────────────────────────


def _closed_loop_nodes() -> np.ndarray:
    """10 nodes on a non-planar closed loop, single half-period."""
    rng = np.random.default_rng(2901)
    # Start from a regular 10-gon in the xy plane (R = 1.5), then perturb z
    # to break planarity so dB is fully populated (and z-derivatives non-trivial).
    angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    R = 1.5
    base = np.stack(
        (R * np.cos(angles), R * np.sin(angles), 0.1 * rng.standard_normal(10)),
        axis=1,
    )
    return np.ascontiguousarray(base, dtype=np.float64)


def _closed_loop_segments() -> np.ndarray:
    """Segments connecting nodes 0->1->...->9->0."""
    n = 10
    segs = np.zeros((n, 2), dtype=np.int32)
    segs[:, 0] = np.arange(n)
    segs[:, 1] = (np.arange(n) + 1) % n
    return segs


def _off_loop_eval_points(count: int = 64) -> np.ndarray:
    """Cartesian points well away from the closed loop's wire."""
    rng = np.random.default_rng(290)
    R_obs = rng.uniform(0.4, 1.0, size=count)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=count)
    z = rng.uniform(-0.5, 0.5, size=count)
    return np.stack((R_obs * np.cos(phi), R_obs * np.sin(phi), z), axis=1).astype(
        np.float64, copy=False
    )


# ── Closed-form NumPy reference (single segment) ─────────────────────


def _finite_wire_B_numpy_closed_form(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, current: float
) -> np.ndarray:
    """B at ``point`` from a single finite wire ``a -> b`` carrying ``current``.

    Uses the explicit ``(cos theta_1 - cos theta_2)`` form, distinct from
    the ``(|r1| + |r2|) / (...)`` arithmetic used internally by the JAX
    kernel. The two forms are mathematically identical -- agreement
    therefore certifies the algebraic identity used in the port.
    """
    mu0_over_4pi = 1e-7
    dl = b - a
    L = np.linalg.norm(dl)
    r1 = point - a
    r2 = point - b
    # Perpendicular distance from point to the infinite line carrying the
    # segment.
    proj = np.dot(r1, dl) / (L * L)
    foot = a + proj * dl
    perp = point - foot
    rho = np.linalg.norm(perp)
    if rho == 0.0:
        return np.zeros(3, dtype=np.float64)
    # cos theta_1, cos theta_2 measured along the segment direction.
    cos1 = np.dot(dl, r1) / (L * np.linalg.norm(r1))
    cos2 = np.dot(dl, r2) / (L * np.linalg.norm(r2))
    # B direction: dl x perp / (L rho) (the azimuthal direction around dl).
    azimuthal = np.cross(dl, perp) / (L * rho)
    magnitude = mu0_over_4pi * current / rho * (cos1 - cos2)
    return magnitude * azimuthal


# ── Tests ────────────────────────────────────────────────────────────


def test_closed_loop_B_dB_parity():
    """10-segment closed loop, 64 off-axis points; bit-identical to C++.

    Oracle: ``sopp.WireframeField``. Tolerance lane: ``direct_kernel``.
    """
    nodes_hp = _closed_loop_nodes()
    segments = _closed_loop_segments()
    seg_signs = [1.0]
    rng = np.random.default_rng(290109)
    currents = rng.uniform(1.0e4, 1.0e6, size=segments.shape[0]).astype(
        np.float64, copy=False
    )

    cpp = sopp.WireframeField([nodes_hp], segments, seg_signs, currents)
    points = _off_loop_eval_points(count=64)
    cpp.set_points(points)
    B_cpp = np.asarray(cpp.B(), dtype=np.float64)
    dB_cpp = np.asarray(cpp.dB_by_dX(), dtype=np.float64)

    nodes = np.stack([nodes_hp], axis=0)
    B_jax = np.asarray(
        wireframe_B(points, nodes, segments, seg_signs, currents), dtype=np.float64
    )
    dB_jax = np.asarray(
        wireframe_dB_by_dX(points, nodes, segments, seg_signs, currents),
        dtype=np.float64,
    )

    np.testing.assert_allclose(B_jax, B_cpp, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpp, rtol=_RTOL, atol=_ATOL)


def test_single_segment_closed_form_parity():
    """Single z-axis segment vs. (cos theta_1 - cos theta_2) NumPy reference.

    Cross-validates the algebraic identity used inside the JAX kernel
    against the textbook ``(cos theta_1 - cos theta_2)`` form. Uses
    ``direct_kernel`` tolerances even though the arithmetic differs,
    because the identity holds exactly in real arithmetic and only loses
    a few ulp under finite precision.
    """
    nodes_hp = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    segments = np.array([[0, 1]], dtype=np.int32)
    seg_signs = [1.0]
    current = 7.5e5
    currents = np.array([current], dtype=np.float64)

    # 50 off-axis points (rho > 0)
    rng = np.random.default_rng(2902)
    R = rng.uniform(0.2, 1.0, size=50)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=50)
    z = rng.uniform(-0.6, 0.6, size=50)
    points = np.stack((R * np.cos(phi), R * np.sin(phi), z), axis=1).astype(
        np.float64, copy=False
    )

    B_ref = np.stack(
        [
            _finite_wire_B_numpy_closed_form(
                points[i], nodes_hp[0], nodes_hp[1], current
            )
            for i in range(points.shape[0])
        ],
        axis=0,
    )

    nodes = np.stack([nodes_hp], axis=0)
    B_jax = np.asarray(
        wireframe_B(points, nodes, segments, seg_signs, currents), dtype=np.float64
    )

    np.testing.assert_allclose(B_jax, B_ref, rtol=_RTOL, atol=_ATOL)


def test_multi_halfperiod_seg_signs_parity():
    """nfp=2, stellarator-symmetric, with seg_signs=[1, -1, 1, -1].

    Exercises the per-half-period sign weighting in
    ``wireframe_segment_*_contributions``. Oracle: C++ ``WireframeField``.
    """
    nodes_hp = _closed_loop_nodes()
    n_nodes = nodes_hp.shape[0]
    # Reflect across z=0 for the second half-period.
    nodes_refl = nodes_hp.copy()
    nodes_refl[:, 1] = -nodes_refl[:, 1]
    nodes_refl[:, 2] = -nodes_refl[:, 2]
    # nfp=2 -> rotate by 2 pi/nfp = pi for the next field period; combined
    # with the reflected sub-half-period, this is four half-periods total.
    nfp = 2
    nodes_list = [nodes_hp, nodes_refl]
    seg_signs_list = [1.0, -1.0]
    for i in range(1, nfp):
        phi_rot = 2.0 * i * np.pi / nfp
        c, s = np.cos(phi_rot), np.sin(phi_rot)
        for base, sign in ((nodes_hp, 1.0), (nodes_refl, -1.0)):
            rot = np.zeros_like(base)
            rot[:, 0] = c * base[:, 0] - s * base[:, 1]
            rot[:, 1] = s * base[:, 0] + c * base[:, 1]
            rot[:, 2] = base[:, 2]
            nodes_list.append(rot)
            seg_signs_list.append(sign)

    segments = _closed_loop_segments()
    rng = np.random.default_rng(290203)
    currents = rng.uniform(1.0e4, 1.0e6, size=segments.shape[0]).astype(
        np.float64, copy=False
    )

    cpp = sopp.WireframeField(nodes_list, segments, seg_signs_list, currents)
    points = _off_loop_eval_points(count=50)
    cpp.set_points(points)
    B_cpp = np.asarray(cpp.B(), dtype=np.float64)
    dB_cpp = np.asarray(cpp.dB_by_dX(), dtype=np.float64)

    nodes = np.stack(nodes_list, axis=0)
    assert nodes.shape == (2 * nfp, n_nodes, 3)
    B_jax = np.asarray(
        wireframe_B(points, nodes, segments, seg_signs_list, currents),
        dtype=np.float64,
    )
    dB_jax = np.asarray(
        wireframe_dB_by_dX(points, nodes, segments, seg_signs_list, currents),
        dtype=np.float64,
    )

    np.testing.assert_allclose(B_jax, B_cpp, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(dB_jax, dB_cpp, rtol=_RTOL, atol=_ATOL)


def test_combined_B_and_dB_matches_separate():
    """``wireframe_B_and_dB_by_dX`` agrees with the standalone entry points.

    Validates the combined entry point that the future ``Optimizable``
    wrapper (item 30) will rely on for caching efficiency. Tolerance
    lane: ``direct_kernel``.
    """
    nodes_hp = _closed_loop_nodes()
    segments = _closed_loop_segments()
    seg_signs = [1.0]
    rng = np.random.default_rng(290301)
    currents = rng.uniform(1.0e4, 1.0e6, size=segments.shape[0]).astype(
        np.float64, copy=False
    )
    points = _off_loop_eval_points(count=33)

    nodes = np.stack([nodes_hp], axis=0)
    B_only = np.asarray(
        wireframe_B(points, nodes, segments, seg_signs, currents), dtype=np.float64
    )
    dB_only = np.asarray(
        wireframe_dB_by_dX(points, nodes, segments, seg_signs, currents),
        dtype=np.float64,
    )
    B_pair, dB_pair = wireframe_B_and_dB_by_dX(
        points, nodes, segments, seg_signs, currents
    )

    np.testing.assert_allclose(
        np.asarray(B_pair, dtype=np.float64), B_only, rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(dB_pair, dtype=np.float64), dB_only, rtol=_RTOL, atol=_ATOL
    )


def test_total_field_kernels_stream_over_segments():
    """Total B/dB kernels scan over segments instead of staging contribution cubes."""

    nodes_hp = _closed_loop_nodes()
    segments = _closed_loop_segments()
    seg_signs = jnp.asarray([1.0], dtype=jnp.float64)
    currents = jnp.asarray(np.linspace(1.0e4, 2.0e4, segments.shape[0]))
    points = jnp.asarray(_off_loop_eval_points(count=8), dtype=jnp.float64)
    nodes = jnp.asarray(np.stack([nodes_hp], axis=0), dtype=jnp.float64)
    segments_jax = jnp.asarray(segments, dtype=jnp.int32)

    kernels = (wireframe_B, wireframe_dB_by_dX, wireframe_B_and_dB_by_dX)
    for kernel in kernels:
        jaxpr = jax.make_jaxpr(kernel)(points, nodes, segments_jax, seg_signs, currents)
        assert count_jaxpr_primitives(jaxpr, "scan") == 2, kernel.__name__


def test_dB_layout_convention_via_finite_difference():
    """Confirm ``dB[p, k, m] = d_m B_k`` (component-first, derivative second).

    The C++ kernel stores ``dB_by_dX(p, k, m) = fak * dB_dX_i[k].m``;
    this property must hold in the JAX port for bit-identity to remain
    meaningful. The finite-difference probe is done at
    ``derivative_heavy`` lane tolerance so the FD noise (O(eps**2)) is
    not asked to satisfy the exact-arithmetic ``direct_kernel`` lane.
    """
    derivative_heavy = parity_ladder_tolerances("derivative_heavy")
    rtol = derivative_heavy["first_derivative_rtol"]
    atol = derivative_heavy["first_derivative_atol"]

    nodes_hp = _closed_loop_nodes()
    segments = _closed_loop_segments()
    seg_signs = [1.0]
    rng = np.random.default_rng(290501)
    currents = rng.uniform(1.0e4, 1.0e6, size=segments.shape[0]).astype(
        np.float64, copy=False
    )
    point = np.array([0.7, -0.3, 0.2], dtype=np.float64)
    nodes = np.stack([nodes_hp], axis=0)
    eps = 1.0e-5
    dB = np.asarray(
        wireframe_dB_by_dX(point[None, :], nodes, segments, seg_signs, currents),
        dtype=np.float64,
    )[0]
    # FD for d_m B (each m gives a 3-vector of d_m B_0, d_m B_1, d_m B_2).
    for m in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[m] += eps
        minus[m] -= eps
        B_plus = np.asarray(
            wireframe_B(plus[None, :], nodes, segments, seg_signs, currents),
            dtype=np.float64,
        )[0]
        B_minus = np.asarray(
            wireframe_B(minus[None, :], nodes, segments, seg_signs, currents),
            dtype=np.float64,
        )[0]
        fd = (B_plus - B_minus) / (2.0 * eps)
        # dB[:, m] should be ∂_m B; dB[m, :] (transposed convention) is not.
        np.testing.assert_allclose(dB[:, m], fd, rtol=rtol, atol=atol)


def test_wireframe_runs_under_strict_transfer_guard():
    """Kernels run cleanly under ``jax.transfer_guard("disallow")``.

    Inputs are placed on the device before entering the guard scope so
    the strict-guard region only measures the compiled kernels. Any
    implicit host transfer inside the JAX paths would raise.
    """
    nodes_hp = _closed_loop_nodes()
    segments = _closed_loop_segments()
    seg_signs_list = [1.0]
    rng = np.random.default_rng(290601)
    currents = rng.uniform(1.0e4, 1.0e6, size=segments.shape[0]).astype(
        np.float64, copy=False
    )
    points = _off_loop_eval_points(count=50)
    nodes = np.stack([nodes_hp], axis=0)

    points_dev = jnp.asarray(points, dtype=jnp.float64)
    nodes_dev = jnp.asarray(nodes, dtype=jnp.float64)
    segments_dev = jnp.asarray(segments, dtype=jnp.int32)
    seg_signs_dev = jnp.asarray(seg_signs_list, dtype=jnp.float64)
    currents_dev = jnp.asarray(currents, dtype=jnp.float64)
    for arr in (points_dev, nodes_dev, segments_dev, seg_signs_dev, currents_dev):
        arr.block_until_ready()

    with jax.transfer_guard("disallow"):
        wireframe_B(
            points_dev, nodes_dev, segments_dev, seg_signs_dev, currents_dev
        ).block_until_ready()
        wireframe_dB_by_dX(
            points_dev, nodes_dev, segments_dev, seg_signs_dev, currents_dev
        ).block_until_ready()
        B_pair, dB_pair = wireframe_B_and_dB_by_dX(
            points_dev, nodes_dev, segments_dev, seg_signs_dev, currents_dev
        )
        B_pair.block_until_ready()
        dB_pair.block_until_ready()


# ── Shape / dtype sanity ─────────────────────────────────────────────


def test_output_shapes_and_dtypes():
    """Kernel output shapes and float64 dtype contract."""
    nodes_hp = _closed_loop_nodes()
    segments = _closed_loop_segments()
    seg_signs = [1.0]
    currents = np.ones(segments.shape[0], dtype=np.float64)
    points = _off_loop_eval_points(count=12)
    nodes = np.stack([nodes_hp], axis=0)

    B = wireframe_B(points, nodes, segments, seg_signs, currents)
    dB = wireframe_dB_by_dX(points, nodes, segments, seg_signs, currents)
    assert B.shape == (12, 3)
    assert dB.shape == (12, 3, 3)
    assert B.dtype == jnp.float64
    assert dB.dtype == jnp.float64
