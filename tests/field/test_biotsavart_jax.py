"""
Parity tests for the JAX Biot-Savart implementation.

Validates against:
1. Analytical on-axis field of a circular current loop.
2. Maxwell's equation ∇·B = 0 (trace of dB/dX).
3. C++ reference (when simsoptpp is available).
"""

import inspect
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import types

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path(__file__).resolve().parents[2] / "src")

from conftest import parity_acceptance_modes, parity_mode_case
from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt_jax.backend import invalidate_backend_cache
from simsopt_jax.core.field import (
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_lists,
)
from simsopt_jax.core.specs import CoilGroupSpec, GroupedCoilSetSpec
from simsopt_jax.core import sharding as sharding_core

import simsopt_jax.field.biotsavart as _bs


def _load_with_backend_mode(_mode: str):
    return _bs


def _load_chunked_biotsavart():
    return _load_with_backend_mode("jax_cpu_parity")


@contextmanager
def _kernel_tuning_env(
    mode: str,
    *,
    coil_chunk_size: int | None = None,
    quadrature_block_size: int | None = None,
    point_chunk_size: int | None = None,
):
    previous = {name: os.environ.get(name) for name in _KERNEL_TUNING_ENV_VARS}
    os.environ["SIMSOPT_BACKEND_MODE"] = mode
    if coil_chunk_size is None:
        os.environ.pop("SIMSOPT_JAX_COIL_CHUNK_SIZE", None)
    else:
        os.environ["SIMSOPT_JAX_COIL_CHUNK_SIZE"] = str(coil_chunk_size)
    if quadrature_block_size is None:
        os.environ.pop("SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE", None)
    else:
        os.environ["SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE"] = str(quadrature_block_size)
    if point_chunk_size is None:
        os.environ.pop("SIMSOPT_JAX_POINT_CHUNK_SIZE", None)
    else:
        os.environ["SIMSOPT_JAX_POINT_CHUNK_SIZE"] = str(point_chunk_size)
    invalidate_backend_cache()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        invalidate_backend_cache()


biot_savart_B = _bs.biot_savart_B
biot_savart_dB_by_dX = _bs.biot_savart_dB_by_dX
biot_savart_B_and_dB = _bs.biot_savart_B_and_dB
biot_savart_A = _bs.biot_savart_A
biot_savart_dA_by_dX = _bs.biot_savart_dA_by_dX
grouped_biot_savart_A = _bs.grouped_biot_savart_A
grouped_biot_savart_B = _bs.grouped_biot_savart_B

MU0 = 4.0 * np.pi * 1e-7
_DIRECT_KERNEL_TOLS = parity_ladder_tolerances("direct-kernel")
_DERIVATIVE_HEAVY_TOLS = parity_ladder_tolerances("derivative-heavy")
_KERNEL_TUNING_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
    "SIMSOPT_JAX_POINT_CHUNK_SIZE",
)
_BIOTSAVART_CHUNKED_DENSE_PARITY_MODES = parity_acceptance_modes(
    "biotsavart_chunked_dense",
    "jax_cpu_parity",
    "jax_gpu_parity",
)
_BIOTSAVART_ACCUMULATION_ORDER_PARITY_MODES = parity_acceptance_modes(
    "biotsavart_accumulation_order",
    "jax_cpu_parity",
    "jax_gpu_parity",
)


def _make_circular_coil(R=1.0, nquad=128):
    """Create a single circular coil of radius R centred at the origin in the xy-plane."""
    phi = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
    gamma = np.stack([R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)], axis=-1)
    # dγ/dφ where φ ∈ [0,1) → chain rule factor 2π already present
    # but simsopt parameterises φ ∈ [0,1), so gammadash = dγ/d(φ_01) = 2π·dγ/dφ_rad
    # Actually the quadrature spacing is 1/nquad, so:
    # gammadash = dγ/dφ_01 = dγ/d(φ_rad) * d(φ_rad)/d(φ_01)
    #           = dγ/d(φ_rad) * 2π
    gammadash = np.stack(
        [
            -R * np.sin(phi) * 2 * np.pi,
            R * np.cos(phi) * 2 * np.pi,
            np.zeros_like(phi),
        ],
        axis=-1,
    )
    return (
        jnp.array(gamma[None, :, :]),  # (1, nquad, 3)
        jnp.array(gammadash[None, :, :]),  # (1, nquad, 3)
    )


def _make_shifted_circular_coils(ncoils: int, *, R: float = 1.0, nquad: int = 128):
    gamma, gammadash = _make_circular_coil(R=R, nquad=nquad)
    z_offsets = jnp.linspace(-0.4, 0.4, ncoils, dtype=jnp.float64)
    gamma_stack = jnp.concatenate(
        [gamma + jnp.array([[[0.0, 0.0, offset]]]) for offset in z_offsets],
        axis=0,
    )
    gammadash_stack = jnp.concatenate([gammadash] * ncoils, axis=0)
    currents = jnp.linspace(5e4, 5e4 + 1e3 * (ncoils - 1), ncoils, dtype=jnp.float64)
    return gamma_stack, gammadash_stack, currents


def _make_random_fixture(
    *,
    seed: int,
    ncoils: int = 33,
    nquad: int = 130,
    npoints: int = 17,
):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(npoints, 3))
    points[:, 0] -= 2.0
    gammas = rng.normal(size=(ncoils, nquad, 3))
    gammas[:, :, 0] += 1.5
    gammadashs = rng.normal(size=(ncoils, nquad, 3))
    currents = rng.normal(loc=1.0e5, scale=2.0e4, size=(ncoils,))
    return (
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(gammas, dtype=jnp.float64),
        jnp.asarray(gammadashs, dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
    )


def _host_array(value):
    return np.asarray(jax.device_get(jax.block_until_ready(value)))


def _make_accumulation_order_fixture(
    *,
    seed: int,
    ncoils: int = 53,
    nquad: int = 193,
    npoints: int = 41,
):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, nquad, endpoint=False)
    twopi_t = 2.0 * np.pi * t

    gammas = np.empty((ncoils, nquad, 3), dtype=np.float64)
    gammadashs = np.empty_like(gammas)
    currents = np.empty((ncoils,), dtype=np.float64)

    for coil_index in range(ncoils):
        phase = 0.031 * coil_index
        theta = twopi_t + phase
        dtheta_dt = 2.0 * np.pi
        radial_mode = 5.0 * twopi_t + 0.37 * coil_index
        vertical_mode = 3.0 * twopi_t + 0.19 * coil_index

        base_radius = 0.72 + 0.018 * ((coil_index % 7) - 3)
        radius = base_radius + 8.0e-4 * np.cos(radial_mode)
        z = 4.0e-3 * (coil_index - 0.5 * (ncoils - 1)) + 6.0e-4 * np.sin(vertical_mode)

        d_radius_dt = -8.0e-4 * (2.0 * np.pi * 5.0) * np.sin(radial_mode)
        dz_dt = 6.0e-4 * (2.0 * np.pi * 3.0) * np.cos(vertical_mode)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        gammas[coil_index, :, 0] = radius * cos_theta
        gammas[coil_index, :, 1] = radius * sin_theta
        gammas[coil_index, :, 2] = z

        gammadashs[coil_index, :, 0] = (
            d_radius_dt * cos_theta - radius * sin_theta * dtheta_dt
        )
        gammadashs[coil_index, :, 1] = (
            d_radius_dt * sin_theta + radius * cos_theta * dtheta_dt
        )
        gammadashs[coil_index, :, 2] = dz_dt
        currents[coil_index] = ((-1.0) ** coil_index) * (5.0e4 + 750.0 * coil_index)

    point_radius = 0.34 + 0.08 * rng.random(npoints)
    point_phi = 2.0 * np.pi * rng.random(npoints)
    points = np.stack(
        (
            point_radius * np.cos(point_phi),
            point_radius * np.sin(point_phi),
            0.03 * (rng.random(npoints) - 0.5),
        ),
        axis=-1,
    )
    return (
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(gammas, dtype=jnp.float64),
        jnp.asarray(gammadashs, dtype=jnp.float64),
        jnp.asarray(currents, dtype=jnp.float64),
    )


def _dense_reference_fields(module, points, gammas, gammadashs, currents):
    """Evaluate B, A, dB/dX, dA/dX through the same JAX kernel without chunking.

    This is a chunked-vs-dense self-consistency helper, NOT a C++ parity
    oracle: the "dense" path runs the exact same JAX integrand
    (``module._one_point_dense`` with ``module._biot_savart_B_integrand``
    /``module._biot_savart_A_integrand``) under ``jax.vmap`` and
    ``jax.jacfwd``. Comparing chunked output against this reference
    verifies that chunking does not perturb the reduction, not that the
    JAX implementation matches the C++ ``simsoptpp.BiotSavart`` symbol.
    Direct C++ parity assertions live in ``TestBiotSavartJaxCppParity``.
    """

    def _dense_B(x):
        return module._one_point_dense(
            x,
            gammas,
            gammadashs,
            currents,
            integrand=module._biot_savart_B_integrand,
        )

    def _dense_A(x):
        return module._one_point_dense(
            x,
            gammas,
            gammadashs,
            currents,
            integrand=module._biot_savart_A_integrand,
        )

    dense_B = _dense_B_reference(module, points, gammas, gammadashs, currents)
    dense_A = jax.vmap(_dense_A)(points)
    dense_dB = jax.vmap(lambda x: jnp.swapaxes(jax.jacfwd(_dense_B)(x), -1, -2))(points)
    dense_dA = jax.vmap(lambda x: jnp.swapaxes(jax.jacfwd(_dense_A)(x), -1, -2))(points)
    return dense_B, dense_A, dense_dB, dense_dA


def _dense_B_reference(module, points, gammas, gammadashs, currents):
    """Run the same JAX B-integrand through ``jax.vmap`` without chunking.

    Self-consistency helper (see ``_dense_reference_fields``). Not a C++
    parity oracle.
    """
    return jax.vmap(
        lambda x: module._one_point_dense(
            x,
            gammas,
            gammadashs,
            currents,
            integrand=module._biot_savart_B_integrand,
        )
    )(points)


def _dense_B_vjp(module, points, v, gammas, gammadashs, currents):
    """VJP through the same JAX kernel without chunking, via ``jax.vjp``.

    Self-consistency helper for chunking probes: returns ``pullback(v)``
    where the forward pass is the dense (non-chunked) JAX integrand.
    Not a C++ parity oracle for ``BiotSavart.B_vjp``; the direct
    ``BiotSavart.B_vjp`` parity assertion lives in
    ``TestBiotSavartJaxCppParity``.
    """

    def _dense_B(group_gammas, group_gammadashs, group_currents):
        return jax.vmap(
            lambda x: module._one_point_dense(
                x,
                group_gammas,
                group_gammadashs,
                group_currents,
                integrand=module._biot_savart_B_integrand,
            )
        )(points)

    _, pullback = jax.vjp(_dense_B, gammas, gammadashs, currents)
    return pullback(v)


def _evaluate_field_family(module, points, gammas, gammadashs, currents):
    B = module.biot_savart_B(points, gammas, gammadashs, currents)
    A = module.biot_savart_A(points, gammas, gammadashs, currents)
    dB = module.biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
    dA = module.biot_savart_dA_by_dX(points, gammas, gammadashs, currents)
    B_combo, dB_combo = module.biot_savart_B_and_dB(
        points,
        gammas,
        gammadashs,
        currents,
    )
    return B, A, dB, dA, B_combo, dB_combo


def _ncsx_biotsavart_parity_fixture():
    from simsopt.configs import get_data
    from simsopt.field import BiotSavart, coils_via_symmetries

    curves, currents_objs, _, nfp, _ = get_data("ncsx")
    coils = coils_via_symmetries(curves, currents_objs, nfp, stellsym=True)
    bs = BiotSavart(coils)

    npoints = 50
    rng = np.random.default_rng(42)
    points_np = rng.standard_normal((npoints, 3)) * 0.3
    points_np[:, 0] += 1.0  # shift near torus

    bs.set_points(points_np)
    gammas_np = np.array([coil.curve.gamma() for coil in coils])
    gds_np = np.array([coil.curve.gammadash() for coil in coils])
    currents_np = np.array([coil.current.get_value() for coil in coils])
    return bs, points_np, gammas_np, gds_np, currents_np


def _cart_points_to_cyl(points):
    return np.ascontiguousarray(
        np.stack(
            (
                np.sqrt(points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]),
                np.arctan2(points[:, 1], points[:, 0]),
                points[:, 2],
            ),
            axis=1,
        )
    )


def _assert_cylindrical_points_match_cpu(jax_field, cpu_field):
    np.testing.assert_allclose(
        np.asarray(jax_field.get_points_cyl()),
        np.asarray(cpu_field.get_points_cyl()),
        rtol=0.0,
        atol=1.0e-15,
    )


def _assert_cylindrical_accessors_match_cpu(jax_field, cpu_field):
    _assert_cylindrical_points_match_cpu(jax_field, cpu_field)
    np.testing.assert_allclose(
        np.asarray(jax_field.AbsB()),
        np.asarray(cpu_field.AbsB()),
        rtol=_DIRECT_KERNEL_TOLS["rtol"],
        atol=_DIRECT_KERNEL_TOLS["atol"],
    )
    assert np.asarray(jax_field.AbsB()).shape == np.asarray(cpu_field.AbsB()).shape
    for method_name in ("B_cyl", "A_cyl"):
        np.testing.assert_allclose(
            np.asarray(getattr(jax_field, method_name)()),
            np.asarray(getattr(cpu_field, method_name)()),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
    np.testing.assert_allclose(
        np.asarray(jax_field.GradAbsB_cyl()),
        np.asarray(cpu_field.GradAbsB_cyl()),
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )


class TestBiotSavartJaxAnalytical:
    """Test against the known on-axis field of a circular current loop."""

    def test_on_axis_field(self):
        """B_z at the centre of a circular loop = μ₀ I / (2R)."""
        R = 1.0
        I = 1e6  # 1 MA
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        points = jnp.array([[0.0, 0.0, 0.0]])
        B = biot_savart_B(points, gammas, gammadashs, currents)

        B_analytical = MU0 * I / (2.0 * R)
        analytical_rel_tol = 1e-12
        symmetry_abs_tol = 1e-14

        np.testing.assert_allclose(
            float(B[0, 2]),
            B_analytical,
            rtol=analytical_rel_tol,
        )
        # Bx and By should be zero by symmetry
        np.testing.assert_allclose(float(B[0, 0]), 0.0, atol=symmetry_abs_tol)
        np.testing.assert_allclose(float(B[0, 1]), 0.0, atol=symmetry_abs_tol)

    def test_on_axis_field_offset_z(self):
        """B_z at z=h on axis: B_z = μ₀IR²/(2(R²+h²)^{3/2})."""
        R = 1.0
        I = 1e6
        h = 0.5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        points = jnp.array([[0.0, 0.0, h]])
        B = biot_savart_B(points, gammas, gammadashs, currents)

        B_analytical = MU0 * I * R**2 / (2.0 * (R**2 + h**2) ** 1.5)
        analytical_rel_tol = 1e-12
        np.testing.assert_allclose(
            float(B[0, 2]),
            B_analytical,
            rtol=analytical_rel_tol,
        )

    def test_div_B_zero(self):
        """∇·B = Tr(dB/dX) should be zero (Maxwell)."""
        R = 1.0
        I = 1e5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        # Off-axis points
        points = jnp.array(
            [
                [0.3, 0.0, 0.0],
                [0.0, 0.3, 0.1],
                [0.5, 0.5, 0.2],
            ]
        )
        dB = biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
        div_B = jnp.trace(dB, axis1=1, axis2=2)  # (npoints,)
        divergence_abs_tol = 1e-14
        np.testing.assert_allclose(np.array(div_B), 0.0, atol=divergence_abs_tol)

    def test_B_and_dB_consistency(self):
        """Tier-4 self-consistency: fused B/dB matches separate JAX calls."""
        R = 1.0
        I = 1e5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=128)
        currents = jnp.array([I])

        points = jnp.array(
            [
                [0.3, 0.1, 0.0],
                [0.0, 0.5, 0.2],
            ]
        )

        B_ref = biot_savart_B(points, gammas, gammadashs, currents)
        dB_ref = biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
        B_combo, dB_combo = biot_savart_B_and_dB(points, gammas, gammadashs, currents)

        np.testing.assert_allclose(np.array(B_combo), np.array(B_ref), atol=1e-14)
        np.testing.assert_allclose(np.array(dB_combo), np.array(dB_ref), atol=1e-14)

    def test_dB_dX_finite_difference(self):
        """dB/dX matches centred finite differences (SIMSOPT convention)."""
        R = 1.0
        I = 1e5
        gammas, gammadashs = _make_circular_coil(R=R, nquad=256)
        currents = jnp.array([I])

        x0 = jnp.array([[0.4, 0.1, 0.05]])
        # SIMSOPT convention: dB_dX[j, l] = ∂_j B_l
        dB_jax = biot_savart_dB_by_dX(x0, gammas, gammadashs, currents)[0]

        eps = 1e-5
        dB_fd = np.zeros((3, 3))
        for j in range(3):
            xp = x0.at[0, j].add(eps)
            xm = x0.at[0, j].add(-eps)
            Bp = biot_savart_B(xp, gammas, gammadashs, currents)[0]
            Bm = biot_savart_B(xm, gammas, gammadashs, currents)[0]
            # Row j = all B components differentiated w.r.t. x_j
            dB_fd[j, :] = (np.array(Bp) - np.array(Bm)) / (2 * eps)

        fd_rel_tol = 1e-8
        fd_abs_tol = 5e-11
        np.testing.assert_allclose(
            np.array(dB_jax),
            dB_fd,
            rtol=fd_rel_tol,
            atol=fd_abs_tol,
        )

    def test_multiple_coils(self):
        """Superposition: field of two coils equals sum of individual fields."""
        R = 1.0
        gammas1, gammadashs1 = _make_circular_coil(R=R, nquad=128)
        gammas2 = gammas1 + jnp.array([[[0.0, 0.0, 0.5]]])
        gammadashs2 = gammadashs1.copy()

        currents = jnp.array([1e5, -5e4])
        gammas = jnp.concatenate([gammas1, gammas2], axis=0)
        gammadashs = jnp.concatenate([gammadashs1, gammadashs2], axis=0)

        points = jnp.array([[0.0, 0.0, 0.25]])

        B_total = biot_savart_B(points, gammas, gammadashs, currents)
        B1 = biot_savart_B(points, gammas1, gammadashs1, jnp.array([currents[0]]))
        B2 = biot_savart_B(points, gammas2, gammadashs2, jnp.array([currents[1]]))

        np.testing.assert_allclose(np.array(B_total), np.array(B1 + B2), atol=1e-14)

    def test_point_on_coil_field_surfaces_singularity(self):
        """Point-on-coil inputs expose the Biot-Savart singularity."""
        gammas, gammadashs = _make_circular_coil(R=1.0, nquad=64)
        currents = jnp.array([1e5])
        points = gammas[0, :1, :]

        B = biot_savart_B(points, gammas, gammadashs, currents)

        assert not bool(np.all(np.asarray(jnp.isfinite(B))))

    def test_grouped_biot_savart_A_host_helper_matches_dense_kernel(self):
        """Direct host-helper pin for grouped vector potential accumulation."""
        points = jnp.array(
            [
                [0.2, 0.1, -0.3],
                [-0.1, 0.35, 0.2],
                [0.3, -0.25, 0.15],
            ],
            dtype=jnp.float64,
        )
        gammas, gammadashs, currents = _make_shifted_circular_coils(4, nquad=32)
        coil_arrays = (
            (gammas[:1], gammadashs[:1], currents[:1]),
            (gammas[1:3], gammadashs[1:3], currents[1:3]),
            (gammas[3:], gammadashs[3:], currents[3:]),
        )

        grouped_A = grouped_biot_savart_A(points, coil_arrays)
        dense_A = biot_savart_A(points, gammas, gammadashs, currents)

        np.testing.assert_allclose(
            np.asarray(grouped_A), np.asarray(dense_A), atol=1e-14
        )

    def test_grouped_biot_savart_B_jit_handles_mixed_quadrature_groups(self):
        """Mixed-quadrature grouped field keeps group count static under JIT."""
        points = jnp.array(
            [
                [0.2, 0.1, -0.3],
                [-0.1, 0.35, 0.2],
                [0.3, -0.25, 0.15],
            ],
            dtype=jnp.float64,
        )
        gammas_16, gammadashs_16, currents_16 = _make_shifted_circular_coils(
            2,
            nquad=16,
        )
        gammas_32, gammadashs_32, currents_32 = _make_shifted_circular_coils(
            1,
            R=1.2,
            nquad=32,
        )
        coil_arrays = (
            (gammas_16, gammadashs_16, currents_16),
            (gammas_32, gammadashs_32, currents_32),
        )

        grouped_B = grouped_biot_savart_B(points, coil_arrays)
        expected_B = biot_savart_B(
            points,
            gammas_16,
            gammadashs_16,
            currents_16,
        ) + biot_savart_B(
            points,
            gammas_32,
            gammadashs_32,
            currents_32,
        )
        outer_jit_B = jax.jit(
            lambda eval_points, groups: grouped_biot_savart_B(
                eval_points,
                groups,
            )
        )(points, coil_arrays)

        np.testing.assert_allclose(
            np.asarray(grouped_B),
            np.asarray(expected_B),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(outer_jit_B),
            np.asarray(expected_B),
            atol=1e-14,
        )


class TestBiotSavartJaxCppParity:
    """Compare against the C++ simsoptpp kernel (skipped if unavailable)."""

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        sopp = pytest.importorskip("simsoptpp")
        if not hasattr(sopp, "BiotSavart"):
            pytest.skip("simsoptpp compiled extensions not available")
        pytest.importorskip("simsopt")

    def test_B_parity_ncsx(self):
        """``biot_savart_B`` matches ``BiotSavart.B()`` on the NCSX fixture.

        Oracle: C++ reference symbol ``simsoptpp::biot_savart_B`` accessed
        through ``simsopt.field.biotsavart.BiotSavart.B`` (acceptable
        oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``). Lane:
        ``direct-kernel`` value tolerances from the validation-ladder
        SSOT (``benchmarks/validation_ladder_contract.py::
        PARITY_LADDER_TOLERANCES``).
        """
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        B_ref = bs.B()

        B_jax = biot_savart_B(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(B_jax),
            B_ref,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_A_parity_ncsx(self):
        """``biot_savart_A`` matches ``BiotSavart.A()`` on the NCSX fixture.

        Oracle: C++ reference symbol ``simsoptpp::BiotSavart::A`` accessed
        through ``simsopt.field.biotsavart.BiotSavart.A`` (acceptable
        oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``). Lane:
        ``direct-kernel`` value tolerances from the validation-ladder SSOT.
        """
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        A_ref = bs.A()

        A_jax = biot_savart_A(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(A_jax),
            A_ref,
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_dB_by_dX_parity_ncsx(self):
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        dB_ref = bs.dB_by_dX()

        dB_jax = biot_savart_dB_by_dX(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(dB_jax),
            dB_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_dA_by_dX_kernel_parity_ncsx(self):
        """``biot_savart_dA_by_dX`` matches ``BiotSavart.dA_by_dX()``."""
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        dA_ref = bs.dA_by_dX()

        dA_jax = biot_savart_dA_by_dX(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(
            np.array(dA_jax),
            dA_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_cylindrical_public_accessors_parity_ncsx(self):
        """``BiotSavartJAX`` owns cylindrical public accessors on its boundary."""
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = _cart_points_to_cyl(points_np)
        points_cyl[:, 1] += 2.0 * np.pi
        bs.set_points_cyl(points_cyl)

        bs_jax = BiotSavartJAX(list(bs._coils))
        assert bs_jax.set_points_cyl(points_cyl) is bs_jax
        assert bs_jax.set_points_cart(points_np) is bs_jax
        assert bs_jax.set_points_cyl(points_cyl) is bs_jax

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cylindrical_public_accessors_use_cached_phi_basis_ncsx(self):
        """``B_cyl`` / ``A_cyl`` / ``GradAbsB_cyl`` use cached cylindrical phi."""
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = np.ascontiguousarray(
            np.array(
                [
                    [0.0, 1.234, 0.1],
                    [0.0, -1.5, -0.2],
                    [0.7, 2.0 * np.pi + 0.4, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cyl(points_cyl)

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points_cyl(points_cyl)

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cartesian_public_accessors_normalize_cylindrical_phi_ncsx(self):
        """Cartesian point conversion follows C++ ``get_points_cyl_impl``."""
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cart = np.ascontiguousarray(
            np.array(
                [
                    [0.5, -0.5, 0.1],
                    [-0.5, -0.25, -0.2],
                    [0.0, 0.0, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cart(points_cart)

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points_cart(points_cart)

        _assert_cylindrical_points_match_cpu(bs_jax, bs)

    def test_spec_backed_cylindrical_public_accessors_parity_ncsx(self):
        """``SpecBackedBiotSavartJAX`` exposes the same cylindrical contract."""
        from simsopt_jax_adapters.field.biotsavart_backend import (
            BiotSavartJAX,
            SpecBackedBiotSavartJAX,
        )
        from simsopt_jax.core.specs import make_biot_savart_spec

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = np.ascontiguousarray(
            np.array(
                [
                    [0.0, 1.234, 0.1],
                    [0.5, 2.0 * np.pi + 0.4, -0.2],
                    [0.7, -1.2, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cyl(points_cyl)

        bs_jax = BiotSavartJAX(list(bs._coils))
        spec = make_biot_savart_spec(
            coil_dof_extraction=bs_jax.coil_dof_extraction_spec(),
            coil_dofs=np.asarray(bs_jax.x, dtype=np.float64),
        )
        spec_backed = SpecBackedBiotSavartJAX(spec)
        assert spec_backed.set_points_cyl(points_cyl) is spec_backed

        _assert_cylindrical_accessors_match_cpu(spec_backed, bs)

    def test_B_vjp_parity_ncsx(self):
        """``BiotSavartJAX.B_vjp(v)`` matches ``BiotSavart.B_vjp(v)`` per coil.

        Oracle: C++ reference symbol ``simsoptpp.biot_savart_vjp_graph``
        invoked through ``simsopt.field.biotsavart.BiotSavart.B_vjp``
        (acceptable oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``).
        Both ``Derivative`` objects are evaluated against each coil to
        compare the per-coil cotangent contributions on identical
        coils/points/cotangent. Lane: ``derivative_heavy`` first-derivative
        tolerances from the validation-ladder SSOT.
        """
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs_cpu, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        coils = list(bs_cpu._coils)

        bs_jax = BiotSavartJAX(coils)
        bs_jax.set_points(points_np)

        v = np.asarray(bs_cpu.B(), dtype=np.float64).copy()
        deriv_cpu = bs_cpu.B_vjp(v)
        deriv_jax = bs_jax.B_vjp(v)

        for coil in coils:
            np.testing.assert_allclose(
                np.asarray(deriv_jax(coil)),
                np.asarray(deriv_cpu(coil)),
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
                err_msg=(
                    "BiotSavartJAX.B_vjp() does not match BiotSavart.B_vjp() "
                    "on the NCSX parity fixture"
                ),
            )

    def test_dA_by_dX_parity_ncsx(self):
        """``BiotSavartJAX.dA_by_dX()`` matches ``BiotSavart.dA_by_dX()``.

        Oracle: C++ reference symbol ``simsoptpp::BiotSavart::dA_by_dX``
        accessed through ``simsopt.field.biotsavart.BiotSavart.dA_by_dX``
        (acceptable oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``).
        Lane: ``derivative-heavy`` first-derivative tolerances from the
        validation-ladder SSOT.
        """
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        dA_ref = bs.dA_by_dX()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        dA_jax = bs_jax.dA_by_dX()

        np.testing.assert_allclose(
            np.array(dA_jax),
            dA_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d2B_by_dXdX_parity_ncsx(self):
        """``BiotSavartJAX.d2B_by_dXdX()`` matches ``BiotSavart.d2B_by_dXdX()``.

        Oracle: C++ reference symbol ``simsoptpp::BiotSavart::d2B_by_dXdX``
        accessed through ``simsopt.field.biotsavart.BiotSavart.d2B_by_dXdX``
        (acceptable oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``).
        Lane: ``derivative-heavy`` second-derivative tolerances from the
        validation-ladder SSOT.
        """
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        d2B_ref = bs.d2B_by_dXdX()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        d2B_jax = bs_jax.d2B_by_dXdX()

        np.testing.assert_allclose(
            np.array(d2B_jax),
            d2B_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    def test_d2A_by_dXdX_parity_ncsx(self):
        """``BiotSavartJAX.d2A_by_dXdX()`` matches ``BiotSavart.d2A_by_dXdX()``.

        Oracle: C++ reference symbol ``simsoptpp::BiotSavart::d2A_by_dXdX``
        accessed through ``simsopt.field.biotsavart.BiotSavart.d2A_by_dXdX``
        (acceptable oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``).
        Lane: ``derivative-heavy`` second-derivative tolerances from the
        validation-ladder SSOT.
        """
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        d2A_ref = bs.d2A_by_dXdX()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        d2A_jax = bs_jax.d2A_by_dXdX()

        np.testing.assert_allclose(
            np.array(d2A_jax),
            d2A_ref,
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    def test_d2B_contracted_helper_matches_dense_hessian_contraction(self, monkeypatch):
        """Private d2B contraction matches the C++ dense-Hessian oracle.

        Oracle: C++ reference symbol ``simsoptpp::BiotSavart::d2B_by_dXdX``
        accessed through ``simsopt.field.biotsavart.BiotSavart.d2B_by_dXdX``
        (acceptable oracle type 1, see ``tests/REVIEWER_ORACLE_LINT.md``).
        Lane: ``derivative-heavy`` second-derivative tolerances from the
        validation-ladder SSOT.
        """
        from simsopt_jax.core import biotsavart as core_bs

        bs, points_np, gammas_np, gammadashs_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        points_np = points_np[:3]
        bs.set_points(points_np)
        cxx_d2B = bs.d2B_by_dXdX()
        points = jnp.asarray(points_np, dtype=jnp.float64)
        gammas = jnp.asarray(gammas_np, dtype=jnp.float64)
        gammadashs = jnp.asarray(gammadashs_np, dtype=jnp.float64)
        currents = jnp.asarray(currents_np, dtype=jnp.float64)
        left_directions = jnp.asarray(
            (
                ((0.8, -0.1, 0.3), (-0.2, 0.5, 0.4)),
                ((0.1, 0.9, -0.4), (0.6, -0.3, 0.2)),
                ((-0.5, 0.7, 0.1), (0.3, 0.2, -0.8)),
            ),
            dtype=jnp.float64,
        )
        right_directions = jnp.asarray(
            (
                ((-0.3, 0.4, 0.7), (0.5, 0.1, -0.6)),
                ((0.2, -0.8, 0.3), (-0.4, 0.6, 0.5)),
                ((0.7, 0.2, -0.1), (-0.6, -0.2, 0.4)),
            ),
            dtype=jnp.float64,
        )

        for tuning in ((0, 0, 0), (3, 5, 2)):
            monkeypatch.setattr(core_bs, "_read_tuning_config", lambda: tuning)
            core_bs.invalidate_kernel_cache()
            expected = jnp.einsum(
                "pjkl,paj,pbk->pabl",
                jnp.asarray(cxx_d2B, dtype=jnp.float64),
                left_directions,
                right_directions,
                precision=jax.lax.Precision.HIGHEST,
            )
            actual = core_bs._biot_savart_d2B_by_dXdX_contract(
                points,
                gammas,
                gammadashs,
                currents,
                left_directions,
                right_directions,
            )
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
            )

        assert core_bs._make_d2B_contracted_kernel.cache_info().currsize > 0
        core_bs.invalidate_kernel_cache()
        assert core_bs._make_d2B_contracted_kernel.cache_info().currsize == 0


class TestBiotSavartJaxCppCoilCurrentParity:
    """Compare JAX coil-current ladder against the C++ simsoptpp lists.

    Oracle: C++ reference symbols
    ``simsoptpp::BiotSavart::{dB,dA}_by_dcoilcurrents``,
    ``simsoptpp::BiotSavart::{d2B,d2A}_by_dXdcoilcurrents``,
    ``simsoptpp::BiotSavart::{d3B,d3A}_by_dXdXdcoilcurrents`` accessed
    through the matching ``simsopt.field.biotsavart.BiotSavart`` Python
    methods (acceptable oracle type 1, see
    ``tests/REVIEWER_ORACLE_LINT.md``). Each test compares the JAX list
    against the C++ list element-by-element on the NCSX parity fixture,
    using tolerances from ``benchmarks/validation_ladder_contract.py::
    PARITY_LADDER_TOLERANCES``.
    """

    @pytest.fixture(autouse=True)
    def _require_simsoptpp(self):
        sopp = pytest.importorskip("simsoptpp")
        if not hasattr(sopp, "BiotSavart"):
            pytest.skip("simsoptpp compiled extensions not available")
        pytest.importorskip("simsopt")

    @staticmethod
    def _assert_coil_current_list_parity(cache_method, list_method, *, rtol, atol):
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        # Populate the matching C++ fieldcache entries before pulling the
        # per-coil list, so ordering is deterministic.
        getattr(bs, cache_method)()
        cpu_list = getattr(bs, list_method)()

        bs_jax = BiotSavartJAX(list(bs._coils))
        bs_jax.set_points(points_np)
        jax_list = getattr(bs_jax, list_method)()

        assert len(jax_list) == len(cpu_list)
        for k, (j_entry, c_entry) in enumerate(zip(jax_list, cpu_list)):
            np.testing.assert_allclose(
                np.array(j_entry),
                c_entry,
                rtol=rtol,
                atol=atol,
                err_msg=f"coil {k}",
            )

    def test_dB_by_dcoilcurrents_parity_ncsx(self):
        """``BiotSavartJAX.dB_by_dcoilcurrents()`` matches CPU list per coil.

        Oracle: C++ reference symbol
        ``simsoptpp::BiotSavart::dB_by_dcoilcurrents`` accessed through
        ``simsopt.field.biotsavart.BiotSavart.dB_by_dcoilcurrents``
        (acceptable oracle type 1). Lane: ``direct-kernel`` value
        tolerances from the validation-ladder SSOT.
        """
        self._assert_coil_current_list_parity(
            "B",
            "dB_by_dcoilcurrents",
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_per_coil_unit_field_contract_under_coil_group_sharding(self, monkeypatch):
        """Per-coil current derivatives stay list-shaped under sharding envs."""
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        bs.B()
        cpu_list = bs.dB_by_dcoilcurrents()

        monkeypatch.setenv("SIMSOPT_JAX_SHARDING", "coil_groups")
        invalidate_backend_cache()
        try:
            bs_jax = BiotSavartJAX(list(bs._coils))
            bs_jax.set_points(points_np)
            jax_list = bs_jax.dB_by_dcoilcurrents()
        finally:
            invalidate_backend_cache()

        assert isinstance(jax_list, list)
        assert len(jax_list) == len(cpu_list)
        for j_entry, c_entry in zip(jax_list, cpu_list):
            np.testing.assert_allclose(
                np.asarray(j_entry),
                np.asarray(c_entry),
                rtol=_DIRECT_KERNEL_TOLS["rtol"],
                atol=_DIRECT_KERNEL_TOLS["atol"],
            )

    def test_per_coil_unit_field_vectorizes_within_quadrature_group(self):
        from simsopt_jax_adapters.field.biotsavart_backend import _per_coil_unit_field

        points = jnp.asarray([[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]], dtype=jnp.float64)
        group0_gammas = jnp.arange(18, dtype=jnp.float64).reshape(2, 3, 3)
        group0_gammadashs = group0_gammas + 0.5
        group1_gammas = jnp.arange(9, dtype=jnp.float64).reshape(1, 3, 3) - 2.0
        group1_gammadashs = group1_gammas - 0.25
        coil_set_spec = GroupedCoilSetSpec(
            groups=(
                CoilGroupSpec(
                    gammas=group0_gammas,
                    gammadashs=group0_gammadashs,
                    currents=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
                    coil_indices=(2, 0),
                ),
                CoilGroupSpec(
                    gammas=group1_gammas,
                    gammadashs=group1_gammadashs,
                    currents=jnp.asarray([5.0], dtype=jnp.float64),
                    coil_indices=(1,),
                ),
            )
        )
        calls = []

        def kernel(kernel_points, gammas, gammadashs, currents):
            calls.append(gammas.shape)
            value = jnp.sum(gammas) + jnp.sum(gammadashs) + jnp.sum(currents)
            return jnp.broadcast_to(value, kernel_points.shape)

        results = _per_coil_unit_field(points, coil_set_spec, kernel)

        assert len(calls) == 2
        assert calls == [(1, 3, 3), (1, 3, 3)]
        assert len(results) == 3
        np.testing.assert_allclose(
            np.asarray(results[0]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[1]))
                + np.sum(np.asarray(group0_gammadashs[1]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[1]),
            np.broadcast_to(
                np.sum(np.asarray(group1_gammas[0]))
                + np.sum(np.asarray(group1_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[2]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[0]))
                + np.sum(np.asarray(group0_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )

    def test_dA_by_dcoilcurrents_parity_ncsx(self):
        """``BiotSavartJAX.dA_by_dcoilcurrents()`` matches CPU list per coil.

        Oracle: C++ reference symbol
        ``simsoptpp::BiotSavart::dA_by_dcoilcurrents`` accessed through
        ``simsopt.field.biotsavart.BiotSavart.dA_by_dcoilcurrents``
        (acceptable oracle type 1). Lane: ``direct-kernel`` value
        tolerances from the validation-ladder SSOT.
        """
        self._assert_coil_current_list_parity(
            "A",
            "dA_by_dcoilcurrents",
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_d2B_by_dXdcoilcurrents_parity_ncsx(self):
        """``BiotSavartJAX.d2B_by_dXdcoilcurrents()`` matches CPU list per coil.

        Oracle: C++ reference symbol
        ``simsoptpp::BiotSavart::d2B_by_dXdcoilcurrents`` accessed
        through ``simsopt.field.biotsavart.BiotSavart.d2B_by_dXdcoilcurrents``
        (acceptable oracle type 1). Lane: ``derivative-heavy``
        first-derivative tolerances from the validation-ladder SSOT.
        """
        self._assert_coil_current_list_parity(
            "dB_by_dX",
            "d2B_by_dXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d2A_by_dXdcoilcurrents_parity_ncsx(self):
        """``BiotSavartJAX.d2A_by_dXdcoilcurrents()`` matches CPU list per coil.

        Oracle: C++ reference symbol
        ``simsoptpp::BiotSavart::d2A_by_dXdcoilcurrents`` accessed
        through ``simsopt.field.biotsavart.BiotSavart.d2A_by_dXdcoilcurrents``
        (acceptable oracle type 1). Lane: ``derivative-heavy``
        first-derivative tolerances from the validation-ladder SSOT.
        """
        self._assert_coil_current_list_parity(
            "dA_by_dX",
            "d2A_by_dXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d3B_by_dXdXdcoilcurrents_parity_ncsx(self):
        """``BiotSavartJAX.d3B_by_dXdXdcoilcurrents()`` matches CPU list per coil.

        Oracle: C++ reference symbol
        ``simsoptpp::BiotSavart::d3B_by_dXdXdcoilcurrents`` accessed
        through
        ``simsopt.field.biotsavart.BiotSavart.d3B_by_dXdXdcoilcurrents``
        (acceptable oracle type 1). Lane: ``derivative-heavy``
        second-derivative tolerances from the validation-ladder SSOT.
        """
        self._assert_coil_current_list_parity(
            "d2B_by_dXdX",
            "d3B_by_dXdXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    def test_d3A_by_dXdXdcoilcurrents_parity_ncsx(self):
        """``BiotSavartJAX.d3A_by_dXdXdcoilcurrents()`` matches CPU list per coil.

        Oracle: C++ reference symbol
        ``simsoptpp::BiotSavart::d3A_by_dXdXdcoilcurrents`` accessed
        through
        ``simsopt.field.biotsavart.BiotSavart.d3A_by_dXdXdcoilcurrents``
        (acceptable oracle type 1). Lane: ``derivative-heavy``
        second-derivative tolerances from the validation-ladder SSOT.
        """
        self._assert_coil_current_list_parity(
            "d2A_by_dXdX",
            "d3A_by_dXdXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )


class TestBiotSavartJaxChunkedSelfConsistency:
    """Chunked-vs-dense JAX self-consistency for the low-level kernels.

    This class checks that chunking (coil chunks, quadrature blocks,
    point chunks, mesh sharding) does not perturb the JAX reduction
    against the SAME JAX kernel evaluated dense (no chunking). The dense
    reference is the JAX kernel itself (``module._one_point_dense``
    under ``jax.vmap`` / ``jax.jacfwd`` / ``jax.vjp``), not the C++
    ``simsoptpp.BiotSavart`` symbol — so these tests are explicit
    Tier-4 self-consistency probes per
    ``tests/REVIEWER_ORACLE_LINT.md``. Direct C++ parity assertions for
    ``B``, ``dB/dX``, and ``B_vjp`` live in
    ``TestBiotSavartJaxCppParity`` above.
    """

    def test_per_coil_unit_field_batch_matches_unbounded_vmap_reference(self):
        from simsopt_jax_adapters.field.biotsavart_backend import (
            _per_coil_unit_field_with_batch_size,
        )
        from simsopt_jax.core.biotsavart import (
            biot_savart_B,
            biot_savart_dB_by_dX,
            biot_savart_d2B_by_dXdX,
        )

        gammas, gammadashs, currents = _make_shifted_circular_coils(5, R=0.72, nquad=12)
        points = jnp.asarray(
            (
                (0.18, -0.12, 0.07),
                (-0.22, 0.15, -0.05),
            ),
            dtype=jnp.float64,
        )
        coil_set_spec = GroupedCoilSetSpec(
            groups=(
                CoilGroupSpec(
                    gammas=gammas[jnp.asarray((3, 0, 4))],
                    gammadashs=gammadashs[jnp.asarray((3, 0, 4))],
                    currents=currents[jnp.asarray((3, 0, 4))],
                    coil_indices=(3, 0, 4),
                ),
                CoilGroupSpec(
                    gammas=gammas[jnp.asarray((1, 2))],
                    gammadashs=gammadashs[jnp.asarray((1, 2))],
                    currents=currents[jnp.asarray((1, 2))],
                    coil_indices=(1, 2),
                ),
            )
        )

        def unbounded_vmap_reference(kernel):
            result_by_index = {}
            for group in coil_set_spec.groups:
                unit_current = jnp.ones((1,), dtype=group.currents.dtype)

                def evaluate_single(gamma, gammadash):
                    return kernel(
                        points,
                        gamma[jnp.newaxis, ...],
                        gammadash[jnp.newaxis, ...],
                        unit_current,
                    )

                group_results = jax.vmap(evaluate_single)(
                    group.gammas,
                    group.gammadashs,
                )
                for position, coil_index in enumerate(group.coil_indices):
                    result_by_index[int(coil_index)] = group_results[position]
            return [result_by_index[index] for index in range(len(currents))]

        for batch_size in (0, 2):
            for kernel in (
                biot_savart_B,
                biot_savart_dB_by_dX,
                biot_savart_d2B_by_dXdX,
            ):
                actual_entries = _per_coil_unit_field_with_batch_size(
                    points,
                    coil_set_spec,
                    kernel,
                    batch_size=batch_size,
                )
                expected_entries = unbounded_vmap_reference(kernel)
                assert len(actual_entries) == len(expected_entries)
                for actual, expected in zip(actual_entries, expected_entries, strict=True):
                    np.testing.assert_allclose(
                        np.asarray(actual),
                        np.asarray(expected),
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_analytic_B_and_dB_matches_linearized_reference(self, monkeypatch):
        from simsopt_jax.core import biotsavart as core_bs

        def linearized_reference(points, gammas, gammadashs, currents, tuning):
            coil_cs, quad_bs, point_cs = tuning

            def one_point(x, group_gammas, group_gammadashs, group_currents):
                return core_bs._coil_chunk_reduce(
                    group_gammas,
                    group_gammadashs,
                    group_currents,
                    chunk_size=coil_cs,
                    zero=core_bs._zeros((3,), dtype=jnp.float64),
                    reduce_chunk=lambda cg, cgd, cc: core_bs._one_point_dense(
                        x,
                        cg,
                        cgd,
                        cc,
                        integrand=core_bs._biot_savart_B_integrand,
                        quadrature_block_size=quad_bs,
                    ),
                )

            def per_point(x):
                f = lambda xx: one_point(xx, gammas, gammadashs, currents)
                primals, tangents_fn = jax.linearize(f, x)
                basis = core_bs._eye(3, dtype=jnp.float64)
                return primals, jax.vmap(tangents_fn, in_axes=(0,))(basis)

            def chunk_fn(chunk_points):
                return jax.vmap(per_point, in_axes=(0,))(chunk_points)

            return core_bs._point_chunk_reduce(points, chunk_fn, point_cs)

        points, gammas, gammadashs, currents = _make_random_fixture(
            seed=71,
            ncoils=9,
            nquad=25,
            npoints=7,
        )
        for tuning in ((0, 0, 0), (4, 0, 0), (0, 7, 0), (4, 7, 3)):
            monkeypatch.setattr(core_bs, "_read_tuning_config", lambda: tuning)
            core_bs.invalidate_kernel_cache()
            actual_B, actual_dB = core_bs.biot_savart_B_and_dB(
                points,
                gammas,
                gammadashs,
                currents,
            )
            expected_B, expected_dB = linearized_reference(
                points,
                gammas,
                gammadashs,
                currents,
                tuning,
            )
            np.testing.assert_allclose(
                np.asarray(actual_B),
                np.asarray(expected_B),
                rtol=1e-12,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                np.asarray(actual_dB),
                np.asarray(expected_dB),
                rtol=1e-12,
                atol=1e-14,
            )

    def test_kernel_factories_do_not_key_equivalent_kernels_by_platform(self):
        from simsopt_jax.core import biotsavart as core_bs

        make_kernel_params = tuple(
            inspect.signature(core_bs._make_kernel.__wrapped__).parameters
        )
        make_b_vjp_kernel_params = tuple(
            inspect.signature(core_bs._make_B_vjp_kernel.__wrapped__).parameters
        )

        assert make_kernel_params == (
            "integrand_key",
            "diff_mode",
            "coil_cs",
            "quad_bs",
            "point_cs",
            "point_vma_axis_name",
        )
        assert make_b_vjp_kernel_params == ("coil_cs", "quad_bs", "point_cs")
        assert tuple(
            inspect.signature(
                core_bs._make_d2B_contracted_kernel.__wrapped__
            ).parameters
        ) == ("coil_cs", "quad_bs", "point_cs")

    def test_kernel_factory_lru_capacities_cover_mode_sweeps(self):
        from simsopt_jax.core import biotsavart as core_bs

        assert core_bs._make_kernel.cache_info().maxsize == 256
        assert core_bs._make_B_vjp_kernel.cache_info().maxsize == 64
        assert core_bs._make_d2B_contracted_kernel.cache_info().maxsize == 64

    def test_backend_cache_invalidation_clears_kernel_cache(self):
        with _kernel_tuning_env("jax_cpu_parity"):
            from simsopt_jax.core import biotsavart as core_bs

            core_bs.invalidate_kernel_cache()
            gammas, gammadashs, currents = _make_shifted_circular_coils(4, nquad=16)
            points = jnp.array([[0.2, -0.1, 0.05]], dtype=jnp.float64)
            v = jnp.array([[0.3, -0.2, 0.1]], dtype=jnp.float64)

            core_bs.biot_savart_B(points, gammas, gammadashs, currents)
            core_bs.biot_savart_B_vjp(points, v, gammas, gammadashs, currents)
            assert core_bs._make_kernel.cache_info().currsize > 0
            assert core_bs._make_B_vjp_kernel.cache_info().currsize > 0

            invalidate_backend_cache()

            assert core_bs._make_kernel.cache_info().currsize == 0
            assert core_bs._make_B_vjp_kernel.cache_info().currsize == 0

    def test_B_vjp_rebuilds_when_tuning_changes_in_process(self, monkeypatch):
        with _kernel_tuning_env("jax_cpu_parity"):
            chunked_bs = _load_chunked_biotsavart()
            from simsopt_jax.core import biotsavart as core_bs

            points, gammas, gammadashs, currents = _make_random_fixture(
                seed=11,
                ncoils=7,
                nquad=19,
                npoints=4,
            )
            v = jnp.linspace(0.2, 1.3, points.shape[0] * 3, dtype=jnp.float64).reshape(
                points.shape[0],
                3,
            )

            monkeypatch.setattr(core_bs, "_read_tuning_config", lambda: (3, 5, 0))
            core_bs.invalidate_kernel_cache()
            first_vjp = core_bs.biot_savart_B_vjp(
                points,
                v,
                gammas,
                gammadashs,
                currents,
            )
            assert core_bs._make_B_vjp_kernel.cache_info().currsize == 1

            monkeypatch.setattr(core_bs, "_read_tuning_config", lambda: (2, 4, 0))
            second_vjp = core_bs.biot_savart_B_vjp(
                points,
                v,
                gammas,
                gammadashs,
                currents,
            )
            assert core_bs._make_B_vjp_kernel.cache_info().currsize == 2

            dense_vjp = _dense_B_vjp(
                chunked_bs,
                points,
                v,
                gammas,
                gammadashs,
                currents,
            )
            for chunked_out in (first_vjp, second_vjp):
                for chunked_leaf, dense_leaf in zip(chunked_out, dense_vjp):
                    np.testing.assert_allclose(
                        np.asarray(chunked_leaf),
                        np.asarray(dense_leaf),
                        atol=1e-14,
                    )

    def test_two_chunk_coil_and_quadrature_paths_match_dense_reference(
        self, monkeypatch
    ):
        with _kernel_tuning_env("jax_cpu_parity"):
            chunked_bs = _load_chunked_biotsavart()
            from simsopt_jax.core import biotsavart as core_bs

            monkeypatch.setattr(core_bs, "_read_tuning_config", lambda: (3, 5, 0))
            core_bs.invalidate_kernel_cache()

            gammas, gammadashs, currents = _make_shifted_circular_coils(6, nquad=9)
            points = jnp.array(
                [
                    [0.2, 0.1, -0.3],
                    [0.1, -0.4, 0.0],
                    [-0.3, 0.2, 0.35],
                ],
                dtype=jnp.float64,
            )

            dense_B, dense_A, dense_dB, dense_dA = _dense_reference_fields(
                chunked_bs,
                points,
                gammas,
                gammadashs,
                currents,
            )

            B, A, dB, dA, B_combo, dB_combo = _evaluate_field_family(
                chunked_bs,
                points,
                gammas,
                gammadashs,
                currents,
            )

            assert core_bs._read_tuning_config() == (3, 5, 0)
            np.testing.assert_allclose(np.asarray(B), np.asarray(dense_B), atol=1e-14)
            np.testing.assert_allclose(np.asarray(A), np.asarray(dense_A), atol=1e-14)
            np.testing.assert_allclose(np.asarray(dB), np.asarray(dense_dB), atol=1e-14)
            np.testing.assert_allclose(np.asarray(dA), np.asarray(dense_dA), atol=1e-14)
            np.testing.assert_allclose(
                np.asarray(B_combo),
                np.asarray(dense_B),
                atol=1e-14,
            )
            np.testing.assert_allclose(
                np.asarray(dB_combo),
                np.asarray(dense_dB),
                atol=1e-14,
            )

    def test_chunked_B_and_dB_match_dense_reference(self):
        with _kernel_tuning_env("jax_cpu_parity"):
            chunked_bs = _load_chunked_biotsavart()
            from simsopt_jax.backend import get_coil_chunk_size

            assert get_coil_chunk_size("jax_cpu_parity") > 0

            gammas, gammadashs, currents = _make_shifted_circular_coils(20, nquad=96)
            points = jnp.array(
                [
                    [0.2, 0.1, -0.3],
                    [0.1, -0.4, 0.0],
                    [-0.3, 0.2, 0.35],
                ],
                dtype=jnp.float64,
            )

            def _dense_B(x):
                return chunked_bs._one_point_dense(
                    x,
                    gammas,
                    gammadashs,
                    currents,
                    integrand=chunked_bs._biot_savart_B_integrand,
                )

            dense_B = jax.vmap(_dense_B)(points)
            dense_dB = jax.vmap(
                lambda x: jnp.swapaxes(jax.jacfwd(_dense_B)(x), -1, -2)
            )(points)

            B = chunked_bs.biot_savart_B(points, gammas, gammadashs, currents)
            dB = chunked_bs.biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
            B_combo, dB_combo = chunked_bs.biot_savart_B_and_dB(
                points,
                gammas,
                gammadashs,
                currents,
            )

            np.testing.assert_allclose(np.asarray(B), np.asarray(dense_B), atol=1e-14)
            np.testing.assert_allclose(np.asarray(dB), np.asarray(dense_dB), atol=1e-14)
            np.testing.assert_allclose(
                np.asarray(B_combo),
                np.asarray(dense_B),
                atol=1e-14,
            )
            np.testing.assert_allclose(
                np.asarray(dB_combo),
                np.asarray(dense_dB),
                atol=1e-14,
            )

    def test_chunked_A_matches_dense_reference(self):
        with _kernel_tuning_env("jax_cpu_parity"):
            chunked_bs = _load_chunked_biotsavart()
            from simsopt_jax.backend import get_coil_chunk_size

            assert get_coil_chunk_size("jax_cpu_parity") > 0

            gammas, gammadashs, currents = _make_shifted_circular_coils(20, nquad=96)
            points = jnp.array(
                [
                    [0.15, 0.05, -0.25],
                    [-0.05, -0.25, 0.1],
                ],
                dtype=jnp.float64,
            )

            dense_A = jax.vmap(
                lambda x: chunked_bs._one_point_dense(
                    x,
                    gammas,
                    gammadashs,
                    currents,
                    integrand=chunked_bs._biot_savart_A_integrand,
                )
            )(points)
            A = chunked_bs.biot_savart_A(points, gammas, gammadashs, currents)

            np.testing.assert_allclose(np.asarray(A), np.asarray(dense_A), atol=1e-14)

    @pytest.mark.parametrize(
        ("mode", "rtol", "atol"),
        _BIOTSAVART_CHUNKED_DENSE_PARITY_MODES,
    )
    def test_chunked_B_matches_dense_reference_under_accumulation_stress(
        self, mode, rtol, atol
    ):
        with _kernel_tuning_env(
            mode,
            coil_chunk_size=5,
            quadrature_block_size=17,
        ):
            stressed_bs = _load_with_backend_mode(mode)
            points, gammas, gammadashs, currents = _make_random_fixture(
                seed=23,
                ncoils=37,
                nquad=149,
                npoints=113,
            )
            dense_B = _dense_B_reference(
                stressed_bs,
                points,
                gammas,
                gammadashs,
                currents,
            )
            chunked_B = stressed_bs.biot_savart_B(
                points,
                gammas,
                gammadashs,
                currents,
            )

            np.testing.assert_allclose(
                _host_array(chunked_B),
                _host_array(dense_B),
                rtol=rtol,
                atol=atol,
            )

    @pytest.mark.parametrize(
        ("mode", "rtol", "atol"),
        _BIOTSAVART_ACCUMULATION_ORDER_PARITY_MODES,
    )
    def test_many_coil_many_quadrature_reduction_order_matches_dense_reference(
        self, mode, rtol, atol
    ):
        points, gammas, gammadashs, currents = _make_accumulation_order_fixture(seed=41)

        with _kernel_tuning_env(
            mode,
            coil_chunk_size=0,
            quadrature_block_size=19,
        ):
            quadrature_chunked_bs = _load_with_backend_mode(mode)
            dense_B = _dense_B_reference(
                quadrature_chunked_bs,
                points,
                gammas,
                gammadashs,
                currents,
            )
            quadrature_chunked_B = quadrature_chunked_bs.biot_savart_B(
                points,
                gammas,
                gammadashs,
                currents,
            )

        with _kernel_tuning_env(
            mode,
            coil_chunk_size=7,
            quadrature_block_size=19,
        ):
            fully_chunked_bs = _load_with_backend_mode(mode)
            fully_chunked_B = fully_chunked_bs.biot_savart_B(
                points,
                gammas,
                gammadashs,
                currents,
            )

        dense_host = _host_array(dense_B)
        quadrature_chunked_host = _host_array(quadrature_chunked_B)
        fully_chunked_host = _host_array(fully_chunked_B)

        np.testing.assert_allclose(
            quadrature_chunked_host,
            dense_host,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            fully_chunked_host,
            dense_host,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            fully_chunked_host,
            quadrature_chunked_host,
            rtol=rtol,
            atol=atol,
        )

    def test_point_chunked_B_A_dB_dA_match_dense_reference(self, monkeypatch):
        with _kernel_tuning_env("jax_cpu_parity"):
            chunked_bs = _load_chunked_biotsavart()
            from simsopt_jax.core import biotsavart as core_bs

            monkeypatch.setattr(core_bs, "_read_tuning_config", lambda: (0, 0, 2))
            core_bs.invalidate_kernel_cache()

            gammas, gammadashs, currents = _make_shifted_circular_coils(6, nquad=32)
            points = jnp.array(
                [
                    [0.2, 0.1, -0.3],
                    [0.1, -0.4, 0.0],
                    [-0.3, 0.2, 0.35],
                    [0.05, 0.25, -0.15],
                    [-0.2, -0.1, 0.1],
                ],
                dtype=jnp.float64,
            )

            dense_B, dense_A, dense_dB, dense_dA = _dense_reference_fields(
                chunked_bs,
                points,
                gammas,
                gammadashs,
                currents,
            )

            B = chunked_bs.biot_savart_B(points, gammas, gammadashs, currents)
            A = chunked_bs.biot_savart_A(points, gammas, gammadashs, currents)
            dB = chunked_bs.biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
            dA = chunked_bs.biot_savart_dA_by_dX(points, gammas, gammadashs, currents)
            B_combo, dB_combo = chunked_bs.biot_savart_B_and_dB(
                points,
                gammas,
                gammadashs,
                currents,
            )

            assert core_bs._read_tuning_config() == (0, 0, 2)
            np.testing.assert_allclose(np.asarray(B), np.asarray(dense_B), atol=1e-14)
            np.testing.assert_allclose(np.asarray(A), np.asarray(dense_A), atol=1e-14)
            np.testing.assert_allclose(np.asarray(dB), np.asarray(dense_dB), atol=1e-14)
            np.testing.assert_allclose(np.asarray(dA), np.asarray(dense_dA), atol=1e-14)
            np.testing.assert_allclose(
                np.asarray(B_combo),
                np.asarray(dense_B),
                atol=1e-14,
            )
            np.testing.assert_allclose(
                np.asarray(dB_combo),
                np.asarray(dense_dB),
                atol=1e-14,
            )

    def test_grouped_biot_savart_accepts_explicit_point_sharding(self, monkeypatch):
        invalidate_backend_cache()
        monkeypatch.setattr(
            sharding_core,
            "get_sharding_tuning",
            lambda mode=None: types.SimpleNamespace(
                active=False,
                strategy="none",
                min_points_to_shard=1 << 30,
                min_coils_to_shard=1 << 30,
            ),
        )
        mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("d",))
        points = jax.device_put(
            np.array(
                [
                    [0.2, 0.1, -0.3],
                    [0.1, -0.4, 0.0],
                    [-0.3, 0.2, 0.35],
                    [0.05, 0.25, -0.15],
                ],
                dtype=np.float64,
            ),
            NamedSharding(mesh, P("d", None)),
        )
        gammas, gammadashs, currents = _make_shifted_circular_coils(3, nquad=16)
        coil_spec = grouped_coil_set_spec_from_lists(
            [gammas[0], gammas[1], gammas[2]],
            [gammadashs[0], gammadashs[1], gammadashs[2]],
            [currents[0], currents[1], currents[2]],
        )

        dense_B = biot_savart_B(points, gammas, gammadashs, currents)
        grouped_B = grouped_biot_savart_B_from_spec(points, coil_spec)

        np.testing.assert_allclose(
            np.asarray(grouped_B), np.asarray(dense_B), atol=1e-14
        )
        assert isinstance(grouped_B.sharding, NamedSharding)

    def test_grouped_biot_savart_jit_accepts_forced_point_sharding(self, monkeypatch):
        monkeypatch.setattr(
            sharding_core,
            "get_sharding_tuning",
            lambda mode=None: types.SimpleNamespace(
                active=True,
                strategy="points",
                min_points_to_shard=1,
                platform="cpu",
                mesh_axis_name="d",
            ),
        )

        points = jnp.array(
            [
                [0.2, 0.1, -0.3],
                [0.1, -0.4, 0.0],
                [-0.3, 0.2, 0.35],
                [0.05, 0.25, -0.15],
            ],
            dtype=jnp.float64,
        )
        gammas, gammadashs, currents = _make_shifted_circular_coils(2, nquad=16)
        coil_spec = grouped_coil_set_spec_from_lists(
            [gammas[0], gammas[1]],
            [gammadashs[0], gammadashs[1]],
            [currents[0], currents[1]],
        )

        result = jax.jit(grouped_biot_savart_B_from_spec)(points, coil_spec)

        assert result.shape == (4, 3)
        assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize(
        ("mode", "rtol", "atol"),
        [
            ("jax_cpu_parity", 1e-12, 1e-14),
            parity_mode_case("jax_gpu_fast", 1e-11, 1e-13),
        ],
    )
    def test_randomized_B_A_dB_dA_match_dense_reference(self, mode, rtol, atol):
        with _kernel_tuning_env(mode):
            tuned_bs = _load_with_backend_mode(mode)
            points, gammas, gammadashs, currents = _make_random_fixture(seed=7)
            dense_B, dense_A, dense_dB, dense_dA = _dense_reference_fields(
                tuned_bs,
                points,
                gammas,
                gammadashs,
                currents,
            )

            B = tuned_bs.biot_savart_B(points, gammas, gammadashs, currents)
            A = tuned_bs.biot_savart_A(points, gammas, gammadashs, currents)
            dB = tuned_bs.biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
            dA = tuned_bs.biot_savart_dA_by_dX(points, gammas, gammadashs, currents)

            np.testing.assert_allclose(
                np.asarray(B), np.asarray(dense_B), rtol=rtol, atol=atol
            )
            np.testing.assert_allclose(
                np.asarray(A), np.asarray(dense_A), rtol=rtol, atol=atol
            )
            np.testing.assert_allclose(
                np.asarray(dB), np.asarray(dense_dB), rtol=rtol, atol=atol
            )
            np.testing.assert_allclose(
                np.asarray(dA), np.asarray(dense_dA), rtol=rtol, atol=atol
            )


class TestGroupCoilDataOrdering:
    """``group_coil_data`` must yield groups in stable first-input order.

    Cross-group floating-point summation must preserve the same coarse coil
    order as the input-loop CPU reference without relying on dictionary
    iteration as the ordering mechanism.
    """

    @staticmethod
    def _build_uniform_coil(nquad: int, current: float, *, seed: int):
        rng = np.random.default_rng(seed)
        gamma = rng.standard_normal((nquad, 3))
        gammadash = rng.standard_normal((nquad, 3))
        return gamma, gammadash, current

    def test_groups_returned_in_first_input_then_input_index_order(self):
        from simsopt_jax.core import group_coil_data

        # Mixed-quadrature input: positions [0, 3] use 128-point quadrature,
        # [1, 2] use 15-point. Group order must follow each group's first
        # occurrence in the input list so cross-group summation keeps the same
        # coarse order as the CPU loop.
        coil_specs = [(128, 1.0), (15, 2.0), (15, 3.0), (128, 4.0)]
        gammas, gammadashs, currents = [], [], []
        for i, (nquad, current) in enumerate(coil_specs):
            g, gd, c = self._build_uniform_coil(nquad, current, seed=i)
            gammas.append(g)
            gammadashs.append(gd)
            currents.append(c)

        groups = group_coil_data(gammas, gammadashs, currents)
        assert len(groups) == 2

        first_gammas, _, first_currents, first_indices = groups[0]
        second_gammas, _, second_currents, second_indices = groups[1]

        assert first_gammas.shape[1] == 128
        assert second_gammas.shape[1] == 15
        assert tuple(first_indices) == (0, 3)
        assert tuple(second_indices) == (1, 2)
        np.testing.assert_array_equal(
            np.asarray(first_currents), np.asarray([1.0, 4.0])
        )
        np.testing.assert_array_equal(
            np.asarray(second_currents), np.asarray([2.0, 3.0])
        )


class TestBiotSavartJAXCoilStateToken:
    """Traceable runtime cache invalidation is keyed to coil DOF state."""

    @staticmethod
    def _make_two_basic_coils():
        from simsopt.field.coil import Coil, Current
        from simsopt.geo.curvexyzfourier import CurveXYZFourier

        coils = []
        for current_amp in (1.0e6, -1.0e6):
            curve = CurveXYZFourier(quadpoints=16, order=1)
            curve.x = np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            coils.append(Coil(curve, Current(current_amp)))
        return coils

    @staticmethod
    def _coil_arrays_in_original_order(coil_set_spec):
        coil_count = sum(len(group.coil_indices) for group in coil_set_spec.groups)
        gammas = [None] * coil_count
        gammadashs = [None] * coil_count
        currents = [None] * coil_count
        for group in coil_set_spec.groups:
            for position, coil_index in enumerate(group.coil_indices):
                gammas[coil_index] = group.gammas[position]
                gammadashs[coil_index] = group.gammadashs[position]
                currents[coil_index] = group.currents[position]
        return gammas, gammadashs, currents

    @staticmethod
    def _assert_per_coil_entries_equal(actual_entries, expected_entries):
        assert len(actual_entries) == len(expected_entries)
        for actual, expected in zip(actual_entries, expected_entries, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=0.0,
                atol=0.0,
            )

    def test_current_derivative_methods_preserve_compute_derivatives_keyword_contract(
        self,
    ):
        from simsopt_jax_adapters.field.biotsavart_backend import (
            BiotSavartJAX,
            SpecBackedBiotSavartJAX,
        )
        from simsopt_jax.core.specs import make_biot_savart_spec

        method_defaults = (
            ("dB_by_dcoilcurrents", 0),
            ("d2B_by_dXdcoilcurrents", 1),
            ("d3B_by_dXdXdcoilcurrents", 2),
            ("dA_by_dcoilcurrents", 0),
            ("d2A_by_dXdcoilcurrents", 1),
            ("d3A_by_dXdXdcoilcurrents", 2),
        )
        points = np.asarray(
            (
                (1.25, 0.1, -0.2),
                (0.9, -0.3, 0.4),
            ),
            dtype=np.float64,
        )
        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        bs_jax.set_points(points)
        spec = make_biot_savart_spec(
            coil_dof_extraction=bs_jax.coil_dof_extraction_spec(),
            coil_dofs=np.asarray(bs_jax.x, dtype=np.float64),
        )
        spec_backed = SpecBackedBiotSavartJAX(spec)
        spec_backed.set_points(points)

        for field_adapter in (bs_jax, spec_backed):
            for method_name, expected_default in method_defaults:
                method = getattr(field_adapter, method_name)
                parameter = inspect.signature(method).parameters["compute_derivatives"]
                assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                assert parameter.default == expected_default

                default_entries = method()
                for compute_derivatives in (0, 1, 2):
                    keyword_entries = method(
                        compute_derivatives=compute_derivatives,
                    )
                    self._assert_per_coil_entries_equal(
                        keyword_entries,
                        default_entries,
                    )

    def test_coil_set_spec_uses_uniform_curve_xyz_fourier_fastpath(self, monkeypatch):
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        bs_jax = BiotSavartJAX(self._make_two_basic_coils())
        assert bs_jax._uses_uniform_curve_xyz_fourier_fastpath

        fast_gammas, fast_gammadashs, fast_currents = (
            bs_jax._coil_arrays_in_order_from_dofs(bs_jax.x)
        )

        def raise_if_immutable_lane_used(_coil_dofs):
            raise AssertionError("immutable-spec lane used")

        monkeypatch.setattr(
            bs_jax,
            "_coil_set_spec_from_dofs_immutable_specs",
            raise_if_immutable_lane_used,
        )

        spec_gammas, spec_gammadashs, spec_currents = (
            self._coil_arrays_in_original_order(bs_jax.coil_set_spec())
        )

        for actual, expected in zip(spec_gammas, fast_gammas, strict=True):
            np.testing.assert_allclose(actual, expected)
        for actual, expected in zip(spec_gammadashs, fast_gammadashs, strict=True):
            np.testing.assert_allclose(actual, expected)
        for actual, expected in zip(spec_currents, fast_currents, strict=True):
            np.testing.assert_allclose(actual, expected)

    def test_biotsavart_jax_advances_coil_dof_state_token_on_x_update(self):
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        initial_token = bs_jax._coil_dof_state_token

        bs_jax.x = np.asarray(bs_jax.x, dtype=np.float64)

        assert bs_jax._coil_dof_state_token != initial_token
        assert bs_jax._coil_dofs_generation == 1

        next_token = bs_jax._coil_dof_state_token
        bs_jax.full_x = np.asarray(bs_jax.full_x, dtype=np.float64)

        assert bs_jax._coil_dof_state_token != next_token
        assert bs_jax._coil_dofs_generation == 2

    def test_biotsavart_jax_advances_coil_dof_state_token_on_parent_update(self):
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        initial_token = bs_jax._coil_dof_state_token
        curve_dofs = np.asarray(coils[0].curve.x, dtype=np.float64)
        curve_dofs[0] += 1.0e-4

        coils[0].curve.x = curve_dofs

        assert bs_jax._coil_dof_state_token != initial_token
        assert bs_jax._coil_dofs_generation == 1

    def test_spec_backed_biotsavart_jax_advances_coil_dof_state_token_on_update(self):
        from simsopt_jax_adapters.field.biotsavart_backend import (
            BiotSavartJAX,
            SpecBackedBiotSavartJAX,
        )
        from simsopt_jax.core.specs import make_biot_savart_spec

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        spec = make_biot_savart_spec(
            coil_dof_extraction=bs_jax.coil_dof_extraction_spec(),
            coil_dofs=np.asarray(bs_jax.x, dtype=np.float64),
        )

        spec_backed_a = SpecBackedBiotSavartJAX(spec)
        initial_token = spec_backed_a._coil_dof_state_token

        spec_backed_a.x = np.asarray(spec_backed_a.x, dtype=np.float64)

        assert spec_backed_a._coil_dof_state_token != initial_token
        assert spec_backed_a._coil_dofs_generation == 1

    def test_spec_backed_biotsavart_jax_advances_layout_version_on_fix(self):
        from simsopt_jax_adapters.field.biotsavart_backend import (
            BiotSavartJAX,
            SpecBackedBiotSavartJAX,
        )
        from simsopt_jax.core.specs import make_biot_savart_spec

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        spec = make_biot_savart_spec(
            coil_dof_extraction=bs_jax.coil_dof_extraction_spec(),
            coil_dofs=np.asarray(bs_jax.x, dtype=np.float64),
        )
        spec_backed = SpecBackedBiotSavartJAX(spec)
        initial_version = spec_backed.dof_layout_version

        spec_backed.fix(0)

        assert spec_backed.dof_layout_version > initial_version

    def test_biotsavart_extraction_spec_changes_only_for_captured_dof_contract(self):
        from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

        coils = self._make_two_basic_coils()
        field = BiotSavartJAX(list(coils))
        curve = coils[0].curve
        initial_spec = field.coil_dof_extraction_spec()
        initial_value = float(curve.local_full_x[0])

        curve.set(0, initial_value + 1.0e-3)
        assert field.coil_dof_extraction_spec() is initial_spec

        curve.fix(0)
        fixed_spec = field.coil_dof_extraction_spec()
        assert fixed_spec is not initial_spec

        curve.set(0, initial_value + 2.0e-3)
        assert field.coil_dof_extraction_spec() is not fixed_spec

    def test_spec_backed_biotsavart_x_setter_writes_free_dofs(self):
        from simsopt_jax_adapters.field.biotsavart_backend import (
            BiotSavartJAX,
            SpecBackedBiotSavartJAX,
        )
        from simsopt_jax.core.specs import make_biot_savart_spec

        class RecordingDofs:
            def __init__(self):
                self.free_x_written = None
                self.full_x_written = None

            @property
            def free_x(self):
                return self.free_x_written

            @free_x.setter
            def free_x(self, values):
                self.free_x_written = np.asarray(values, dtype=np.float64)

            @property
            def full_x(self):
                return self.full_x_written

            @full_x.setter
            def full_x(self, values):
                self.full_x_written = np.asarray(values, dtype=np.float64)

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        spec = make_biot_savart_spec(
            coil_dof_extraction=bs_jax.coil_dof_extraction_spec(),
            coil_dofs=np.asarray(bs_jax.x, dtype=np.float64),
        )
        spec_backed = SpecBackedBiotSavartJAX(spec)
        recorder = RecordingDofs()
        spec_backed._dofs = recorder
        updated = np.asarray(spec_backed.x, dtype=np.float64) + 1.0e-4

        spec_backed.x = updated

        np.testing.assert_allclose(recorder.free_x_written, updated)
        assert recorder.full_x_written is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
