"""
Parity tests for the JAX Biot-Savart implementation.

Validates against:
1. Analytical on-axis field of a circular current loop.
2. Maxwell's equation ∇·B = 0 (trace of dB/dX).
3. C++ reference (when simsoptpp is available).
"""

import importlib.util
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

from conftest import parity_acceptance_modes
from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.backend import invalidate_backend_cache
from simsopt.jax_core.field import (
    grouped_biot_savart_B_from_spec,
    grouped_coil_set_spec_from_lists,
)
from simsopt.jax_core import sharding as sharding_core

# Load JAX module directly (avoids simsopt/__init__.py → simsoptpp dep)
_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_with_backend_mode(mode: str):
    return _load(f"biotsavart_jax_{mode}", "field/biotsavart_jax.py")


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


_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
biot_savart_B = _bs.biot_savart_B
biot_savart_dB_by_dX = _bs.biot_savart_dB_by_dX
biot_savart_B_and_dB = _bs.biot_savart_B_and_dB
biot_savart_A = _bs.biot_savart_A
biot_savart_dA_by_dX = _bs.biot_savart_dA_by_dX

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
    np.random.seed(42)
    points_np = np.random.randn(npoints, 3) * 0.3
    points_np[:, 0] += 1.0  # shift near torus

    bs.set_points(points_np)
    gammas_np = np.array([coil.curve.gamma() for coil in coils])
    gds_np = np.array([coil.curve.gammadash() for coil in coils])
    currents_np = np.array([coil.current.get_value() for coil in coils])
    return bs, points_np, gammas_np, gds_np, currents_np


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
        """biot_savart_B_and_dB returns same values as separate calls."""
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
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

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
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

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
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

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
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

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
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

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

    def test_backend_cache_invalidation_clears_kernel_cache(self):
        with _kernel_tuning_env("jax_cpu_parity"):
            from simsopt.jax_core import biotsavart as core_bs

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
            from simsopt.jax_core import biotsavart as core_bs

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
            from simsopt.jax_core import biotsavart as core_bs

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
            from simsopt.backend import get_coil_chunk_size

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
            from simsopt.backend import get_coil_chunk_size

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
            from simsopt.jax_core import biotsavart as core_bs

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
            ("jax_gpu_fast", 1e-11, 1e-13),
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
        from simsopt.jax_core import group_coil_data

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


class TestBiotSavartJAXCacheToken:
    """``BiotSavartJAX`` and ``SpecBackedBiotSavartJAX`` must produce a unique
    UUID ``_cache_token`` per instance.

    The traceable runtime cache key (``surfaceobjectives_jax``) relies on this
    token to discriminate independently-constructed adapters even when CPython
    recycles the ``id()`` of a just-garbage-collected predecessor (W4.2 / E4).
    """

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

    def test_biotsavart_jax_assigns_unique_cache_token(self):
        import uuid

        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

        coils = self._make_two_basic_coils()
        bs_a = BiotSavartJAX(list(coils))
        bs_b = BiotSavartJAX(list(coils))
        assert isinstance(bs_a._cache_token, uuid.UUID)
        assert isinstance(bs_b._cache_token, uuid.UUID)
        assert bs_a._cache_token != bs_b._cache_token

    def test_spec_backed_biotsavart_jax_assigns_unique_cache_token(self):
        import uuid

        from simsopt.field.biotsavart_jax_backend import (
            BiotSavartJAX,
            SpecBackedBiotSavartJAX,
        )
        from simsopt.jax_core.specs import make_biot_savart_spec

        coils = self._make_two_basic_coils()
        bs_jax = BiotSavartJAX(list(coils))
        spec = make_biot_savart_spec(
            coil_dof_extraction=bs_jax.coil_dof_extraction_spec(),
            coil_dofs=np.asarray(bs_jax.x, dtype=np.float64),
        )

        spec_backed_a = SpecBackedBiotSavartJAX(spec)
        spec_backed_b = SpecBackedBiotSavartJAX(spec)
        assert isinstance(spec_backed_a._cache_token, uuid.UUID)
        assert isinstance(spec_backed_b._cache_token, uuid.UUID)
        assert spec_backed_a._cache_token != spec_backed_b._cache_token


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
