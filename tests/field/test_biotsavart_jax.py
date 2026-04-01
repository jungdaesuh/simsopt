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

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path(__file__).resolve().parents[2] / "src")

from simsopt.backend import invalidate_backend_cache

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
_KERNEL_TUNING_ENV_VARS = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_JAX_COIL_CHUNK_SIZE",
    "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
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


def _dense_reference_fields(module, points, gammas, gammadashs, currents):
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

    dense_B = jax.vmap(_dense_B)(points)
    dense_A = jax.vmap(_dense_A)(points)
    dense_dB = jax.vmap(lambda x: jnp.swapaxes(jax.jacfwd(_dense_B)(x), -1, -2))(points)
    dense_dA = jax.vmap(lambda x: jnp.swapaxes(jax.jacfwd(_dense_A)(x), -1, -2))(points)
    return dense_B, dense_A, dense_dB, dense_dA


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

        np.testing.assert_allclose(np.array(B_jax), B_ref, rtol=1e-10)

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

        np.testing.assert_allclose(np.array(dB_jax), dB_ref, rtol=1e-10, atol=1e-13)


class TestBiotSavartJaxChunkedParity:
    """Directly compare chunked low-level kernels against dense references."""

    def test_backend_cache_invalidation_clears_kernel_cache(self):
        with _kernel_tuning_env("jax_cpu_parity"):
            from simsopt.jax_core import biotsavart as core_bs

            core_bs.invalidate_kernel_cache()
            gammas, gammadashs, currents = _make_shifted_circular_coils(4, nquad=16)
            points = jnp.array([[0.2, -0.1, 0.05]], dtype=jnp.float64)

            core_bs.biot_savart_B(points, gammas, gammadashs, currents)
            assert core_bs._make_kernel.cache_info().currsize > 0

            invalidate_backend_cache()

            assert core_bs._make_kernel.cache_info().currsize == 0

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
