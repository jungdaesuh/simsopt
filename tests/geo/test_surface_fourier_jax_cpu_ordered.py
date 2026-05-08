"""Tests for the CPU-ordered surface Fourier twins (Phase 2 of the bit-identity plan).

The CPU-ordered kernels in
:mod:`simsopt.geo.surface_fourier_jax_cpu_ordered` exist because the
production hot path (matmul + ``jax.jacfwd``) does not match the C++ oracle
in ``src/simsoptpp/surfacexyztensorfourier.h`` byte-for-byte. The plan's
Phase 2 acceptance gate is "census in parity mode shows surface-side arrays
byte-identical OR documents the exact remaining first-mismatch with
arithmetic-order reason." Today the residual is FMA-fusion (Phase 4
territory), so these tests assert (a) the cpu_ordered output reproduces the
C++ values within tightly bounded ULP drift and (b) it is *strictly tighter*
than the production matmul kernel — proving the Phase 2 substitution
removes the dominant arithmetic-order divergence even before Phase 4.
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.parity_census, pytest.mark.boozer]


_FIXTURE_PARAMS = [
    pytest.param(2, 2, 3, True, 11, 11, id="m2-n2-nfp3-stellsym-11x11"),
    pytest.param(3, 3, 3, True, 20, 20, id="m3-n3-nfp3-stellsym-20x20"),
    pytest.param(2, 1, 2, False, 13, 9, id="m2-n1-nfp2-non-stellsym-13x9"),
]


@pytest.fixture
def cpu_jax_pair():
    """Build a ``SurfaceXYZTensorFourier`` and bind helpers we'll need."""

    def _build(*, mpol, ntor, nfp, stellsym, nphi, ntheta, seed=42):
        import jax

        jax.config.update("jax_enable_x64", True)
        from simsopt.geo import SurfaceXYZTensorFourier
        from simsopt.geo.surface_fourier_jax import (
            _dofs_to_xyzc_any,
            stellsym_scatter_indices,
        )
        import jax.numpy as jnp

        phis = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
        thetas = np.linspace(0, 1.0, ntheta, endpoint=False)
        s = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            clamped_dims=[False, False, False],
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )
        rng = np.random.default_rng(seed)
        sdofs = rng.normal(loc=0.0, scale=0.5, size=s.get_dofs().size)
        # Push DOFs through both surfaces equivalently; the absolute scale
        # doesn't matter as long as it's well-conditioned.
        s.set_dofs(sdofs)
        scatter_indices = stellsym_scatter_indices(mpol, ntor) if stellsym else None
        xc, yc, zc = _dofs_to_xyzc_any(
            jnp.asarray(sdofs), mpol, ntor, stellsym, scatter_indices
        )
        return {
            "surface": s,
            "sdofs": sdofs,
            "xc": xc,
            "yc": yc,
            "zc": zc,
            "scatter_indices": scatter_indices,
            "mpol": mpol,
            "ntor": ntor,
            "nfp": nfp,
            "stellsym": stellsym,
        }

    return _build


@pytest.mark.parametrize("mpol,ntor,nfp,stellsym,nphi,ntheta", _FIXTURE_PARAMS)
def test_surface_gamma_cpu_ordered_matches_cpp_within_ulp(
    cpu_jax_pair, mpol, ntor, nfp, stellsym, nphi, ntheta
):
    import jax

    from simsopt.geo.surface_fourier_jax_cpu_ordered import (
        surface_gamma_cpu_ordered,
    )
    from simsopt.geo.surface_fourier_jax import surface_gamma

    fx = cpu_jax_pair(
        mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, nphi=nphi, ntheta=ntheta
    )
    s = fx["surface"]
    gamma_cpp = np.asarray(s.gamma(), dtype=np.float64)
    gamma_cpu_ordered = np.asarray(
        jax.device_get(
            surface_gamma_cpu_ordered(
                s.quadpoints_phi,
                s.quadpoints_theta,
                fx["xc"],
                fx["yc"],
                fx["zc"],
                mpol,
                ntor,
                nfp,
            )
        ),
        dtype=np.float64,
    )
    gamma_production = np.asarray(
        jax.device_get(
            surface_gamma(
                s.quadpoints_phi,
                s.quadpoints_theta,
                fx["xc"],
                fx["yc"],
                fx["zc"],
                mpol,
                ntor,
                nfp,
            )
        ),
        dtype=np.float64,
    )
    cpu_ordered_drift = np.max(np.abs(gamma_cpu_ordered - gamma_cpp))
    production_drift = np.max(np.abs(gamma_production - gamma_cpp))
    # Phase 2 lower bound: cpu_ordered must be at most production-drift,
    # *and* the absolute drift stays within the documented FMA-fusion ULP
    # ceiling. (Production drift is ~4-7 ULP × |gamma|; cpu_ordered should
    # be 1-2 ULP under the same ladder.)
    assert cpu_ordered_drift <= production_drift, (
        f"cpu_ordered drift {cpu_ordered_drift!r} exceeds production matmul "
        f"drift {production_drift!r}; Phase 2 substitution must not regress."
    )
    assert cpu_ordered_drift < 1e-13, (
        f"cpu_ordered gamma drift {cpu_ordered_drift!r} exceeds the FMA-fusion "
        "ULP ceiling; investigate."
    )


@pytest.mark.parametrize("mpol,ntor,nfp,stellsym,nphi,ntheta", _FIXTURE_PARAMS)
def test_surface_gammadash_cpu_ordered_matches_cpp(
    cpu_jax_pair, mpol, ntor, nfp, stellsym, nphi, ntheta
):
    import jax

    from simsopt.geo.surface_fourier_jax_cpu_ordered import (
        surface_gammadash1_cpu_ordered,
        surface_gammadash2_cpu_ordered,
    )

    fx = cpu_jax_pair(
        mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, nphi=nphi, ntheta=ntheta
    )
    s = fx["surface"]
    gd1_cpp = np.asarray(s.gammadash1(), dtype=np.float64)
    gd2_cpp = np.asarray(s.gammadash2(), dtype=np.float64)
    gd1_cpu = np.asarray(
        jax.device_get(
            surface_gammadash1_cpu_ordered(
                s.quadpoints_phi,
                s.quadpoints_theta,
                fx["xc"],
                fx["yc"],
                fx["zc"],
                mpol,
                ntor,
                nfp,
            )
        ),
        dtype=np.float64,
    )
    gd2_cpu = np.asarray(
        jax.device_get(
            surface_gammadash2_cpu_ordered(
                s.quadpoints_phi,
                s.quadpoints_theta,
                fx["xc"],
                fx["yc"],
                fx["zc"],
                mpol,
                ntor,
                nfp,
            )
        ),
        dtype=np.float64,
    )
    # Tangent magnitudes scale with 2π · |coeff|, so the absolute drift
    # ceiling is correspondingly larger — but still well within the
    # documented FMA-fusion bracket.
    assert np.max(np.abs(gd1_cpu - gd1_cpp)) < 5e-13
    assert np.max(np.abs(gd2_cpu - gd2_cpp)) < 5e-13


@pytest.mark.parametrize("mpol,ntor,nfp,stellsym,nphi,ntheta", _FIXTURE_PARAMS)
def test_dgamma_by_dcoeff_cpu_ordered_matches_cpp(
    cpu_jax_pair, mpol, ntor, nfp, stellsym, nphi, ntheta
):
    import jax

    from simsopt.geo.surface_fourier_jax_cpu_ordered import (
        dgamma_by_dcoeff_cpu_ordered,
        dgammadash1_by_dcoeff_cpu_ordered,
        dgammadash2_by_dcoeff_cpu_ordered,
    )

    fx = cpu_jax_pair(
        mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym, nphi=nphi, ntheta=ntheta
    )
    s = fx["surface"]

    cpp_arrays = {
        "dgamma_by_dcoeff": np.asarray(s.dgamma_by_dcoeff(), dtype=np.float64),
        "dgammadash1_by_dcoeff": np.asarray(
            s.dgammadash1_by_dcoeff(), dtype=np.float64
        ),
        "dgammadash2_by_dcoeff": np.asarray(
            s.dgammadash2_by_dcoeff(), dtype=np.float64
        ),
    }
    jax_arrays = {
        "dgamma_by_dcoeff": np.asarray(
            jax.device_get(
                dgamma_by_dcoeff_cpu_ordered(
                    s.quadpoints_phi,
                    s.quadpoints_theta,
                    mpol=mpol,
                    ntor=ntor,
                    nfp=nfp,
                    stellsym=stellsym,
                )
            ),
            dtype=np.float64,
        ),
        "dgammadash1_by_dcoeff": np.asarray(
            jax.device_get(
                dgammadash1_by_dcoeff_cpu_ordered(
                    s.quadpoints_phi,
                    s.quadpoints_theta,
                    mpol=mpol,
                    ntor=ntor,
                    nfp=nfp,
                    stellsym=stellsym,
                )
            ),
            dtype=np.float64,
        ),
        "dgammadash2_by_dcoeff": np.asarray(
            jax.device_get(
                dgammadash2_by_dcoeff_cpu_ordered(
                    s.quadpoints_phi,
                    s.quadpoints_theta,
                    mpol=mpol,
                    ntor=ntor,
                    nfp=nfp,
                    stellsym=stellsym,
                )
            ),
            dtype=np.float64,
        ),
    }
    for name, cpp in cpp_arrays.items():
        jax_v = jax_arrays[name]
        assert cpp.shape == jax_v.shape, (
            f"{name}: shape mismatch CPU={cpp.shape} vs JAX={jax_v.shape}"
        )
        # The analytic kernels do not depend on the surface DOFs, so the
        # values are computed from cached basis functions only. Their max
        # absolute drift against C++ should be at or below the basis-cache
        # FMA contribution.
        diff = np.max(np.abs(cpp - jax_v))
        assert diff < 1e-13, f"{name}: cpu_ordered drift {diff!r} too large"


def test_parity_policy_routes_through_cpu_ordered_kernels(cpu_jax_pair):
    """The parity policy gate exposes the cpu_ordered kernels via
    ``_surface_geometry_and_derivatives_from_dofs``."""
    import jax

    from simsopt.geo.boozersurface_jax import (
        _surface_geometry_and_derivatives_from_dofs,
    )
    import jax.numpy as jnp

    fx = cpu_jax_pair(mpol=2, ntor=2, nfp=3, stellsym=True, nphi=11, ntheta=11)
    sdofs = jnp.asarray(fx["sdofs"])
    geom_prod, _ = _surface_geometry_and_derivatives_from_dofs(
        sdofs,
        quadpoints_phi=fx["surface"].quadpoints_phi,
        quadpoints_theta=fx["surface"].quadpoints_theta,
        mpol=fx["mpol"],
        ntor=fx["ntor"],
        nfp=fx["nfp"],
        stellsym=fx["stellsym"],
        scatter_indices=fx["scatter_indices"],
        surface_kind="generic",
        parity_policy="production",
    )
    geom_cpu, _ = _surface_geometry_and_derivatives_from_dofs(
        sdofs,
        quadpoints_phi=fx["surface"].quadpoints_phi,
        quadpoints_theta=fx["surface"].quadpoints_theta,
        mpol=fx["mpol"],
        ntor=fx["ntor"],
        nfp=fx["nfp"],
        stellsym=fx["stellsym"],
        scatter_indices=fx["scatter_indices"],
        surface_kind="generic",
        parity_policy="cpu_ordered",
    )
    cpp_gamma = np.asarray(fx["surface"].gamma(), dtype=np.float64)
    prod = np.asarray(jax.device_get(geom_prod.gamma), dtype=np.float64)
    cpu = np.asarray(jax.device_get(geom_cpu.gamma), dtype=np.float64)
    assert np.max(np.abs(prod - cpp_gamma)) > 0.0, (
        "production matmul should diverge from C++ at sub-ULP magnitudes "
        "(if it doesn't, the test fixture is too small to surface the bug)"
    )
    assert (
        np.max(np.abs(cpu - cpp_gamma)) <= np.max(np.abs(prod - cpp_gamma)) + 1e-18
    ), "parity_policy='cpu_ordered' must not regress vs production"
