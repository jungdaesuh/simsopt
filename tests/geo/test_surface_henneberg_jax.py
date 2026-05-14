"""N04b parity tests for ``SurfaceHenneberg`` -> JAX spec kernels.

Oracle: the CPU ``simsopt.geo.surfacehenneberg.SurfaceHenneberg`` host
class (``src/simsopt/geo/surfacehenneberg.py:588-739``). Every numerical
assertion in this file compares the spec-driven JAX kernel output
(``surface_henneberg_*_from_spec``) against ``surface.gamma()``,
``surface.gammadash1()``, ``surface.gammadash2()``,
``surface.normal()``, ``surface.unitnormal()``, ``surface.area()``, or
``surface.volume()`` from the host class.

Parity-ladder lane
------------------
``direct-kernel`` for ``gamma`` and the two surface derivatives
(``rtol=1e-12, atol=1e-14``); the same lane applies to derived
geometry (normal, unitnormal, area, volume) because the host class
inherits those from ``sopp.Surface`` which composes the same
``gamma*`` kernels evaluated above.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from simsopt.geo.surfacehenneberg import SurfaceHenneberg
from simsopt.jax_core import (
    SurfaceHennebergSpec,
    make_surface_henneberg_spec,
    surface_henneberg_area_from_spec,
    surface_henneberg_gamma_from_spec,
    surface_henneberg_gammadash1_from_spec,
    surface_henneberg_gammadash2_from_spec,
    surface_henneberg_normal_from_spec,
    surface_henneberg_unitnormal_from_spec,
    surface_henneberg_volume_from_spec,
)


_PARITY_RTOL = 1e-12
_PARITY_ATOL = 1e-14
_DERIVED_RTOL = 1e-11
_DERIVED_ATOL = 1e-13


_SHAPE_FIXTURES = [
    pytest.param(1, 1, id="mmax1-nmax1"),
    pytest.param(2, 2, id="mmax2-nmax2"),
    pytest.param(3, 1, id="mmax3-nmax1"),
    pytest.param(1, 3, id="mmax1-nmax3"),
]

_NFP_FIXTURES = [1, 2, 3, 5]
_ALPHA_FIXTURES = [-1, 0, 1]


@pytest.fixture(autouse=True)
def _require_simsoptpp():
    pytest.importorskip("simsoptpp")


def _seed_surface(
    *,
    nfp: int,
    alpha_fac: int,
    mmax: int,
    nmax: int,
    seed: int,
) -> SurfaceHenneberg:
    """Build a SurfaceHenneberg and seed its DOFs with a deterministic RNG.

    The host class initialises ``R0nH[0] = 1.0``, ``bn[0] = 0.1``, and
    ``rhomn(1, 0) = 0.1`` (see surfacehenneberg.py:123-125). We perturb
    every DOF with a small Gaussian noise (scale 1e-2) on top of those
    physically-meaningful initial values, so the surface remains a
    valid plasma boundary and every (m, n) cell exercises a non-trivial
    coefficient.
    """
    surface = SurfaceHenneberg(nfp=nfp, alpha_fac=alpha_fac, mmax=mmax, nmax=nmax)
    rng = np.random.default_rng(seed)
    x = np.asarray(surface.local_full_x, dtype=np.float64)
    x = x + 1e-2 * rng.standard_normal(x.shape)
    surface.local_full_x = x
    return surface


# ---------------------------------------------------------------------------
# Section 1 — to_spec() field-by-field round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mmax", "nmax"), _SHAPE_FIXTURES)
@pytest.mark.parametrize("nfp", _NFP_FIXTURES)
def test_to_spec_round_trips_fields(mmax: int, nmax: int, nfp: int) -> None:
    """``to_spec`` returns a spec whose fields mirror the host class state.

    Oracle: the host class state arrays themselves (type-3 pinned-state
    snapshot at construction time; the spec is constructed *from* those
    arrays so equality here only certifies the round-trip plumbing, not
    the geometry math).
    """
    surface = _seed_surface(nfp=nfp, alpha_fac=1, mmax=mmax, nmax=nmax, seed=2026)
    spec = surface.to_spec()

    assert isinstance(spec, SurfaceHennebergSpec)
    assert spec.nfp == surface.nfp
    assert spec.alpha_fac == surface.alpha_fac
    assert spec.mmax == surface.mmax
    assert spec.nmax == surface.nmax

    np.testing.assert_array_equal(
        np.asarray(spec.R0nH), np.asarray(surface.R0nH, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(spec.Z0nH), np.asarray(surface.Z0nH, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(spec.bn), np.asarray(surface.bn, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(spec.rhomn), np.asarray(surface.rhomn, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(spec.quadpoints_phi),
        np.asarray(surface.quadpoints_phi, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(spec.quadpoints_theta),
        np.asarray(surface.quadpoints_theta, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Section 2 — gamma parity vs CPU oracle across (nfp, alpha_fac, mmax, nmax)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_fac", _ALPHA_FIXTURES)
@pytest.mark.parametrize(("mmax", "nmax"), _SHAPE_FIXTURES)
def test_gamma_matches_cpu_oracle(mmax: int, nmax: int, alpha_fac: int) -> None:
    """``surface_henneberg_gamma_from_spec`` matches CPU ``gamma()``.

    Oracle: ``SurfaceHenneberg.gamma_impl`` (surfacehenneberg.py:626-640)
    via the public ``surface.gamma()`` accessor.
    Lane: direct-kernel, rtol=1e-12, atol=1e-14.
    """
    surface = _seed_surface(nfp=3, alpha_fac=alpha_fac, mmax=mmax, nmax=nmax, seed=2027)
    gamma_cpu = np.asarray(surface.gamma(), dtype=np.float64)

    spec = surface.to_spec()
    gamma_jax = np.asarray(jax.jit(surface_henneberg_gamma_from_spec)(spec))

    assert gamma_jax.shape == gamma_cpu.shape
    np.testing.assert_allclose(
        gamma_jax, gamma_cpu, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )


@pytest.mark.parametrize("nfp", _NFP_FIXTURES)
def test_gamma_matches_cpu_oracle_across_nfp(nfp: int) -> None:
    """``gamma`` parity across multiple field-period counts.

    Oracle: ``SurfaceHenneberg.gamma_impl``.
    Lane: direct-kernel, rtol=1e-12, atol=1e-14.
    """
    surface = _seed_surface(nfp=nfp, alpha_fac=1, mmax=2, nmax=2, seed=2028 + nfp)
    gamma_cpu = np.asarray(surface.gamma(), dtype=np.float64)

    spec = surface.to_spec()
    gamma_jax = np.asarray(jax.jit(surface_henneberg_gamma_from_spec)(spec))

    np.testing.assert_allclose(
        gamma_jax, gamma_cpu, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )


# ---------------------------------------------------------------------------
# Section 3 — gammadash1 / gammadash2 parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_fac", _ALPHA_FIXTURES)
@pytest.mark.parametrize(("mmax", "nmax"), _SHAPE_FIXTURES)
def test_gammadash1_matches_cpu_oracle(mmax: int, nmax: int, alpha_fac: int) -> None:
    """``surface_henneberg_gammadash1_from_spec`` matches CPU ``gammadash1()``.

    Oracle: ``SurfaceHenneberg.gammadash1_impl`` (surfacehenneberg.py:642-704).
    Lane: direct-kernel, rtol=1e-12, atol=1e-14.
    """
    surface = _seed_surface(nfp=3, alpha_fac=alpha_fac, mmax=mmax, nmax=nmax, seed=2029)
    expected = np.asarray(surface.gammadash1(), dtype=np.float64)

    spec = surface.to_spec()
    actual = np.asarray(jax.jit(surface_henneberg_gammadash1_from_spec)(spec))

    np.testing.assert_allclose(actual, expected, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)


@pytest.mark.parametrize("alpha_fac", _ALPHA_FIXTURES)
@pytest.mark.parametrize(("mmax", "nmax"), _SHAPE_FIXTURES)
def test_gammadash2_matches_cpu_oracle(mmax: int, nmax: int, alpha_fac: int) -> None:
    """``surface_henneberg_gammadash2_from_spec`` matches CPU ``gammadash2()``.

    Oracle: ``SurfaceHenneberg.gammadash2_impl`` (surfacehenneberg.py:706-739).
    Lane: direct-kernel, rtol=1e-12, atol=1e-14.
    """
    surface = _seed_surface(nfp=3, alpha_fac=alpha_fac, mmax=mmax, nmax=nmax, seed=2030)
    expected = np.asarray(surface.gammadash2(), dtype=np.float64)

    spec = surface.to_spec()
    actual = np.asarray(jax.jit(surface_henneberg_gammadash2_from_spec)(spec))

    np.testing.assert_allclose(actual, expected, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)


# ---------------------------------------------------------------------------
# Section 4 — derived geometry parity (normal, unitnormal, area, volume)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_fac", _ALPHA_FIXTURES)
def test_normal_matches_cpu_oracle(alpha_fac: int) -> None:
    """Surface normal parity.

    Oracle: ``simsopt.geo.surface.Surface.normal`` (the host class
    derives it from ``gammadash1 x gammadash2``). Direct-kernel lane.
    """
    surface = _seed_surface(nfp=3, alpha_fac=alpha_fac, mmax=2, nmax=2, seed=2031)
    expected = np.asarray(surface.normal(), dtype=np.float64)

    spec = surface.to_spec()
    actual = np.asarray(jax.jit(surface_henneberg_normal_from_spec)(spec))

    np.testing.assert_allclose(actual, expected, rtol=_DERIVED_RTOL, atol=_DERIVED_ATOL)


def test_unitnormal_matches_cpu_oracle() -> None:
    """Surface unit-normal parity.

    Oracle: ``simsopt.geo.surface.Surface.unitnormal``. Direct-kernel lane.
    """
    surface = _seed_surface(nfp=3, alpha_fac=1, mmax=2, nmax=2, seed=2032)
    expected = np.asarray(surface.unitnormal(), dtype=np.float64)

    spec = surface.to_spec()
    actual = np.asarray(jax.jit(surface_henneberg_unitnormal_from_spec)(spec))

    np.testing.assert_allclose(actual, expected, rtol=_DERIVED_RTOL, atol=_DERIVED_ATOL)


def test_area_matches_cpu_oracle() -> None:
    """Surface area parity.

    Oracle: ``simsopt.geo.surface.Surface.area`` (riemann-sum of |normal|).
    Direct-kernel lane.
    """
    surface = _seed_surface(nfp=3, alpha_fac=1, mmax=2, nmax=2, seed=2033)
    expected = float(surface.area())

    spec = surface.to_spec()
    actual = float(jax.jit(surface_henneberg_area_from_spec)(spec))

    assert actual == pytest.approx(expected, rel=_DERIVED_RTOL, abs=_DERIVED_ATOL)


def test_volume_matches_cpu_oracle() -> None:
    """Enclosed-volume parity.

    Oracle: ``simsopt.geo.surface.Surface.volume`` (riemann-sum of γ·n).
    Direct-kernel lane.
    """
    surface = _seed_surface(nfp=3, alpha_fac=1, mmax=2, nmax=2, seed=2034)
    expected = float(surface.volume())

    spec = surface.to_spec()
    actual = float(jax.jit(surface_henneberg_volume_from_spec)(spec))

    assert actual == pytest.approx(expected, rel=_DERIVED_RTOL, abs=_DERIVED_ATOL)


# ---------------------------------------------------------------------------
# Section 5 — closed-form / axisymmetric oracles (alpha_fac == 0 special case)
# ---------------------------------------------------------------------------


def test_axisymmetric_default_torus_matches_analytic() -> None:
    """Default-initialised axisymmetric surface matches the closed-form torus.

    With ``alpha_fac=0``, ``R0nH[0]=1.0``, ``bn[0]=0.1``, ``rhomn(1,0)=0.1``,
    and all other DOFs zero, the parameterisation reduces to
    ``R(θ,φ) = 1.0 + 0.1·cos(θ)``, ``Z(θ,φ) = 0.1·sin(θ)``.

    Oracle: closed-form analytic torus (type 2). Direct-kernel lane.
    """
    surface = SurfaceHenneberg(nfp=1, alpha_fac=0, mmax=1, nmax=0)
    # Default state: R0nH[0]=1.0, bn[0]=0.1, rhomn(1,0)=0.1.
    spec = surface.to_spec()
    gamma_jax = np.asarray(jax.jit(surface_henneberg_gamma_from_spec)(spec))

    qp_phi = np.asarray(surface.quadpoints_phi, dtype=np.float64)
    qp_theta = np.asarray(surface.quadpoints_theta, dtype=np.float64)
    phi_rad = qp_phi * 2 * np.pi
    theta_rad = qp_theta * 2 * np.pi
    phi_grid, theta_grid = np.meshgrid(phi_rad, theta_rad, indexing="ij")

    # With alpha=0: rho=0.1*cos(theta), zeta=0.1*sin(theta),
    # R = 1.0 + rho, Z = zeta.
    R_expected = 1.0 + 0.1 * np.cos(theta_grid)
    Z_expected = 0.1 * np.sin(theta_grid)
    expected = np.stack(
        (R_expected * np.cos(phi_grid), R_expected * np.sin(phi_grid), Z_expected),
        axis=-1,
    )

    np.testing.assert_allclose(
        gamma_jax, expected, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )


def test_axisymmetric_gammadash2_matches_analytic() -> None:
    """Default-initialised axisymmetric ``gammadash2`` matches the closed form.

    With the configuration above, ``∂γ/∂(quadpoint_theta)`` is
    ``2π · (-0.1·sin(θ)·cos(φ), -0.1·sin(θ)·sin(φ), 0.1·cos(θ))``.

    Oracle: closed-form analytic differentiation (type 2).
    Direct-kernel lane.
    """
    surface = SurfaceHenneberg(nfp=1, alpha_fac=0, mmax=1, nmax=0)
    spec = surface.to_spec()
    actual = np.asarray(jax.jit(surface_henneberg_gammadash2_from_spec)(spec))

    qp_phi = np.asarray(surface.quadpoints_phi, dtype=np.float64)
    qp_theta = np.asarray(surface.quadpoints_theta, dtype=np.float64)
    phi_rad = qp_phi * 2 * np.pi
    theta_rad = qp_theta * 2 * np.pi
    phi_grid, theta_grid = np.meshgrid(phi_rad, theta_rad, indexing="ij")
    dR_dtheta = -0.1 * np.sin(theta_grid)
    dZ_dtheta = 0.1 * np.cos(theta_grid)
    expected = (
        2.0
        * np.pi
        * np.stack(
            (
                dR_dtheta * np.cos(phi_grid),
                dR_dtheta * np.sin(phi_grid),
                dZ_dtheta,
            ),
            axis=-1,
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)


# ---------------------------------------------------------------------------
# Section 6 — Validation & immutability
# ---------------------------------------------------------------------------


def test_make_spec_rejects_invalid_alpha_fac() -> None:
    """The factory raises ``ValueError`` on alpha_fac outside {-1, 0, +1}.

    Oracle: the spec docstring contract.
    """
    surface = SurfaceHenneberg(nfp=2, alpha_fac=1, mmax=1, nmax=1)
    with pytest.raises(ValueError, match="alpha_fac must be one of"):
        make_surface_henneberg_spec(
            R0nH=np.asarray(surface.R0nH, dtype=np.float64),
            Z0nH=np.asarray(surface.Z0nH, dtype=np.float64),
            bn=np.asarray(surface.bn, dtype=np.float64),
            rhomn=np.asarray(surface.rhomn, dtype=np.float64),
            quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64),
            quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64),
            nfp=2,
            alpha_fac=2,
            mmax=1,
            nmax=1,
        )


def test_make_spec_rejects_shape_mismatch() -> None:
    """The factory raises ``ValueError`` if the DOF shape disagrees with (mmax, nmax).

    Oracle: the spec docstring contract.
    """
    surface = SurfaceHenneberg(nfp=2, alpha_fac=1, mmax=2, nmax=2)
    bad_rhomn = np.zeros((surface.mmax + 1, 2 * surface.nmax))  # wrong width
    with pytest.raises(ValueError, match="rhomn shape mismatch"):
        make_surface_henneberg_spec(
            R0nH=np.asarray(surface.R0nH, dtype=np.float64),
            Z0nH=np.asarray(surface.Z0nH, dtype=np.float64),
            bn=np.asarray(surface.bn, dtype=np.float64),
            rhomn=bad_rhomn,
            quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64),
            quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64),
            nfp=surface.nfp,
            alpha_fac=surface.alpha_fac,
            mmax=surface.mmax,
            nmax=surface.nmax,
        )


def test_spec_is_frozen_dataclass() -> None:
    """The spec dataclass must be frozen: attribute assignment raises.

    Oracle: ``@dataclass(frozen=True)`` contract (host-side invariant).
    """
    surface = SurfaceHenneberg(nfp=2, alpha_fac=1, mmax=1, nmax=1)
    spec = surface.to_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        # Direct attribute assignment on a frozen dataclass must fail.
        spec.alpha_fac = 0  # type: ignore[misc]


def test_spec_register_dataclass_keys_match_host_class() -> None:
    """``jax.tree_util`` registration matches the documented (data, meta) layout.

    Oracle: explicit field-list in ``specs.py`` registration call (the
    contract that downstream JIT cache keys depend on). The registered
    ``meta_fields`` are ``("nfp", "alpha_fac", "mmax", "nmax")``, in
    that order.
    """
    surface = SurfaceHenneberg(nfp=2, alpha_fac=1, mmax=1, nmax=1)
    spec = surface.to_spec()
    leaves, treedef = jax.tree_util.tree_flatten(spec)
    # The data_fields list is:
    #   R0nH, Z0nH, bn, rhomn, quadpoints_phi, quadpoints_theta -> 6 leaves
    assert len(leaves) == 6
    # node_data() returns (cls, meta_tuple) where meta_tuple contains the
    # meta_field values in registration order.
    node_data = treedef.node_data()
    assert node_data is not None
    cls, meta_tuple = node_data
    assert cls is SurfaceHennebergSpec
    assert meta_tuple == (2, 1, 1, 1)


def test_spec_jit_round_trip_preserves_meta_fields() -> None:
    """Round-tripping a spec through ``jax.jit`` preserves all meta_fields.

    Oracle: ``jax.tree_util.register_dataclass`` contract — meta_fields
    must persist across a JIT boundary as static compile-time values.
    """
    surface = SurfaceHenneberg(nfp=3, alpha_fac=-1, mmax=2, nmax=2)
    spec = surface.to_spec()

    @jax.jit
    def identity(s: SurfaceHennebergSpec) -> jax.Array:
        return s.R0nH + 0.0  # touches a data_field

    out = np.asarray(identity(spec))
    np.testing.assert_array_equal(out, np.asarray(spec.R0nH))
    # The spec object itself is unchanged.
    assert spec.nfp == 3
    assert spec.alpha_fac == -1
    assert spec.mmax == 2
    assert spec.nmax == 2


# ---------------------------------------------------------------------------
# Section 7 — JIT cache discrimination & strict transfer guard
# ---------------------------------------------------------------------------


def test_jit_cache_discriminates_alpha_fac() -> None:
    """Different ``alpha_fac`` values must compile to different traces.

    Oracle: JAX docs — meta_fields participate in the JIT cache key.
    A change in ``alpha_fac`` (a meta_field) must produce a non-identical
    array, since the α coefficient enters every mode of ρ and ζ. The
    same DOFs with different ``alpha_fac`` must NOT give the same γ.
    """
    jit_kernel = jax.jit(surface_henneberg_gamma_from_spec)

    surface_pos = SurfaceHenneberg(nfp=3, alpha_fac=1, mmax=2, nmax=2)
    rng = np.random.default_rng(2099)
    x = np.asarray(surface_pos.local_full_x, dtype=np.float64)
    x = x + 1e-2 * rng.standard_normal(x.shape)
    surface_pos.local_full_x = x

    surface_neg = SurfaceHenneberg(nfp=3, alpha_fac=-1, mmax=2, nmax=2)
    surface_neg.local_full_x = x  # same DOFs as surface_pos

    gamma_pos = np.asarray(jit_kernel(surface_pos.to_spec()))
    gamma_neg = np.asarray(jit_kernel(surface_neg.to_spec()))
    gamma_pos_cpu = np.asarray(surface_pos.gamma(), dtype=np.float64)
    gamma_neg_cpu = np.asarray(surface_neg.gamma(), dtype=np.float64)

    # Each result must match its own CPU oracle:
    np.testing.assert_allclose(
        gamma_pos, gamma_pos_cpu, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
    np.testing.assert_allclose(
        gamma_neg, gamma_neg_cpu, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )
    # The two outputs must be observably different (else the JIT cache
    # would be silently reusing one compiled function regardless of the
    # alpha_fac meta_field, which would be a bug).
    assert np.max(np.abs(gamma_pos - gamma_neg)) > 1e-6


def test_kernels_run_under_strict_transfer_guard() -> None:
    """All four primary kernels run under ``jax.transfer_guard('disallow')``.

    Oracle: JAX transfer-guard contract — the kernels must keep their
    inputs on-device once the spec has been constructed (host side).
    """
    surface = _seed_surface(nfp=3, alpha_fac=1, mmax=2, nmax=2, seed=2040)
    spec = surface.to_spec()

    jit_gamma = jax.jit(surface_henneberg_gamma_from_spec)
    jit_gd1 = jax.jit(surface_henneberg_gammadash1_from_spec)
    jit_gd2 = jax.jit(surface_henneberg_gammadash2_from_spec)
    jit_normal = jax.jit(surface_henneberg_normal_from_spec)

    with jax.transfer_guard("disallow"):
        gamma = jit_gamma(spec)
        gd1 = jit_gd1(spec)
        gd2 = jit_gd2(spec)
        normal = jit_normal(spec)
        gamma.block_until_ready()
        gd1.block_until_ready()
        gd2.block_until_ready()
        normal.block_until_ready()

    # Smoke: outputs must have the expected shape.
    nphi = len(surface.quadpoints_phi)
    ntheta = len(surface.quadpoints_theta)
    assert gamma.shape == (nphi, ntheta, 3)
    assert gd1.shape == (nphi, ntheta, 3)
    assert gd2.shape == (nphi, ntheta, 3)
    assert normal.shape == (nphi, ntheta, 3)


# ---------------------------------------------------------------------------
# Section 8 — Spec is a pytree (can be jax.jit'd and tree-mapped)
# ---------------------------------------------------------------------------


def test_spec_can_be_jitted_through_pytree_registration() -> None:
    """The spec round-trips through ``jax.tree_util.tree_map``.

    Oracle: ``jax.tree_util.register_dataclass`` contract — every
    data_field leaf must survive a tree-map, and the meta_fields must
    be preserved on the rebuilt instance.
    """
    surface = _seed_surface(nfp=2, alpha_fac=1, mmax=1, nmax=1, seed=2050)
    spec = surface.to_spec()

    def negate_then_negate(x: jax.Array) -> jax.Array:
        # On-device only: no Python scalar promotion, no host->device transfer.
        return -(-x)

    mapped = jax.tree_util.tree_map(negate_then_negate, spec)
    assert isinstance(mapped, SurfaceHennebergSpec)
    assert mapped.nfp == spec.nfp
    assert mapped.alpha_fac == spec.alpha_fac
    assert mapped.mmax == spec.mmax
    assert mapped.nmax == spec.nmax
    np.testing.assert_array_equal(np.asarray(mapped.rhomn), np.asarray(spec.rhomn))
    np.testing.assert_array_equal(np.asarray(mapped.R0nH), np.asarray(spec.R0nH))


# ---------------------------------------------------------------------------
# Section 9 — Non-default quadpoint grids
# ---------------------------------------------------------------------------


def test_gamma_parity_with_custom_quadpoints() -> None:
    """Parity holds for non-default ``quadpoints_phi`` / ``quadpoints_theta``.

    Oracle: CPU ``SurfaceHenneberg.gamma()`` on a fine custom grid.
    """
    qp_phi = np.linspace(0.0, 1.0 / 3, 17, endpoint=False)
    qp_theta = np.linspace(0.0, 1.0, 19, endpoint=False)
    surface = SurfaceHenneberg(
        nfp=3,
        alpha_fac=1,
        mmax=2,
        nmax=2,
        quadpoints_phi=qp_phi,
        quadpoints_theta=qp_theta,
    )
    rng = np.random.default_rng(2060)
    x = np.asarray(surface.local_full_x, dtype=np.float64)
    x = x + 1e-2 * rng.standard_normal(x.shape)
    surface.local_full_x = x

    gamma_cpu = np.asarray(surface.gamma(), dtype=np.float64)
    spec = surface.to_spec()
    gamma_jax = np.asarray(jax.jit(surface_henneberg_gamma_from_spec)(spec))

    assert gamma_jax.shape == (len(qp_phi), len(qp_theta), 3)
    np.testing.assert_allclose(
        gamma_jax, gamma_cpu, rtol=_PARITY_RTOL, atol=_PARITY_ATOL
    )


def test_gammadash1_parity_with_custom_quadpoints() -> None:
    """``gammadash1`` parity on a custom grid.

    Oracle: CPU ``SurfaceHenneberg.gammadash1()``.
    """
    qp_phi = np.linspace(0.0, 1.0 / 2, 11, endpoint=False)
    qp_theta = np.linspace(0.0, 1.0, 13, endpoint=False)
    surface = SurfaceHenneberg(
        nfp=2,
        alpha_fac=1,
        mmax=2,
        nmax=2,
        quadpoints_phi=qp_phi,
        quadpoints_theta=qp_theta,
    )
    rng = np.random.default_rng(2061)
    x = np.asarray(surface.local_full_x, dtype=np.float64)
    x = x + 1e-2 * rng.standard_normal(x.shape)
    surface.local_full_x = x

    expected = np.asarray(surface.gammadash1(), dtype=np.float64)
    spec = surface.to_spec()
    actual = np.asarray(jax.jit(surface_henneberg_gammadash1_from_spec)(spec))

    np.testing.assert_allclose(actual, expected, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)
