"""Item 17 closeout parity test for ``simsopt.field.normal_field``.

``src/simsopt/field/normal_field.py`` is byte-identical to upstream SIMSOPT
SHA ``1b0cc3a96063197cdbdd01559e04c25456fbe6ff`` and uses no JAX. This
closeout file locks in five complementary CPU-oracle invariants at the
``direct_kernel`` parity-ladder lane (tolerances imported from
``parity_ladder_tolerances``) under both the explicit
``jax.transfer_guard("disallow")`` context AND the process-wide
``SIMSOPT_JAX_TRANSFER_GUARD=disallow`` discipline:

1. ``test_fourier_pair_identity_at_production_scale`` — Round-trip
   ``surface.fourier_transform_scalar(surface.inverse_fourier_transform_scalar(*))``
   reproduces an arbitrary ``(Vns, Vnc)`` bit-tight on the production-scale
   surface grid for both ``stellsym=True`` and ``stellsym=False``.
   Hand-derived from the upstream docstring at
   ``src/simsopt/geo/surfacerzfourier.py:2249-2253`` which guarantees the
   inverse-Fourier-transform / Fourier-transform pair is identity on the
   band-limited mode set.

2. ``test_coil_normal_field_vns_vnc_match_direct_cpu_oracle`` —
   ``CoilNormalField.vns`` / ``.vnc`` reproduce a hand-rolled CPU oracle
   built from ``BiotSavart.B()`` reshaped to ``(nphi, ntheta, 3)``,
   reduced as ``np.sum(B * surface.normal() * -1, axis=2)``, and
   transformed by ``surface.fourier_transform_scalar`` at the
   SPEC-convention ``(2*pi)**2`` normalization. Bit-tight at production
   scale for both symmetry branches.

3. ``test_coil_normal_field_recompute_bell_invalidates_cache`` — Changing
   a ``CoilSet`` DOF clears ``_vns`` / ``_vnc`` and the next access
   reproduces the new hand-rolled oracle bit-tight. Catches a regression
   where the property cache fails to invalidate.

4. ``test_coil_normal_field_negative_control_wrong_sign_breaks_parity`` —
   Negative control: dropping the ``-1`` reduction sign produces a
   tolerance-busting Vns deviation, proving the parity gate would catch a
   wrong-sign reduction.

5. ``test_normal_field_get_real_space_field_matches_hand_rolled_formula`` —
   ``NormalField.get_real_space_field()`` reproduces the hand-rolled
   ``-1 * inverse_fourier_transform_scalar(...) / |surface.normal()|``
   formula bit-tight. Validates the public real-space accessor for both
   symmetry branches.

Tolerances are imported from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances("direct_kernel")``;
no ``atol`` / ``rtol`` literal appears inside the test bodies.
"""

import jax
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import (
    Coil,
    CoilNormalField,
    CoilSet,
    Current,
    NormalField,
)
from simsopt.geo import SurfaceRZFourier


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

# Production-scale fixture floor (section 4c): nphi >= 16, ntheta >= 8,
# ncoils >= 4. The toroidal symmetry expansion (nfp=2,
# coils_per_period=4) yields 2 * 4 = 8 coils total after
# ``coils_via_symmetries``, well above the floor.
_NFP = 2
_NPHI = 32
_NTHETA = 16
_COILS_PER_PERIOD = 4
_CURRENT_A = 1.0e5
_COIL_ORDER = 4
_DOF_BUMP = 1.0e-3
_VNS_AMPLITUDE = 1.0e-3
_NEGATIVE_CONTROL_REL_THRESHOLD = 1.0e-2

_SYMMETRY_FIXTURES = (
    pytest.param(True, 4, 3, id="stellsym_true_mpol4_ntor3"),
    pytest.param(False, 3, 2, id="stellsym_false_mpol3_ntor2"),
)


def _build_surface(stellsym, mpol, ntor):
    """Production-scale SurfaceRZFourier matching the fixture floor."""
    return SurfaceRZFourier(
        nfp=_NFP,
        stellsym=stellsym,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=np.linspace(0.0, 1.0, _NPHI, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, _NTHETA, endpoint=False),
    )


def _build_coilset(surface):
    """Production-scale CoilSet built from circular coils around the surface."""
    base_curves = CoilSet._circlecurves_around_surface(
        surface,
        coils_per_period=_COILS_PER_PERIOD,
        order=_COIL_ORDER,
    )
    base_coils = [Coil(curve, Current(_CURRENT_A)) for curve in base_curves]
    return CoilSet(base_coils=base_coils, surface=surface)


def _populate_admissible_harmonics(stellsym, mpol, ntor, seed):
    """Build (Vns, Vnc) arrays populated only on the admissible mode set.

    Sine series excludes ``(m=0, n<=0)``; cosine series excludes
    ``(m=0, n<0)``. Modes outside the admissible set are zeroed so the
    round-trip identity is well-defined.
    """
    rng = np.random.default_rng(seed)
    Vns_in = np.zeros((mpol + 1, 2 * ntor + 1), dtype=np.float64)
    Vnc_in = np.zeros((mpol + 1, 2 * ntor + 1), dtype=np.float64)
    for m in range(0, mpol + 1):
        for n in range(-ntor, ntor + 1):
            if not (m == 0 and n <= 0):
                Vns_in[m, n + ntor] = float(rng.normal()) * _VNS_AMPLITUDE
            if not stellsym and not (m == 0 and n < 0):
                Vnc_in[m, n + ntor] = float(rng.normal()) * _VNS_AMPLITUDE
    return Vns_in, Vnc_in


def _coil_normal_field_direct_oracle(cnf):
    """Hand-rolled CPU oracle for ``CoilNormalField.vns`` / ``.vnc``.

    Reproduces the SSOT formula at
    ``src/simsopt/field/normal_field.py:576-577`` and lines 589-590:
    ``np.sum(coilset.bs.B().reshape((nphi, ntheta, 3)) * surface.normal()
    * -1, axis=2)`` reduced through
    ``surface.fourier_transform_scalar(...)`` at the
    SPEC-convention ``(2*pi)**2`` normalization.
    """
    phisize = cnf.surface.quadpoints_phi.size
    thetasize = cnf.surface.quadpoints_theta.size
    B = cnf.coilset.bs.B().reshape((phisize, thetasize, 3))
    bn_unnormalized = np.sum(B * cnf.surface.normal() * -1.0, axis=2)
    return cnf.surface.fourier_transform_scalar(
        bn_unnormalized,
        normalization=(2.0 * np.pi) ** 2,
        stellsym=cnf.stellsym,
    )


@pytest.mark.parametrize("stellsym, mpol, ntor", _SYMMETRY_FIXTURES)
def test_fourier_pair_identity_at_production_scale(stellsym, mpol, ntor):
    """SurfaceRZFourier IFT . FT round-trip is identity on the
    band-limited mode set at production scale.

    Oracle source: upstream docstring at
    ``src/simsopt/geo/surfacerzfourier.py:2249-2253`` documents that
    ``inverse_fourier_transform_scalar(fourier_transform_scalar(*))`` is
    identity when the input is band-limited to ``(mpol, ntor)``. The
    closeout exercises the equivalent ``FT . IFT`` direction on an
    admissible ``(Vns, Vnc)`` mode set.
    """
    surface = _build_surface(stellsym, mpol, ntor)
    seed = 1729 + (1 if stellsym else 0)
    Vns_in, Vnc_in = _populate_admissible_harmonics(stellsym, mpol, ntor, seed)

    with jax.transfer_guard("disallow"):
        bn_real = surface.inverse_fourier_transform_scalar(
            Vns_in,
            Vnc_in,
            normalization=(2.0 * np.pi) ** 2,
            stellsym=stellsym,
        )
        Vns_back, Vnc_back = surface.fourier_transform_scalar(
            bn_real,
            normalization=(2.0 * np.pi) ** 2,
            stellsym=stellsym,
        )

    np.testing.assert_allclose(Vns_back, Vns_in, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(Vnc_back, Vnc_in, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("stellsym, mpol, ntor", _SYMMETRY_FIXTURES)
def test_coil_normal_field_vns_vnc_match_direct_cpu_oracle(stellsym, mpol, ntor):
    """``CoilNormalField.vns`` and ``.vnc`` match the hand-rolled CPU
    oracle bit-tight at production scale for both symmetry branches.
    """
    surface = _build_surface(stellsym, mpol, ntor)
    coilset = _build_coilset(surface)
    cnf = CoilNormalField(coilset)

    with jax.transfer_guard("disallow"):
        Vns_direct, Vnc_direct = _coil_normal_field_direct_oracle(cnf)
        Vns_cnf = cnf.vns
        Vnc_cnf = cnf.vnc

    np.testing.assert_allclose(Vns_cnf, Vns_direct, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(Vnc_cnf, Vnc_direct, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("stellsym, mpol, ntor", _SYMMETRY_FIXTURES)
def test_coil_normal_field_recompute_bell_invalidates_cache(stellsym, mpol, ntor):
    """``CoilNormalField`` cache invalidates on parent DOF change.

    Verifies the recompute-bell contract documented at
    ``src/simsopt/field/normal_field.py:639-641``. After modifying a
    ``CoilSet`` DOF, the next ``cnf.vns`` access must reproduce the
    new hand-rolled oracle bit-tight.
    """
    surface = _build_surface(stellsym, mpol, ntor)
    coilset = _build_coilset(surface)
    cnf = CoilNormalField(coilset)

    with jax.transfer_guard("disallow"):
        # Prime the cache.
        _ = cnf.vns
        assert cnf._vns is not None
        assert cnf._vnc is not None

        # Bump a single DOF; this rings the ``Optimizable`` recompute bell.
        new_x = np.array(coilset.x, copy=True)
        new_x[0] += _DOF_BUMP
        coilset.x = new_x

        # The cache must be cleared by ``recompute_bell``.
        assert cnf._vns is None
        assert cnf._vnc is None

        Vns_direct, Vnc_direct = _coil_normal_field_direct_oracle(cnf)
        Vns_after = cnf.vns
        Vnc_after = cnf.vnc

    np.testing.assert_allclose(Vns_after, Vns_direct, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(Vnc_after, Vnc_direct, rtol=_RTOL, atol=_ATOL)


def test_coil_normal_field_negative_control_wrong_sign_breaks_parity():
    """Negative control: dropping the ``-1`` reduction sign yields a
    tolerance-busting Vns deviation.

    Catches a regression where ``CoilNormalField.vns`` flips the SPEC
    sign convention or drops the multiplicative ``-1`` in the
    ``np.sum(B * surface.normal() * -1, axis=2)`` reduction.
    """
    surface = _build_surface(True, 4, 3)
    coilset = _build_coilset(surface)
    cnf = CoilNormalField(coilset)

    with jax.transfer_guard("disallow"):
        Vns_correct = np.array(cnf.vns, copy=True)
        phisize = surface.quadpoints_phi.size
        thetasize = surface.quadpoints_theta.size
        B = cnf.coilset.bs.B().reshape((phisize, thetasize, 3))
        # Hand-rolled oracle WITHOUT the upstream ``-1`` sign.
        bn_wrong = np.sum(B * surface.normal(), axis=2)
        Vns_wrong, _Vnc_wrong = surface.fourier_transform_scalar(
            bn_wrong,
            normalization=(2.0 * np.pi) ** 2,
            stellsym=True,
        )

    correct_norm = float(np.max(np.abs(Vns_correct)))
    assert correct_norm > 0.0, (
        "Negative control fixture has degenerate Vns: cnf.vns is zero. "
        "The hand-rolled negative control would silently pass; the "
        "fixture must produce a non-trivial Vns scale."
    )
    relative_deviation = float(np.max(np.abs(Vns_wrong - Vns_correct)) / correct_norm)
    assert relative_deviation > _NEGATIVE_CONTROL_REL_THRESHOLD, (
        "Negative control failed: dropping the ``-1`` reduction sign "
        f"produced a relative Vns deviation of {relative_deviation:.3e}, "
        f"which is below the threshold of "
        f"{_NEGATIVE_CONTROL_REL_THRESHOLD:.3e}. The CoilNormalField "
        "reduction must be tied to the SPEC sign convention."
    )


@pytest.mark.parametrize("stellsym, mpol, ntor", _SYMMETRY_FIXTURES)
def test_normal_field_get_real_space_field_matches_hand_rolled_formula(
    stellsym, mpol, ntor
):
    """``NormalField.get_real_space_field()`` reproduces the hand-rolled
    ``-1 * inverse_fourier_transform_scalar(...) / |normal|`` formula
    bit-tight.

    Locks in the SSOT for the public real-space accessor at
    ``src/simsopt/field/normal_field.py:510-519`` against the underlying
    Fourier-pair helper at
    ``src/simsopt/geo/surfacerzfourier.py:2269-2323``.
    """
    surface = _build_surface(stellsym, mpol, ntor)
    nf = NormalField(
        nfp=_NFP,
        stellsym=stellsym,
        mpol=mpol,
        ntor=ntor,
        surface=surface,
    )

    rng = np.random.default_rng(2026 + (1 if stellsym else 0))
    for m in range(0, nf.mpol + 1):
        for n in range(-nf.ntor, nf.ntor + 1):
            if m == 0 and n < 0:
                continue
            if not (m == 0 and n == 0):
                nf.set_vns(m, n, float(rng.normal()) * _VNS_AMPLITUDE)
            if not stellsym:
                nf.set_vnc(m, n, float(rng.normal()) * _VNS_AMPLITUDE)

    with jax.transfer_guard("disallow"):
        real_space_via_api = nf.get_real_space_field()
        vns, vnc = nf.get_vns_vnc_asarray()
        bn_unnormalized = surface.inverse_fourier_transform_scalar(
            vns,
            vnc,
            normalization=(2.0 * np.pi) ** 2,
            stellsym=stellsym,
        )
        real_space_hand = (
            -1.0 * bn_unnormalized / np.linalg.norm(surface.normal(), axis=-1)
        )

    np.testing.assert_allclose(
        real_space_via_api,
        real_space_hand,
        rtol=_RTOL,
        atol=_ATOL,
    )
