"""Regression coverage for the §7 in-bundle quadrature canonicalization.

The plan at
``.artifacts/jax-silent-fallback-removal-2026-05-13/PLAN.md`` (§7,
"Surface quadrature -- in-bundle canonicalization") records that
``_canonicalize_traceable_exact_quadrature`` in
``src/simsopt/geo/surfaceobjectives_jax.py`` replaced a silent
``try/except ValueError`` rescue with one explicit detection step.
The contract is:

* Stellsym phi grids matching
  ``Surface.get_phi_quadpoints(nphi, RANGE_HALF_PERIOD, nfp)`` are
  substituted with the unshifted canonical ``half_phi`` / ``full_theta``
  family so the downstream
  ``_compute_stellsym_mask_indices_for_grid`` mask builder accepts them.
* Stellsym grids matching the canonical ``half_phi`` / ``full_theta``
  family already pass through ``_compute_stellsym_mask_indices_for_grid``
  unchanged.
* Stellsym grids that are neither canonical nor the named VMEC
  half-period shifted family raise ``ValueError`` from
  ``_compute_stellsym_mask_indices_for_grid``. No silent rescue.
* Non-stellsym grids are not subject to canonicalization at all.

These tests pin the canonicalization function as a focused unit test
(``types.SimpleNamespace`` stand-in for the BoozerSurfaceJAX inputs the
function reads) so the loud-failure contract cannot regress to a silent
``try/except``.
"""

import types

import numpy as np
import pytest

from simsopt.geo._surface_stellsym import (
    compute_stellsym_mask_indices_for_grid,
)
from simsopt.geo.surface import Surface
from simsopt.geo.surfaceobjectives_jax import (
    _canonicalize_traceable_exact_quadrature,
)


def _canonical_half_phi(*, nfp, ntor):
    """Canonical unshifted half-period phi grid expected by the mask."""
    return np.linspace(0.0, 0.5 / float(nfp), ntor + 1, endpoint=False)


def _canonical_full_theta(*, mpol):
    """Canonical unshifted full-theta grid expected by the mask."""
    return np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)


def _make_fake_booz(*, mpol, ntor, nfp, stellsym, quadpoints_phi, quadpoints_theta):
    """SimpleNamespace stand-in exposing the booz_jax attributes the function reads."""
    return types.SimpleNamespace(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=np.asarray(quadpoints_phi, dtype=float),
        quadpoints_theta=np.asarray(quadpoints_theta, dtype=float),
    )


def test_canonical_half_phi_full_theta_grid_passes_through():
    """Canonical unshifted half_phi / full_theta inputs are not substituted."""
    mpol, ntor, nfp = 3, 2, 4
    half_phi = _canonical_half_phi(nfp=nfp, ntor=ntor)
    full_theta = _canonical_full_theta(mpol=mpol)

    booz = _make_fake_booz(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=half_phi,
        quadpoints_theta=full_theta,
    )

    quadpoints_phi, quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz)
    )

    np.testing.assert_array_equal(np.asarray(quadpoints_phi), half_phi)
    np.testing.assert_array_equal(np.asarray(quadpoints_theta), full_theta)
    assert np.asarray(mask_indices).ndim == 1
    assert np.asarray(mask_indices).size > 0


def test_vmec_half_period_shifted_grid_canonicalized():
    """The named VMEC ``RANGE_HALF_PERIOD`` family is rewritten to canonical."""
    mpol, ntor, nfp = 2, 2, 5
    nphi = 31  # nphi differs from ntor + 1; canonicalization must rewrite it.

    shifted_phi = np.asarray(
        Surface.get_phi_quadpoints(
            nphi=nphi,
            range=Surface.RANGE_HALF_PERIOD,
            nfp=nfp,
        ),
        dtype=float,
    )
    shifted_theta = np.asarray(
        Surface.get_theta_quadpoints(ntheta=16),
        dtype=float,
    )
    # Sanity: the production formula really is half-cell shifted, so the
    # detection step is exercising the documented family rather than an
    # already-canonical grid.
    assert shifted_phi[0] != 0.0

    booz = _make_fake_booz(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=shifted_phi,
        quadpoints_theta=shifted_theta,
    )

    quadpoints_phi, quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz)
    )

    np.testing.assert_allclose(
        np.asarray(quadpoints_phi),
        _canonical_half_phi(nfp=nfp, ntor=ntor),
    )
    np.testing.assert_allclose(
        np.asarray(quadpoints_theta),
        _canonical_full_theta(mpol=mpol),
    )
    assert np.asarray(mask_indices).ndim == 1
    assert np.asarray(mask_indices).size > 0


def test_unknown_non_canonical_phi_grid_raises():
    """Unrecognized stellsym grids surface ValueError from the mask builder."""
    mpol, ntor, nfp = 2, 2, 5
    arbitrary_phi = np.array([0.0, 0.1, 0.2, 0.4], dtype=float)
    full_theta = _canonical_full_theta(mpol=mpol)

    booz = _make_fake_booz(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=arbitrary_phi,
        quadpoints_theta=full_theta,
    )

    with pytest.raises(ValueError):
        _canonicalize_traceable_exact_quadrature(booz)


def test_non_stellsym_passes_through_without_substitution():
    """Non-stellsym grids are not canonicalized; shifted VMEC grid flows through."""
    mpol, ntor, nfp = 2, 2, 5
    shifted_phi = np.asarray(
        Surface.get_phi_quadpoints(
            nphi=31,
            range=Surface.RANGE_HALF_PERIOD,
            nfp=nfp,
        ),
        dtype=float,
    )
    shifted_theta = np.asarray(
        Surface.get_theta_quadpoints(ntheta=16),
        dtype=float,
    )

    booz = _make_fake_booz(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=False,
        quadpoints_phi=shifted_phi,
        quadpoints_theta=shifted_theta,
    )

    quadpoints_phi, quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz)
    )

    np.testing.assert_array_equal(np.asarray(quadpoints_phi), shifted_phi)
    np.testing.assert_array_equal(np.asarray(quadpoints_theta), shifted_theta)
    # Non-stellsym mask is full (no symmetry reduction, no (0,0,0) drop).
    assert np.asarray(mask_indices).size == shifted_phi.size * shifted_theta.size * 3


def test_canonicalized_grid_matches_compute_stellsym_mask_indices_for_grid():
    """The canonicalized grid is what the downstream mask builder accepts.

    Without canonicalization the shifted VMEC grid raises in
    ``compute_stellsym_mask_indices_for_grid``; once canonicalized, the
    same builder produces the identical mask indices the canonicalization
    helper returns.
    """
    mpol, ntor, nfp = 2, 2, 5

    shifted_phi = np.asarray(
        Surface.get_phi_quadpoints(
            nphi=31,
            range=Surface.RANGE_HALF_PERIOD,
            nfp=nfp,
        ),
        dtype=float,
    )
    shifted_theta = np.asarray(
        Surface.get_theta_quadpoints(ntheta=16),
        dtype=float,
    )

    # Without the in-bundle canonicalization, the loud builder rejects
    # the shifted grid directly. This pins the precondition for §7.
    with pytest.raises(ValueError):
        compute_stellsym_mask_indices_for_grid(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=True,
            quadpoints_phi=shifted_phi,
            quadpoints_theta=shifted_theta,
        )

    booz = _make_fake_booz(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=shifted_phi,
        quadpoints_theta=shifted_theta,
    )
    quadpoints_phi, quadpoints_theta, mask_indices = (
        _canonicalize_traceable_exact_quadrature(booz)
    )

    # Same call against the canonical grids the helper returned must
    # succeed and produce byte-identical mask indices.
    direct_mask = compute_stellsym_mask_indices_for_grid(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=np.asarray(quadpoints_phi),
        quadpoints_theta=np.asarray(quadpoints_theta),
    )

    np.testing.assert_array_equal(
        np.asarray(mask_indices),
        np.asarray(direct_mask),
    )
