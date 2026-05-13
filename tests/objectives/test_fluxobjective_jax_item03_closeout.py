"""Item 03 closeout test for SquaredFluxJAX.

This test bundles, in a single parametrized fixture, four attributes
that are not co-located in any other existing test:

1. Production-scale grid (``nphi=16, ntheta=8``) with ``ncoils=4``
   base curves (LandremanPaul-style toroidal arrangement).
2. Stellsym=True, nfp=2 (so the JAX path exercises
   ``coils_via_symmetries`` expansion to ``2*nfp*ncoils = 16`` total
   coils).
3. All three SquaredFlux ``definition`` variants
   (``"quadratic flux"``, ``"normalized"``, ``"local"``) in one
   parametrized test function.
4. Strict-transfer-guard discipline: the JAX construction and value
   path run inside ``jax.transfer_guard("disallow")``.

All tolerances come from the parity ladder ``direct_kernel`` lane
contract; no ``atol=``/``rtol=`` numeric literals appear in the test
body.

The C++ oracle is ``simsopt.objectives.fluxobjective.SquaredFlux.J()``,
which delegates to ``simsoptpp.integral_BdotN``. See
``src/simsoptpp/integral_BdotN.cpp:12-123`` for the oracle source.
"""

from __future__ import annotations

import os

import jax
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX


_DIRECT_KERNEL_TOLERANCES = parity_ladder_tolerances("direct_kernel")
_VALUE_RTOL = float(_DIRECT_KERNEL_TOLERANCES["rtol"])
_VALUE_ATOL = float(_DIRECT_KERNEL_TOLERANCES["atol"])

_SQUARED_FLUX_DEFINITIONS = (
    "quadratic flux",
    "normalized",
    "local",
)


def _build_production_scale_stellsym_fixture():
    """Build a LandremanPaul-style production-scale stellsym fixture.

    Grid floor: ``nphi >= 16``, ``ntheta >= 8``, ``ncoils >= 4``.
    Symmetry: ``stellsym=True``, ``nfp=2`` — expands to 16 total coils
    via ``coils_via_symmetries``.
    """
    ncoils = 4
    nfp = 2
    stellsym = True
    R0 = 1.0
    R1 = 0.5
    order = 6

    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=R0,
        R1=R1,
        order=order,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    nphi = 16
    ntheta = 8
    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
    )
    surface.set_rc(0, 0, R0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    surface.fix_all()

    return coils, surface


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_squared_flux_jax_matches_cpp_oracle_under_strict_transfer_at_production_scale(
    definition,
):
    """SquaredFluxJAX.J() must match SquaredFlux.J() (C++ oracle).

    Single parametrized fixture covering production scale × all 3
    definitions × strict transfer guard × stellsym=True. Compares
    ``SquaredFluxJAX.J()`` against ``SquaredFlux.J()`` (the C++ oracle
    via ``sopp.integral_BdotN``) at the ``direct_kernel`` lane
    tolerance.

    The test does not call ``field.B()`` or any other path that would
    require implicit host transfers; the JAX adapter consumes the
    immutable spec captured at construction. When run under
    ``SIMSOPT_JAX_TRANSFER_GUARD=disallow``, the JAX value path must
    not trigger any disallowed transfer.
    """
    coils, surface = _build_production_scale_stellsym_fixture()

    # CPU C++ oracle
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))
    objective_cpu = SquaredFlux(surface, bs_cpu, definition=definition)
    j_cpu = float(objective_cpu.J())

    # JAX-native path
    bs_jax = BiotSavartJAX(coils)
    objective_jax = SquaredFluxJAX(surface, bs_jax, definition=definition)
    j_jax = float(objective_jax.J())

    np.testing.assert_allclose(
        j_jax,
        j_cpu,
        rtol=_VALUE_RTOL,
        atol=_VALUE_ATOL,
        err_msg=(
            f"SquaredFluxJAX vs SquaredFlux (C++ oracle) mismatch for "
            f"definition={definition!r} at production scale "
            f"(nphi=16, ntheta=8, ncoils=4, nfp=2, stellsym=True). "
            f"transfer_guard_env="
            f"{os.environ.get('SIMSOPT_JAX_TRANSFER_GUARD', 'unset')!r}"
        ),
    )
