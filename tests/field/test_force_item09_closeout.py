"""Item 09 closeout: production-scale LpCurveForce Taylor parity test.

Closes documentation/coverage for `src/simsopt/field/force.py` per the
JAX-port goal prompt item 09. Imports lane tolerances from
`benchmarks.validation_ladder_contract.parity_ladder_tolerances` so the
parity assertion lives on the documented contract instead of inline
numeric literals. Runs under strict transfer-guard discipline at
``ncoils=4`` base coils (expanded to 24 via ``coils_via_symmetries``)
with ``numquadpoints=64`` per coil.
"""

from __future__ import annotations

import jax
import numpy as np

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import (
    Current,
    LpCurveForce,
    coils_via_symmetries,
)
from simsopt.field.selffield import regularization_circ
from simsopt.geo.curve import create_equally_spaced_curves

_FD_GRADIENT = parity_ladder_tolerances("fd_gradient")
_FD_RTOL = _FD_GRADIENT["directional_fd_rtol"]
_FD_ATOL = _FD_GRADIENT["directional_fd_atol"]
_FD_FLOOR = _FD_GRADIENT["directional_derivative_floor"]
_FD_SEED = _FD_GRADIENT["direction_seed"]


def _build_lp_curve_force_objective() -> LpCurveForce:
    """Build a production-scale LpCurveForce fixture.

    ncoils=4 base coils expanded by coils_via_symmetries with nfp=3,
    stellsym=True (=> 24 expanded coils), numquadpoints=64 each. The
    target is base_coils[0]; sources are the full expanded set with
    overlapping target removed inside the wrapper.
    """
    nfp = 3
    ncoils = 4
    current_amplitude = 1.7e4
    numquadpoints = 64
    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=True,
        numquadpoints=numquadpoints,
    )
    base_currents = [Current(current_amplitude) for _ in range(ncoils)]
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        nfp,
        stellsym=True,
        regularizations=[regularization_circ(0.05)] * ncoils,
    )
    return LpCurveForce(
        coils[0],
        coils,
        p=2.5,
        threshold=1.0e-3,
    )


def test_lp_curve_force_production_scale_taylor_parity_under_strict_transfer_guard():
    """Production-scale Taylor parity for LpCurveForce.

    Oracle: central two-step finite-difference directional derivative of
    `obj.J()` along a fixed random direction with a documented seed.
    Compared against the analytic `obj.dJ()` projection onto the same
    direction. The forward `J()` path is re-evaluated under
    `jax.transfer_guard("disallow")` after the host fixture is built so
    no implicit host-to-device transfer is permitted during the compiled
    boundary. The reverse `dJ()` path runs outside the strict guard
    because the public `Optimizable` derivative projection
    (`_assemble_curve_current_derivative` at `force.py:781`) iterates
    over per-coil derivative slices in Python with `dgamma[i]` indexing,
    which crosses a host-to-device boundary by construction. The
    matching subprocess smoke
    `tests/subprocess/import_smoke_cases.py::case_transfer_guard_disallow_allows_lpcurveforce_shared_state_packing`
    also exercises `J()` under `transfer_guard("disallow")` but not
    `dJ()`; this test follows the same boundary discipline.
    """
    objective = _build_lp_curve_force_objective()
    x0 = np.asarray(objective.x, dtype=np.float64).copy()
    assert x0.size > 0

    with jax.transfer_guard("disallow"):
        value = objective.J()
        value.block_until_ready()
    gradient = np.asarray(objective.dJ(), dtype=np.float64)

    value_host = float(value)
    assert np.isfinite(value_host)
    assert gradient.shape == (x0.size,)
    assert np.isfinite(gradient).all()

    rng = np.random.default_rng(int(_FD_SEED))
    direction = rng.standard_normal(x0.size)
    direction = direction / np.linalg.norm(direction)
    analytic_directional = float(np.dot(gradient, direction))

    assert abs(analytic_directional) > float(_FD_FLOOR), (
        "directional derivative below lane floor; choose a different seed "
        "or expand fixture so that the FD direction is not degenerate"
    )

    eps = 1.0e-5
    objective.x = x0 + eps * direction
    j_plus = float(objective.J())
    objective.x = x0 - eps * direction
    j_minus = float(objective.J())
    objective.x = x0
    fd_directional = (j_plus - j_minus) / (2.0 * eps)

    np.testing.assert_allclose(
        fd_directional,
        analytic_directional,
        rtol=float(_FD_RTOL),
        atol=float(_FD_ATOL),
    )
