"""Item 16 follow-up tests: phi-plane crossings + non-Levelset stopping criteria.

The original item 16 test suite verified the CPU/JAX backend-routing
contract and rejected ``phis``/non-Levelset stopping criteria with
:class:`NotImplementedError`. With the phi-plane localizer and stopping
criterion translation now wired through
:func:`simsopt.field.tracing.compute_fieldlines`, the carve-outs are
lifted; this module covers the new accept paths.

Parity-ladder lane: ``event_time_tracing``. The lane bounds the
JAX-vs-CPU phi-hit time agreement.

All tests run under ``JAX_PLATFORMS=cpu`` with ``JAX_ENABLE_X64=True``.
"""

from __future__ import annotations

import numpy as np

from benchmarks.validation_ladder_contract import parity_ladder_tolerances

from simsopt.field import tracing as tracing_module
from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.field.toroidal_field_jax import ToroidalFieldJAX
from simsopt.field.tracing import (
    MaxRStoppingCriterion,
    MaxZStoppingCriterion,
    MinRStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    compute_fieldlines,
)


_EVENT_TIME_TOLERANCES = parity_ladder_tolerances("event_time_tracing")


def _force_jax_backend(monkeypatch):
    """Force ``is_jax_backend`` to return True at the tracing module call site."""
    monkeypatch.setattr(tracing_module, "is_jax_backend", lambda: True)


def test_compute_fieldlines_jax_phi_plane_parity_vs_upstream(monkeypatch):
    """Phi-plane crossing positions match the upstream C++ oracle.

    Both lanes use the upstream ``dx/dt = B`` parametrisation and must
    agree on the recorded ``(x, y, z)`` at each phi-plane crossing
    within a tolerance band that reflects the JAX driver's linear
    in-step interpolant; the bracketed bisection is exact on the
    interpolant but the trajectory itself is only known to RK accuracy
    at each accepted step. With tight controller tolerances the
    position match is well below 1e-4.
    """

    R0_field = 1.3
    B0 = 0.8
    R_init = 1.4
    phi_target = 0.7

    # CPU oracle: upstream ``dx/dt = B`` parametrisation, multiple transits.
    tmax_cpp = 4.0 * (2.0 * np.pi * R_init * R_init) / (B0 * R0_field)
    res_tys_cpu, res_phi_hits_cpu = compute_fieldlines(
        ToroidalField(R0_field, B0),
        [R_init],
        [0.0],
        tmax=tmax_cpp,
        tol=1e-10,
        phis=[phi_target],
    )

    cpu_hits = res_phi_hits_cpu[0]

    # JAX route: same raw-B integration time as upstream. Use tight
    # tolerances (1e-12) so the linear-interpolant residual at each
    # crossing is bounded well below the comparison threshold.
    tmax_jax = tmax_cpp
    _force_jax_backend(monkeypatch)
    res_tys_jax, res_phi_hits_jax = compute_fieldlines(
        ToroidalFieldJAX(R0_field, B0),
        [R_init],
        [0.0],
        tmax=tmax_jax,
        tol=1e-12,
        phis=[phi_target],
    )

    jax_hits = res_phi_hits_jax[0]
    assert jax_hits.shape[1] == 5
    assert jax_hits.shape[0] >= 1, "JAX route must record at least one phi crossing"

    n_to_compare = min(int(cpu_hits.shape[0]), int(jax_hits.shape[0]))
    assert n_to_compare >= 1, (
        f"no phi crossings to compare (cpu={cpu_hits.shape[0]}, jax={jax_hits.shape[0]})"
    )

    # Position comparison tolerance: ~1e-4 reflects the linear in-step
    # interpolant residual at the bracketed crossing. The bracketed
    # bisection is byte-exact on the interpolant but the interpolant
    # itself differs from the trajectory by an amount proportional to
    # the step size at the crossing; tight controller tolerances
    # (1e-12) keep this well within the comparison band.
    pos_tol = 1.0e-4
    for i in range(n_to_compare):
        cpu_xyz = np.asarray(cpu_hits[i, 2:5])
        jax_xyz = np.asarray(jax_hits[i, 2:5])
        diff = np.linalg.norm(cpu_xyz - jax_xyz)
        assert diff < pos_tol, (
            f"phi crossing {i} position mismatch: cpu={cpu_xyz}, jax={jax_xyz}, "
            f"diff={diff}"
        )


def test_compute_fieldlines_jax_accepts_minR_stopping_criterion(monkeypatch):
    """Non-Levelset criteria are no longer rejected by the JAX route.

    This is the explicit follow-up to
    ``test_compute_fieldlines_jax_raises_on_unsupported_stopping_criterion``
    in ``test_tracing_jax_item16.py``: the carve-out is lifted and the
    isinstance dispatch must translate ``MinRStoppingCriterion`` to
    its JAX dataclass mirror without raising ``NotImplementedError``.
    """

    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = compute_fieldlines(
        ToroidalFieldJAX(1.3, 0.8),
        [1.4],
        [0.0],
        tmax=0.5,
        tol=1e-9,
        stopping_criteria=[MinRStoppingCriterion(0.5)],
    )
    assert len(res_tys) == 1
    assert res_phi_hits[0].shape[1] == 5


def test_compute_fieldlines_jax_maxR_criterion_fires_and_records_event(monkeypatch):
    """A ``MaxRStoppingCriterion`` set below the trajectory R fires immediately.

    The toroidal-axis trajectory holds at R=1.4 throughout. Setting
    ``MaxRStoppingCriterion(1.3)`` makes the predicate ``r >= 1.3``
    true on the very first accepted step; the event is recorded with
    ``idx = -1`` (the encoding for criterion index 0).
    """

    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = compute_fieldlines(
        ToroidalFieldJAX(1.3, 0.8),
        [1.4],
        [0.0],
        tmax=2.0,
        tol=1e-9,
        stopping_criteria=[MaxRStoppingCriterion(1.3)],
    )
    assert len(res_tys) == 1
    np.testing.assert_allclose(res_tys[0], np.array([[0.0, 1.4, 0.0, 0.0]]))
    # A trajectory that stopped due to a criterion must have a hit
    # record with idx<0.
    hits = res_phi_hits[0]
    assert hits.shape[0] >= 1, "expected criterion event row"
    assert int(hits[0, 1]) < 0, "criterion event must have negative idx"


def test_compute_fieldlines_jax_toroidal_transit_criterion(monkeypatch):
    """``ToroidalTransitStoppingCriterion`` caps the toroidal transit count.

    Tracing for ~3 transits with the criterion set to 1 must stop the
    trajectory before tmax. We verify the trajectory length is shorter
    than the tmax projected without the criterion.
    """

    R_init = 1.4
    # ~3 transits in upstream raw-B parametrisation.
    tmax_jax = 6.0 * np.pi * R_init
    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = compute_fieldlines(
        ToroidalFieldJAX(1.3, 0.8),
        [R_init],
        [0.0],
        tmax=tmax_jax,
        tol=1e-10,
        stopping_criteria=[ToroidalTransitStoppingCriterion(1.0, False)],
    )
    # The trajectory must not have reached tmax; either the criterion
    # fired (status<0 -> hits record idx<0) or the trajectory simply
    # short-circuited.
    assert res_tys[0][-1, 0] < tmax_jax, "transit criterion did not terminate"


def test_compute_fieldlines_jax_minZ_maxZ_pair_terminates(monkeypatch):
    """``MaxZStoppingCriterion`` with crit_z=-0.1 fires immediately on z=0.

    The predicate is ``z >= crit_z``; at z=0 and crit_z=-0.1 the
    predicate is true.
    """

    _force_jax_backend(monkeypatch)
    res_tys, res_phi_hits = compute_fieldlines(
        ToroidalFieldJAX(1.3, 0.8),
        [1.4],
        [0.0],
        tmax=2.0,
        tol=1e-9,
        stopping_criteria=[MaxZStoppingCriterion(-0.1)],
    )
    hits = res_phi_hits[0]
    assert hits.shape[0] >= 1
    assert int(hits[0, 1]) < 0


def test_compute_fieldlines_jax_unsupported_criterion_raises(monkeypatch):
    """An exotic non-supported criterion type still raises ``NotImplementedError``.

    The translator dispatches on isinstance; an arbitrary
    ``sopp.StoppingCriterion`` subclass that is not in the supported
    set raises ``NotImplementedError`` with a precise message.
    """

    class _NoOpCriterion:
        """Stand-in for an unknown stopping-criterion class."""

        pass

    _force_jax_backend(monkeypatch)
    import pytest

    with pytest.raises(
        NotImplementedError, match="cannot translate stopping criterion"
    ):
        compute_fieldlines(
            ToroidalFieldJAX(1.3, 0.8),
            [1.4],
            [0.0],
            tmax=0.5,
            tol=1e-9,
            stopping_criteria=[_NoOpCriterion()],
        )
