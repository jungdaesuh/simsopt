"""A stopping criterion just OUTSIDE the interpolation domain.

Particle 54 of the shipped ``native-tracing-particle`` case leaves the
interpolation box (in ``z``) a few steps before upstream's level-set criterion
fires, and upstream keeps integrating there because its interpolant leaves its
output buffer untouched outside the domain instead of returning zero (see
``tests/jax/core/test_tracing_field_cache.py`` for that contract and its C++
citations). A port that returns zero gets ``|B| = 0``, an all-``nan``
right-hand side and a lane that fails where upstream stops on its criterion.

This is the same situation at unit scale, with the same two lanes and the
OFFICIAL lane as the authority: a purely toroidal field whose grad-B drift
carries the particle straight out of the top of the interpolation box, and a
``MaxZStoppingCriterion`` one accepted step beyond that boundary.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import numpy as np
from simsopt.field import InterpolatedField, MaxZStoppingCriterion, ToroidalField
from simsopt.field.tracing import trace_particles as native_trace_particles
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX
from simsopt_jax_adapters.field.tracing import trace_particles_with_status

R0 = 1.0
B0 = 2.5
DEGREE = 3
GRID = 8
RRANGE = (0.8, 1.2, GRID)
PHIRANGE = (0.0, 2.0 * np.pi, 2 * GRID)
# The boundary the particle leaves: the grad-B drift of a purely toroidal field
# is vertical, so the box top is the domain exit.
ZMAX_BOX = 0.2
ZRANGE = (-ZMAX_BOX, ZMAX_BOX, GRID)

MASS = 1.6726219e-27
CHARGE = 1.602176634e-19
SPEED_TOTAL = 4.2e5
SPEED_PAR = 1.7e5
EKIN = 0.5 * MASS * SPEED_TOTAL * SPEED_TOTAL
TOL = 1.0e-9
TMAX = 1.0e-3
XYZ_INIT = np.array([[1.0, 0.0, 0.15]])


def _native_field():
    return InterpolatedField(
        ToroidalField(R0=R0, B0=B0), DEGREE, RRANGE, PHIRANGE, ZRANGE, True
    )


def _jax_field():
    return InterpolatedFieldJAX(
        ToroidalFieldJAX(R0=R0, B0=B0), DEGREE, RRANGE, PHIRANGE, ZRANGE, True
    )


def _native_lane(crit_z: float):
    return native_trace_particles(
        _native_field(),
        XYZ_INIT,
        np.array([SPEED_PAR]),
        tmax=TMAX,
        mass=MASS,
        charge=CHARGE,
        Ekin=EKIN,
        tol=TOL,
        comm=None,
        phis=[],
        stopping_criteria=[MaxZStoppingCriterion(crit_z)],
        mode="gc_vac",
        forget_exact_path=False,
    )


def _jax_lane(crit_z: float):
    return trace_particles_with_status(
        _jax_field(),
        XYZ_INIT,
        np.array([SPEED_PAR]),
        tmax=TMAX,
        mass=MASS,
        charge=CHARGE,
        Ekin=EKIN,
        tol=TOL,
        comm=None,
        phis=(),
        stopping_criteria=[MaxZStoppingCriterion(crit_z)],
        mode="gc_vac",
        forget_exact_path=False,
        max_steps=None,
    )


def _boundary_reference():
    """The official lane stopped AT the box top: the last in-domain step.

    Returns ``(z_crit, step_seconds, step_metres)``: the criterion one accepted
    step beyond the interpolation boundary, and that step's length in time and
    in ``z``. All three are measured from the official lane, so nothing here is
    a chosen number.
    """
    trajectory = _native_lane(ZMAX_BOX)[0][0]
    assert trajectory.shape[0] >= 3, trajectory.shape
    dz_step = float(trajectory[-1, 3] - trajectory[-2, 3])
    dt_step = float(trajectory[-1, 0] - trajectory[-2, 0])
    assert dz_step > 0.0, dz_step
    return ZMAX_BOX + dz_step, dt_step, dz_step


def test_a_criterion_one_step_outside_the_domain_stops_both_lanes():
    z_crit, step_seconds, step_metres = _boundary_reference()
    assert z_crit > ZMAX_BOX

    native_trajectories, native_hits = _native_lane(z_crit)
    native_hit = native_hits[0]
    assert native_hit.size > 0, "the official lane must reach the criterion"
    assert int(native_hit[-1, 1]) == -1
    assert np.all(np.isfinite(native_hit))
    assert np.all(np.isfinite(native_trajectories[0]))
    native_t = float(native_hit[-1, 0])
    # The official lane really did leave the interpolation box before it
    # stopped: its last accepted state is above the box top.
    assert float(native_trajectories[0][-1, 3]) > ZMAX_BOX

    trajectories, hits, statuses = _jax_lane(z_crit)
    assert int(statuses[0]) == -1, (
        "the JAX lane must stop on the criterion, like the official lane; "
        f"got status {int(statuses[0])}"
    )
    hit = hits[0]
    assert hit.size > 0
    assert int(hit[-1, 1]) == -1
    assert np.all(np.isfinite(hit))
    assert np.all(np.isfinite(trajectories[0]))
    assert float(trajectories[0][-1, 3]) > ZMAX_BOX
    # The two lanes must stop within one accepted step of each other; the step
    # is the official lane's own, measured above.
    assert abs(float(hit[-1, 0]) - native_t) <= step_seconds
    # Upstream records the criterion row at the POST-step state (``solve`` in
    # ``legacy native extension/tracing.cpp`` pushes ``y``, not the bracketed
    # crossing), so both rows sit just beyond ``z_crit`` -- and within one
    # accepted step of each other.
    assert float(native_hit[-1, 4]) >= z_crit
    assert float(hit[-1, 4]) >= z_crit
    assert abs(float(hit[-1, 4]) - float(native_hit[-1, 4])) <= step_metres
