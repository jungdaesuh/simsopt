"""Ports boundary: native geometry owns the constraint, JAX consumes it.

The ``wireframe_rcls_with_ports`` workflow builds a ``ToroidalWireframe`` and
constrains the segments that collide with a ``PortSet`` entirely on the host
(native) side, via ``wireframe.constrain_colliding_segments(ports.collides,
gap=...)`` (``src/simsopt/geo/wireframe_toroidal.py``). The JAX RCLS adapter
never re-derives that geometry: ``rcls_wireframe_jax``
(``src/simsopt_jax_adapters/solve/wireframe.py``) consumes
``wframe.unconstrained_segments()`` on the same host wireframe object. This
module pins that hybrid boundary at two levels:

1. ``test_native_constrained_segments_match_independent_collision_oracle``
   verifies the wireframe's own bookkeeping (segment indexing / dedup /
   implicit-constraint expansion) against an ORACLE computed independently
   at test level: the same linspace-per-segment sampling contract
   documented in ``constrain_colliding_segments``, applied directly to the
   wireframe's public ``nodes``/``segments`` arrays and a freshly built
   ``PortSet.collides`` predicate -- never by calling
   ``constrain_colliding_segments``, ``constrained_segments()``, or
   ``unconstrained_segments()`` internally.

2. ``test_jax_rcls_solution_currents_match_native_reduced_system``
   exercises the real JAX lane -- calls ``rcls_wireframe_jax`` on the
   12x22 port wireframe and independently solves the native reduced system
   over the complete native-free index set. It compares every JAX free
   current to that reference while retaining constrained-zero and
   finite/nonzero guards. This catches a constrained index added to the solve
   and a native-free index dropped from it.

NOT independently re-verified here: the pure set-complement arithmetic
inside ``unconstrained_segments()`` itself (``free_segs[constrained] =
False; return where(free_segs)``) is guaranteed correct by construction once
``constrained_segments()`` is correct (test 1) and ``n_segments`` is right
(confirmed by test 2, which solves directly over the full native-free set and
compares every resulting current).

For the reader: on the 12x22 wireframe grid there are 528 segments, 31
native-constrained, 497 free. These numbers are documentation, not
assertions.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import CircularPort, PortSet, SurfaceRZFourier, ToroidalWireframe
from simsopt.solve.wireframe_optimization import (
    bnorm_obj_matrices,
    regularized_constrained_least_squares,
)
from simsopt_jax_adapters.solve.wireframe import rcls_wireframe_jax

_SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

# The source example's port workflow at its reduced plasma resolution.
_PLASMA_RESOLUTION = 16
_WIREFRAME_NPHI = 12
_WIREFRAME_NTHETA = 22
_WIREFRAME_SURFACE_DISTANCE = 0.3
_FIELD_ON_AXIS = 1.0
_REGULARIZATION_WEIGHT = 1.0e-10
_ASSUME_NO_CROSSINGS = False
_PORT_GAP = 0.04

# Matches the ``pts_per_seg`` default of
# ``ToroidalWireframe.constrain_colliding_segments``, which the workflow calls
# without overriding it.
_PTS_PER_SEG = 10


def _wireframe_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_vmec_input(str(_SURFACE_INPUT))
    surface.extend_via_projected_normal(_WIREFRAME_SURFACE_DISTANCE)
    return surface


def _ports_on_surface(surface: SurfaceRZFourier) -> PortSet:
    ports = PortSet()
    gamma = surface.gamma()
    normal = surface.normal()
    for phi in (np.pi / 8.0, 3.0 * np.pi / 8.0):
        phi_index = int(np.argmin(np.abs((0.5 / np.pi) * phi - surface.quadpoints_phi)))
        for theta in (np.pi / 4.0, 7.0 * np.pi / 4.0):
            theta_index = int(
                np.argmin(np.abs((0.5 / np.pi) * theta - surface.quadpoints_theta))
            )
            origin = gamma[phi_index, theta_index]
            axis = normal[phi_index, theta_index]
            ports.add_ports(
                [
                    CircularPort(
                        ox=origin[0],
                        oy=origin[1],
                        oz=origin[2],
                        ax=axis[0],
                        ay=axis[1],
                        az=axis[2],
                        ir=0.1,
                        thick=0.005,
                        l0=-0.15,
                        l1=0.15,
                    )
                ]
            )
    return ports.repeat_via_symmetries(surface.nfp, True)


def _build_geometry() -> tuple[SurfaceRZFourier, ToroidalWireframe]:
    plasma = SurfaceRZFourier.from_vmec_input(
        str(_SURFACE_INPUT),
        nphi=_PLASMA_RESOLUTION,
        ntheta=_PLASMA_RESOLUTION,
        range="half period",
    )
    wireframe_surface = _wireframe_surface()
    wireframe = ToroidalWireframe(
        wireframe_surface,
        _WIREFRAME_NPHI,
        _WIREFRAME_NTHETA,
    )
    wireframe.constrain_colliding_segments(
        _ports_on_surface(wireframe_surface).collides,
        gap=_PORT_GAP,
    )
    mu0 = 4.0 * np.pi * 1.0e-7
    wireframe.set_poloidal_current(
        -2.0 * np.pi * plasma.get_rc(0, 0) * _FIELD_ON_AXIS / mu0
    )
    return plasma, wireframe


def _independent_constrained_segments(wireframe, ports, gap: float) -> np.ndarray:
    """Re-derive colliding-segment indices from public geometry + the collision predicate.

    Duplicates only the SAMPLING CONTRACT documented in
    ``ToroidalWireframe.constrain_colliding_segments``: ``_PTS_PER_SEG``
    linearly-spaced test points along the two-node segment defined by
    ``wireframe.nodes[0][wireframe.segments[:, k], :]``, tested against
    ``ports.collides``. It never calls ``constrain_colliding_segments``,
    ``constrained_segments()``, or ``unconstrained_segments()``.
    """
    pos = np.linspace(0.0, 1.0, _PTS_PER_SEG).reshape((_PTS_PER_SEG, 1, 1))
    point0 = wireframe.nodes[0][wireframe.segments[:, 0], :]
    point1 = wireframe.nodes[0][wireframe.segments[:, 1], :]
    test_points = point0 + pos * (point1 - point0)
    colliding = ports.collides(
        test_points[:, :, 0], test_points[:, :, 1], test_points[:, :, 2], gap=gap
    )
    return np.where(np.any(colliding, axis=0))[0].astype(np.int64)


def test_native_constrained_segments_match_independent_collision_oracle() -> None:
    """``wireframe.constrained_segments()`` equals an independently-sampled oracle."""
    _plasma, wireframe = _build_geometry()

    # A fresh surface, deliberately NOT ``wireframe.surface``: the wireframe
    # re-parameterizes its own copy on its (n_phi, n_theta) grid.
    ports = _ports_on_surface(_wireframe_surface())
    expected_constrained = _independent_constrained_segments(
        wireframe, ports, gap=_PORT_GAP
    )

    native_constrained = np.asarray(wireframe.constrained_segments(), dtype=np.int64)

    assert native_constrained.size > 0, (
        "the port geometry did not constrain any segments at this scale; "
        "the boundary this test exists to guard cannot be exercised"
    )
    np.testing.assert_array_equal(
        np.sort(native_constrained), np.sort(expected_constrained)
    )


def test_jax_rcls_solution_currents_match_native_reduced_system() -> None:
    """JAX RCLS consumes the complete native-free segment set at the ports boundary."""
    plasma, wireframe = _build_geometry()
    response, target = bnorm_obj_matrices(
        wireframe, plasma, area_weighted=True, verbose=False
    )

    constrained_segments = np.asarray(wireframe.constrained_segments(), dtype=np.int64)
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    constraints, constraint_targets = wireframe.constraint_matrices(
        assume_no_crossings=_ASSUME_NO_CROSSINGS,
        remove_constrained_segments=True,
    )
    native_free_currents = np.asarray(
        regularized_constrained_least_squares(
            np.asarray(response)[:, free_segments],
            np.asarray(target),
            _REGULARIZATION_WEIGHT,
            constraints,
            constraint_targets,
        ),
        dtype=np.float64,
    ).reshape(-1)

    assert constrained_segments.size > 0, (
        "the port geometry did not constrain any segments at this scale; "
        "the boundary this test exists to guard cannot be exercised"
    )
    assert np.all(np.isfinite(native_free_currents)), (
        "the native reduced-system reference returned non-finite currents; "
        "the free-index boundary cannot be certified"
    )
    assert np.all(native_free_currents != 0.0), (
        "a native-free segment has zero reference current, so dropping that "
        "index would not be observable in this boundary fixture"
    )

    result = rcls_wireframe_jax(
        wireframe,
        jnp.asarray(response),
        jnp.asarray(target),
        _REGULARIZATION_WEIGHT,
        assume_no_crossings=_ASSUME_NO_CROSSINGS,
    )
    currents = np.asarray(jax.device_get(result.x), dtype=np.float64).reshape(-1)

    assert np.all(np.isfinite(currents)), (
        "the JAX RCLS solve returned non-finite currents; a degenerate solve "
        "cannot certify the constrained/free placement boundary"
    )

    np.testing.assert_array_equal(
        currents[constrained_segments], np.zeros(constrained_segments.size)
    )

    nonzero_segments = np.flatnonzero(currents)
    assert nonzero_segments.size > 0, (
        "the JAX RCLS solve returned an all-zero current vector; a degenerate "
        "solve cannot certify the constrained/free placement boundary"
    )
    assert np.all(np.isin(nonzero_segments, free_segments)), (
        "the JAX RCLS solution carries nonzero current on a segment the "
        "native wireframe marked constrained"
    )
    np.testing.assert_allclose(
        currents[free_segments], native_free_currents, rtol=1.0e-6, atol=0.0
    )
