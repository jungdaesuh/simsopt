"""The tracing right-hand side outside the interpolation domain.

An ``InterpolatedField`` does NOT return zero outside its interpolation domain.
``RegularGridInterpolant3D::evaluate_local``
(``legacy native C++ source regular_grid_interpolant_3d_impl.h``) returns
WITHOUT writing its output slot when the cell is missing and ``extrapolate`` is
set, and ``CachedTensor::get_or_create_and_fill``
(``legacy native C++ source cachedtensor.h``) reuses the same buffer while the
point count is unchanged. Every right-hand side in
``legacy native extension/tracing.cpp`` queries ONE point per call, so such a
query returns the PREVIOUS query's cylindrical value, re-flipped by the current
point's symmetry and re-rotated by the current point's ``phi``.

The authority in every assertion below is the OFFICIAL
:class:`simsopt.field.InterpolatedField` object, queried in the same one point
at a time sequence the C++ right-hand side uses -- not the other JAX lane.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.field import InterpolatedField, ToroidalField
from simsopt_jax.core.interpolated_field import (
    interpolated_field_cyl_cache_zeros,
    interpolated_field_state_B_GradAbsB,
    interpolated_field_state_B_GradAbsB_cached,
)
from simsopt_jax.core.tracing import (
    guiding_center_vacuum_rhs_cached,
)
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX
from simsopt_jax_adapters.field.tracing import (
    _resolve_jax_field_B_GradAbsB_cached,
)

R0 = 1.0
B0 = 2.5
DEGREE = 3
RRANGE = (0.75, 1.25, 6)
PHIRANGE = (0.0, 2.0 * np.pi, 8)
ZRANGE = (-0.2, 0.2, 6)

MASS = 1.6726219e-27
CHARGE = 1.602176634e-19
MU = 3.0e10

# Two in-domain points with different ``phi`` and a point above ``ZRANGE``.
INSIDE_A = np.array([0.95, 0.10, 0.05])
INSIDE_B = np.array([0.20, -0.98, -0.12])
OUTSIDE = np.array([1.02, 0.12, 0.31])
V_PAR = 1.7e5


def _native_field():
    return InterpolatedField(
        ToroidalField(R0=R0, B0=B0), DEGREE, RRANGE, PHIRANGE, ZRANGE, True
    )


def _jax_field():
    return InterpolatedFieldJAX(
        ToroidalFieldJAX(R0=R0, B0=B0), DEGREE, RRANGE, PHIRANGE, ZRANGE, True
    )


def _cyl_row(point: np.ndarray) -> np.ndarray:
    """``GuidingCenterVacuumRHS::operator()``'s own ``rphiz`` row."""
    phi = float(np.arctan2(point[1], point[0]))
    if phi < 0.0:
        phi += 2.0 * np.pi
    return np.ascontiguousarray(
        np.array([[float(np.hypot(point[0], point[1])), phi, float(point[2])]])
    )


def _native_B_GradAbsB(field, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One query of the official object, in the C++ right-hand side's order."""
    field.set_points_cyl(_cyl_row(point))
    grad_abs_B = np.asarray(field.GradAbsB(), dtype=np.float64)[0]
    B = np.asarray(field.B(), dtype=np.float64)[0]
    return B, grad_abs_B


def _upstream_dydt(B: np.ndarray, grad_abs_B: np.ndarray, v_par: float) -> np.ndarray:
    """Literal transcription of ``GuidingCenterVacuumRHS::operator()``.

    Every scalar stays a ``np.float64`` so the divisions follow IEEE-754 like
    the C++ ``double`` arithmetic does; a Python ``float`` would raise instead
    of producing ``inf``.
    """
    abs_B = np.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
    v_par = np.float64(v_par)
    cross = np.array(
        [
            B[1] * grad_abs_B[2] - B[2] * grad_abs_B[1],
            B[2] * grad_abs_B[0] - B[0] * grad_abs_B[2],
            B[0] * grad_abs_B[1] - B[1] * grad_abs_B[0],
        ]
    )
    with np.errstate(all="ignore"):
        v_perp2 = 2.0 * MU * abs_B
        fak1 = v_par / abs_B
        fak2 = (MASS / (CHARGE * abs_B**3)) * (0.5 * v_perp2 + v_par * v_par)
        dydt = np.empty(4, dtype=np.float64)
        dydt[:3] = fak1 * B + fak2 * cross
        dydt[3] = (
            -MU
            * (B[0] * grad_abs_B[0] + B[1] * grad_abs_B[1] + B[2] * grad_abs_B[2])
            / abs_B
        )
    return dydt


def _cached_sequence(points):
    """Query the tracing field function along ``points``, threading the buffer."""
    field = _jax_field()
    state = field.jax_tracing_state()
    cache = interpolated_field_cyl_cache_zeros()
    out = []
    for point in points:
        B, grad_abs_B, cache = interpolated_field_state_B_GradAbsB_cached(
            state, jnp.asarray(point, jnp.float64), cache
        )
        out.append(
            (np.asarray(B, dtype=np.float64), np.asarray(grad_abs_B, dtype=np.float64))
        )
    return out


def test_the_outside_value_is_the_previous_query_not_zero():
    """Upstream's own object, queried the way its right-hand side queries it."""
    native = _native_field()
    B_in, grad_in = _native_B_GradAbsB(native, INSIDE_A)
    B_out, grad_out = _native_B_GradAbsB(native, OUTSIDE)
    assert np.all(np.isfinite(B_in)) and np.linalg.norm(B_in) > 0.0
    # The official object does not return zero outside its own domain: the
    # cylindrical MAGNITUDE is the previous query's, only the rotation changed.
    assert np.linalg.norm(B_out) == pytest.approx(np.linalg.norm(B_in), rel=1e-14)
    assert np.linalg.norm(grad_out) == pytest.approx(np.linalg.norm(grad_in), rel=1e-14)
    # And it depends on the history, which is what makes it a buffer.
    native2 = _native_field()
    _ = _native_B_GradAbsB(native2, INSIDE_B)
    B_out2, _grad_out2 = _native_B_GradAbsB(native2, OUTSIDE)
    assert not np.allclose(B_out, B_out2)


def test_the_tracing_field_function_reproduces_the_official_outside_value():
    """Same sequences through the port's tracing entry point."""
    native = _native_field()
    expected = [
        _native_B_GradAbsB(native, INSIDE_A),
        _native_B_GradAbsB(native, OUTSIDE),
    ]
    got = _cached_sequence([INSIDE_A, OUTSIDE])
    for (B_ref, grad_ref), (B, grad) in zip(expected, got, strict=True):
        np.testing.assert_allclose(B, B_ref, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(grad, grad_ref, rtol=1e-12, atol=1e-13)

    native2 = _native_field()
    expected2 = [
        _native_B_GradAbsB(native2, INSIDE_B),
        _native_B_GradAbsB(native2, OUTSIDE),
    ]
    got2 = _cached_sequence([INSIDE_B, OUTSIDE])
    for (B_ref, grad_ref), (B, grad) in zip(expected2, got2, strict=True):
        np.testing.assert_allclose(B, B_ref, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(grad, grad_ref, rtol=1e-12, atol=1e-13)
    # The two histories really do give different values at the same point, so
    # the agreement above cannot come from a point-only function.
    assert not np.allclose(got[1][0], got2[1][0])


def test_the_zero_filling_entry_point_is_what_makes_the_rhs_non_finite():
    """The contrast, measured rather than asserted from memory.

    ``interpolated_field_state_B_GradAbsB`` is the zero-filling entry point the
    guiding-centre route used before this fix. It is kept for callers that want
    a point-only function; this test records what it does outside the domain
    and what the guiding-centre arithmetic then produces, which is the defect
    this module's other tests forbid.
    """
    field = _jax_field()
    state = field.jax_tracing_state()
    B, grad = interpolated_field_state_B_GradAbsB(
        state, jnp.asarray(OUTSIDE, jnp.float64)
    )
    assert np.all(np.asarray(B) == 0.0)
    assert np.all(np.asarray(grad) == 0.0)
    dydt = _upstream_dydt(np.zeros(3), np.zeros(3), V_PAR)
    assert np.all(np.isnan(dydt))


def test_the_rhs_matches_upstream_term_by_term_outside_the_domain():
    """Value AND finiteness pattern, component by component, at both points."""
    native = _native_field()
    jax_field = _jax_field()
    state = jax_field.jax_tracing_state()
    rhs = guiding_center_vacuum_rhs_cached(
        lambda point, cache: interpolated_field_state_B_GradAbsB_cached(
            state, point, cache
        ),
        MASS,
        CHARGE,
        MU,
    )

    cache = interpolated_field_cyl_cache_zeros()
    for point in (INSIDE_A, OUTSIDE):
        B_ref, grad_ref = _native_B_GradAbsB(native, point)
        expected = _upstream_dydt(B_ref, grad_ref, V_PAR)
        y = jnp.asarray(np.append(point, V_PAR), jnp.float64)
        dydt_device, cache = rhs(jnp.asarray(0.0, jnp.float64), y, cache)
        dydt = np.asarray(dydt_device, dtype=np.float64)
        assert np.isfinite(dydt).tolist() == np.isfinite(expected).tolist()
        np.testing.assert_allclose(dydt[:3], expected[:3], rtol=1e-11, atol=1e-11)
        # ``dv_par`` is ``-mu (B . grad|B|) / |B|``, and for a purely toroidal
        # field ``B`` and ``grad|B|`` are orthogonal in exact arithmetic, so
        # that dot product is a complete cancellation sitting at the rounding
        # floor. The bound is therefore derived from the terms that cancel --
        # the standard ``n * eps * sum |x_i y_i|`` dot-product bound plus one
        # rounding of the gradient itself -- not from a chosen tolerance. The
        # three drift components above carry the same ``grad|B|`` without
        # cancellation and are checked tightly.
        eps = np.finfo(np.float64).eps
        abs_B_ref = float(np.linalg.norm(B_ref))
        cancellation = float(np.dot(np.abs(B_ref), np.abs(grad_ref)))
        bound = MU * (
            3.0 * eps * cancellation / abs_B_ref + eps * float(np.linalg.norm(grad_ref))
        )
        assert abs(dydt[3] - expected[3]) <= bound

    # Upstream is FINITE at the outside point -- reached the way its own
    # right-hand side reaches it, one query after an in-domain one, which is
    # the whole point. (A FRESH object has a zero-filled buffer there, which is
    # the state a batch probe sees and is what
    # ``test_the_zero_filling_entry_point_is_what_makes_the_rhs_non_finite``
    # records.)
    sequenced = _native_field()
    _native_B_GradAbsB(sequenced, INSIDE_A)
    B_out, grad_out = _native_B_GradAbsB(sequenced, OUTSIDE)
    assert np.all(np.isfinite(_upstream_dydt(B_out, grad_out, V_PAR)))
    B_fresh, grad_fresh = _native_B_GradAbsB(_native_field(), OUTSIDE)
    assert np.all(B_fresh == 0.0) and np.all(grad_fresh == 0.0)


def test_the_guiding_centre_route_resolves_the_buffered_contract():
    """The wiring: an interpolated field gets the buffered field function."""
    field_fn, state, cache = _resolve_jax_field_B_GradAbsB_cached(_jax_field())
    assert field_fn is interpolated_field_state_B_GradAbsB_cached
    assert state is not None
    assert cache is not None
    assert np.all(np.asarray(cache.B_cyl) == 0.0)
    assert np.all(np.asarray(cache.GradAbsB_cyl) == 0.0)
    # An analytic field writes every output row, so it has no buffer to mirror.
    assert _resolve_jax_field_B_GradAbsB_cached(ToroidalFieldJAX(R0=R0, B0=B0)) == (
        None,
        None,
        None,
    )


def test_the_initial_buffer_is_host_resident_and_built_under_the_strict_guard():
    """The strict GPU lane runs under ``jax_transfer_guard=disallow``; the initial buffer must survive it twice.

    The guiding-centre route captures the initial buffer in the closure of its
    compiled chunk program. Measured on the JAX GPU lane of
    ``native-tracing-particle``: (1) built with an eager ``jnp.zeros`` it died in
    set-up with ``Disallowed host-to-device transfer`` (the scalar fill value is
    staged implicitly); (2) built as a device array it died at the first
    lowering with ``Disallowed device-to-host transfer``, because XLA copies a
    captured device array back to make its literal. Host leaves do neither,
    which is the rule of ``simsopt_jax.core.specs.host_resident_spec``.

    The CPU backend enforces the guard for (1) exactly like CUDA, so the guarded
    block needs no GPU. It cannot see (2) -- a CPU array is already on the host --
    so host residency is asserted directly. The field is built outside the guard:
    constructing the interpolant is not the route under test, resolving its
    tracing contract is.
    """
    field = _jax_field()
    with jax.transfer_guard("disallow"):
        cache = interpolated_field_cyl_cache_zeros()
        _field_fn, _state, resolved = _resolve_jax_field_B_GradAbsB_cached(field)
    assert resolved is not None
    for buffer in (
        cache.B_cyl,
        cache.GradAbsB_cyl,
        resolved.B_cyl,
        resolved.GradAbsB_cyl,
    ):
        assert isinstance(buffer, np.ndarray)
        assert not isinstance(buffer, jax.Array)
        assert buffer.shape == (1, 3)
        assert buffer.dtype == np.float64
        assert np.all(buffer == 0.0)
        # The buffer stands for the C++ tensor BEFORE the first evaluation and is read by a compiled program;
        # nothing may write into it in place, so the flag the constructor sets is pinned here.
        assert buffer.flags.writeable is False
