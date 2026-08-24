"""JAX mirror of the native self-field-force suite.

Native oracle suite: ``tests/field/test_selffieldforces.py`` (23 test functions,
classes ``SpecialFunctionsTests`` and ``CoilForcesTest``).  Implementation under
test: ``simsopt_jax_adapters.field.force``.  Unit 1 of the mirror-wave plan
(formerly ``docs/jax_native_test_mirror_wave_implementation_plan.md``, commit
2221b542a; removed by the 2026-08-24 docs curation).

What the JAX side actually implements
-------------------------------------
``simsopt_jax_adapters.field.force`` re-implements the *objective* layer
(``B2Energy``, ``NetFluxes``, ``LpCurveForce``, ``SquaredMeanForce``,
``LpCurveTorque``, ``SquaredMeanTorque``), the inductance / induced-current pure
functions, and the JAX-only ``curve_force_norms_pure`` export.  It does **not**
re-implement two native surfaces:

* ``simsopt.field.selffield`` (k, delta, ``regularization_circ/rect``,
  ``B_regularized_pure``) — that module is already ``jax.numpy``-based and the
  adapter imports it directly (``force.py:22``), so there is one shared
  implementation rather than two.  The tests below therefore pin it against
  published/closed-form values and against its differentiability under
  ``jax.grad``, which is what the adapter's ``dJ`` depends on.
* the ``RegularizedCoil`` coil-method layer
  (``B_regularized``/``self_force``/``force``/``torque``/``net_force``/``net_torque``)
  and its ``Circular``/``Rectangular`` subclasses, which return force *vectors*
  and stay native.  The JAX lane exposes force *magnitudes* only
  (``curve_force_norms_pure``) plus the net-force/net-torque objectives, so the
  vector-valued behavior is mirrored to that boundary and the remainder is a
  manifest fact, not a faked test.

Oracle protocol
---------------
Closed-form or published values wherever the native suite uses them (k, delta,
regularizations, the analytic circular-coil self-force, mutual inductance of
concentric and coaxial circular filaments).  Otherwise the native
implementation evaluated in this same process — direct ``import simsopt.field
.force as native_force`` calls alongside the adapter under test, at whatever
configuration each test below states.
The native HSX self-force oracle is itself pinned to the CoilForces.jl benchmark
by ``tests/field/test_selffieldforces.py::CoilForcesTest::test_hsx_coil``.

Every test runs under the ``parity_lane`` fixture, so the suite executes on GPU
when CUDA is present and skips that lane cleanly otherwise.
"""

import numpy as np
import pytest
from jax import grad, jit
from scipy import constants
from scipy.interpolate import interp1d
from scipy.special import ellipe, ellipk
from simsopt.configs import get_data
from simsopt.field import (
    CircularRegularizedCoil,
    Coil,
    Current,
    RectangularRegularizedCoil,
    RegularizedCoil,
    coils_via_symmetries,
)
from simsopt.field.selffield import (
    _rectangular_xsection_delta,
    _rectangular_xsection_k,
    regularization_circ,
    regularization_rect,
)
from simsopt.geo import CurvePlanarFourier, CurveXYZFourier
from simsopt.geo.curve import create_equally_spaced_curves

import simsopt.field.force as native_force
import simsopt_jax_adapters.field.force as jax_force

from conftest import parity_default_device
from simsopt_jax.parity_tolerances import parity_ladder_tolerances

_DERIVATIVE_HEAVY = parity_ladder_tolerances("derivative_heavy")
_OBJECTIVE_VALUE_RTOL = _DERIVATIVE_HEAVY["scalar_value_rtol"]
_OBJECTIVE_GRADIENT_RTOL = _DERIVATIVE_HEAVY["first_derivative_rtol"]

# Landreman/Hurwitz/Antonsen reduced-model constants for a square cross-section,
# reproduced from the native suite's own published truth values.
_K_SQUARE = 2.556493222766492
_DELTA_SQUARE = 0.19985294779417703

_LP_POWER = 2.5
_FORCE_THRESHOLD_MN_PER_M = 1e-3


@pytest.fixture(autouse=True)
def _parity_device_scope(parity_lane):
    with parity_default_device(parity_lane):
        yield


def _relative_infinity_error(actual, reference):
    """Return ``max|actual - reference| / max|reference|`` for array parity.

    Reference arrays here (objective gradients) contain entries spanning many
    orders of magnitude, so a per-entry ``rtol`` would be dominated by the
    smallest components; this file adopts the infinity-norm ratio, scaled by
    the reference's own largest entry, as its own criterion for those arrays
    (``simsopt_jax.parity_tolerances`` states per-entry ``rtol``/``atol``
    pairs, not this reduction).
    """
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    scale = np.max(np.abs(reference))
    assert scale > 0.0, "reference gradient is identically zero; pick a live case"
    return float(np.max(np.abs(actual - reference)) / scale)


def _planar_circle(radius, quadpoint_count, *, center_z=0.0):
    """Circle of the given radius in a plane normal to z, as CurvePlanarFourier."""
    curve = CurvePlanarFourier(quadpoint_count, 0)
    dofs = np.zeros(8)
    dofs[0] = radius
    dofs[1] = 1.0  # unit quaternion for the identity rotation (normal = +z)
    dofs[7] = center_z
    curve.set_dofs(dofs)
    return curve


def _xyz_circle(radius, quadpoint_count, *, tilt=0.0):
    """Circle of the given radius in the x-y plane, as CurveXYZFourier.

    ``tilt`` adds a z_s1 harmonic (absolute, not scaled by ``radius``) that
    lifts the curve out of the x-y plane.  A set of concentric, coplanar
    circles (``tilt=0.0`` everywhere) has zero pointwise mutual torque by
    rotational symmetry, which both makes Lp-torque objectives vacuously zero
    and makes JAX's ``jnp.linalg.norm`` gradient at that exact-zero vector
    ``nan``; a nonzero, per-coil-distinct ``tilt`` avoids both.
    """
    curve = CurveXYZFourier(quadpoint_count, 1)
    dofs = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * radius
    dofs[8] += tilt
    curve.x = dofs
    return curve


def _distant_source_curve(quadpoint_count):
    """A curve far from every fixture coil, used as a zero-current source slot."""
    return create_equally_spaced_curves(
        1,
        1,
        False,
        R0=50.0,
        R1=1.0,
        numquadpoints=quadpoint_count,
    )[0]


def _stacked(curves, attribute):
    return np.stack([np.asarray(getattr(curve, attribute)()) for curve in curves])


def _jax_force_norms_mn_per_m(
    target_curve,
    target_current,
    regularization,
    *,
    source_curves,
    source_currents,
    downsample=1,
):
    """Pointwise |dF/dl| in MN/m for one target coil, through the JAX lane."""
    norms = jax_force.curve_force_norms_pure(
        np.asarray(target_curve.gamma())[None],
        _stacked(source_curves, "gamma"),
        np.asarray(target_curve.gammadash())[None],
        _stacked(source_curves, "gammadash"),
        np.asarray(target_curve.gammadashdash())[None],
        np.asarray(target_curve.quadpoints)[None],
        np.asarray([target_current], dtype=np.float64),
        np.asarray(source_currents, dtype=np.float64),
        np.asarray([regularization], dtype=np.float64),
        downsample,
    )
    return np.asarray(norms)[0]


def _jax_self_force_norms_mn_per_m(curve, current, regularization):
    """Self-force magnitude through the JAX lane, isolated from mutual sources.

    ``curve_force_norms_pure`` always consumes a source group; the mutual field
    is linear in the source current, so a single distant source carrying exactly
    zero current contributes exactly zero and the result is the pure self-force.
    """
    return _jax_force_norms_mn_per_m(
        curve,
        current,
        regularization,
        source_curves=[_distant_source_curve(len(curve.quadpoints))],
        source_currents=[0.0],
    )


def _symmetric_regularized_coils(
    *,
    base_curve_count=4,
    nfp=3,
    current=1.7e4,
    regularization=None,
    quadpoint_count=None,
    twist=None,
):
    """Return ``(base_curves, coils)``: the owning base curves, then the
    stellarator-symmetric ``RegularizedCoil`` set expanded from them."""
    if regularization is None:
        regularization = regularization_circ(0.05)
    kwargs = {} if quadpoint_count is None else {"numquadpoints": quadpoint_count}
    base_curves = create_equally_spaced_curves(base_curve_count, nfp, True, **kwargs)
    if twist is not None:
        for base_curve in base_curves:
            twisted = base_curve.x.copy()
            twisted[3] += twist
            base_curve.x = twisted
    coils = coils_via_symmetries(
        base_curves,
        [Current(current) for _ in range(base_curve_count)],
        nfp,
        True,
        regularizations=[regularization] * base_curve_count,
    )
    return base_curves, coils


def _objective_kwargs(objective_name, *, threshold=_FORCE_THRESHOLD_MN_PER_M):
    if objective_name.startswith("Lp"):
        return {"p": _LP_POWER, "threshold": threshold}
    return {}


_OBJECTIVE_NAMES = (
    "LpCurveForce",
    "SquaredMeanForce",
    "LpCurveTorque",
    "SquaredMeanTorque",
)


# --------------------------------------------------------------------------
# SpecialFunctionsTests mirror — the shared jax.numpy reduced-model functions.
# --------------------------------------------------------------------------


def test_rectangular_xsection_k_matches_published_square_value():
    """k(a, a) reproduces the published square-cross-section constant."""
    np.testing.assert_allclose(float(_rectangular_xsection_k(0.3, 0.3)), _K_SQUARE)
    np.testing.assert_allclose(float(_rectangular_xsection_k(2.7, 2.7)), _K_SQUARE)


def test_rectangular_xsection_delta_matches_published_square_value():
    """delta(a, a) reproduces the published square-cross-section constant."""
    np.testing.assert_allclose(
        float(_rectangular_xsection_delta(0.3, 0.3)), _DELTA_SQUARE
    )
    np.testing.assert_allclose(
        float(_rectangular_xsection_delta(2.7, 2.7)), _DELTA_SQUARE
    )


@pytest.mark.parametrize("aspect_ratio", (0.1, 3.7))
def test_rectangular_xsection_functions_are_symmetric_in_a_and_b(aspect_ratio):
    """Swapping conductor width and height leaves k and delta unchanged."""
    geometric_mean = 0.01
    width = geometric_mean * aspect_ratio
    height = geometric_mean / aspect_ratio

    np.testing.assert_allclose(
        float(_rectangular_xsection_k(width, height)),
        float(_rectangular_xsection_k(height, width)),
    )
    np.testing.assert_allclose(
        float(_rectangular_xsection_delta(width, height)),
        float(_rectangular_xsection_delta(height, width)),
    )


@pytest.mark.parametrize("aspect_ratio", (1.1e6, 2.2e4, 3.5e5))
@pytest.mark.parametrize("short_side", (0.2, 1.0, 7.3))
def test_rectangular_xsection_functions_match_thin_strip_limits(
    aspect_ratio, short_side
):
    """For a >> b (and the mirrored case) k -> 7/6 + log(a/b), delta -> a/(b e^3)."""
    long_side = short_side * aspect_ratio

    np.testing.assert_allclose(
        float(_rectangular_xsection_k(long_side, short_side)),
        (7.0 / 6) + np.log(aspect_ratio),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        float(_rectangular_xsection_delta(long_side, short_side)),
        aspect_ratio / np.exp(3),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        float(_rectangular_xsection_k(short_side, long_side)),
        (7.0 / 6) + np.log(aspect_ratio),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        float(_rectangular_xsection_delta(short_side, long_side)),
        aspect_ratio / np.exp(3),
        rtol=1e-3,
    )


def test_regularization_circ_matches_closed_form_and_scales_quadratically():
    """regularization_circ(a) = a^2 / sqrt(e) and scales as a^2."""
    small_radius, large_radius = 0.01, 0.05
    small = float(regularization_circ(small_radius))
    large = float(regularization_circ(large_radius))

    np.testing.assert_allclose(small, small_radius**2 / np.sqrt(np.e), rtol=1e-10)
    np.testing.assert_allclose(large, large_radius**2 / np.sqrt(np.e), rtol=1e-10)
    np.testing.assert_allclose(
        large / small, (large_radius / small_radius) ** 2, rtol=1e-10
    )


def test_regularization_rect_is_area_times_delta_and_symmetric():
    """regularization_rect(a, b) = a b delta(a, b), symmetric and monotone in area."""
    side = 0.01
    np.testing.assert_allclose(
        float(regularization_rect(side, side)), side**2 * _DELTA_SQUARE, rtol=1e-10
    )

    width, height = 0.01, 0.023
    value = float(regularization_rect(width, height))
    np.testing.assert_allclose(
        value,
        width * height * float(_rectangular_xsection_delta(width, height)),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        value, float(regularization_rect(height, width)), rtol=1e-10
    )
    assert value > 0.0
    assert float(regularization_rect(0.02, 0.03)) > value


def test_regularizations_stay_traceable_under_jit_and_grad():
    """The adapter's dJ path differentiates through the regularization functions."""
    step = 1e-6
    width, height = 0.3, 0.7

    jitted_rect = jit(regularization_rect)
    np.testing.assert_allclose(
        float(jitted_rect(0.01, 0.023)),
        float(regularization_rect(0.01, 0.023)),
        rtol=1e-14,
    )

    analytic = float(grad(_rectangular_xsection_k, argnums=0)(width, height))
    finite_difference = float(
        (
            _rectangular_xsection_k(width + step, height)
            - _rectangular_xsection_k(width - step, height)
        )
        / (2.0 * step)
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6)

    analytic_circ = float(grad(regularization_circ)(0.05))
    np.testing.assert_allclose(analytic_circ, 2 * 0.05 / np.sqrt(np.e), rtol=1e-12)


# --------------------------------------------------------------------------
# CoilForcesTest mirror — analytic anchors.
# --------------------------------------------------------------------------


def test_jax_self_force_on_circular_coil_matches_analytic_hoop_force():
    """Circular-centerline hoop force matches Landreman/Hurwitz closed forms."""
    major_radius, current = 1.7, 1e5
    conductor_width, conductor_height = 0.01, 0.023
    curve = _xyz_circle(major_radius, 200)

    analytic_circ = (
        constants.mu_0
        * current
        / (4 * np.pi * major_radius)
        * (np.log(8 * major_radius / conductor_width) - 3 / 4)
        * current
    )
    analytic_rect = (
        constants.mu_0
        * current
        / (4 * np.pi * major_radius)
        * (
            np.log(8 * major_radius / np.sqrt(conductor_width * conductor_height))
            + 13.0 / 12
            - float(_rectangular_xsection_k(conductor_width, conductor_height)) / 2
        )
        * current
    )

    circ_norms = _jax_self_force_norms_mn_per_m(
        curve, current, regularization_circ(conductor_width)
    )
    rect_norms = _jax_self_force_norms_mn_per_m(
        curve,
        current,
        regularization_rect(conductor_width, conductor_height),
    )

    np.testing.assert_allclose(circ_norms * 1e6, analytic_circ, rtol=1e-10)
    np.testing.assert_allclose(rect_norms * 1e6, analytic_rect, rtol=1e-10)


def test_jax_inductances_match_analytic_concentric_and_coaxial_filaments():
    """Mutual inductance reproduces the two closed-form circular-filament cases."""
    inner_radius, outer_radius = 1.7, 40.0
    coaxial_radius, axial_offset = 3.0, 5.0
    regularizations = np.array([regularization_circ(0.01)] * 2)
    quadpoint_count = 200

    inner = _planar_circle(inner_radius, quadpoint_count)
    outer = _planar_circle(outer_radius, quadpoint_count)
    coaxial = _planar_circle(coaxial_radius, quadpoint_count, center_z=axial_offset)

    concentric_analytic = constants.mu_0 * np.pi * inner_radius**2 / (2 * outer_radius)
    modulus = np.sqrt(
        4.0
        * inner_radius
        * coaxial_radius
        / ((inner_radius + coaxial_radius) ** 2 + axial_offset**2)
    )
    coaxial_analytic = (
        constants.mu_0
        * np.sqrt(inner_radius * coaxial_radius)
        * (
            (2 / modulus - modulus) * ellipk(modulus**2)
            - (2 / modulus) * ellipe(modulus**2)
        )
    )

    for downsample in (1, 2, 4):
        concentric = np.asarray(
            jax_force._coil_coil_inductances_pure(
                _stacked([inner, outer], "gamma"),
                _stacked([inner, outer], "gammadash"),
                downsample=downsample,
                regularizations=regularizations,
            )
        )
        np.testing.assert_allclose(concentric[1, 0], concentric_analytic, rtol=1e-2)
        # The matrix is symmetric and swapping the coil order transposes it.
        swapped = np.asarray(
            jax_force._coil_coil_inductances_pure(
                _stacked([outer, inner], "gamma"),
                _stacked([outer, inner], "gammadash"),
                downsample=downsample,
                regularizations=regularizations,
            )
        )
        np.testing.assert_allclose(swapped[1, 0], concentric[0, 1], rtol=1e-12)
        np.testing.assert_allclose(swapped[0, 0], concentric[1, 1], rtol=1e-12)

    coaxial_matrix = np.asarray(
        jax_force._coil_coil_inductances_pure(
            _stacked([inner, coaxial], "gamma"),
            _stacked([inner, coaxial], "gammadash"),
            downsample=2,
            regularizations=regularizations,
        )
    )
    np.testing.assert_allclose(coaxial_matrix[1, 0], coaxial_analytic, rtol=1e-6)

    inverse = np.asarray(
        jax_force._coil_coil_inductances_inv_pure(
            _stacked([inner, coaxial], "gamma"),
            _stacked([inner, coaxial], "gammadash"),
            downsample=2,
            regularizations=regularizations,
        )
    )
    np.testing.assert_allclose(
        inverse, np.linalg.inv(coaxial_matrix), rtol=1e-10, atol=0.0
    )


def test_jax_inductance_and_induced_current_kernels_match_native():
    """Inductance, its Cholesky inverse, and induced currents match native."""
    regularizations = np.array([regularization_circ(0.01)] * 2)
    inner = _planar_circle(1.7, 200)
    outer = _planar_circle(40.0, 200)
    gammas = _stacked([inner, outer], "gamma")
    gammadashs = _stacked([inner, outer], "gammadash")

    for downsample in (1, 2, 4):
        native = np.asarray(
            native_force._coil_coil_inductances_pure(
                gammas, gammadashs, downsample, regularizations
            )
        )
        jax_values = np.asarray(
            jax_force._coil_coil_inductances_pure(
                gammas, gammadashs, downsample, regularizations
            )
        )
        assert _relative_infinity_error(jax_values, native) <= 1e-12

    native_inverse = np.asarray(
        native_force._coil_coil_inductances_inv_pure(
            gammas, gammadashs, 2, regularizations
        )
    )
    jax_inverse = np.asarray(
        jax_force._coil_coil_inductances_inv_pure(
            gammas, gammadashs, 2, regularizations
        )
    )
    assert _relative_infinity_error(jax_inverse, native_inverse) <= 1e-10

    single_regularization = np.array([regularization_circ(0.01)])
    native_currents = np.asarray(
        native_force._induced_currents_pure(
            gammas[:1],
            gammadashs[:1],
            gammas[1:],
            gammadashs[1:],
            np.array([1e6]),
            2,
            single_regularization,
        )
    )
    jax_currents = np.asarray(
        jax_force._induced_currents_pure(
            gammas[:1],
            gammadashs[:1],
            gammas[1:],
            gammadashs[1:],
            np.array([1e6]),
            2,
            single_regularization,
        )
    )
    assert np.all(np.abs(jax_currents) > 1e3)
    assert _relative_infinity_error(jax_currents, native_currents) <= 1e-12


def test_jax_force_norms_on_hsx_coil_match_native_self_force():
    """HSX coil 1 self-force magnitudes match the native (CoilForces.jl-pinned) oracle."""
    base_curves, _, _, _, _ = get_data("hsx")
    curve = base_curves[0]
    assert len(curve.quadpoints) == 160
    current, width, height = 150e3, 0.01, 0.023

    for coil, regularization in (
        (
            CircularRegularizedCoil(curve, Current(current), width),
            regularization_circ(width),
        ),
        (
            RectangularRegularizedCoil(curve, Current(current), width, height),
            regularization_rect(width, height),
        ),
    ):
        native_norms = np.linalg.norm(np.asarray(coil.self_force()), axis=1) / 1e6
        jax_norms = _jax_self_force_norms_mn_per_m(curve, current, regularization)
        assert _relative_infinity_error(jax_norms, native_norms) <= 1e-12


def test_jax_force_norms_converge_with_quadrature_resolution():
    """Self-force profile is resolution independent, as in the native convergence test."""
    current, conductor_radius = 1.5e3, 0.01
    regularization = regularization_circ(conductor_radius)
    reference_interpolant = None
    reference_budget = None

    for index, points_per_period in enumerate([8, 4, 2, 7, 5]):
        base_curves, _, _, _, _ = get_data("hsx", points_per_period=points_per_period)
        curve = base_curves[0]
        norms = _jax_self_force_norms_mn_per_m(curve, current, regularization) * 1e6
        if index == 0:
            reference_interpolant = interp1d(curve.quadpoints, norms, axis=0)
            reference_budget = np.max(np.abs(norms)) / 60
        else:
            np.testing.assert_allclose(
                norms,
                reference_interpolant(curve.quadpoints),
                atol=reference_budget,
                rtol=0.0,
            )


# --------------------------------------------------------------------------
# CoilForcesTest mirror — native-oracle parity for the objective layer.
# --------------------------------------------------------------------------


def test_jax_force_norms_match_native_coil_force_with_mutual_sources():
    """Force magnitudes include self plus mutual contributions, as native's force()."""
    _, coils = _symmetric_regularized_coils()
    target = coils[0]
    sources = [coil for coil in coils if coil is not target]

    native_norms = np.linalg.norm(np.asarray(target.force(coils)), axis=1) / 1e6
    jax_norms = _jax_force_norms_mn_per_m(
        target.curve,
        target.current.get_value(),
        target.regularization,
        source_curves=[coil.curve for coil in sources],
        source_currents=[coil.current.get_value() for coil in sources],
    )

    assert _relative_infinity_error(jax_norms, native_norms) <= 1e-8


@pytest.mark.parametrize("objective_name", _OBJECTIVE_NAMES)
@pytest.mark.parametrize("downsample", (1, 2))
def test_force_and_torque_objective_values_match_native(objective_name, downsample):
    """JAX objective values equal the native objective on identical coils."""
    _, coils = _symmetric_regularized_coils(base_curve_count=2, twist=0.1)
    kwargs = _objective_kwargs(objective_name, threshold=0.0)
    native_value = float(
        getattr(native_force, objective_name)(
            coils[0], coils, downsample=downsample, **kwargs
        ).J()
    )
    jax_value = float(
        getattr(jax_force, objective_name)(
            coils[0], coils, downsample=downsample, **kwargs
        ).J()
    )

    assert abs(native_value) > 0.0
    np.testing.assert_allclose(
        jax_value, native_value, rtol=_OBJECTIVE_VALUE_RTOL, atol=0.0
    )


@pytest.mark.parametrize("objective_name", _OBJECTIVE_NAMES)
def test_force_and_torque_objective_gradients_match_native(objective_name):
    """JAX objective gradients equal the native gradients, with native's shape."""
    base_curves, coils = _symmetric_regularized_coils(base_curve_count=2, twist=0.1)
    kwargs = _objective_kwargs(objective_name, threshold=0.0)
    native_gradient = np.asarray(
        getattr(native_force, objective_name)(coils[0], coils, **kwargs).dJ()
    )
    jax_gradient = np.asarray(
        getattr(jax_force, objective_name)(coils[0], coils, **kwargs).dJ()
    )

    expected_size = len(base_curves) * len(coils[0].x)
    assert native_gradient.shape == (expected_size,)
    assert jax_gradient.shape == (expected_size,)
    assert (
        _relative_infinity_error(jax_gradient, native_gradient)
        <= _OBJECTIVE_GRADIENT_RTOL
    )


def test_b2energy_and_netfluxes_match_native_values_and_gradients():
    """The energy and flux objectives match native in value and gradient."""
    _, coils = _symmetric_regularized_coils(base_curve_count=2, twist=0.1)

    for native_objective, jax_objective in (
        (native_force.B2Energy(coils), jax_force.B2Energy(coils)),
        (
            native_force.NetFluxes(coils[0], coils[1:]),
            jax_force.NetFluxes(coils[0], coils[1:]),
        ),
        (
            native_force.B2Energy(coils, downsample=2),
            jax_force.B2Energy(coils, downsample=2),
        ),
    ):
        native_value = float(native_objective.J())
        assert abs(native_value) > 0.0
        np.testing.assert_allclose(
            float(jax_objective.J()),
            native_value,
            rtol=_OBJECTIVE_VALUE_RTOL,
            atol=0.0,
        )
        assert (
            _relative_infinity_error(
                np.asarray(jax_objective.dJ()), np.asarray(native_objective.dJ())
            )
            <= _OBJECTIVE_GRADIENT_RTOL
        )


@pytest.mark.parametrize(
    "regularization", (regularization_circ(0.05), regularization_rect(0.01, 0.023))
)
def test_squared_mean_force_equals_native_net_force_integral(regularization):
    """SquaredMeanForce reproduces ||net force||^2 in MN^2.

    The objective and ``RegularizedCoil.net_force`` reach the mutual field by
    different quadratures, so native's own analogous check (a 4-base-curve,
    untwisted fixture in ``test_force_objectives``) budgets ``rtol=1e-6``;
    this test reuses that budget on a smaller 2-base-curve, untwisted
    fixture, which is not identical to native's but exercises the same
    quadrature-mismatch mechanism.
    """
    _, coils = _symmetric_regularized_coils(
        base_curve_count=2, regularization=regularization
    )
    target = coils[0]
    net_force_mn = np.asarray(target.net_force(coils)) / 1e6
    assert np.linalg.norm(net_force_mn) > 0.0

    np.testing.assert_allclose(
        float(jax_force.SquaredMeanForce(target, coils).J()),
        float(net_force_mn @ net_force_mn),
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "regularization", (regularization_circ(0.05), regularization_rect(0.01, 0.023))
)
def test_squared_mean_torque_equals_native_net_torque_integral(regularization):
    """SquaredMeanTorque reproduces ||net torque||^2 in MN^2 for a twisted coil set.

    The coil orientations are scrambled exactly as the native suite does so
    the net torque is nonzero; native's own analogous single-target check
    budgets ``rtol=1e-2``, and this comparison is held about 10x tighter, at
    ``rtol=1e-3``.
    """
    _, coils = _symmetric_regularized_coils(
        base_curve_count=2, regularization=regularization, twist=0.1
    )
    target = coils[0]
    net_torque_mn = np.asarray(target.net_torque(coils)) / 1e6
    assert np.linalg.norm(net_torque_mn) > 0.0

    np.testing.assert_allclose(
        float(jax_force.SquaredMeanTorque(target, coils).J()),
        float(net_torque_mn @ net_torque_mn),
        rtol=1e-3,
        atol=0.0,
    )


def test_lp_objectives_equal_independent_force_and_torque_integrals():
    """Lp objectives equal (1/p) integral of max(|density| - threshold, 0)^p.

    Uses native's own 4-base-curve ``test_force_objectives`` configuration
    (``ncoils=4``, ``nfp=3``, ``I=1.7e4``): with 2 base curves the mutual
    force never clears ``_FORCE_THRESHOLD_MN_PER_M`` (measured
    max|F|=6.4e-4 MN/m against the threshold's 1e-3), so the LpCurveForce
    comparison below would be a vacuous ``0.0 == 0.0``.
    """
    _, coils = _symmetric_regularized_coils(base_curve_count=4, twist=0.1)
    target = coils[0]
    speed = np.linalg.norm(np.asarray(target.curve.gammadash()), axis=1)

    force_norms_mn = np.linalg.norm(np.asarray(target.force(coils)), axis=1) / 1e6
    expected_force = (
        (1 / _LP_POWER)
        * np.sum(
            np.maximum(force_norms_mn - _FORCE_THRESHOLD_MN_PER_M, 0) ** _LP_POWER
            * speed
        )
        / speed.shape[0]
    )
    assert expected_force > 0.0
    np.testing.assert_allclose(
        float(
            jax_force.LpCurveForce(
                target,
                coils,
                p=_LP_POWER,
                threshold=_FORCE_THRESHOLD_MN_PER_M,
            ).J()
        ),
        expected_force,
        rtol=1e-6,
    )

    torque_norms_mn = np.linalg.norm(np.asarray(target.torque(coils)), axis=1) / 1e6
    expected_torque = (
        (1 / _LP_POWER) * np.sum(torque_norms_mn**_LP_POWER * speed) / speed.shape[0]
    )
    assert expected_torque > 0.0
    np.testing.assert_allclose(
        float(jax_force.LpCurveTorque(target, coils, p=_LP_POWER, threshold=0.0).J()),
        expected_torque,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("objective_name", "downsample", "rtol"),
    (
        # SquaredMeanForce ds=2 and LpCurveTorque ds=2 are held to native's
        # own exact budgets (rtol=1e-6 and 1e-4 respectively, from
        # ``test_force_objectives``'s downsample=2 checks); measured on this
        # fixture, SquaredMeanForce ds=2 drifts ~2.0e-9 (500x headroom) and
        # LpCurveTorque ds=2 drifts ~9.3e-5 (a tight ~8% margin against its
        # 1e-4 budget). The other rows already matched native and are
        # unchanged.
        ("SquaredMeanForce", 2, 1e-6),
        ("SquaredMeanForce", 3, 1e-3),
        ("LpCurveForce", 2, 1e-2),
        ("LpCurveForce", 3, 1e-2),
        ("SquaredMeanTorque", 2, 1e-2),
        ("SquaredMeanTorque", 3, 1e-2),
        ("LpCurveTorque", 2, 1e-4),
        ("LpCurveTorque", 3, 1e-2),
    ),
)
def test_downsampled_objectives_track_full_resolution(objective_name, downsample, rtol):
    """Downsampling trades accuracy for speed within the native suite's budgets."""
    _, coils = _symmetric_regularized_coils(base_curve_count=2, twist=0.1)
    objective_class = getattr(jax_force, objective_name)
    kwargs = _objective_kwargs(objective_name, threshold=0.0)

    full = float(objective_class(coils[0], coils, **kwargs).J())
    reduced = float(
        objective_class(coils[0], coils, downsample=downsample, **kwargs).J()
    )

    assert full > 0.0
    np.testing.assert_allclose(reduced, full, rtol=rtol)


@pytest.mark.parametrize("objective_name", _OBJECTIVE_NAMES + ("B2Energy", "NetFluxes"))
def test_taylor_test_confirms_jax_objective_gradients(objective_name):
    """Central finite differences converge quadratically onto the analytic dJ.

    Matches native's own factor: quartering the step should quarter the
    error (ideal ratio 0.25); measured on this fixture, every objective's
    worst step-to-step ratio is <= 0.252, comfortably inside native's 0.3
    budget.
    """
    _, coils = _symmetric_regularized_coils(
        base_curve_count=2, quadpoint_count=30, twist=0.1
    )
    if objective_name == "B2Energy":
        objective = jax_force.B2Energy(coils)
    elif objective_name == "NetFluxes":
        objective = jax_force.NetFluxes(coils[0], coils[1:])
    else:
        objective = getattr(jax_force, objective_name)(
            coils[0], coils, **_objective_kwargs(objective_name, threshold=0.0)
        )

    dofs = np.copy(objective.x)
    direction = np.ones_like(dofs)
    directional_derivative = float(np.sum(np.asarray(objective.dJ()) * direction))
    assert abs(directional_derivative) > 0.0

    previous_error = None
    for exponent in range(10, 16):
        step = 0.5**exponent
        objective.x = dofs + step * direction
        plus = float(objective.J())
        objective.x = dofs - step * direction
        minus = float(objective.J())
        error = abs((plus - minus) / (2.0 * step) - directional_derivative) / abs(
            directional_derivative
        )
        if previous_error is not None:
            assert error <= 0.3 * previous_error, (
                f"{objective_name}: Taylor error did not shrink at step {step:.3e} "
                f"(previous {previous_error:.3e}, current {error:.3e})"
            )
        previous_error = error
    objective.x = dofs

    # A gradient of the wrong magnitude can still decay; pin the residual too.
    assert previous_error < 1e-4, (
        f"{objective_name}: finite differences never reached the analytic dJ "
        f"(residual {previous_error:.3e})"
    )


# --------------------------------------------------------------------------
# CoilForcesTest mirror — mixed-quadpoint composition and coarse/fine sources.
# --------------------------------------------------------------------------


# Distinct, nonzero per-coil tilts for the concentric circles below: a fully
# coplanar set has zero mutual torque by rotational symmetry (see
# ``_xyz_circle``), which starves both the Lp-torque value (identically
# clipped to zero by any positive threshold) and its gradient (``nan`` from
# ``jnp.linalg.norm`` at the exact-zero torque vector).
_MIXED_QUADPOINT_TILT_STEP = 0.3

# The threshold budget for the base force fixture (``_FORCE_THRESHOLD_MN_PER_M``
# = 1e-3) exceeds every mutual-force/torque magnitude this weaker, small-current
# concentric-circle fixture produces (measured max ~6e-4 MN/m even at full
# 4-coil coupling, and under 1e-4 MN/m for the weakest single-pair combination
# below); a lower, fixture-specific threshold keeps the Lp objectives live.
_MIXED_QUADPOINT_LP_THRESHOLD = 1e-5


def _mixed_quadpoint_coils():
    """Two coarse (40-quadpoint) and two fine (60-quadpoint) circular coils.

    Each of the four coils gets a distinct ``_xyz_circle`` tilt so mutual
    forces and torques are genuinely nonzero rather than an artifact of four
    concentric, coplanar circles (see ``_MIXED_QUADPOINT_TILT_STEP``).
    """
    regularization = regularization_circ(0.05)
    current = 1.7e4
    coarse = [
        RegularizedCoil(
            _xyz_circle(scale, 40, tilt=(index + 1) * _MIXED_QUADPOINT_TILT_STEP),
            Current(current),
            regularization,
        )
        for index, scale in enumerate((1.0, 1.2))
    ]
    fine = [
        RegularizedCoil(
            _xyz_circle(scale, 60, tilt=(index + 3) * _MIXED_QUADPOINT_TILT_STEP),
            Current(current),
            regularization,
        )
        for index, scale in enumerate((0.8, 1.5))
    ]
    return coarse, fine


@pytest.mark.parametrize("objective_name", _OBJECTIVE_NAMES)
def test_objectives_are_finite_across_quadpoint_groups(objective_name):
    """Targets and sources may live on different quadrature grids.

    Lp objectives are additionally checked ``> 0`` rather than merely
    finite: with the tilted fixture and ``_MIXED_QUADPOINT_LP_THRESHOLD``
    every one of the four target/source combinations below clears the
    threshold, so a zero value would signal a real regression rather than a
    fixture that never exercised the threshold subtraction.
    """
    group_a, group_b = _mixed_quadpoint_coils()
    objective_class = getattr(jax_force, objective_name)
    kwargs = _objective_kwargs(objective_name, threshold=_MIXED_QUADPOINT_LP_THRESHOLD)
    is_lp = objective_name.startswith("Lp")

    for targets, sources in (
        (group_a[0], group_b[0]),
        (group_a[0], group_b),
        (group_a, group_b),
        (group_b[0], group_a),
    ):
        value = float(objective_class(targets, sources, **kwargs).J())
        if is_lp:
            assert value > 0.0
        else:
            assert np.isfinite(value)

    gradient = np.asarray(objective_class(group_a[0], group_b, **kwargs).dJ())
    assert len(gradient) > 0
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("objective_name", _OBJECTIVE_NAMES)
def test_per_coil_objectives_sum_to_the_combined_objective(objective_name):
    """Summing single-target objectives reproduces the multi-target objective.

    Values and gradients are asserted finite before comparison: on an
    untilted fixture ``LpCurveTorque``'s gradient was silently all-``nan``
    (``jnp.linalg.norm``'s gradient at an exact-zero torque vector) and
    ``np.testing.assert_allclose`` defaults to ``equal_nan=True``, so that
    comparison passed without ever exercising real numbers.  The tolerances
    below are measured margins against the tilted, nondegenerate fixture,
    not native budgets — this is a pure-JAX self-consistency check (no
    native comparison here), and per-coil-vs-combined evaluation reorders
    floating-point accumulation, which the native suite's own analogous
    check documents as a source of small but real drift.
    """
    group_a, _ = _mixed_quadpoint_coils()
    external = [
        RegularizedCoil(
            _xyz_circle(scale, 40, tilt=(index + 5) * _MIXED_QUADPOINT_TILT_STEP),
            Current(1.7e4),
            regularization_circ(0.05),
        )
        for index, scale in enumerate((0.8, 1.5))
    ]
    objective_class = getattr(jax_force, objective_name)
    kwargs = _objective_kwargs(objective_name, threshold=_MIXED_QUADPOINT_LP_THRESHOLD)

    per_coil_terms = [
        objective_class(
            coil,
            [other for other in group_a if other is not coil] + external,
            **kwargs,
        )
        for coil in group_a
    ]
    per_coil_values = np.array([float(term.J()) for term in per_coil_terms])
    assert np.all(np.isfinite(per_coil_values)), per_coil_values

    combined = objective_class(group_a, external, **kwargs)
    combined_value = float(combined.J())
    assert np.isfinite(combined_value)
    assert combined_value > 0.0
    np.testing.assert_allclose(
        per_coil_values.sum(), combined_value, rtol=1e-6, atol=0.0
    )

    per_coil_gradient_terms = [term.dJ(partials=True) for term in per_coil_terms]
    combined_gradient = np.asarray(combined.dJ())
    assert np.all(np.isfinite(combined_gradient))
    per_coil_gradient = sum(per_coil_gradient_terms)
    per_coil_gradient_array = np.asarray(per_coil_gradient(combined))
    assert np.all(np.isfinite(per_coil_gradient_array))
    np.testing.assert_allclose(
        combined_gradient,
        per_coil_gradient_array,
        rtol=1e-4,
        atol=1e-20,
    )


@pytest.mark.parametrize("objective_name", _OBJECTIVE_NAMES)
def test_coarse_fine_source_split_matches_all_coarse_sources(objective_name):
    """Splitting sources into coarse and fine groups leaves J and dJ unchanged."""
    _, coils = _symmetric_regularized_coils(
        base_curve_count=3, nfp=2, quadpoint_count=30, twist=0.1
    )
    target = coils[0]
    sources = coils[1:]
    split_index = len(sources) // 2
    objective_class = getattr(jax_force, objective_name)
    kwargs = _objective_kwargs(objective_name, threshold=0.0)

    all_coarse = objective_class(target, sources, **kwargs)
    split = objective_class(
        target,
        sources[:split_index],
        source_coils_fine=sources[split_index:],
        **kwargs,
    )

    np.testing.assert_allclose(
        float(split.J()), float(all_coarse.J()), rtol=1e-8, atol=1e-30
    )
    np.testing.assert_allclose(
        np.asarray(split.dJ()), np.asarray(all_coarse.dJ()), rtol=1e-6, atol=2e-22
    )

    empty_fine = objective_class(target, sources, source_coils_fine=[], **kwargs)
    np.testing.assert_allclose(
        float(empty_fine.J()), float(all_coarse.J()), rtol=1e-10, atol=1e-30
    )


# --------------------------------------------------------------------------
# CoilForcesTest mirror — guard behaviors.
# --------------------------------------------------------------------------


def _downsample_guard_coils():
    regularization = regularization_circ(0.05)
    current = 1.7e4
    indivisible = RegularizedCoil(
        _xyz_circle(1.0, 20), Current(current), regularization
    )
    divisible = RegularizedCoil(_xyz_circle(1.0, 21), Current(current), regularization)
    divisible_other = RegularizedCoil(
        _xyz_circle(1.1, 21), Current(current), regularization
    )
    return indivisible, divisible, divisible_other


def test_downsample_must_divide_target_quadpoints():
    """A downsample factor that does not divide the target grid is rejected."""
    indivisible, divisible, divisible_other = _downsample_guard_coils()
    sources = [divisible, divisible_other]

    for module in (native_force, jax_force):
        with pytest.raises(ValueError):
            module.B2Energy([indivisible], downsample=7)
        with pytest.raises(ValueError):
            module.NetFluxes(indivisible, [divisible], downsample=7)
        for objective_name in _OBJECTIVE_NAMES:
            with pytest.raises(ValueError):
                getattr(module, objective_name)(
                    indivisible,
                    sources,
                    downsample=7,
                    **_objective_kwargs(objective_name),
                )


def test_downsample_must_divide_source_quadpoints():
    """A downsample factor that does not divide the source grid is rejected."""
    indivisible, divisible, divisible_other = _downsample_guard_coils()
    sources = [indivisible, divisible_other]

    for module in (native_force, jax_force):
        with pytest.raises(ValueError):
            module.NetFluxes(divisible, [indivisible], downsample=7)
        for objective_name in _OBJECTIVE_NAMES:
            with pytest.raises(ValueError):
                getattr(module, objective_name)(
                    divisible,
                    sources,
                    downsample=7,
                    **_objective_kwargs(objective_name),
                )


def test_mixed_quadpoints_within_a_coil_list_are_rejected():
    """A single target or source list may not mix quadrature grids."""
    group_a, group_b = _mixed_quadpoint_coils()

    for module in (native_force, jax_force):
        with pytest.raises(ValueError):
            module.B2Energy([group_a[0], group_b[0]])
        for objective_name in _OBJECTIVE_NAMES:
            kwargs = _objective_kwargs(objective_name)
            with pytest.raises(ValueError):
                getattr(module, objective_name)(
                    [group_a[0], group_b[0]], group_a[1], **kwargs
                )
            with pytest.raises(ValueError):
                getattr(module, objective_name)(
                    group_a[0], [group_a[1], group_b[0]], **kwargs
                )


def test_self_field_objectives_require_regularized_coils():
    """Objectives that read a self-field reject plain Coil targets; net ones accept them."""
    regularization = regularization_circ(0.05)
    current = Current(1.7e4)
    plain = Coil(_xyz_circle(1.0, 20), current)
    regularized = RegularizedCoil(_xyz_circle(1.0, 20), current, regularization)
    regularized_other = RegularizedCoil(_xyz_circle(1.1, 20), current, regularization)

    for module in (native_force, jax_force):
        with pytest.raises(ValueError):
            module.LpCurveForce(
                plain, [regularized], p=_LP_POWER, threshold=_FORCE_THRESHOLD_MN_PER_M
            )
        with pytest.raises(ValueError):
            module.LpCurveTorque(
                plain, [regularized], p=_LP_POWER, threshold=_FORCE_THRESHOLD_MN_PER_M
            )
        with pytest.raises(ValueError):
            module.B2Energy([plain, regularized])

        # Net force and net torque carry no self-field term, so plain coils are fine.
        assert np.isfinite(float(module.SquaredMeanForce(plain, [regularized]).J()))
        assert np.isfinite(float(module.SquaredMeanTorque(plain, [regularized]).J()))
        assert np.isfinite(
            float(
                module.LpCurveForce(
                    regularized,
                    [regularized, regularized_other],
                    p=_LP_POWER,
                    threshold=_FORCE_THRESHOLD_MN_PER_M,
                ).J()
            )
        )
        assert np.isfinite(float(module.B2Energy([regularized]).J()))


@pytest.mark.parametrize("objective_name", _OBJECTIVE_NAMES)
def test_sources_must_contain_a_coil_outside_the_target_group(objective_name):
    """A source list that only repeats the target leaves no mutual interaction."""
    group_a, _ = _mixed_quadpoint_coils()
    kwargs = _objective_kwargs(objective_name)

    for module in (native_force, jax_force):
        with pytest.raises(ValueError):
            getattr(module, objective_name)(
                group_a[0], [group_a[0], group_a[0]], **kwargs
            )


def test_regularized_coil_force_quantities_have_native_shapes_and_are_finite():
    """The JAX force-norm lane returns one finite magnitude per target quadpoint."""
    _, coils = _symmetric_regularized_coils(base_curve_count=2, twist=0.1)
    target = coils[0]
    sources = [coil for coil in coils if coil is not target]
    quadpoint_count = len(target.curve.quadpoints)

    norms = _jax_force_norms_mn_per_m(
        target.curve,
        target.current.get_value(),
        target.regularization,
        source_curves=[coil.curve for coil in sources],
        source_currents=[coil.current.get_value() for coil in sources],
    )

    assert norms.shape == (quadpoint_count,)
    assert np.all(np.isfinite(norms))
    assert np.all(norms > 0.0)

    # Downsampling thins the target grid *and* the source quadrature, so the
    # reduced profile approximates - rather than subsets - the full one.
    downsampled = _jax_force_norms_mn_per_m(
        target.curve,
        target.current.get_value(),
        target.regularization,
        source_curves=[coil.curve for coil in sources],
        source_currents=[coil.current.get_value() for coil in sources],
        downsample=2,
    )
    assert downsampled.shape == (quadpoint_count // 2,)
    np.testing.assert_allclose(downsampled, norms[::2], rtol=1e-3)
