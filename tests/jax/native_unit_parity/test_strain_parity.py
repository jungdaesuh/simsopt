"""Native-oracle unit-parity tests for the JAX strain-optimization example.

Mirrors the three ``unittest`` methods in ``tests/geo/test_strainopt.py``
(``test_strain_opt``, ``test_torsion``, ``test_binormal_curvature``) against
``src/simsopt_jax/examples/strain_optimization.py``, using the native
implementation (``simsopt.geo.strain_optimization`` /
``simsopt.geo.framedcurve``) as the in-process oracle.

Scope note (see the coverage manifest rows ST-1/ST-2/ST-3 in
``tests/fixtures/jax_native_unit_coverage_manifest.json`` for the full
manifest-fact list): the native suite exercises BOTH the coil-centroid and
Frenet reference frames, and its LP strain penalties differentiate with
respect to the full free-DOF vector of ``LPTorsionalStrainPenalty``/
``LPBinormalCurvatureStrainPenalty`` -- which includes the underlying
curve's shape DOFs whenever the curve itself is not fixed. The JAX module
only implements the coil-CENTROID frame (``rotated_centroid_frame``/
``rotated_centroid_frame_dash`` in ``simsopt_jax.core.framedcurve``, a
line-for-line port of the native centroid kernels) and only differentiates
its combined strain objective with respect to the rotation DOFs:
``gamma``/``gammadash``/``gammadashdash`` are consumed as fixed device
arrays, never generated from curve DOFs under JAX autodiff. Every test below
therefore fixes the native curve's DOFs (``curve.fix_all()``, matching
native's own ``test_strain_opt`` and isolating exactly the rotation-DOF
subspace the JAX example implements) and restricts value/gradient
comparisons to the centroid frame. The Frenet-frame and curve-shape-gradient
gaps are NOT silently skipped -- they are recorded as manifest-facts
(jax_missing / jax_partial) in ST-1 (Frenet-frame branch, disposition
jax_partial) and ST-2 (curve-shape-DOF gradients, disposition jax_missing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_strict_parity_backend, parity_default_device

from simsopt.configs.zoo import get_data
from simsopt.geo import (
    CoilStrain,
    FrameRotation,
    FramedCurveCentroid,
    LPBinormalCurvatureStrainPenalty,
    LPTorsionalStrainPenalty,
    ZeroRotation,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.examples.strain_optimization import (
    _strain_values,
    solve_strain_rotation,
)
from simsopt_jax.parity_tolerances import parity_ladder_tolerances

# The centroid-frame kernels underneath ``_strain_values`` are a line-for-line
# JAX port of the native centroid math (see
# ``simsopt_jax/core/framedcurve.py`` module docstring), so raw strain
# *values* are compared at the tight same-state ``direct_kernel`` tier.
_DIRECT_KERNEL_TOLS = parity_ladder_tolerances("direct_kernel")
_VALUE_RTOL = _DIRECT_KERNEL_TOLS["rtol"]
_VALUE_ATOL = _DIRECT_KERNEL_TOLS["atol"]

# Tape width shared by both the native ``CoilStrain``/LP-penalty
# construction and the JAX ``objective_width``/``reporting_width`` argument,
# matching the width used throughout ``tests/geo/test_strainopt.py``.
_WIDTH = 1.0e-3


@pytest.fixture(autouse=True)
def _strict_parity_lane(monkeypatch, request, parity_lane):
    """Make every test in this module GPU-capable via ``parity_lane``.

    ``parity_lane`` is parametrized ``("cpu", "gpu")`` by
    ``tests/conftest.py``; the "gpu" case skips cleanly
    (``pytest.skip("CUDA GPU not available")``) whenever no CUDA device is
    registered. The GPU lane activates automatically the moment a CUDA
    device is visible to JAX in the running environment -- it is not
    permanently disabled, just currently unexercised because this wave's
    validation environment has no CUDA device registered.
    """
    enable_strict_parity_backend(monkeypatch, request, parity_lane)
    with parity_default_device(parity_lane):
        yield


def _fixed_ncsx_curve():
    """Return NCSX base curve 0, DOFs fixed (matches JAX's fixed-gamma scope)."""
    base_curves, _base_currents, _axis, _nfp, _field = get_data(
        "ncsx", coil_order=6, points_per_period=120
    )
    curve = base_curves[0]
    curve.fix_all()
    return curve


def _curve_arrays(curve):
    return (
        np.asarray(curve.quadpoints, dtype=np.float64),
        np.asarray(curve.gamma(), dtype=np.float64),
        np.asarray(curve.gammadash(), dtype=np.float64),
        np.asarray(curve.gammadashdash(), dtype=np.float64),
    )


def _native_rotation_and_dofs(curve, order):
    """Build the native rotation object and its matching JAX DOF encoding.

    ``order=1`` mirrors native's explicit ``rotation.x = [0, 0.1, 0.3]``
    case; ``order=None`` mirrors native's ``ZeroRotation`` case, encoded on
    the JAX side as ``rotation_order=0`` with a single zero DOF (JAX's
    ``rotation_alpha`` broadcasts ``dofs[0]`` when ``order=0``, giving the
    same identically-zero angle as ``ZeroRotation``).
    """
    if order == 1:
        rotation = FrameRotation(curve.quadpoints, 1)
        rotation.x = np.array([0.0, 0.1, 0.3])
        return rotation, 1, np.array([0.0, 0.1, 0.3])
    assert order is None
    rotation = ZeroRotation(curve.quadpoints)
    return rotation, 0, np.array([0.0])


def _assert_strain_values_match_native(order):
    curve = _fixed_ncsx_curve()
    quadpoints, gamma, gammadash, gammadashdash = _curve_arrays(curve)
    rotation, rotation_order, rotation_dofs = _native_rotation_and_dofs(curve, order)

    framedcurve = FramedCurveCentroid(curve, rotation)
    native_strain = CoilStrain(framedcurve, _WIDTH)
    native_torsion = np.asarray(native_strain.torsional_strain(), dtype=np.float64)
    native_binormal = np.asarray(
        native_strain.binormal_curvature_strain(), dtype=np.float64
    )

    jax_torsion, jax_binormal, _arc_length = _strain_values(
        quadpoints,
        gamma,
        gammadash,
        gammadashdash,
        jnp.asarray(rotation_dofs),
        rotation_order=rotation_order,
        width=_WIDTH,
    )

    np.testing.assert_allclose(
        np.asarray(jax_torsion),
        native_torsion,
        rtol=_VALUE_RTOL,
        atol=_VALUE_ATOL,
        err_msg=f"torsional_strain mismatch at order={order}",
    )
    np.testing.assert_allclose(
        np.asarray(jax_binormal),
        native_binormal,
        rtol=_VALUE_RTOL,
        atol=_VALUE_ATOL,
        err_msg=f"binormal_curvature_strain mismatch at order={order}",
    )
    return (
        curve,
        rotation,
        framedcurve,
        native_strain,
        (
            quadpoints,
            gamma,
            gammadash,
            gammadashdash,
        ),
    )


def _jax_component_gradient(
    *,
    quadpoints,
    gamma,
    gammadash,
    gammadashdash,
    jax_component_index,
    rotation_dofs,
):
    """Autodiff gradient of ``sum(_strain_values(...)[jax_component_index])``.

    NOTE on what this differentiates (see module docstring's Scope note and
    the per-component asserters below for the full disclosure): this is the
    gradient of the RAW, un-thresholded, un-penalized strain sum -- NOT the
    gradient of native's ``LPTorsionalStrainPenalty``/
    ``LPBinormalCurvatureStrainPenalty`` (the ``p``-th-power, thresholded,
    arc-length-weighted penalty that native's own ``test_torsion``/
    ``test_binormal_curvature`` Taylor-check via ``J.dJ()``). The two
    quantities are deliberately different functionals; see
    ``_assert_torsion_gradient_matches_native`` and
    ``_assert_binormal_gradient_matches_native`` for why, and do not read
    agreement here as evidence the LP-penalty gradient path is covered.
    """

    def jax_reduction(dofs: jax.Array) -> jax.Array:
        return jnp.sum(
            _strain_values(
                quadpoints,
                gamma,
                gammadash,
                gammadashdash,
                dofs,
                rotation_order=1,
                width=_WIDTH,
            )[jax_component_index]
        )

    return np.asarray(
        jax.grad(jax_reduction)(jnp.asarray(rotation_dofs)), dtype=np.float64
    )


def _assert_torsion_gradient_matches_native(
    *,
    quadpoints,
    gamma,
    gammadash,
    gammadashdash,
    rotation,
    native_strain_component,
    rotation_dofs,
):
    """Cross-check the JAX torsion gradient against a fixed-tolerance native FD.

    Substitution disclosure: this Taylor-checks the gradient of the RAW
    summed torsional strain (``CoilStrain.torsional_strain()``), not native's
    ``LPTorsionalStrainPenalty(..., p=2, threshold=1e-8).dJ()``
    (``tests/geo/test_strainopt.py:118``), which is what native's own
    ``test_torsion`` Taylor-checks. The JAX example exposes only the fused,
    hard-wired-threshold combined objective
    (``_strain_objective``, ``src/simsopt_jax/examples/strain_optimization.py:128-168``)
    as a differentiable primitive -- there is no JAX-side standalone
    LP-penalty gradient for one strain component to check against native's
    ``dJ()`` at native's own threshold. This is a deliberate, disclosed
    substitution, not a claim of equivalence with native's LP-penalty
    gradient check; the strain domain is `jax_partial` in the coverage
    manifest (ST-1/ST-3) precisely because of gaps like this one.

    Reference derivative: a central finite difference of the NATIVE
    ``CoilStrain.torsional_strain()`` sum (a true cross-implementation
    oracle), not a JAX-side finite difference -- this checks that JAX's
    autodiff gradient of ``_strain_values`` agrees with native's
    centroid-frame math, not merely that JAX agrees with itself.

    Tolerance rationale (fixed floor, not native's shrinking-``eps``
    protocol): a rotated centroid frame's torsion is exactly
    ``torsion(alpha=0) + dalpha/dl`` (an identity of the rotation
    construction), i.e. affine in the rotation DOFs, so
    ``torsional_strain = torsion**2 * width**2/12`` is an exactly quadratic
    polynomial in the DOFs. A central difference of an exactly-quadratic
    function has ZERO truncation error at every step size (its third
    derivative is identically zero) -- native's shrinking-error assertion
    (``tests/geo/test_strainopt.py:80-89``, ``errf < 0.3 * errf_old``) would
    then be testing pure floating-point round-off noise, which does not
    shrink monotonically, and would false-reject a correct gradient
    (measured empirically: the round-off floor sits at ~1e-19..1e-15 for
    every ``eps`` from ``0.5**2`` down to ``0.5**19``, never showing the
    O(eps**2) trend native's protocol assumes -- confirmed non-monotonic at
    several of those steps). Instead this asserts a fixed floor of
    ``1e-12``: the measured error at ``eps in (11, 12, 13)`` sits at
    ~1.3e-17..1.3e-16 (a ~1e4-1e5 margin below the bound), while a
    deliberately injected 1% gradient error raises the same error to
    ~5.2e-7, comfortably above the bound -- so the tighter fixed floor
    (unlike the module's prior ``max(1e-6, 1e-3*|df|)`` tolerance, which a
    1% wrong gradient passed) has real discriminating power.
    """
    jax_grad = _jax_component_gradient(
        quadpoints=quadpoints,
        gamma=gamma,
        gammadash=gammadash,
        gammadashdash=gammadashdash,
        jax_component_index=0,
        rotation_dofs=rotation_dofs,
    )
    rotation.x = rotation_dofs

    # DOF 0 is the constant term of the rotation angle's Fourier series
    # (``rotation.x = [const, cos(1*theta), sin(1*theta)]``). Torsion depends
    # on the rotation only through ``dalpha/dl``, and the derivative of a
    # constant is identically zero, so ``d(torsional_strain)/d(dof_0)`` is
    # STRUCTURALLY zero -- not merely small. A regression that made this
    # component nonzero (e.g. a sign/index error mixing up which DOF is the
    # constant term) would not be caught by the directional check below,
    # since a random probe direction only rarely isolates one axis; assert
    # it directly. Measured value ~9.7e-21; floor chosen with an ~1e6 margin.
    assert abs(jax_grad[0]) < 1.0e-15, (
        f"torsion gradient w.r.t. the constant rotation-angle DOF should be "
        f"structurally zero (see comment above); got {jax_grad[0]!r}"
    )

    def native_reduction(dofs: np.ndarray) -> float:
        rotation.x = dofs
        return float(np.sum(np.asarray(native_strain_component())))

    np.random.seed(1)
    h = np.random.standard_normal(size=rotation_dofs.shape)
    df = float(np.sum(jax_grad * h))

    for i in (11, 12, 13):
        eps = 0.5**i
        f1 = native_reduction(rotation_dofs + eps * h)
        f2 = native_reduction(rotation_dofs - eps * h)
        errf = abs((f1 - f2) / (2 * eps) - df)
        assert errf < 1.0e-12, (
            f"JAX torsion gradient disagrees with native finite difference "
            f"at eps={eps}: errf={errf} df={df}"
        )

    # Restore the DOFs the value comparison left the rotation object at.
    rotation.x = rotation_dofs


def _assert_binormal_gradient_matches_native(
    *,
    quadpoints,
    gamma,
    gammadash,
    gammadashdash,
    rotation,
    native_strain_component,
    rotation_dofs,
):
    """Cross-check the JAX binormal-curvature gradient via native's own FD protocol.

    Substitution disclosure: this Taylor-checks the gradient of the RAW
    summed binormal-curvature strain (``CoilStrain.binormal_curvature_strain()``),
    not native's ``LPBinormalCurvatureStrainPenalty(..., p=2,
    threshold=1e-4).dJ()`` (``tests/geo/test_strainopt.py:76``) -- see
    ``_assert_torsion_gradient_matches_native``'s docstring for why (same
    reasoning applies here; the JAX example has no standalone LP-penalty
    gradient primitive to check against native's ``dJ()`` at native's
    threshold).

    Unlike the torsion component, binormal curvature's rotation dependence
    is trigonometric, not affine, so the central-difference truncation error
    is NOT identically zero and DOES shrink like O(eps**2) as eps shrinks --
    this mirrors native's own shrinking-``eps`` protocol exactly
    (``tests/geo/test_strainopt.py:80-89``: ``eps in 0.5**range(9, 14)``,
    ``errf < 0.3 * errf_old``) rather than substituting a fixed tolerance.
    Measured: the uncorrupted gradient shrinks monotonically at every step
    (errf ~1.1e-3 -> ~4.1e-9 across the five steps); a deliberately injected
    1% gradient error breaks the shrink (errf plateaus at ~2.4e-3, since the
    finite difference converges to the TRUE derivative while the asserted
    ``df`` is off by a fixed 1%, so ``errf < 0.3 * errf_old`` fails once the
    truncation error drops below the injected bias).
    """
    jax_grad = _jax_component_gradient(
        quadpoints=quadpoints,
        gamma=gamma,
        gammadash=gammadash,
        gammadashdash=gammadashdash,
        jax_component_index=1,
        rotation_dofs=rotation_dofs,
    )
    rotation.x = rotation_dofs

    def native_reduction(dofs: np.ndarray) -> float:
        rotation.x = dofs
        return float(np.sum(np.asarray(native_strain_component())))

    np.random.seed(1)
    h = np.random.standard_normal(size=rotation_dofs.shape)
    df = float(np.sum(jax_grad * h))

    errf_old = 1.0e10
    for i in range(9, 14):
        eps = 0.5**i
        f1 = native_reduction(rotation_dofs + eps * h)
        f2 = native_reduction(rotation_dofs - eps * h)
        errf = abs((f1 - f2) / (2 * eps) - df)
        assert errf < 0.3 * errf_old, (
            f"JAX binormal-curvature gradient FD error failed to shrink at "
            f"eps={eps}: errf={errf} errf_old={errf_old} df={df}"
        )
        errf_old = errf

    # Restore the DOFs the value comparison left the rotation object at.
    rotation.x = rotation_dofs


def test_torsional_strain_matches_native_and_gradient_taylor_checks():
    """Mirrors native ``test_torsion`` (centroid frame; rotation-DOF scope)."""
    for order in (None, 1):
        (
            curve,
            rotation,
            framedcurve,
            native_strain,
            arrays,
        ) = _assert_strain_values_match_native(order)
        if order is None:
            # ZeroRotation carries zero free DOFs: native's own Taylor check
            # degenerates to a no-op vector of length 0 in this case, so we
            # stop at the value comparison already performed above.
            continue
        quadpoints, gamma, gammadash, gammadashdash = arrays
        _assert_torsion_gradient_matches_native(
            quadpoints=quadpoints,
            gamma=gamma,
            gammadash=gammadash,
            gammadashdash=gammadashdash,
            rotation=rotation,
            native_strain_component=native_strain.torsional_strain,
            rotation_dofs=np.array([0.0, 0.1, 0.3]),
        )


def test_binormal_curvature_strain_matches_native_and_gradient_taylor_checks():
    """Mirrors native ``test_binormal_curvature`` (centroid frame; rotation-DOF scope)."""
    for order in (None, 1):
        (
            curve,
            rotation,
            framedcurve,
            native_strain,
            arrays,
        ) = _assert_strain_values_match_native(order)
        if order is None:
            continue
        quadpoints, gamma, gammadash, gammadashdash = arrays
        _assert_binormal_gradient_matches_native(
            quadpoints=quadpoints,
            gamma=gamma,
            gammadash=gammadash,
            gammadashdash=gammadashdash,
            rotation=rotation,
            native_strain_component=native_strain.binormal_curvature_strain,
            rotation_dofs=np.array([0.0, 0.1, 0.3]),
        )


def test_strain_optimization_vanishes_matches_native_centroid_frame():
    """Mirrors native ``test_strain_opt`` (centroid branch only).

    Native asserts that a circular coil's torsional and binormal-curvature
    LP strain penalties can be driven below ``1e-12`` by optimizing only the
    rotation DOFs. This test runs the JAX on-device solver
    (``solve_strain_rotation``) on the identical curve/rotation setup and
    then plugs the JAX-optimized rotation DOFs back into the NATIVE
    ``LPTorsionalStrainPenalty``/``LPBinormalCurvatureStrainPenalty``
    objectives (the native oracle) to confirm the vanishing-strain claim
    holds under native's own penalty definition, not only under JAX's.

    Threshold notes (measured, both oracle-level bars tightened to match
    native's own ``< 1e-12`` bar in ``tests/geo/test_strainopt.py``):
    ``native_Jt.J()`` at the JAX optimum measures ~3.08e-49 and
    ``native_Jb.J()`` measures ~1.86e-26 -- both pass ``1e-12`` with an
    enormous margin (>1e14x). ``final.maximum_torsional_strain`` (JAX's own
    diagnostic, the raw per-point strain max, a DIFFERENT quantity from the
    LP-penalty integral above) measures ~7.0e-23 and also tightens cleanly
    to ``1e-12``. ``final.maximum_binormal_curvature_strain`` measures
    ~1.36e-11 at convergence -- reproducibly, independent of ``maxiter``
    (identical to 5 significant figures at maxiter in
    {200, 1000, 5000, 20000}; the JAX L-BFGS solver plateaus at 28 iterations
    regardless of budget) -- so it is a genuine convergence floor of this
    solver on this problem, not an iteration-budget artifact, and CANNOT be
    tightened to ``1e-12`` without producing a test that fails against
    correct, converged output. It is tightened to ``1e-9`` instead: still
    four orders of magnitude tighter than the original ``1e-8``, with a
    healthy ~74x margin above the measured floor.
    """
    quadpoints = np.linspace(0, 1, 10, endpoint=False)
    curve = CurveXYZFourier(quadpoints, order=1)
    curve.set("xc(1)", 1e-4)
    curve.set("ys(1)", 1e-4)
    curve.fix_all()

    rotation_order = 2
    np.random.seed(1)
    rotation = FrameRotation(quadpoints, rotation_order)
    rotation.x = np.random.standard_normal(size=(2 * rotation_order + 1,))
    initial_parameters = np.asarray(rotation.x, dtype=np.float64)

    framedcurve = FramedCurveCentroid(curve, rotation)
    native_Jt = LPTorsionalStrainPenalty(framedcurve, width=_WIDTH, p=2, threshold=0)
    native_Jb = LPBinormalCurvatureStrainPenalty(
        framedcurve, width=_WIDTH, p=2, threshold=0
    )

    gamma = np.asarray(curve.gamma(), dtype=np.float64)
    gammadash = np.asarray(curve.gammadash(), dtype=np.float64)
    gammadashdash = np.asarray(curve.gammadashdash(), dtype=np.float64)

    result = solve_strain_rotation(
        quadpoints=quadpoints,
        gamma=gamma,
        gammadash=gammadash,
        gammadashdash=gammadashdash,
        initial_parameters=initial_parameters,
        rotation_order=rotation_order,
        objective_width=_WIDTH,
        reporting_width=_WIDTH,
        torsional_threshold=0.0,
        curvature_threshold=0.0,
        maxiter=200,
        maxfun=15000,
        gtol=1.0e-20,
        ftol=1.0e-20,
        maxcor=10,
        maxls=20,
    )
    final = jax.device_get(result.final)

    assert bool(np.isfinite(final.objective))
    assert float(final.maximum_torsional_strain) < 1.0e-12
    assert float(final.maximum_binormal_curvature_strain) < 1.0e-9

    rotation.x = np.asarray(final.parameters, dtype=np.float64)
    assert native_Jt.J() < 1.0e-12
    assert native_Jb.J() < 1.0e-12
