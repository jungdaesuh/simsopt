"""JAX curve specs against native curves and independent references.

Each test builds a native curve, converts it to an immutable JAX spec and
compares the spec's geometry, derivatives, pullbacks and errors with the native
curve or with a closed-form NumPy reference that shares no code with the JAX
kernels. One class per curve family:

* ``TestCurveSubclassesParity``: ``CurveHelical``, ``CurvePlanarFourier``,
  ``CurveRZFourier`` and the native ``JaxCurve`` twins, on the CPU and GPU lanes.
* ``TestCurveWrapperSpecs``: ``CurvePerturbed`` and ``CurveFilament`` wrappers.
* ``TestCurvexyzfouriersymmetriesSpecJax``: ``CurveXYZFourierSymmetries``.
* ``TestOrientedcurveJaxSpec``: the oriented ``CurveXYZFourier`` spec.
"""

from __future__ import annotations

from unittest_jax_support import (
    JAX_IMPORT_ERROR,
    JaxTestCase,
    host_array,
    optional_jax_imports,
    parity_default_device,
    parity_rng,
    require_cuda,
)

with optional_jax_imports():
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend import set_backend
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core import (
        CurveXYZFourierSymmetriesSpec,
        OrientedCurveXYZFourierSpec,
        curve_gamma_and_gammadash_from_dofs,
        curve_gamma_and_gammadash_from_spec,
        curve_gamma_vjp_from_dofs,
        curve_gammadash_vjp_from_dofs,
        curve_gammadashdash_vjp_from_dofs,
        curve_gammadashdashdash_vjp_from_dofs,
        curve_geometry_from_dofs,
        curve_pullback_from_dofs,
        curve_spec_from_curve,
        make_curve_filament_spec,
        make_curve_xyzfouriersymmetries_spec,
        make_optimizable_dof_map_spec,
        make_oriented_curve_xyzfourier_spec,
    )
    from simsopt_jax.core.curve_geometry import _slice_1d_static, _update_1d_static
    from simsopt_jax.core.curve_planar_fourier import curveplanarfourier_pure
    from simsopt_jax.core.specs import CurveFilamentSpec, curve_spec_kind
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

from typing import cast

import numpy as np
from numpy.random import PCG64DXSM, Generator
from simsopt.geo import (
    CurveFilament,
    CurvePerturbed,
    CurveXYZFourier,
    FrameRotation,
    GaussianSampler,
    PerturbationSample,
    RotatedCurve,
)
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.framedcurve import FramedCurve, FramedCurveCentroid, FramedCurveFrenet


_GAMMA_ATOL = 1e-10
_GAMMA_RTOL = 1e-10
_DERIV_ATOL = 1e-9
_DERIV_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Curve construction (native is the oracle; dofs are randomized but seeded)
# ---------------------------------------------------------------------------


def _make_helical_curve(seed: int) -> CurveHelical:
    order, m, ell, R0, r = 4, 5, 2, 1.0, 0.3
    curve = CurveHelical(24, order, m, ell, R0, r)
    dofs = np.zeros(curve.dof_size)
    dofs[0] = np.pi / 2
    dofs += 0.01 * parity_rng(seed).normal(size=dofs.shape)
    curve.x = dofs
    return curve


def _make_planar_curve(seed: int) -> CurvePlanarFourier:
    order = 3
    curve = CurvePlanarFourier(20, order)
    dofs = np.zeros(curve.dof_size)
    dofs[0] = 1.0
    dofs[1] = 0.1
    dofs[order + 1] = 0.1
    # The C++ CurvePlanarFourier normalizes every nonzero quaternion by its
    # exact norm, without epsilon regularization; the JAX kernel does likewise.
    q_start = 2 * order + 1
    dofs[q_start : q_start + 4] = np.array([0.9, 0.2, -0.15, 0.05])
    dofs[-3:] = np.array([2.0, -0.5, 0.3])
    dofs += 0.01 * parity_rng(seed).normal(size=dofs.shape)
    curve.x = dofs
    return curve


def _make_native_jax_curves(seed: int) -> tuple[JaxCurveXYZFourier, JaxCurvePlanarFourier]:
    """Native JaxCurve twins of CurveXYZFourier and CurvePlanarFourier, same DOF layouts."""
    xyz = JaxCurveXYZFourier(24, 3)
    dofs = 0.05 * parity_rng(seed).normal(size=xyz.dof_size)
    dofs[[xyz.local_dof_names.index(name) for name in ("xc(1)", "ys(1)", "zs(1)")]] += (1.0, 1.0, 0.3)
    xyz.x = dofs
    planar = JaxCurvePlanarFourier(20, 3)
    planar.x = _make_planar_curve(seed).get_dofs()
    return xyz, planar


def _make_rzfourier_curve(seed: int, *, stellsym: bool) -> CurveRZFourier:
    order, nfp = 3, 3
    curve = CurveRZFourier(20, order, nfp, stellsym)
    dofs = np.zeros(curve.dof_size)
    dofs[0] = 1.3
    dofs[1] = 0.15
    dofs += 0.01 * parity_rng(seed).normal(size=dofs.shape)
    curve.x = dofs
    return curve


# ---------------------------------------------------------------------------
# Assertion helpers (shared numeric comparison; construction stays DAMP above)
# ---------------------------------------------------------------------------


def _assert_pure_kernel_gamma_matches_native(curve, pure_fn, *static_args) -> None:
    """Compare the raw ``*_pure`` kernel directly against native ``gamma()``."""
    quadpoints = np.asarray(curve.quadpoints)
    gamma_jax = host_array(
        pure_fn(np.asarray(curve.get_dofs()), quadpoints, *static_args)
    )
    np.testing.assert_allclose(
        gamma_jax, curve.gamma(), rtol=_GAMMA_RTOL, atol=_GAMMA_ATOL
    )


def _assert_spec_geometry_matches_native(curve, spec) -> None:
    """Compare position, tangent, and second derivative to the native oracle."""
    gamma, gammadash, gammadashdash = curve_geometry_from_dofs(spec, spec.dofs)
    np.testing.assert_allclose(
        host_array(gamma), curve.gamma(), rtol=_GAMMA_RTOL, atol=_GAMMA_ATOL
    )
    np.testing.assert_allclose(
        host_array(gammadash),
        curve.gammadash(),
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
    )
    np.testing.assert_allclose(
        host_array(gammadashdash),
        curve.gammadashdash(),
        rtol=_DERIV_RTOL,
        atol=_DERIV_ATOL,
    )


def _assert_dof_round_trip(curve, *, seed: int) -> None:
    """Adapter DOF pass-through, then geometry re-verified at two DOF configurations.

    The ``assert_array_equal`` checks below confirm that
    ``curve_spec_from_adapter_curve`` (the production adapter boundary, not
    the hand-built factories) neither reorders, truncates, nor rounds the
    DOF vector it is constructed from: each ``spec.dofs`` is checked against
    the exact ``curve.get_dofs()`` array the constructor was just given, so
    these are pass-through checks on the constructor's input handling, not a
    round trip through independent state. The substantive verification is
    ``_assert_spec_geometry_matches_native``, run once at the curve's
    starting dofs and again after mutating the native curve's dofs in place
    to a second, perturbed configuration, confirming the adapter's derived
    geometry tracks native at more than one DOF value.
    """
    original_dofs = np.array(curve.get_dofs(), copy=True)
    spec = curve_spec_from_adapter_curve(curve)
    np.testing.assert_array_equal(host_array(spec.dofs), original_dofs)
    _assert_spec_geometry_matches_native(curve, spec)

    perturbed_dofs = original_dofs + 0.02 * parity_rng(seed).normal(
        size=original_dofs.shape
    )
    curve.x = perturbed_dofs
    updated_spec = curve_spec_from_adapter_curve(curve)
    np.testing.assert_array_equal(host_array(updated_spec.dofs), curve.get_dofs())
    _assert_spec_geometry_matches_native(curve, updated_spec)


class TestCurveSubclassesParity(JaxTestCase):
    """Helical, planar-Fourier and RZ-Fourier specs against the native curve.

    ``curve_spec_from_adapter_curve`` (the conversion the JAX Biot-Savart path
    uses) must pass the native DOFs through unchanged, and the spec's
    ``gamma``, ``gammadash`` and ``gammadashdash`` (built on the pure
    ``*_pure`` kernels) must match the native curve at two DOF configurations,
    on the CPU and GPU lanes. Bitwise equality is not required: the two
    implementations reassociate the trigonometric sums differently, so values
    are compared with a tolerance several orders of magnitude above the
    observed noise floor (~1e-13 for second derivatives, ~1e-15 for positions).
    """

    def test_curve_helical_dof_round_trip_matches_native(self):
        """Helical native DOF setters round-trip to matching immutable spec geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_helical_dof_round_trip_matches_native(
                    parity_lane, patches
                )

    def _case_curve_helical_dof_round_trip_matches_native(self, parity_lane, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_helical_curve(seed=103)
        _assert_dof_round_trip(curve, seed=203)

    def test_curve_planarfourier_dof_round_trip_matches_native(self):
        """Planar Fourier native DOF setters round-trip to matching immutable spec
        geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_planarfourier_dof_round_trip_matches_native(
                    parity_lane, patches
                )

    def _case_curve_planarfourier_dof_round_trip_matches_native(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_planar_curve(seed=113)
        _assert_dof_round_trip(curve, seed=213)

    def test_curve_planarfourier_small_quaternion_matches_native(self):
        """Planar Fourier geometry with a very small quaternion matches native normalized
        rotation."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_curve_planarfourier_small_quaternion_matches_native(
                    parity_lane, patches
                )

    def _case_curve_planarfourier_small_quaternion_matches_native(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_planar_curve(seed=114)
        dofs = np.array(curve.get_dofs(), copy=True)
        q_start = 2 * curve.order + 1
        dofs[q_start : q_start + 4] = np.array([1.0, 2.0, 3.0, 4.0]) * 1e-12
        curve.x = dofs
        _assert_pure_kernel_gamma_matches_native(
            curve, curveplanarfourier_pure, curve.order
        )
        _assert_spec_geometry_matches_native(
            curve, curve_spec_from_adapter_curve(curve)
        )

    def test_curve_rzfourier_dof_round_trip_matches_native(self):
        """RZ Fourier native DOF setters round-trip to matching immutable spec geometry."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for case_id_1, stellsym in zip(
                ["stellsym", "non_stellsym"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    parity_lane=parity_lane,
                    case_id_1=case_id_1,
                    stellsym=stellsym,
                ), self.case() as patches:
                    self._case_curve_rzfourier_dof_round_trip_matches_native(
                        parity_lane, stellsym, patches
                    )

    def _case_curve_rzfourier_dof_round_trip_matches_native(
        self, parity_lane, stellsym, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        curve = _make_rzfourier_curve(seed=125 if stellsym else 126, stellsym=stellsym)
        _assert_dof_round_trip(curve, seed=225 if stellsym else 226)

    def test_native_jax_curve_twins_round_trip_to_matching_specs(self):
        """JaxCurveXYZFourier and JaxCurvePlanarFourier map to their direct curve specs."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, parity_lane=parity_lane
            ), self.case() as patches:
                self._case_native_jax_curve_twins_round_trip_to_matching_specs(
                    parity_lane, patches
                )

    def _case_native_jax_curve_twins_round_trip_to_matching_specs(
        self, parity_lane, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        patches.enter_context(parity_default_device(parity_lane))
        for curve in _make_native_jax_curves(seed=130):
            _assert_dof_round_trip(curve, seed=230)


# ---------------------------------------------------------------------------
# Perturbed and filament wrappers
# ---------------------------------------------------------------------------

_NQUAD = 128
# Spectral-derivative references resolve the Frenet frame to round-off only
# from 256 points (at 128 the spectral error of gammadashdash is 2e-8).
_SPECTRAL_NQUAD = 256
_XYZ_ORDER = 3
_FRAMES = (FramedCurveCentroid, FramedCurveFrenet)


def _xyz_curve(coefficients: dict[str, float], nquad: int = _NQUAD) -> CurveXYZFourier:
    curve = CurveXYZFourier(nquad, _XYZ_ORDER)
    dofs = np.zeros(curve.dof_size)
    for name, value in coefficients.items():
        dofs[curve.local_dof_names.index(name)] = value
    curve.x = dofs
    return curve


_BASE = {"xc(1)": 1.0, "ys(1)": 1.0, "zs(2)": 0.2, "xs(3)": 0.05}
_PERTURBATION = {"xs(2)": 0.03, "yc(3)": -0.02, "zs(1)": 0.04}


def _gaussian_sample(curve, n_derivs: int) -> PerturbationSample:
    """Seeded Gaussian-process sample, so value comparisons see one fixed draw."""
    sampler = GaussianSampler(curve.quadpoints, 0.05, 0.3, n_derivs=n_derivs)
    return PerturbationSample(sampler, randomgen=Generator(PCG64DXSM(670)))


def _perturbed_and_equivalent() -> tuple[CurvePerturbed, CurveXYZFourier]:
    """Return base + Fourier sample (with three exact derivatives) and the summed curve."""
    perturbation = _xyz_curve(_PERTURBATION)
    sample = [
        perturbation.gamma(),
        perturbation.gammadash(),
        perturbation.gammadashdash(),
        perturbation.gammadashdashdash(),
    ]
    perturbed = CurvePerturbed(_xyz_curve(_BASE), PerturbationSample(None, sample=sample))
    # The two coefficient sets share no mode, so their union is their sum.
    return perturbed, _xyz_curve({**_BASE, **_PERTURBATION})


def _filament(curve, frame) -> CurveFilament:
    # A non-unit scale multiplies every rotation-angle derivative the frame uses.
    rotation = FrameRotation(curve.quadpoints, 2, scale=0.7)
    rotation.x = np.array([0.1, 0.2, 0.3, -0.1, 0.05])
    return CurveFilament(frame(curve, rotation), 0.02, 0.03)


def _spectral_derivative(values: np.ndarray) -> np.ndarray:
    """d/dphi of periodic samples on the uniform grid phi in [0, 1)."""
    wavenumbers = np.fft.fftfreq(values.shape[0], d=1.0 / values.shape[0])
    spectrum = 2j * np.pi * wavenumbers[:, None] * np.fft.fft(values, axis=0)
    return np.real(np.fft.ifft(spectrum, axis=0))


def _terms(spec, order: int) -> list[np.ndarray]:
    evaluate = curve_gamma_and_gammadash_from_dofs if order == 1 else curve_geometry_from_dofs
    return [host_array(term) for term in evaluate(spec, spec.dofs)]

class TestCurveWrapperSpecs(JaxTestCase):
    """Perturbed and filament specs follow native geometry and native errors.

    A perturbation sample that is an exact Fourier series makes
    ``CurvePerturbed`` the same curve as one ``CurveXYZFourier`` with summed
    coefficients, so filament geometry over the perturbed curve has an
    independent analytic reference. The second parameter derivative, which
    native filaments do not provide, is checked against the spectral derivative
    of the first.
    """

    def test_filament_over_perturbed_curve_matches_summed_curve(self):
        """Order-1 and order-2 filament terms over a perturbed curve include the sample derivatives."""
        perturbed, summed = _perturbed_and_equivalent()
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                filament = _filament(perturbed, frame)
                spec = curve_spec_from_adapter_curve(filament)
                reference = curve_spec_from_adapter_curve(_filament(summed, frame))
                order1 = _terms(spec, 1)
                np.testing.assert_allclose(order1[1], filament.gammadash(), rtol=0, atol=1e-13)
                for term, expected in zip(order1, _terms(reference, 1), strict=True):
                    np.testing.assert_allclose(term, expected, rtol=0, atol=1e-13)
                if frame is FramedCurveFrenet:
                    # The Frenet second derivative needs a fourth sample derivative.
                    with self.assertRaisesRegex(ValueError, "derivative 4 is needed"):
                        curve_geometry_from_dofs(spec, spec.dofs)
                    continue
                order2 = _terms(spec, 2)
                for term, expected in zip(order2, _terms(reference, 2), strict=True):
                    np.testing.assert_allclose(term, expected, rtol=0, atol=1e-11)
                np.testing.assert_allclose(order2[1], filament.gammadash(), rtol=0, atol=1e-13)

    def test_filament_second_derivative_is_derivative_of_tangent(self):
        """gammadashdash equals the spectral derivative of gammadash for both frames."""
        curve = _xyz_curve({**_BASE, **_PERTURBATION}, nquad=_SPECTRAL_NQUAD)
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                gamma, gammadash, gammadashdash = _terms(
                    curve_spec_from_adapter_curve(_filament(curve, frame)), 2
                )
                np.testing.assert_allclose(_spectral_derivative(gamma), gammadash, rtol=0, atol=1e-11)
                np.testing.assert_allclose(_spectral_derivative(gammadash), gammadashdash, rtol=0, atol=1e-11)

    def test_filament_over_filament_derivatives_are_derivatives_of_lower_terms(self):
        """Nested filaments: each derivative is the spectral derivative of the term below.

        Native filaments provide no second derivative, so native evaluates only
        the position of a centroid filament over another filament.
        """
        inner = _filament(_xyz_curve(_BASE, nquad=_SPECTRAL_NQUAD), FramedCurveCentroid)
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                outer = _filament(inner, frame)
                gamma, gammadash, gammadashdash = _terms(curve_spec_from_adapter_curve(outer), 2)
                if frame is FramedCurveCentroid:
                    np.testing.assert_allclose(gamma, outer.gamma(), rtol=0, atol=1e-13)
                np.testing.assert_allclose(_spectral_derivative(gamma), gammadash, rtol=0, atol=1e-11)
                np.testing.assert_allclose(_spectral_derivative(gammadash), gammadashdash, rtol=0, atol=1e-11)

    def test_sample_without_frame_derivatives_raises_like_native(self):
        """A sample lacking the derivative a frame needs is rejected at conversion, as native raises."""
        base = _xyz_curve(_BASE)
        # (n_derivs, frame) -> native rejects: centroid needs 2, Frenet 3.
        for n_derivs, frame, native_raises in (
            (1, FramedCurveCentroid, True),
            (1, FramedCurveFrenet, True),
            (2, FramedCurveFrenet, True),
            (2, FramedCurveCentroid, False),
        ):
            with self.subTest(n_derivs=n_derivs, frame=frame.__name__):
                filament = _filament(CurvePerturbed(base, _gaussian_sample(base, n_derivs)), frame)
                if not native_raises:
                    spec = curve_spec_from_adapter_curve(filament)
                    np.testing.assert_allclose(_terms(spec, 1)[1], filament.gammadash(), rtol=0, atol=1e-13)
                    continue
                with self.assertRaises(ValueError):
                    filament.gammadash()
                with self.assertRaisesRegex(ValueError, f"has {n_derivs} derivatives"):
                    curve_spec_from_adapter_curve(filament)

    def test_perturbed_curve_without_second_sample_derivative_raises_like_native(self):
        """Order-2 geometry of a perturbed curve needs the sample's second derivative."""
        base = _xyz_curve(_BASE)
        perturbed = CurvePerturbed(base, _gaussian_sample(base, 1))
        spec = curve_spec_from_adapter_curve(perturbed)
        np.testing.assert_allclose(_terms(spec, 1)[1], perturbed.gammadash(), rtol=0, atol=1e-13)
        with self.assertRaises(ValueError):
            perturbed.gammadashdash()
        with self.assertRaisesRegex(ValueError, "has 1 derivatives, but derivative 2"):
            curve_geometry_from_dofs(spec, spec.dofs)

    def test_unsupported_frames_and_wrapper_orders_raise(self):
        """Custom frames, unknown frame kinds and inner RotatedCurve placements are rejected."""
        base = _xyz_curve(_BASE)
        custom_frame = CurveFilament(FramedCurve(base), 0.02, 0.03)
        with self.assertRaisesRegex(NotImplementedError, "FramedCurve"):
            curve_spec_from_adapter_curve(custom_frame)
        valid = cast(
            CurveFilamentSpec,
            curve_spec_from_adapter_curve(_filament(base, FramedCurveCentroid)),
        )
        with self.assertRaisesRegex(ValueError, "frame_kind"):
            make_curve_filament_spec(
                dofs=valid.dofs,
                quadpoints=valid.quadpoints,
                base_curve=valid.base_curve,
                base_curve_map=valid.base_curve_map,
                rotation=valid.rotation,
                rotation_map=valid.rotation_map,
                frame_kind="custom",
                dn=valid.dn,
                db=valid.db,
            )
        rotated = RotatedCurve(base, np.pi / 3, flip=False)
        for inner_rotation in (
            CurvePerturbed(rotated, _gaussian_sample(base, 3)),
            _filament(rotated, FramedCurveCentroid),
        ):
            with self.subTest(wrapper=type(inner_rotation).__name__), self.assertRaisesRegex(
                NotImplementedError, "RotatedCurve"
            ):
                curve_spec_from_adapter_curve(inner_rotation)

    def test_dof_map_segments_must_have_equal_lengths(self):
        """Owner and target ranges of a DOF-map segment must match and fit the template."""
        for segment in ((0, 3, 1, 3), (0, 2, 3, 5)):
            with self.subTest(segment=segment), self.assertRaisesRegex(
                ValueError, "equal-length target range"
            ):
                make_optimizable_dof_map_spec(
                    template_full_dofs=np.zeros(4),
                    owner_segments=(segment,),
                    input_mode="full",
                    input_start=0,
                    input_end=4,
                )

    def test_wrapper_dof_map_templates_follow_the_runtime_device(self):
        """Under the JAX CPU backend in a CUDA process, wrapper DOF-map templates join
        the other spec leaves on the runtime CPU device, not on the GPU."""
        require_cuda("CUDA device required for a runtime device other than the GPU")
        set_backend("jax", device="cpu")
        cpu = jax.devices("cpu")[0]
        base = _xyz_curve(_BASE)
        for wrapper in (CurvePerturbed(base, _gaussian_sample(base, 1)), _filament(base, FramedCurveCentroid)):
            with self.subTest(wrapper=type(wrapper).__name__):
                spec = curve_spec_from_adapter_curve(wrapper)
                devices = {leaf.devices().pop() for leaf in jax.tree.leaves(spec)}
                self.assertEqual(devices, {cpu})


# ---------------------------------------------------------------------------
# CurveXYZFourierSymmetries
# ---------------------------------------------------------------------------

_DIRECT_RTOL = 1e-10
_DIRECT_ATOL = 1e-12
_FIRST_DERIV_RTOL = 1e-08
_FIRST_DERIV_ATOL = 1e-10
_SECOND_DERIV_RTOL = 1e-06
_SECOND_DERIV_ATOL = 1e-08


_NQUADPOINTS = 64
_ORDER = 3
_RAND_SCALE = 1e-2
_RNG_SEED = 1729

if JAX_IMPORT_ERROR is None:
    _CURVE_VJP_CASES = (
        ("gamma", curve_gamma_vjp_from_dofs, "dgamma_by_dcoeff"),
        ("gammadash", curve_gammadash_vjp_from_dofs, "dgammadash_by_dcoeff"),
        (
            "gammadashdash",
            curve_gammadashdash_vjp_from_dofs,
            "dgammadashdash_by_dcoeff",
        ),
        (
            "gammadashdashdash",
            curve_gammadashdashdash_vjp_from_dofs,
            "dgammadashdashdash_by_dcoeff",
        ),
    )


def _stellsym_num_dofs(order: int) -> int:
    return (order + 1) + order + order


def _non_stellsym_num_dofs(order: int) -> int:
    return 3 * (2 * order + 1)


def _seed_stellsym_dofs(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = _stellsym_num_dofs(order)
    dofs = np.zeros(ndofs, dtype=np.float64)
    # xc(0) sets the loop radius scale; xc(1), ys(1), zs(1) are the
    # dominant first-harmonic amplitudes for a helical coil.
    dofs[0] = 1.0
    dofs[1] = 0.25
    dofs[order + 1] = 0.25
    dofs[2 * order + 1] = 0.25
    return dofs + _RAND_SCALE * rng.random(ndofs)


def _seed_non_stellsym_dofs(order: int, rng: np.random.Generator) -> np.ndarray:
    ndofs = _non_stellsym_num_dofs(order)
    dofs = np.zeros(ndofs, dtype=np.float64)
    # xc(0), yc(0) place the loop centre; xc(1) and ys(1) provide the
    # first-harmonic shape.
    dofs[0] = 1.0
    dofs[1] = 0.25
    dofs[2 * order + 1] = 0.0
    dofs[3 * order + 2] = 0.25
    return dofs + _RAND_SCALE * rng.random(ndofs)


def _build_curve(
    *,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
    rng: np.random.Generator,
    nquadpoints: int = _NQUADPOINTS,
) -> CurveXYZFourierSymmetries:
    curve = CurveXYZFourierSymmetries(
        nquadpoints,
        order,
        nfp=nfp,
        stellsym=stellsym,
        ntor=ntor,
    )
    if stellsym:
        dofs = _seed_stellsym_dofs(order, rng)
    else:
        dofs = _seed_non_stellsym_dofs(order, rng)
    curve.x = dofs
    return curve


def _curve_spec(curve: CurveXYZFourierSymmetries) -> CurveXYZFourierSymmetriesSpec:
    return make_curve_xyzfouriersymmetries_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
        ntor=curve.ntor,
    )


_FORWARD_PARITY_CASES = [
    ("3-2-True", (3, 2, True)),
    ("3-2-False", (3, 2, False)),
    ("5-3-True", (5, 3, True)),
    ("5-3-False", (5, 3, False)),
    ("4-1-True", (4, 1, True)),
    ("7-4-True", (7, 4, True)),
]


def _numpy_gamma_oracle(
    dofs: np.ndarray,
    quadpoints: np.ndarray,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
) -> np.ndarray:
    """Pure-NumPy ``gamma`` reference for ``CurveXYZFourierSymmetries``.

    Implements the closed-form Fourier series documented at
    ``src/simsopt/geo/curvexyzfouriersymmetries.py:66-90`` without
    sharing any code with ``jaxXYZFourierSymmetriescurve_pure``. Acts as
    the independent oracle for the JAX kernel parity tests.
    """
    theta = np.asarray(quadpoints, dtype=np.float64)
    angles_full = 2.0 * np.pi * nfp * np.outer(theta, np.arange(order + 1))
    cos_full = np.cos(angles_full)  # (N, order+1)
    sin_tail = np.sin(angles_full[:, 1:])  # (N, order)

    if stellsym:
        xc = dofs[: order + 1]
        ys = dofs[order + 1 : 2 * order + 1]
        zs = dofs[2 * order + 1 :]
        hat_x = cos_full @ xc
        hat_y = sin_tail @ ys
        z = sin_tail @ zs
    else:
        xc = dofs[0 : order + 1]
        xs = dofs[order + 1 : 2 * order + 1]
        yc = dofs[2 * order + 1 : 3 * order + 2]
        ys = dofs[3 * order + 2 : 4 * order + 2]
        zc = dofs[4 * order + 2 : 5 * order + 3]
        zs = dofs[5 * order + 3 :]
        hat_x = cos_full @ xc + sin_tail @ xs
        hat_y = cos_full @ yc + sin_tail @ ys
        z = cos_full @ zc + sin_tail @ zs

    tor = 2.0 * np.pi * theta * ntor
    cos_tor = np.cos(tor)
    sin_tor = np.sin(tor)
    x = hat_x * cos_tor - hat_y * sin_tor
    y = hat_x * sin_tor + hat_y * cos_tor
    return np.stack((x, y, z), axis=1).astype(np.float64)
class TestCurvexyzfouriersymmetriesSpecJax(JaxTestCase):
    """Symmetry-curve geometry and derivatives against native curves.

    The pure kernels consume immutable specs; the native mutable curve supplies
    an independent geometry and derivative oracle at identical DOFs.
    """

    def test_spec_gamma_byte_identical_to_curve_gamma(self):
        """Spec-driven gamma matches ``curve.gamma()`` at the direct-kernel lane."""
        for case_id_0, (nfp, ntor, stellsym) in _FORWARD_PARITY_CASES:
            with self.subTest(
                case_id_0=case_id_0, nfp=nfp, ntor=ntor, stellsym=stellsym
            ), self.case():
                self._case_spec_gamma_byte_identical_to_curve_gamma(nfp, ntor, stellsym)

    def _case_spec_gamma_byte_identical_to_curve_gamma(self, nfp, ntor, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED)
        curve = _build_curve(
            order=_ORDER,
            nfp=nfp,
            stellsym=stellsym,
            ntor=ntor,
            rng=rng,
        )
        spec = _curve_spec(curve)
        self.assertTrue(
            isinstance(spec, CurveXYZFourierSymmetriesSpec),
            "isinstance(spec, CurveXYZFourierSymmetriesSpec)",
        )

        gamma_curve = np.asarray(curve.gamma(), dtype=np.float64)
        gamma_jax = np.asarray(
            curve_geometry_from_dofs(spec, spec.dofs)[0], dtype=np.float64
        )
        # Independent NumPy oracle (closed-form series from the docstring;
        # shares no code with `jaxXYZFourierSymmetriescurve_pure`).
        gamma_numpy = _numpy_gamma_oracle(
            np.asarray(spec.dofs, dtype=np.float64),
            np.asarray(spec.quadpoints, dtype=np.float64),
            order=int(spec.order),
            nfp=int(spec.nfp),
            stellsym=bool(spec.stellsym),
            ntor=int(spec.ntor),
        )

        self.assertTrue(
            gamma_curve.shape == (_NQUADPOINTS, 3),
            "gamma_curve.shape == (_NQUADPOINTS, 3)",
        )
        self.assertTrue(
            gamma_jax.shape == (_NQUADPOINTS, 3),
            "gamma_jax.shape == (_NQUADPOINTS, 3)",
        )
        self.assertTrue(
            gamma_numpy.shape == (_NQUADPOINTS, 3),
            "gamma_numpy.shape == (_NQUADPOINTS, 3)",
        )
        # Both JaxCurve route and the new spec route must agree with the
        # independent oracle (these two paths share the JAX kernel, so an
        # algebraic regression in `jaxXYZFourierSymmetriescurve_pure` would
        # fail the oracle comparison even though they would still agree with
        # each other).
        np.testing.assert_allclose(
            gamma_jax,
            gamma_numpy,
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )
        np.testing.assert_allclose(
            gamma_curve,
            gamma_numpy,
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )

    def test_explicit_spec_round_trips_fields(self):
        """Explicit spec construction mirrors the host class state."""
        for case_id_0, stellsym in zip(
            ["stellsym=True", "stellsym=False"], [True, False], strict=True
        ):
            with self.subTest(case_id_0=case_id_0, stellsym=stellsym), self.case():
                self._case_explicit_spec_round_trips_fields(stellsym)

    def _case_explicit_spec_round_trips_fields(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 1)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)

        self.assertTrue(
            isinstance(spec, CurveXYZFourierSymmetriesSpec),
            "isinstance(spec, CurveXYZFourierSymmetriesSpec)",
        )
        self.assertTrue(spec.order == curve.order, "spec.order == curve.order")
        self.assertTrue(spec.nfp == curve.nfp, "spec.nfp == curve.nfp")
        self.assertTrue(
            spec.stellsym == curve.stellsym, "spec.stellsym == curve.stellsym"
        )
        self.assertTrue(spec.ntor == curve.ntor, "spec.ntor == curve.ntor")

        np.testing.assert_array_equal(
            np.asarray(spec.dofs),
            np.asarray(curve.get_dofs(), dtype=np.float64),
        )
        np.testing.assert_array_equal(
            np.asarray(spec.quadpoints),
            np.asarray(curve.quadpoints, dtype=np.float64),
        )

        # ``curve_spec_kind`` discriminator must recognize the new variant.
        self.assertTrue(
            curve_spec_kind(spec) == "xyz_fourier_symmetries",
            'curve_spec_kind(spec) == "xyz_fourier_symmetries"',
        )

    def test_curve_spec_from_curve_rejects_without_adapter_hook(self):
        """Legacy host curves without adapter hooks stay outside the implicit boundary."""
        rng = np.random.default_rng(_RNG_SEED + 2)
        curve = _build_curve(
            order=_ORDER,
            nfp=5,
            stellsym=True,
            ntor=3,
            rng=rng,
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "CurveXYZFourierSymmetries does not expose an immutable JAX spec",
        ):
            curve_spec_from_curve(curve)

    def test_spec_higher_derivatives_match_curve(self):
        """``curve_geometry_from_dofs`` ladder matches CPU oracle for gamma/dash/dashdash."""
        for case_id_0, stellsym in zip(
            ["stellsym=True", "stellsym=False"], [True, False], strict=True
        ):
            with self.subTest(case_id_0=case_id_0, stellsym=stellsym), self.case():
                self._case_spec_higher_derivatives_match_curve(stellsym)

    def _case_spec_higher_derivatives_match_curve(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 3)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)

        gamma_cpu = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash_cpu = np.asarray(curve.gammadash(), dtype=np.float64)
        gammadashdash_cpu = np.asarray(curve.gammadashdash(), dtype=np.float64)

        gamma_jax, gammadash_jax, gammadashdash_jax = (
            np.asarray(term, dtype=np.float64)
            for term in curve_geometry_from_dofs(spec, spec.dofs)
        )

        np.testing.assert_allclose(
            gamma_jax,
            gamma_cpu,
            rtol=_DIRECT_RTOL,
            atol=_DIRECT_ATOL,
        )
        np.testing.assert_allclose(
            gammadash_jax,
            gammadash_cpu,
            rtol=_FIRST_DERIV_RTOL,
            atol=_FIRST_DERIV_ATOL,
        )
        np.testing.assert_allclose(
            gammadashdash_jax,
            gammadashdash_cpu,
            rtol=_SECOND_DERIV_RTOL,
            atol=_SECOND_DERIV_ATOL,
        )

    def test_curve_pullback_shape(self):
        """VJP cotangent has the shape of ``spec.dofs``."""
        for case_id_0, stellsym in zip(
            ["stellsym=True", "stellsym=False"], [True, False], strict=True
        ):
            with self.subTest(case_id_0=case_id_0, stellsym=stellsym), self.case():
                self._case_curve_pullback_shape(stellsym)

    def _case_curve_pullback_shape(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 4)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)

        gamma_jax, gammadash_jax = curve_gamma_and_gammadash_from_dofs(spec, spec.dofs)
        cotangent_g = jnp.ones_like(gamma_jax)
        cotangent_gd = jnp.ones_like(gammadash_jax)

        coeff_cotangent = curve_pullback_from_dofs(
            spec, spec.dofs, cotangent_g, cotangent_gd
        )
        self.assertTrue(
            coeff_cotangent.shape == spec.dofs.shape,
            "coeff_cotangent.shape == spec.dofs.shape",
        )
        # The all-ones cotangent must produce a finite, non-zero result for a
        # non-degenerate curve. Zero would mean the pullback misrouted.
        coeff_np = np.asarray(coeff_cotangent, dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(coeff_np)), "np.all(np.isfinite(coeff_np))")
        self.assertTrue(
            np.linalg.norm(coeff_np) > 0.0, "np.linalg.norm(coeff_np) > 0.0"
        )

    def test_named_geometry_vjp_wrappers_match_cpu_derivatives(self):
        """Named geometry VJP wrappers match CPU derivative tensor contractions."""
        for term_name, vjp_fn, derivative_method in _CURVE_VJP_CASES:
            for case_id_1, stellsym in zip(
                ["stellsym=True", "stellsym=False"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=term_name,
                    term_name=term_name,
                    vjp_fn=vjp_fn,
                    derivative_method=derivative_method,
                    case_id_1=case_id_1,
                    stellsym=stellsym,
                ), self.case():
                    self._case_named_geometry_vjp_wrappers_match_cpu_derivatives(
                        term_name, vjp_fn, derivative_method, stellsym
                    )

    def _case_named_geometry_vjp_wrappers_match_cpu_derivatives(
        self, term_name, vjp_fn, derivative_method, stellsym
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = np.random.default_rng(_RNG_SEED + 5 + int(stellsym))
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=stellsym,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)
        derivative_cpu = np.asarray(
            getattr(curve, derivative_method)(),
            dtype=np.float64,
        )
        cotangent = rng.normal(size=derivative_cpu.shape[:2])

        vjp_jax = np.asarray(
            vjp_fn(spec, spec.dofs, jnp.asarray(cotangent, dtype=jnp.float64)),
            dtype=np.float64,
        )
        vjp_cpu = np.einsum("ij,ijk->k", cotangent, derivative_cpu)

        np.testing.assert_allclose(
            vjp_jax,
            vjp_cpu,
            rtol=_FIRST_DERIV_RTOL,
            atol=_FIRST_DERIV_ATOL,
            err_msg=f"CurveXYZFourierSymmetries {term_name} named VJP mismatch.",
        )

    def test_spec_geometry_runs_under_strict_transfer_guard(self):
        """Spec-driven geometry executes under ``jax.transfer_guard('disallow')``.

        The spec's leaves are already device arrays, and a compiled callable
        evaluates ``curve_geometry_from_dofs`` without implicit host-to-device
        transfers.
        """
        rng = np.random.default_rng(_RNG_SEED + 5)
        curve = _build_curve(
            order=_ORDER,
            nfp=3,
            stellsym=True,
            ntor=2,
            rng=rng,
        )
        spec = _curve_spec(curve)

        @jit
        def evaluate(spec_arg):
            gamma, gammadash, gammadashdash = curve_geometry_from_dofs(
                spec_arg, spec_arg.dofs
            )
            return gamma, gammadash, gammadashdash

        compiled = evaluate.lower(spec).compile()
        with jax.transfer_guard("disallow"):
            gamma, gammadash, gammadashdash = compiled(spec)

        self.assertTrue(
            gamma.shape == (_NQUADPOINTS, 3), "gamma.shape == (_NQUADPOINTS, 3)"
        )
        self.assertTrue(
            gammadash.shape == (_NQUADPOINTS, 3), "gammadash.shape == (_NQUADPOINTS, 3)"
        )
        self.assertTrue(
            gammadashdash.shape == (_NQUADPOINTS, 3),
            "gammadashdash.shape == (_NQUADPOINTS, 3)",
        )
        self.assertTrue(
            np.all(np.isfinite(np.asarray(gamma))),
            "np.all(np.isfinite(np.asarray(gamma)))",
        )
        self.assertTrue(
            np.all(np.isfinite(np.asarray(gammadash))),
            "np.all(np.isfinite(np.asarray(gammadash)))",
        )
        self.assertTrue(
            np.all(np.isfinite(np.asarray(gammadashdash))),
            "np.all(np.isfinite(np.asarray(gammadashdash)))",
        )

    def test_static_slice_and_update_grad_run_under_strict_transfer_guard(self):
        """Compiled static slices and updates have exact values and gradients under a
        strict transfer guard."""
        values_host = np.linspace(-2.0, 3.0, num=6, dtype=np.float64)
        replacement_host = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        values = jnp.asarray(values_host)
        replacement = jnp.asarray(replacement_host)

        def objective(values_arg, replacement_arg):
            segment = _slice_1d_static(values_arg, 2, 5)
            updated = _update_1d_static(values_arg, 1, replacement_arg)
            return jnp.sum(segment * segment) + jnp.sum(updated)

        value_fn = jit(objective).lower(values, replacement).compile()
        grad_fn = (
            jit(jax.grad(objective, argnums=(0, 1)))
            .lower(values, replacement)
            .compile()
        )

        with jax.transfer_guard("disallow"):
            actual_value = value_fn(values, replacement)
            grad_values, grad_replacement = grad_fn(values, replacement)

        expected_updated = np.concatenate(
            [values_host[:1], replacement_host, values_host[4:]]
        )
        expected_value = np.sum(values_host[2:5] ** 2) + np.sum(expected_updated)
        expected_grad_values = np.zeros_like(values_host)
        expected_grad_values[2:5] += 2.0 * values_host[2:5]
        expected_grad_values[:1] += 1.0
        expected_grad_values[4:] += 1.0

        np.testing.assert_allclose(
            np.asarray(actual_value), expected_value, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(grad_values), expected_grad_values, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(grad_replacement),
            np.ones_like(replacement_host),
            rtol=0.0,
            atol=0.0,
        )

    def test_coprime_invariant_unchanged(self):
        """The host-class constructor still enforces ``gcd(ntor, nfp) == 1``."""
        with self.assertRaisesRegex(Exception, "nfp and ntor must be coprime"):
            CurveXYZFourierSymmetries(
                _NQUADPOINTS, _ORDER, nfp=4, stellsym=True, ntor=2
            )

    def test_spec_constructor_enforces_coprime_invariant(self):
        """The explicit spec constructor preserves the same coprime contract."""
        with self.assertRaisesRegex(ValueError, "requires nfp and ntor coprime"):
            make_curve_xyzfouriersymmetries_spec(
                dofs=np.zeros(_stellsym_num_dofs(_ORDER), dtype=np.float64),
                quadpoints=np.linspace(0.0, 1.0, _NQUADPOINTS, endpoint=False),
                order=_ORDER,
                nfp=4,
                stellsym=True,
                ntor=2,
            )


# ---------------------------------------------------------------------------
# Oriented CurveXYZFourier
# ---------------------------------------------------------------------------

_RTOL = 1e-10
_ATOL = 1e-12


def _oriented_curve_numpy(dofs: np.ndarray, quadpoints: np.ndarray, order: int):
    xyz = dofs[:3]
    yaw, pitch, roll = dofs[3:6]
    coeffs = np.split(dofs[6:], 3)

    yaw_matrix = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pitch_matrix = np.asarray(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    roll_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )
    rotation = yaw_matrix @ pitch_matrix @ roll_matrix

    gamma = np.zeros((quadpoints.size, 3), dtype=np.float64)
    gammadash = np.zeros_like(gamma)
    for component_index, component_coeffs in enumerate(coeffs):
        for mode_index in range(order):
            mode = float(mode_index + 1)
            angle = 2.0 * np.pi * mode * quadpoints
            sin_coeff = component_coeffs[2 * mode_index]
            cos_coeff = component_coeffs[2 * mode_index + 1]
            gamma[:, component_index] += sin_coeff * np.sin(angle)
            gamma[:, component_index] += cos_coeff * np.cos(angle)
            gammadash[:, component_index] += (
                2.0 * np.pi * mode * sin_coeff * np.cos(angle)
            )
            gammadash[:, component_index] -= (
                2.0 * np.pi * mode * cos_coeff * np.sin(angle)
            )

    return gamma @ rotation + xyz, gammadash @ rotation
class TestOrientedcurveJaxSpec(JaxTestCase):
    """Oriented ``CurveXYZFourier`` geometry from its immutable spec."""

    def test_oriented_curve_spec_matches_numpy_geometry_oracle(self):
        """JIT evaluation of oriented Fourier position and tangent matches an independent
        NumPy oracle."""
        quadpoints = np.linspace(0.0, 1.0, 16, endpoint=False)
        dofs = np.array(
            [
                1.2,
                -0.1,
                0.3,
                0.2,
                -0.15,
                0.05,
                0.1,
                -0.04,
                0.03,
                0.2,
                -0.05,
                0.07,
                0.09,
                -0.11,
                0.08,
                0.13,
                -0.02,
                0.06,
            ],
            dtype=np.float64,
        )

        spec = make_oriented_curve_xyzfourier_spec(
            dofs=dofs,
            quadpoints=quadpoints,
            order=2,
        )
        self.assertTrue(
            isinstance(spec, OrientedCurveXYZFourierSpec),
            "isinstance(spec, OrientedCurveXYZFourierSpec)",
        )
        gamma, gammadash = jit(curve_gamma_and_gammadash_from_spec)(spec)
        expected_gamma, expected_gammadash = _oriented_curve_numpy(dofs, quadpoints, 2)

        np.testing.assert_allclose(
            np.asarray(gamma),
            expected_gamma,
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(gammadash),
            expected_gammadash,
            rtol=_RTOL,
            atol=_ATOL,
        )
