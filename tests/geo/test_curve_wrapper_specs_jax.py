"""JAX specs of perturbed and filament curves follow native geometry and native errors.

A perturbation sample that is an exact Fourier series makes ``CurvePerturbed``
the same curve as one ``CurveXYZFourier`` with summed coefficients, so filament
geometry over the perturbed curve has an independent analytic reference. The
second parameter derivative, which native filaments do not provide, is also
checked against the spectral derivative of the first.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, host_array, optional_jax_imports

with optional_jax_imports():
    from simsopt_jax.core import (
        curve_gamma_and_gammadash_from_dofs,
        curve_geometry_from_dofs,
    )
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

import numpy as np
from simsopt.geo import (
    CurveFilament,
    CurvePerturbed,
    CurveXYZFourier,
    FrameRotation,
    GaussianSampler,
    PerturbationSample,
)
from simsopt.geo.framedcurve import FramedCurveCentroid, FramedCurveFrenet

_NQUAD = 128
_ORDER = 3
_FRAMES = (FramedCurveCentroid, FramedCurveFrenet)


def _xyz_curve(coefficients: dict[str, float]) -> CurveXYZFourier:
    curve = CurveXYZFourier(_NQUAD, _ORDER)
    dofs = np.zeros(curve.dof_size)
    for name, value in coefficients.items():
        dofs[curve.local_dof_names.index(name)] = value
    curve.x = dofs
    return curve


_BASE = {"xc(1)": 1.0, "ys(1)": 1.0, "zs(2)": 0.2, "xs(3)": 0.05}
_PERTURBATION = {"xs(2)": 0.03, "yc(3)": -0.02, "zs(1)": 0.04}


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
    rotation = FrameRotation(curve.quadpoints, 2)
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
                np.testing.assert_array_equal(order2[1], order1[1])

    def test_filament_second_derivative_is_derivative_of_tangent(self):
        """gammadashdash equals the spectral derivative of gammadash for both frames."""
        _perturbed, summed = _perturbed_and_equivalent()
        for frame in _FRAMES:
            with self.subTest(frame=frame.__name__):
                gamma, gammadash, gammadashdash = _terms(
                    curve_spec_from_adapter_curve(_filament(summed, frame)), 2
                )
                np.testing.assert_allclose(_spectral_derivative(gamma), gammadash, rtol=0, atol=1e-10)
                np.testing.assert_allclose(_spectral_derivative(gammadash), gammadashdash, rtol=0, atol=1e-7)

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
                sampler = GaussianSampler(base.quadpoints, 0.05, 0.3, n_derivs=n_derivs)
                filament = _filament(CurvePerturbed(base, PerturbationSample(sampler)), frame)
                if not native_raises:
                    spec = curve_spec_from_adapter_curve(filament)
                    np.testing.assert_allclose(_terms(spec, 1)[1], filament.gammadash(), rtol=0, atol=1e-13)
                    continue
                with self.assertRaises(ValueError):
                    filament.gammadash()
                with self.assertRaisesRegex(ValueError, f"has {n_derivs} derivatives"):
                    curve_spec_from_adapter_curve(filament)
