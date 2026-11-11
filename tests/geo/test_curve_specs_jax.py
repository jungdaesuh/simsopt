import unittest

import numpy as np
from simsopt.geo import (
    CurveFilament,
    CurveHelical,
    CurvePerturbed,
    CurvePlanarFourier,
    CurveRZFourier,
    CurveXYZFourier,
    CurveXYZFourierSymmetries,
    FramedCurve,
    FramedCurveCentroid,
    FramedCurveFrenet,
    FrameRotation,
    GaussianSampler,
    PerturbationSample,
    RotatedCurve,
)
from simsopt.geo.curveplanarfourier import JaxCurvePlanarFourier
from simsopt.geo.curvexyzfourier import JaxCurveXYZFourier

try:
    import jax
    from simsopt_jax import core
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve
except ImportError:
    core = None


@unittest.skipIf(core is None, "simsopt_jax needs Python 3.12+ and JAX 0.12+")
class CurveSpecTests(unittest.TestCase):

    def test_spec_geometry_matches_native_curve(self):
        """Spec positions, parameter derivatives and their DOF derivatives equal the native curve's."""
        nquad = 128
        rng = np.random.default_rng(1)

        def shaped(curve):
            curve.x = 0.05 * rng.standard_normal(curve.dof_size)
            for name in ("xc(1)", "ys(1)", "rc(0)", "q0"):
                if name in curve.local_dof_names:
                    curve.set(name, 1.0)
            return curve

        xyz = shaped(CurveXYZFourier(nquad, 3))
        # A Fourier-curve sample has exact derivatives, as the spectral check below needs.
        p = shaped(CurveXYZFourier(nquad, 3))
        sample = [0.05 * term() for term in (p.gamma, p.gammadash, p.gammadashdash, p.gammadashdashdash)]
        perturbed = CurvePerturbed(xyz, PerturbationSample(None, sample=sample))
        rotation = FrameRotation(xyz.quadpoints, 2, scale=0.7)
        rotation.x = 0.1 * rng.standard_normal(rotation.dof_size)
        curves = [xyz, shaped(JaxCurveXYZFourier(nquad, 3)), shaped(CurvePlanarFourier(nquad, 3)),
                  shaped(JaxCurvePlanarFourier(nquad, 3)), shaped(CurveHelical(nquad, 4, 5, 2, 1.0, 0.3)),
                  *(shaped(CurveRZFourier(nquad, order, 3, stellsym))
                    for order, stellsym in ((3, True), (3, False), (0, True))),
                  *(shaped(CurveXYZFourierSymmetries(nquad, 3, 3, stellsym, 2)) for stellsym in (True, False)),
                  perturbed, CurveFilament(FramedCurveCentroid(perturbed, rotation), 0.02, 0.03),
                  CurveFilament(FramedCurveFrenet(xyz), 0.02, 0.03)]
        wavenumbers = 2j * np.pi * np.fft.fftfreq(nquad, 1 / nquad)
        for i, curve in enumerate(curves):
            with self.subTest(i=i, curve=type(curve).__name__):
                spec = curve_spec_from_adapter_curve(curve)
                terms = core.curve_geometry_from_dofs(spec, spec.dofs)
                jacobians = jax.jacfwd(core.curve_geometry_from_dofs, argnums=1)(spec, spec.dofs)
                names = ("gamma", "gammadash", "gammadashdash")
                if isinstance(curve, CurveFilament):
                    # Native filaments have no second derivative: compare with the spectral one.
                    spectral = np.fft.ifft(wavenumbers[:, None] * np.fft.fft(curve.gammadash(), axis=0), axis=0)
                    np.testing.assert_allclose(terms[2], spectral.real, atol=1e-7)
                    names = names[:2]
                v = rng.standard_normal((nquad, 3))
                for name, term, jacobian in zip(names, terms, jacobians):
                    np.testing.assert_allclose(term, getattr(curve, name)(), rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(np.einsum("ij,ijk->k", v, jacobian),
                                               getattr(curve, f"d{name}_by_dcoeff_vjp")(v)(curve),
                                               rtol=1e-10, atol=1e-10)

    def test_unsupported_curves_raise(self):
        """Curves without a JAX spec, or samples lacking a needed derivative, raise instead of giving geometry."""
        curve = CurveXYZFourier(32, 2)
        curve.set("xc(1)", 1.0)
        curve.set("ys(1)", 1.0)
        sample = PerturbationSample(GaussianSampler(curve.quadpoints, 0.05, 0.3, n_derivs=1),
                                    randomgen=np.random.default_rng(0))
        rotated = RotatedCurve(curve, np.pi / 3, flip=False)
        for unsupported in (FramedCurve(curve), CurvePerturbed(rotated, sample),
                            CurveFilament(FramedCurve(curve), 0.02, 0.03)):
            with self.subTest(curve=type(unsupported).__name__), self.assertRaises(NotImplementedError):
                curve_spec_from_adapter_curve(unsupported)
        perturbed = CurvePerturbed(curve, sample)
        with self.assertRaises(ValueError):
            curve_spec_from_adapter_curve(CurveFilament(FramedCurveCentroid(perturbed), 0.02, 0.03))
        spec = curve_spec_from_adapter_curve(perturbed)
        with self.assertRaises(ValueError):
            core.curve_geometry_from_dofs(spec, spec.dofs)


if __name__ == "__main__":
    unittest.main()
