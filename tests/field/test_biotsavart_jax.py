import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from simsopt import load
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.geo import (
    CurveFilament,
    CurveHelical,
    CurvePerturbed,
    CurvePlanarFourier,
    CurveRZFourier,
    CurveXYZFourier,
    CurveXYZFourierSymmetries,
    FramedCurveCentroid,
    FramedCurveFrenet,
    FrameRotation,
    GaussianSampler,
    PerturbationSample,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.objectives import SquaredFlux

try:
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend import set_backend
    from simsopt_jax_adapters.field import JaxBiotSavart
except ImportError:
    JaxBiotSavart = None


@unittest.skipIf(JaxBiotSavart is None, "requires simsopt[jax]: Python 3.12+ and JAX 0.12+")
class JaxBiotSavartTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.curves = create_equally_spaced_curves(2, 3, stellsym=True, R0=1.0, R1=0.4, order=2, numquadpoints=40)
        for curve in self.curves:
            curve.x = curve.x + 0.02 * rng.standard_normal(curve.x.size)
        self.currents = [Current(1e5), Current(-4e4)]
        self.coils = coils_via_symmetries(self.curves, self.currents, 3, True)
        self.points_cyl = np.column_stack((rng.uniform(0.9, 1.1, 7), rng.uniform(-7, 7, 7), rng.uniform(-0.1, 0.1, 7)))
        R, phi, z = self.points_cyl.T
        self.points = np.column_stack((R * np.cos(phi), R * np.sin(phi), z))

    def test_circular_loop_on_axis(self):
        """On a circular loop's axis B_z = mu0 I R^2 / (2 (R^2 + z^2)^(3/2)), A = 0 and grad|B| = dB_z/dz."""
        radius, current = 1.3, 2e5
        curve = CurveXYZFourier(64, 1)
        curve.set("xc(1)", radius)
        curve.set("ys(1)", radius)
        z = np.array([-2.0, -0.4, 0.0, 0.25, 3.0])
        zero = np.zeros_like(z)
        field = JaxBiotSavart([Coil(curve, Current(current))]).set_points(np.column_stack((zero, zero, z)))
        Bz = 4e-7 * np.pi * current * radius**2 / (2 * (radius**2 + z**2) ** 1.5)
        dBz_dz = -3 * z * Bz / (radius**2 + z**2)
        np.testing.assert_allclose(field.B(), np.column_stack((zero, zero, Bz)), rtol=1e-14, atol=1e-14 * Bz.max())
        np.testing.assert_allclose(field.A(), 0, atol=1e-14 * Bz.max() * radius)
        np.testing.assert_allclose(field.GradAbsB(), np.column_stack((zero, zero, dBz_dz)),
                                   rtol=1e-13, atol=1e-13 * np.abs(dBz_dz).max())

    def test_field_invariants(self):
        """dB/dX is traceless and symmetric, B = curl A, d2B/dXdX is symmetric, and B has the coils' symmetries."""
        field = JaxBiotSavart(self.coils).set_points(self.points)
        B, dB, d2B, dA = (np.asarray(f()) for f in (field.B, field.dB_by_dX, field.d2B_by_dXdX, field.dA_by_dX))
        np.testing.assert_allclose(np.trace(dB, axis1=1, axis2=2), 0, atol=1e-13 * np.abs(dB).max())
        np.testing.assert_allclose(dB, dB.transpose(0, 2, 1), atol=1e-13 * np.abs(dB).max())
        np.testing.assert_allclose(d2B, d2B.transpose(0, 2, 1, 3), atol=1e-13 * np.abs(d2B).max())
        curl_A = np.column_stack((dA[:, 1, 2] - dA[:, 2, 1], dA[:, 2, 0] - dA[:, 0, 2], dA[:, 0, 1] - dA[:, 1, 0]))
        np.testing.assert_allclose(curl_A, B, atol=1e-13 * np.abs(B).max())
        angle = 2 * np.pi / 3
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        flip = np.diag([1.0, -1.0, -1.0])
        # Rotation by 2 pi / nfp maps B to R B; stellarator symmetry (x, y, z) -> (x, -y, -z) maps it to -T B.
        for name, transform, sign in (("nfp rotation", rotation, 1), ("stellarator symmetry", flip, -1)):
            with self.subTest(symmetry=name):
                mapped = field.set_points(self.points @ transform.T).B()
                np.testing.assert_allclose(mapped, sign * B @ transform.T, atol=1e-13 * np.abs(B).max())

    def test_vjps_match_central_differences(self):
        """The B, A, dB/dX and dA/dX VJPs in coil and current DOFs are the second-order limit of central differences."""
        field = JaxBiotSavart(self.coils).set_points(self.points)
        rng = np.random.default_rng(1)
        v, vgrad = rng.standard_normal((7, 3)), rng.standard_normal((7, 3, 3))
        x0 = field.x.copy()
        h = 1e-2 * rng.uniform(-1, 1, x0.size) * np.maximum(np.abs(x0), 1)
        objectives = {
            "B": (lambda: np.sum(v * field.B()), lambda: field.B_vjp(v)),
            "A": (lambda: np.sum(v * field.A()), lambda: field.A_vjp(v)),
            "B and dB/dX": (lambda: np.sum(v * field.B()) + np.sum(vgrad * field.dB_by_dX()),
                            lambda: sum(field.B_and_dB_vjp(v, vgrad))),
            "A and dA/dX": (lambda: np.sum(v * field.A()) + np.sum(vgrad * field.dA_by_dX()),
                            lambda: sum(field.A_and_dA_vjp(v, vgrad))),
        }
        for name, (J, dJ) in objectives.items():
            with self.subTest(objective=name):
                field.x = x0
                slope = dJ()(field) @ h
                err = np.inf
                for eps in 0.5 ** np.arange(6):
                    field.x = x0 + eps * h
                    J_plus = J()
                    field.x = x0 - eps * h
                    new_err = abs((J_plus - J()) / (2 * eps) - slope)
                    self.assertLess(new_err, 0.3 * err)
                    err = new_err

    def test_matches_native(self):
        """Fields, derivatives, cylindrical components, per-coil unit-current fields and VJPs equal BiotSavart's."""
        rng = np.random.default_rng(2)
        curves = self.curves
        sample = PerturbationSample(GaussianSampler(curves[0].quadpoints, 1e-2, 0.2, n_derivs=1), randomgen=rng)
        rotation = FrameRotation(curves[1].quadpoints, 1)
        rotation.x = [0.2, -0.1, 0.05]
        symmetric = CurveXYZFourierSymmetries(40, 1, nfp=3, stellsym=True, ntor=1)
        symmetric.x = [1.6, 0.05, 0.05, 0.05]
        planar = CurvePlanarFourier(40, 1)
        planar.x = [0.3, 0.02, 0.01, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.8]
        rz = CurveRZFourier(40, 1, 3, True)
        rz.x = [1.8, 0.05, 0.05]
        first, second = Current(1e5), Current(-3e4)
        mixed = [
            Coil(CurvePerturbed(curves[0], sample), first),
            Coil(CurveXYZFourier(curves[0].quadpoints, 2, dofs=curves[0].dofs), Current(2e4)),
            Coil(CurveFilament(FramedCurveCentroid(curves[1], rotation), 0.01, 0.02), first - second),
            Coil(CurvePerturbed(CurveFilament(FramedCurveFrenet(curves[1], rotation), -0.01, 0.02), sample), -2 * second),
            Coil(symmetric, Current(3e4)),
            Coil(planar, Current(-2e4)),
            Coil(rz, Current(1e4)),
            Coil(CurveHelical(40, 1, 5, 2, 1.0, 0.45), Current(5e3)),
        ]
        fine = create_equally_spaced_curves(2, 3, stellsym=True, R0=1.0, R1=0.4, order=2, numquadpoints=100)
        fine[1] = CurveXYZFourier(150, 2, dofs=fine[1].dofs)
        many_points_cyl = np.column_stack((rng.uniform(0.9, 1.1, 1100), rng.uniform(0, 7, 1100),
                                           rng.uniform(-0.1, 0.1, 1100)))
        # Past 64 quadrature nodes and 1024 points the kernels evaluate in blocks.
        variants = {
            "stellarator-symmetric coils": (self.coils, self.points_cyl),
            "curve and current types": (mixed, self.points_cyl),
            "some DOFs fixed": (mixed, self.points_cyl),
            "all DOFs fixed": (mixed, self.points_cyl),
            "100 and 150 nodes at 1100 points": ([Coil(c, i) for c, i in zip(fine, self.currents)], many_points_cyl),
            "no coils": ([], self.points_cyl),
        }
        for variant, (coils, points_cyl) in variants.items():
            if variant == "some DOFs fixed":
                curves[0].fix(0)
                rotation.fix(0)
                second.fix_all()
            elif variant == "all DOFs fixed":
                for opt in BiotSavart(coils).ancestors:
                    opt.fix_all()
            with self.subTest(variant=variant):
                native = BiotSavart(coils).set_points_cyl(points_cyl)
                field = JaxBiotSavart(coils).set_points_cyl(points_cyl)
                for name in ("get_points_cart", "get_points_cyl"):
                    np.testing.assert_allclose(getattr(field, name)(), getattr(native, name)(), rtol=1e-15, atol=1e-15)
                for name in ("B", "dB_by_dX", "d2B_by_dXdX", "A", "dA_by_dX", "d2A_by_dXdX", "AbsB", "GradAbsB",
                             "B_cyl", "A_cyl", "GradAbsB_cyl"):
                    expected = getattr(native, name)()
                    np.testing.assert_allclose(getattr(field, name)(), expected, rtol=1e-10,
                                               atol=1e-12 * np.abs(expected).max(initial=0))
                for actual, expected in zip(field.B_and_dB(), (native.B(), native.dB_by_dX()), strict=True):
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max(initial=0))
                for name in ("dB_by_dcoilcurrents", "d2B_by_dXdcoilcurrents", "d3B_by_dXdXdcoilcurrents",
                             "dA_by_dcoilcurrents", "d2A_by_dXdcoilcurrents", "d3A_by_dXdXdcoilcurrents"):
                    for actual, expected in zip(getattr(field, name)(), getattr(native, name)(), strict=True):
                        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())
                v, vgrad = rng.standard_normal(points_cyl.shape), rng.standard_normal((len(points_cyl), 3, 3))
                for name, args in (("B_vjp", (v,)), ("A_vjp", (v,)), ("B_and_dB_vjp", (v, vgrad)),
                                   ("A_and_dA_vjp", (v, vgrad))):
                    jax_vjps, native_vjps = getattr(field, name)(*args), getattr(native, name)(*args)
                    if len(args) == 1:
                        jax_vjps, native_vjps = (jax_vjps,), (native_vjps,)
                    for actual, expected in zip(jax_vjps, native_vjps, strict=True):
                        # Without coils the native VJP is the integer 0.
                        expected = expected(native) if coils else np.zeros(0)
                        np.testing.assert_allclose(actual(field), expected, rtol=1e-10,
                                                   atol=1e-12 * np.abs(expected).max(initial=0))

    def test_squared_flux_matches_native(self):
        """SquaredFlux on JaxBiotSavart has the value and gradient it has on BiotSavart."""
        surface = SurfaceRZFourier.from_nphi_ntheta(nphi=12, ntheta=8, nfp=3, range="half period")
        surface.set_rc(1, 0, 0.2)
        surface.set_zs(1, 0, 0.2)
        points = surface.gamma().reshape((-1, 3))
        native = SquaredFlux(surface, BiotSavart(self.coils).set_points(points))
        jax_flux = SquaredFlux(surface, JaxBiotSavart(self.coils).set_points(points))
        np.testing.assert_allclose(jax_flux.J(), native.J(), rtol=1e-12)
        np.testing.assert_allclose(jax_flux.dJ(), native.dJ(), rtol=1e-10, atol=1e-12 * np.abs(native.dJ()).max())

    def test_field_follows_changes(self):
        """After each change of points, free or fixed DOFs, currents, perturbation sample or backend, the field, a copy
        of it and their B VJPs equal a new BiotSavart's."""
        sample = PerturbationSample(GaussianSampler(self.curves[0].quadpoints, 1e-2, 0.2, n_derivs=1),
                                    randomgen=np.random.default_rng(3))
        perturbed = CurvePerturbed(self.curves[0], sample)
        coils = coils_via_symmetries([perturbed, self.curves[1]], self.currents, 3, True)
        field = JaxBiotSavart(coils).set_points(self.points)
        copied = copy.copy(field)
        v = np.random.default_rng(4).standard_normal((7, 3))
        changes = {
            "none": lambda: None,
            "set_points": lambda: field.set_points(jnp.asarray(self.points[::-1])),
            "set_points_cyl": lambda: field.set_points_cyl(field.get_points_cyl() + [0.05, 0.1, 0.02]),
            "field DOFs": lambda: setattr(field, "x", 1.01 * field.x),
            "curve DOFs": lambda: setattr(self.curves[1], "x", self.curves[1].x + 2e-3),
            "current": lambda: setattr(self.currents[0], "x", [2.5e5]),
            "fix a DOF": lambda: self.curves[1].fix(0),
            "change the fixed DOF": lambda: self.curves[1].set(0, self.curves[1].get(0) + 5e-3),
            "resample": perturbed.resample,
            "current scale": lambda: (setattr(coils[-1].current, "scale", 1.5), coils[-1].current.set_recompute_flag()),
            "set_backend": lambda: set_backend("native"),
        }
        for change, apply in changes.items():
            apply()
            for name, f in (("field", field), ("copy", copied)):
                with self.subTest(change=change, field=name):
                    # CurvePerturbed.resample does not reach the native caches of the curve's rotated copies.
                    for coil in coils:
                        coil.curve.invalidate_cache()
                    native = BiotSavart(coils).set_points(f.get_points_cart())
                    np.testing.assert_allclose(f.B(), native.B(), rtol=1e-12, atol=1e-14)
                    np.testing.assert_allclose(f.dB_by_dX(), native.dB_by_dX(), rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(f.B_vjp(v)(f), native.B_vjp(v)(native), rtol=1e-10, atol=1e-14)

    def test_set_points_copies_the_points(self):
        """Editing the buffer behind the points passed to set_points changes neither the points nor the field."""
        native = BiotSavart(self.coils).set_points(self.points)
        for kind in ("NumPy", "JAX"):
            with self.subTest(points=kind):
                # A 64-byte aligned buffer, which JAX on the CPU uses in place.
                storage = np.empty(self.points.size + 8)
                start = (-storage.ctypes.data % 64) // storage.itemsize
                buffer = storage[start:start + self.points.size].reshape(self.points.shape)
                buffer[:] = self.points
                field = JaxBiotSavart(self.coils).set_points_cart(buffer if kind == "NumPy" else jax.device_put(buffer))  # noqa: TID251 test needs a JAX array aliasing a host buffer
                buffer += 0.3
                np.testing.assert_array_equal(field.get_points_cart(), self.points)
                np.testing.assert_allclose(field.dB_by_dX(), native.dB_by_dX(), rtol=1e-10, atol=1e-12)

    def test_save_and_load(self):
        """A saved and loaded field has the same points and B; a field without points cannot be evaluated."""
        field = JaxBiotSavart(self.coils).set_points_cyl(self.points_cyl)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.json"
            field.save(path)
            loaded = load(path)
            np.testing.assert_array_equal(loaded.get_points_cart(), field.get_points_cart())
            np.testing.assert_allclose(loaded.B(), field.B(), rtol=1e-14, atol=1e-16)
            field.clear_points()
            field.save(path)
            for f in (field, load(path)):
                with self.assertRaises(RuntimeError):
                    f.B()

    def test_field_arithmetic_is_rejected(self):
        """Sums and multiples, which need the native field interface, raise TypeError."""
        field = JaxBiotSavart(self.coils)
        operations = {
            "sum": lambda: field + field,
            "multiple": lambda: 2 * field,
            "native sum": lambda: MagneticFieldSum([BiotSavart(self.coils), field]),
        }
        for name, operation in operations.items():
            with self.subTest(operation=name), self.assertRaises(TypeError):
                operation()


if __name__ == "__main__":
    unittest.main()
