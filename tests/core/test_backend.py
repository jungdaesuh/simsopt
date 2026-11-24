import unittest

import numpy as np
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveLength, create_equally_spaced_curves

try:
    import jax
    import jax.extend.backend
    from simsopt_jax.backend import get_backend_config, set_backend
    from simsopt_jax_adapters.field import JaxBiotSavart
except ImportError:
    set_backend = None


@unittest.skipIf(set_backend is None, "simsopt_jax needs Python 3.12+ and JAX 0.12+")
class BackendTests(unittest.TestCase):

    def setUp(self):
        for name in ("jax_enable_x64", "jax_platform_name", "jax_default_device", "jax_default_matmul_precision"):
            self.addCleanup(jax.config.update, name, jax.config.values[name])
        self.addCleanup(set_backend, "native")

    def test_jax_backend_matches_native_on_each_device(self):
        """On each available device JaxBiotSavart gives the native field in float64; a missing GPU raises."""
        curves = create_equally_spaced_curves(2, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32)
        coils = [Coil(curve, Current(1e5)) for curve in curves]
        points = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1], [0.3, 0.4, 0.3]])
        native = BiotSavart(coils).set_points(points).B()
        length = CurveLength(curves[0]).J()
        jax.config.update("jax_enable_x64", False)
        for device, platforms in (("cpu", {"cpu"}), ("gpu", {"cuda", "gpu"})):
            with self.subTest(device=device):
                if not platforms & set(jax.extend.backend.backends()):
                    previous = get_backend_config()
                    with self.assertRaises(RuntimeError):
                        set_backend("jax", device=device)
                    self.assertIs(get_backend_config(), previous)
                    continue
                config = set_backend("jax", device=device)
                self.assertIs(get_backend_config(), config)
                B = JaxBiotSavart(coils).set_points(points).B()
                self.assertEqual({d.platform for d in B.devices()}, {device})
                np.testing.assert_allclose(B, native, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(CurveLength(curves[0]).J(), length, rtol=1e-14)

    def test_invalid_selection_raises(self):
        """Only set_backend("native") and set_backend("jax", device="cpu" or "gpu") are accepted."""
        for backend, device in (("jax", None), ("native", "cpu"), ("jax", "tpu")):
            with self.subTest(backend=backend, device=device), self.assertRaises(ValueError):
                set_backend(backend, device=device)


if __name__ == "__main__":
    unittest.main()
