import importlib.util
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simsopt.geo.surfaceobjectives import boozer_surface_residual, boozer_surface_residual_dB
from simsopt.objectives.utilities import forward_backward


EXAMPLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
TEST_MPOL = 8
TEST_NTOR = 6
TEST_VOL_TARGET = 0.1
TEST_IOTA = 0.15
TEST_G0 = 1.0


def load_single_stage_example_module():
    spec = importlib.util.spec_from_file_location(
        f"single_stage_banana_example_{uuid.uuid4().hex}",
        EXAMPLE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeSurfPrev:
    def __init__(self):
        self.nfp = 5
        self.quadpoints_phi = np.linspace(0, 1 / self.nfp, 13, endpoint=False)
        self.quadpoints_theta = np.linspace(0, 1, 17, endpoint=False)

    def gamma(self):
        return np.zeros((self.quadpoints_phi.size, self.quadpoints_theta.size, 3))


class FakeSurfaceXYZTensorFourier:
    instances = []

    def __init__(
        self,
        *,
        mpol,
        ntor,
        nfp,
        stellsym,
        quadpoints_theta,
        quadpoints_phi,
        dofs=None,
    ):
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        self.quadpoints_theta = np.asarray(quadpoints_theta)
        self.quadpoints_phi = np.asarray(quadpoints_phi)
        self.dofs = np.array([1.0]) if dofs is None else np.asarray(dofs)
        FakeSurfaceXYZTensorFourier.instances.append(self)

    def least_squares_fit(self, gamma):
        self.fitted_gamma = gamma

    def is_self_intersecting(self):
        return False

    def volume(self):
        return 1.0


class FakeVolume:
    def __init__(self, surface):
        self.surface = surface


class FakeBoozerSurface:
    def __init__(self, bs, surface, label, targetlabel, constraint_weight, options=None):
        self.bs = bs
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.options = options or {}
        self.res = {"success": True, "iota": 0.15, "G": 1.0}

    def run_code(self, iota, G):
        return self.res


class SingleStageExampleTests(unittest.TestCase):
    def setUp(self):
        FakeSurfaceXYZTensorFourier.instances = []

    def load_module(self):
        return load_single_stage_example_module()

    def initialize_boozer_surface(self, module, surf_prev, *, constraint_weight):
        with patch.object(module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier), patch.object(
            module, "Volume", FakeVolume
        ), patch.object(module, "BoozerSurface", FakeBoozerSurface):
            return module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=constraint_weight,
                iota=TEST_IOTA,
                G0=TEST_G0,
            )

    def test_exact_boozer_helpers_are_imported(self):
        module = self.load_module()

        self.assertIs(module.boozer_surface_residual, boozer_surface_residual)
        self.assertIs(module.boozer_surface_residual_dB, boozer_surface_residual_dB)
        self.assertIs(module.forward_backward, forward_backward)

    def test_initialize_boozer_surface_exact_uses_ntor_phi_quadrature(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(module, surf_prev, constraint_weight=None)

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 2)

        exact_surface = FakeSurfaceXYZTensorFourier.instances[1]
        expected_phi = np.linspace(0, 1 / surf_prev.nfp, 2 * TEST_NTOR + 1, endpoint=False)

        self.assertEqual(exact_surface.quadpoints_theta.size, 2 * TEST_MPOL + 1)
        self.assertEqual(exact_surface.quadpoints_phi.size, 2 * TEST_NTOR + 1)
        np.testing.assert_allclose(exact_surface.quadpoints_phi, expected_phi)

    def test_initialize_boozer_surface_zero_constraint_weight_keeps_least_squares_path(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(module, surf_prev, constraint_weight=0.0)

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 1)
        self.assertIs(boozer_surface.surface, FakeSurfaceXYZTensorFourier.instances[0])

    def test_fun_fallback_returns_elevated_j_and_same_sign_gradient(self):
        """Issue #2: failed Boozer must return elevated J + same-sign gradient,
        not (J_old, -dJ_old)."""
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)
            def is_self_intersecting(self):
                return False

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": False, "iota": TEST_IOTA, "G": TEST_G0}
            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        module.run_dict = {
            "x_prev": np.zeros(5), "lscount": 0,
            "sdofs": np.ones(3), "iota": TEST_IOTA, "G": TEST_G0,
            "J": last_J, "dJ": last_dJ.copy(),
        }
        module.boozer_surface = _BoozerSurface()
        module.JF = _JF()

        J_out, dJ_out = module.fun(np.ones(5))

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        self.assertIsNot(dJ_out, module.run_dict["dJ"])


class BoozerFallbackLBFGSBTests(unittest.TestCase):
    """Issue #2: elevated-J fallback must not flush L-BFGS-B Hessian memory."""

    def test_elevated_j_stale_gradient_preserves_bfgs_memory(self):
        from scipy.optimize import minimize

        def rosenbrock(x):
            f = sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                    for i in range(len(x) - 1))
            g = np.zeros_like(x)
            for i in range(len(x) - 1):
                g[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
                g[i+1] += 200*(x[i+1] - x[i]**2)
            return f, g

        rng = np.random.RandomState(42)
        x0 = rng.randn(10) * 0.5
        state = {"x_good": x0.copy(), "J": None, "dJ": None}

        def fun_with_fallback(x):
            f, g = rosenbrock(x)
            if np.linalg.norm(x - state["x_good"]) > 0.5 and state["J"] is not None:
                return state["J"] + max(abs(state["J"]), 1.0), state["dJ"].copy()
            state["J"] = f
            state["dJ"] = g.copy()
            state["x_good"] = x.copy()
            return f, g

        res = minimize(fun_with_fallback, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "maxcor": 10})

        self.assertTrue(res.success, f"L-BFGS-B did not converge: {res.message}")
        self.assertGreater(res.hess_inv.n_corrs, 0)


STAGE2_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)


def _clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _segment_segment_distance(P1, P2, Q1, Q2):
    """Pure-Python replica of the Sunday/Lumelsky algorithm in banana_coil_solver.py.

    Used for regression testing independent of numba availability.
    Any change to the deployed algorithm must be mirrored here.
    """
    u = P2 - P1
    v = Q2 - Q1
    w0 = P1 - Q1
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    ZERO_LEN = 1e-30
    PAR_EPS = 1e-10
    if a < ZERO_LEN:
        if c < ZERO_LEN:
            return np.linalg.norm(w0)
        return np.linalg.norm(w0 - _clamp01(e / c) * v)
    if c < ZERO_LEN:
        return np.linalg.norm(w0 + _clamp01(-d / a) * u)
    denom = a * c - b * b
    if denom < PAR_EPS * a * c:
        best_sq = np.inf
        dp = w0 - _clamp01(e / c) * v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq
        dp = w0 + u - _clamp01((e + b) / c) * v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq
        dp = w0 + _clamp01(-d / a) * u
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq
        dp = w0 + _clamp01((b - d) / a) * u - v
        dsq = np.dot(dp, dp)
        if dsq < best_sq:
            best_sq = dsq
        return np.sqrt(best_sq)
    sc = (b * e - c * d) / denom
    tc = (a * e - b * d) / denom
    if sc < 0.0:
        sc = 0.0
        tc = e / c
    elif sc > 1.0:
        sc = 1.0
        tc = (e + b) / c
    if tc < 0.0:
        tc = 0.0
        sc = _clamp01(-d / a)
    elif tc > 1.0:
        tc = 1.0
        sc = _clamp01((b - d) / a)
    dp = w0 + sc * u - tc * v
    return np.sqrt(np.dot(dp, dp))


class SegmentDistanceTests(unittest.TestCase):
    """Issue #5/#6: segment-segment distance with Sunday/Lumelsky re-projection."""

    def _d(self, p1, p2, q1, q2):
        return _segment_segment_distance(
            np.array(p1, dtype=float), np.array(p2, dtype=float),
            np.array(q1, dtype=float), np.array(q2, dtype=float),
        )

    def test_skew_segments_reprojection(self):
        """Issue #5: buggy=1.414, correct=sqrt(1.8) after re-projection."""
        d = self._d([0, 0, 0], [2, 1, 0], [-1, 3, 0], [1, 2, 0])
        self.assertAlmostEqual(d, np.sqrt(1.8), places=10)

    def test_parallel_overlapping_segments(self):
        """Issue #6: buggy=8.06, correct=1.0 for overlapping parallel segments."""
        d = self._d([0, 0, 0], [10, 0, 0], [8, 1, 0], [20, 1, 0])
        self.assertAlmostEqual(d, 1.0, places=10)

    def test_collinear_gap(self):
        d = self._d([0, 0, 0], [1, 0, 0], [3, 0, 0], [5, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_perpendicular_touching(self):
        d = self._d([0, 0, 0], [1, 0, 0], [0.5, 0, 0], [0.5, 1, 0])
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_point_to_segment(self):
        d = self._d([0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_parallel_non_overlapping(self):
        d = self._d([0, 0, 0], [3, 0, 0], [5, 1, 0], [8, 1, 0])
        self.assertAlmostEqual(d, np.sqrt(5.0), places=10)

    def test_t_shaped(self):
        d = self._d([0, 0, 0], [2, 0, 0], [1, 0.5, 0], [1, 3, 0])
        self.assertAlmostEqual(d, 0.5, places=10)

    def test_both_degenerate(self):
        d = self._d([1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6])
        self.assertAlmostEqual(d, np.linalg.norm([3, 3, 3]), places=10)

    def test_random_brute_force(self):
        """Verify against exhaustive interior + edge search on 1000 random pairs."""
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            P1, P2, Q1, Q2 = rng.randn(4, 3)
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            u = P2 - P1
            v = Q2 - Q1
            w0 = P1 - Q1
            a, bv, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
            d_val, e = np.dot(u, w0), np.dot(v, w0)
            cands = []
            # Interior (unclamped line-line solution)
            denom = a * c - bv * bv
            if denom > 1e-30:
                sn = (bv * e - c * d_val) / denom
                tn = (a * e - bv * d_val) / denom
                if 0.0 <= sn <= 1.0 and 0.0 <= tn <= 1.0:
                    dp = w0 + sn * u - tn * v
                    cands.append(np.dot(dp, dp))
            # Four edge optima
            for sf in [0.0, 1.0]:
                to = max(0.0, min(1.0, (e + sf * bv) / c)) if c > 1e-30 else 0.0
                dp = w0 + sf * u - to * v
                cands.append(np.dot(dp, dp))
            for tf in [0.0, 1.0]:
                so = max(0.0, min(1.0, (tf * bv - d_val) / a)) if a > 1e-30 else 0.0
                dp = w0 + so * u - tf * v
                cands.append(np.dot(dp, dp))
            d_brute = np.sqrt(min(cands))
            self.assertAlmostEqual(d_algo, d_brute, places=9,
                                   msg=f"Mismatch: algo={d_algo}, brute={d_brute}")


    def test_replica_matches_deployed_source(self):
        """Guard against the test replica diverging from the deployed @njit function."""
        import inspect, re
        source = STAGE2_MODULE_PATH.read_text()

        def _extract_code_lines(text, func_name):
            """Extract function body lines, stripping docstrings, comments, and blanks."""
            lines = []
            in_func = False
            in_docstring = False
            for line in text.splitlines():
                stripped = line.strip()
                if f"def {func_name}(" in stripped and not in_func:
                    in_func = True
                    continue
                if in_func:
                    if '"""' in stripped:
                        if stripped.count('"""') >= 2:
                            continue
                        in_docstring = not in_docstring
                        continue
                    if in_docstring:
                        continue
                    if stripped and not line.startswith("    ") and not line.startswith("\t"):
                        break
                    code = re.sub(r'\s*#.*$', '', stripped)
                    if code:
                        lines.append(code)
            return lines

        replica = _extract_code_lines(inspect.getsource(_segment_segment_distance), "_segment_segment_distance")
        deployed = _extract_code_lines(source, "segment_segment_distance")
        self.assertEqual(
            replica, deployed,
            "Test replica of segment_segment_distance has diverged from deployed code. "
            "Update the test replica to match banana_coil_solver.py."
        )


class CrossSectionNormalizationTests(unittest.TestCase):
    """Issue #8/#9: cross_section phi argument must be normalized to [0,1]."""

    def test_single_stage_source_divides_by_2pi(self):
        """Verify the deployed source uses phi_slice / (2 * np.pi), not * 2 * np.pi."""
        source = EXAMPLE_MODULE_PATH.read_text()
        self.assertIn("phi_slice / (2 * np.pi)", source)
        self.assertNotIn("phi_slice * 2 * np.pi", source)

    def test_stage2_source_divides_by_2pi(self):
        """Verify banana_coil_solver.py source uses the correct normalization."""
        source = STAGE2_MODULE_PATH.read_text()
        self.assertIn("phi_slice / (2 * np.pi)", source)
        self.assertNotIn("phi_slice * 2 * np.pi", source)


class FtolGtolDefaultTests(unittest.TestCase):
    """Issue #31: ftol/gtol must not be None for any mpol value."""

    def test_ftol_gtol_have_defaults_for_all_mpol(self):
        module = load_single_stage_example_module()
        ftol_by_mpol = module.ftol_by_mpol
        gtol_by_mpol = module.gtol_by_mpol
        for mpol in range(1, 30):
            ftol = ftol_by_mpol.get(mpol, 1e-5 if mpol < 8 else 1e-10)
            gtol = gtol_by_mpol.get(mpol, 1e-2 if mpol < 8 else 1e-7)
            self.assertIsNotNone(ftol, f"ftol is None for mpol={mpol}")
            self.assertIsNotNone(gtol, f"gtol is None for mpol={mpol}")
            self.assertIsInstance(ftol, float, f"ftol not float for mpol={mpol}")
            self.assertIsInstance(gtol, float, f"gtol not float for mpol={mpol}")
            self.assertGreater(ftol, 0, f"ftol not positive for mpol={mpol}")
            self.assertGreater(gtol, 0, f"gtol not positive for mpol={mpol}")

    def test_defaults_match_dictionary_endpoints(self):
        module = load_single_stage_example_module()
        ftol_by_mpol = module.ftol_by_mpol
        gtol_by_mpol = module.gtol_by_mpol
        self.assertEqual(ftol_by_mpol.get(7, 1e-5 if 7 < 8 else 1e-10), 1e-5)
        self.assertEqual(ftol_by_mpol.get(19, 1e-5 if 19 < 8 else 1e-10), 1e-10)
        self.assertEqual(gtol_by_mpol.get(7, 1e-2 if 7 < 8 else 1e-7), 1e-2)
        self.assertEqual(gtol_by_mpol.get(19, 1e-2 if 19 < 8 else 1e-7), 1e-7)

    def test_source_uses_default_argument(self):
        """The deployed .get() calls must include a default, not bare .get(mpol)."""
        source = EXAMPLE_MODULE_PATH.read_text()
        self.assertNotIn("ftol_by_mpol.get(mpol)", source)
        self.assertNotIn("gtol_by_mpol.get(mpol)", source)


if __name__ == "__main__":
    unittest.main()
