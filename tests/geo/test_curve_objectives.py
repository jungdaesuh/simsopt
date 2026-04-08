import unittest
import json

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from _jit_test_state import make_module_jit_hooks
import simsopt.geo.curveobjectives as curveobjectives_module
from simsopt.geo import FrameRotation, FramedCurveCentroid
from simsopt.geo import parameters
from simsopt.geo.curve import (
    RotatedCurve,
    create_equally_spaced_curves,
)
from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curveobjectives import (
    CurveLength,
    LpCurveCurvature,
    LpCurveCurvatureBarrier,
    LpCurveTorsion,
    CurveCurveDistance,
    CurveCurveDistanceBarrier,
    ArclengthVariation,
    MeanSquaredCurvature,
    CurveSurfaceDistance,
    FramedCurveTwist,
    LinkingNumber,
    cc_distance_barrier_pure,
    cc_distance_pure,
    cs_distance_pure,
    max_distance_pure,
    pairwise_min_distance_pure,
)
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.coil import coils_via_symmetries
from simsopt.configs.zoo import get_data
from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON
from simsopt.backend import invalidate_backend_cache
import simsoptpp as sopp

setUpModule, tearDownModule = make_module_jit_hooks(parameters, value=False)


def _make_cache_test_curve(offset=0.0):
    quadpoints = np.linspace(0, 1, 32, endpoint=False)
    curve = CurveXYZFourier(quadpoints, order=3)
    curve.x = np.linspace(
        0.1 + offset,
        0.1 + offset + 0.01 * (curve.dof_size - 1),
        curve.dof_size,
    )
    return curve


def _legacy_curveobjective_jit_attrs(objective):
    return [
        name
        for name in vars(objective)
        if name == "J_jax"
        or name.startswith("thisgrad")
        or name.startswith("frametwist_vjp")
        or name in {"range_grad", "net_grad"}
    ]


def _jit_cache_sizes(*functions):
    return tuple(fn._cache_size() for fn in functions)


@pytest.fixture
def _clear_curveobjective_shared_jit_caches():
    cache_fns = (
        curveobjectives_module.Lp_torsion_pure,
        curveobjectives_module._lp_curve_torsion_grad,
        curveobjectives_module.frametwist_lp_pure,
        curveobjectives_module._frametwist_lp_grad,
        curveobjectives_module._frametwist_vjp,
    )
    for fn in cache_fns:
        fn.clear_cache()
    yield
    for fn in cache_fns:
        fn.clear_cache()


def test_lp_curve_torsion_reuses_shared_jit_kernels(
    _clear_curveobjective_shared_jit_caches,
):
    cache_fns = (
        curveobjectives_module.Lp_torsion_pure,
        curveobjectives_module._lp_curve_torsion_grad,
    )
    objective1 = LpCurveTorsion(_make_cache_test_curve(0.0), p=2, threshold=0.0)
    float(objective1.J())
    np.asarray(objective1.dJ(), dtype=float)

    first_cache_sizes = _jit_cache_sizes(*cache_fns)
    assert _legacy_curveobjective_jit_attrs(objective1) == []

    objective2 = LpCurveTorsion(_make_cache_test_curve(0.02), p=2, threshold=0.0)
    float(objective2.J())
    np.asarray(objective2.dJ(), dtype=float)

    assert _jit_cache_sizes(*cache_fns) == first_cache_sizes
    assert _legacy_curveobjective_jit_attrs(objective2) == []


def test_framed_curve_twist_reuses_shared_jit_kernels(
    _clear_curveobjective_shared_jit_caches,
):
    cache_fns = (
        curveobjectives_module.frametwist_lp_pure,
        curveobjectives_module._frametwist_lp_grad,
        curveobjectives_module._frametwist_vjp,
    )
    curve1 = _make_cache_test_curve(0.03)
    rotation1 = FrameRotation(curve1.quadpoints, order=1)
    rotation1.x = np.array([0.1, -0.2, 0.05])
    objective1 = FramedCurveTwist(FramedCurveCentroid(curve1, rotation1), f="lp", p=2)
    float(objective1.J())
    np.asarray(objective1.dJ(), dtype=float)

    first_cache_sizes = _jit_cache_sizes(*cache_fns)
    assert _legacy_curveobjective_jit_attrs(objective1) == []

    curve2 = _make_cache_test_curve(0.04)
    rotation2 = FrameRotation(curve2.quadpoints, order=1)
    rotation2.x = np.array([-0.3, 0.15, 0.02])
    objective2 = FramedCurveTwist(FramedCurveCentroid(curve2, rotation2), f="lp", p=2)
    float(objective2.J())
    np.asarray(objective2.dJ(), dtype=float)

    assert _jit_cache_sizes(*cache_fns) == first_cache_sizes
    assert _legacy_curveobjective_jit_attrs(objective2) == []


def test_pairwise_penalty_chunking_matches_dense_paths(monkeypatch):
    gamma1 = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.12, 0.01, 0.00],
            [0.24, 0.02, 0.01],
            [0.36, 0.03, 0.02],
        ],
        dtype=np.float64,
    )
    gamma2 = np.array(
        [
            [0.03, 0.01, 0.00],
            [0.15, 0.04, 0.01],
            [0.27, 0.05, 0.02],
        ],
        dtype=np.float64,
    )
    gammadash1 = np.array(
        [
            [0.10, 0.01, 0.00],
            [0.10, 0.01, 0.01],
            [0.10, 0.01, 0.01],
            [0.10, 0.01, 0.02],
        ],
        dtype=np.float64,
    )
    gammadash2 = np.array(
        [
            [0.11, 0.02, 0.00],
            [0.11, 0.02, 0.01],
            [0.11, 0.02, 0.02],
        ],
        dtype=np.float64,
    )
    surface_gamma = np.array(
        [
            [0.00, 0.20, 0.00],
            [0.12, 0.21, 0.01],
            [0.24, 0.22, 0.02],
            [0.36, 0.23, 0.03],
            [0.48, 0.24, 0.04],
        ],
        dtype=np.float64,
    )
    surface_normal = np.array(
        [
            [0.00, 1.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 1.00, 0.00],
        ],
        dtype=np.float64,
    )

    monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "2")
    invalidate_backend_cache()
    try:
        chunked_cc = float(
            cc_distance_pure(gamma1, gammadash1, gamma2, gammadash2, 0.09)
        )
        chunked_cc_barrier = float(
            cc_distance_barrier_pure(gamma1, gammadash1, gamma2, gammadash2, 0.01)
        )
        chunked_cs = float(
            cs_distance_pure(gamma1, gammadash1, surface_gamma, surface_normal, 0.05)
        )
        chunked_max = float(max_distance_pure(gamma1, gamma2, 0.30, -10))
        chunked_min = float(pairwise_min_distance_pure(gamma1, surface_gamma))
    finally:
        monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "0")
        invalidate_backend_cache()

    dense_cc = float(cc_distance_pure(gamma1, gammadash1, gamma2, gammadash2, 0.09))
    dense_cc_barrier = float(
        cc_distance_barrier_pure(gamma1, gammadash1, gamma2, gammadash2, 0.01)
    )
    dense_cs = float(
        cs_distance_pure(gamma1, gammadash1, surface_gamma, surface_normal, 0.05)
    )
    dense_max = float(max_distance_pure(gamma1, gamma2, 0.30, -10))
    dense_min = float(pairwise_min_distance_pure(gamma1, surface_gamma))

    monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
    invalidate_backend_cache()

    assert chunked_cc == pytest.approx(dense_cc, rel=1e-12, abs=1e-12)
    assert chunked_cc_barrier == pytest.approx(dense_cc_barrier, rel=1e-12, abs=1e-12)
    assert chunked_cs == pytest.approx(dense_cs, rel=1e-12, abs=1e-12)
    assert chunked_max == pytest.approx(dense_max, rel=1e-12, abs=1e-12)
    assert chunked_min == pytest.approx(dense_min, rel=1e-12, abs=1e-12)


def test_pairwise_penalty_chunking_preserves_infeasible_barrier_inf(monkeypatch):
    gamma1 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    gamma2 = np.array([[0.02, 0.0, 0.0], [0.12, 0.0, 0.0]], dtype=np.float64)
    gammadash = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)

    monkeypatch.setenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", "1")
    invalidate_backend_cache()
    try:
        value = cc_distance_barrier_pure(gamma1, gammadash, gamma2, gammadash, 0.05)
    finally:
        monkeypatch.delenv("SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE", raising=False)
        invalidate_backend_cache()

    assert np.isposinf(float(value))


def test_pairwise_penalty_accepts_explicit_row_sharding():
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("d",))
    gamma1 = jax.device_put(
        np.array(
            [
                [0.00, 0.00, 0.00],
                [0.12, 0.01, 0.00],
                [0.24, 0.02, 0.01],
                [0.36, 0.03, 0.02],
            ],
            dtype=np.float64,
        ),
        NamedSharding(mesh, P("d", None)),
    )
    gamma2 = np.array(
        [
            [0.03, 0.01, 0.00],
            [0.15, 0.04, 0.01],
            [0.27, 0.05, 0.02],
        ],
        dtype=np.float64,
    )

    rowwise = curveobjectives_module._pairwise_rowwise_min_distance(
        gamma1,
        gamma2,
        chunk_size=2,
    )
    dense_rowwise = curveobjectives_module._pairwise_rowwise_min_distance(
        np.asarray(gamma1),
        gamma2,
        chunk_size=0,
    )
    sharded_min = float(pairwise_min_distance_pure(gamma1, gamma2, chunk_size=2))
    dense_min = float(
        pairwise_min_distance_pure(np.asarray(gamma1), gamma2, chunk_size=0)
    )

    np.testing.assert_allclose(
        np.asarray(rowwise), np.asarray(dense_rowwise), atol=1e-12
    )
    assert sharded_min == pytest.approx(dense_min, rel=1e-12, abs=1e-12)


class Testing(unittest.TestCase):
    curvetypes = [
        "CurveXYZFourier",
        "JaxCurveXYZFourier",
        "CurveRZFourier",
        "CurvePlanarFourier",
        "JaxCurvePlanarFourier",
        "CurveHelical",
    ]
    _BARRIER_TAYLOR_FLOOR = 5e-12

    def create_curve(self, curvetype, rotated):
        np.random.seed(1)
        rand_scale = 0.01
        order = 4
        nquadpoints = 200

        if curvetype == "CurveXYZFourier":
            coil = CurveXYZFourier(nquadpoints, order)
        elif curvetype == "JaxCurveXYZFourier":
            coil = JaxCurveXYZFourier(nquadpoints, order)
        elif curvetype == "CurveRZFourier":
            coil = CurveRZFourier(nquadpoints, order, 2, False)
        elif curvetype == "CurvePlanarFourier":
            coil = CurvePlanarFourier(nquadpoints, order)
        elif curvetype == "JaxCurvePlanarFourier":
            coil = JaxCurvePlanarFourier(nquadpoints, order)
        elif curvetype == "CurveHelical":
            coil = CurveHelical(nquadpoints, order, 5, 1, 1.0, 0.3)
        else:
            # print('Could not find' + curvetype)
            assert False
        dofs = np.zeros((coil.dof_size,))
        if curvetype in ["CurveXYZFourier", "JaxCurveXYZFourier"]:
            dofs[1] = 1.0
            dofs[2 * order + 3] = 1.0
            dofs[4 * order + 3] = 1.0
        elif curvetype in ["CurveRZFourier"]:
            dofs[0] = 1.0
            dofs[1] = 0.1
            dofs[order + 1] = 0.1
        elif curvetype in ["CurvePlanarFourier", "JaxCurvePlanarFourier"]:
            dofs[0] = 1.0
            dofs[: 2 * order + 1] = 0.1
            dofs[2 * order + 1] = 1.0
            dofs[2 * order + 2] = 0.0
            dofs[2 * order + 3] = 0.0
            dofs[2 * order + 4] = 0.0
        elif curvetype in ["CurveHelical"]:
            dofs[0] = np.pi / 2
        else:
            assert False

        coil.x = dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape)
        if rotated:
            coil = RotatedCurve(coil, 0.5, flip=False)
        return coil

    def _assert_barrier_taylor_progress(self, err, err_new):
        if err > 10.0 * self._BARRIER_TAYLOR_FLOOR:
            assert err_new < 0.6 * err
        else:
            assert err_new <= max(1.05 * err, self._BARRIER_TAYLOR_FLOOR)

    def _curve_collection_min_distance(self, curves):
        min_distance = 1e10
        for i in range(len(curves)):
            for j in range(i):
                pair_distance = np.min(
                    np.linalg.norm(
                        curves[i].gamma()[:, None, :] - curves[j].gamma()[None, :, :],
                        axis=2,
                    )
                )
                min_distance = min(min_distance, pair_distance)
        return min_distance

    def subtest_curve_length_taylor_test(self, curve):
        J = CurveLength(curve)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh - J0) / eps
            err_new = np.linalg.norm(deriv_est - deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_length_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_length_taylor_test(curve)

    def subtest_curve_curvature_taylor_test(self, curve):
        J = LpCurveCurvature(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh - J0) / eps
            err_new = np.linalg.norm(deriv_est - deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_curvature_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_curvature_taylor_test(curve)

    def subtest_curve_curvature_barrier_taylor_test(self, curve):
        max_kappa = float(np.max(curve.kappa()))
        feasible_threshold = 2.0 * max_kappa
        J = LpCurveCurvatureBarrier(curve, feasible_threshold)
        assert np.isfinite(J.J())
        Jviol = LpCurveCurvatureBarrier(curve, 0.5 * max_kappa)
        assert np.isposinf(Jviol.J())

        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-2 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh - J0) / eps
            err_new = np.linalg.norm(deriv_est - deriv)
            self._assert_barrier_taylor_progress(err, err_new)
            err = err_new

    def test_curve_curvature_barrier_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_curvature_barrier_taylor_test(curve)

    def subtest_curve_torsion_taylor_test(self, curve):
        J = LpCurveTorsion(curve, p=2)
        J0 = J.J()
        curve_dofs = curve.x
        h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(10, 20):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jh = J.J()
            deriv_est = (Jh - J0) / eps
            err_new = np.linalg.norm(deriv_est - deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_torsion_taylor_test(self):
        for curvetype in self.curvetypes:
            if "CurvePlanarFourier" not in curvetype:
                for rotated in [True, False]:
                    with self.subTest(curvetype=curvetype, rotated=rotated):
                        curve = self.create_curve(curvetype, rotated)
                        self.subtest_curve_torsion_taylor_test(curve)

    def subtest_curve_minimum_distance_taylor_test(self, curve):
        ncurves = 3
        curve_t = (
            curve.curve.__class__.__name__
            if isinstance(curve, RotatedCurve)
            else curve.__class__.__name__
        )
        curves = [curve] + [
            RotatedCurve(self.create_curve(curve_t, False), 0.1 * i, True)
            for i in range(1, ncurves)
        ]
        distance_threshold = 0.4 if curve_t == "CurveHelical" else 0.2
        J = CurveCurveDistance(curves, distance_threshold)
        mindist = 1e10
        for i in range(len(curves)):
            for j in range(i):
                mindist = min(
                    mindist,
                    np.min(
                        np.linalg.norm(
                            curves[i].gamma()[:, None, :]
                            - curves[j].gamma()[None, :, :],
                            axis=2,
                        )
                    ),
                )
        assert abs(J.shortest_distance() - mindist) < 1e-14
        assert mindist > 1e-10

        for k in range(ncurves):
            curve_dofs = curves[k].x
            h = 1e-3 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
            J0 = J.J()
            dJ = J.dJ(partials=True)(
                curves[k].curve if isinstance(curves[k], RotatedCurve) else curves[k]
            )
            deriv = np.sum(dJ * h)
            assert np.abs(deriv) > 1e-10
            err = 1e6
            for i in range(5, 12):
                eps = 0.5**i
                curves[k].x = curve_dofs + eps * h
                Jh = J.J()
                deriv_est = (Jh - J0) / eps
                err_new = np.linalg.norm(deriv_est - deriv)
                assert err_new < 0.6 * err
                err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_minimum_distance_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_minimum_distance_taylor_test(curve)

    def subtest_curve_minimum_distance_barrier_taylor_test(self, curve):
        ncurves = 3
        curve_t = (
            curve.curve.__class__.__name__
            if isinstance(curve, RotatedCurve)
            else curve.__class__.__name__
        )
        curves = [curve] + [
            RotatedCurve(self.create_curve(curve_t, False), 0.1 * i, True)
            for i in range(1, ncurves)
        ]
        mindist = self._curve_collection_min_distance(curves)
        assert mindist > 1e-6

        feasible_threshold = 0.5 * mindist
        J = CurveCurveDistanceBarrier(curves, feasible_threshold)
        assert np.isfinite(J.J())
        Jviol = CurveCurveDistanceBarrier(curves, mindist + 1e-3)
        assert np.isposinf(Jviol.J())

        for k in range(ncurves):
            curve_dofs = curves[k].x
            h = 1e-4 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
            J0 = J.J()
            dJ = J.dJ(partials=True)(
                curves[k].curve if isinstance(curves[k], RotatedCurve) else curves[k]
            )
            deriv = np.sum(dJ * h)
            assert np.abs(deriv) > 1e-10
            err = 1e6
            for i in range(6, 15):
                eps = 0.5**i
                curves[k].x = curve_dofs + eps * h
                Jh = J.J()
                deriv_est = (Jh - J0) / eps
                err_new = np.linalg.norm(deriv_est - deriv)
                self._assert_barrier_taylor_progress(err, err_new)
                err = err_new

    def test_curve_minimum_distance_barrier_taylor_test(self):
        for curvetype in self.curvetypes:
            if curvetype == "CurveHelical":
                continue
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_minimum_distance_barrier_taylor_test(curve)

    def subtest_curve_arclengthvariation_taylor_test(self, curve, nintervals):
        if isinstance(curve, CurveXYZFourier):
            J = ArclengthVariation(curve, nintervals=nintervals)
        else:
            J = ArclengthVariation(curve, nintervals=2)

        curve_dofs = curve.x
        h = 1e-1 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(2, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jp = J.J()
            curve.x = curve_dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp - Jm) / (2 * eps)
            err_new = np.linalg.norm(deriv_est - deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.3 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_arclengthvariation_taylor_test(self):
        for curvetype in self.curvetypes:
            for nintervals in ["full", "partial", 2]:
                with self.subTest(curvetype=curvetype, nintervals=nintervals):
                    curve = self.create_curve(curvetype, False)
                    self.subtest_curve_arclengthvariation_taylor_test(curve, nintervals)

    def test_arclength_variation_circle(self):
        """For a circle, the arclength variation should be 0."""
        c = CurveXYZFourier(16, 1)
        c.set("xc(1)", 4.0)
        c.set("ys(1)", 4.0)
        for nintervals in ["full", "partial", 2]:
            a = ArclengthVariation(c, nintervals=nintervals)
            assert np.abs(a.J()) < 1.0e-12

    def subtest_curve_meansquaredcurvature_taylor_test(self, curve):
        J = MeanSquaredCurvature(curve)
        curve_dofs = curve.x
        h = 1e-1 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 10):
            eps = 0.5**i
            curve.x = curve_dofs + eps * h
            Jp = J.J()
            curve.x = curve_dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp - Jm) / (2 * eps)
            err_new = np.linalg.norm(deriv_est - deriv)
            # print("err_new %s" % (err_new))
            assert err_new < 0.3 * err
            err = err_new
        J_str = json.dumps(SIMSON(J), cls=GSONEncoder)
        J_regen = json.loads(J_str, cls=GSONDecoder)
        self.assertAlmostEqual(J.J(), J_regen.J())

    def test_curve_meansquaredcurvature_taylor_test(self):
        for curvetype in self.curvetypes:
            for rotated in [True, False]:
                with self.subTest(curvetype=curvetype, rotated=rotated):
                    curve = self.create_curve(curvetype, rotated)
                    self.subtest_curve_meansquaredcurvature_taylor_test(curve)

    def test_minimum_distance_candidates_one_collection(self):
        np.random.seed(0)
        n_clouds = 4
        pointClouds = [
            np.random.uniform(low=-1.0, high=+1.0, size=(5, 3)) for _ in range(n_clouds)
        ]
        true_min_dists = {}
        from scipy.spatial.distance import cdist

        for i in range(n_clouds):
            for j in range(i):
                true_min_dists[(i, j)] = np.min(cdist(pointClouds[i], pointClouds[j]))

        threshold = max(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
            pointClouds, threshold, n_clouds
        )
        assert len(candidates) == len(true_min_dists)

        threshold = min(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
            pointClouds, threshold, n_clouds
        )
        assert len(candidates) == 1

    def test_minimum_distance_candidates_two_collections(self):
        np.random.seed(0)
        n_clouds = 4
        pointCloudsA = [
            np.random.uniform(low=-1.0, high=+1.0, size=(5, 3)) for _ in range(n_clouds)
        ]
        pointCloudsB = [
            np.random.uniform(low=-1.0, high=+1.0, size=(5, 3)) for _ in range(n_clouds)
        ]
        true_min_dists = {}
        from scipy.spatial.distance import cdist

        for i in range(n_clouds):
            for j in range(n_clouds):
                true_min_dists[(i, j)] = np.min(cdist(pointCloudsA[i], pointCloudsB[j]))

        threshold = max(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
            pointCloudsA, pointCloudsB, threshold
        )
        assert len(candidates) == len(true_min_dists)

        threshold = min(true_min_dists.values()) * 1.0001
        candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
            pointCloudsA, pointCloudsB, threshold
        )
        assert len(candidates) == 1

    def test_minimum_distance_candidates_symmetry(self):
        from scipy.spatial.distance import cdist

        base_curves, base_currents, _, _, _ = get_data("ncsx", coil_order=10)
        curves = [
            c.curve for c in coils_via_symmetries(base_curves, base_currents, 3, True)
        ]
        for t in np.linspace(0.05, 0.5, num=10):
            Jnosym = CurveCurveDistance(curves, t)
            Jsym = CurveCurveDistance(curves, t, num_basecurves=3)
            assert (
                abs(
                    Jnosym.shortest_distance_among_candidates()
                    - Jsym.shortest_distance_among_candidates()
                )
                < 1e-15
            )
            print(
                len(Jnosym.candidates),
                len(Jsym.candidates),
                Jnosym.shortest_distance_among_candidates(),
            )
            distsnosym = [
                np.min(cdist(Jnosym.curves[i].gamma(), Jnosym.curves[j].gamma()))
                for i, j in Jnosym.candidates
            ]
            distssym = [
                np.min(cdist(Jsym.curves[i].gamma(), Jsym.curves[j].gamma()))
                for i, j in Jsym.candidates
            ]
            print("distsnosym", distsnosym)
            print("distssym", distssym)
            print((Jnosym.candidates), (Jsym.candidates))

            assert np.allclose(
                np.unique(np.round(distsnosym, 8)), np.unique(np.round(distssym, 8))
            )

    def test_curve_surface_distance(self):
        np.random.seed(0)
        base_curves, base_currents, _, _, _ = get_data("ncsx", coil_order=10)
        curves = [
            c.curve for c in coils_via_symmetries(base_curves, base_currents, 3, True)
        ]
        ntor = 0
        surface = SurfaceRZFourier.from_nphi_ntheta(
            nfp=3, nphi=32, ntheta=32, ntor=ntor
        )
        surface.set(f"rc(0,{ntor})", 1.6)
        surface.set(f"rc(1,{ntor})", 0.2)
        surface.set(f"zs(1,{ntor})", 0.2)

        last_num_candidates = 0
        for t in np.linspace(0.01, 1.0, num=10):
            J = CurveSurfaceDistance(curves, surface, t)
            J.compute_candidates()
            assert len(J.candidates) >= last_num_candidates
            last_num_candidates = len(J.candidates)
            if len(J.candidates) == 0:
                assert J.shortest_distance() > J.shortest_distance_among_candidates()
            else:
                assert J.shortest_distance() == J.shortest_distance_among_candidates()

        assert last_num_candidates == len(curves)
        threshold = 1.0
        J = CurveSurfaceDistance(curves, surface, threshold)

        curve_dofs = J.x
        h = 1e-1 * np.random.rand(len(curve_dofs)).reshape(curve_dofs.shape)
        dJ = J.dJ()
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 12):
            eps = 0.5**i
            J.x = curve_dofs + eps * h
            Jp = J.J()
            J.x = curve_dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp - Jm) / (2 * eps)
            err_new = np.linalg.norm(deriv_est - deriv)
            print("err_new %s" % (err_new))
            print(err_new / err)
            assert err_new < 0.3 * err
            err = err_new

    def test_linking_number(self):
        for downsample in [1, 2, 5]:
            curves1 = create_equally_spaced_curves(
                2, 1, stellsym=True, R0=1, R1=0.5, order=5, numquadpoints=120
            )
            curve1 = CurveXYZFourier(200, 3)
            coeffs = curve1.dofs_matrix
            coeffs[1][0] = 1.0
            coeffs[1][1] = 0.5
            coeffs[2][2] = 0.5
            curve1.set_dofs(np.concatenate(coeffs))

            curve2 = CurveXYZFourier(150, 3)
            coeffs = curve2.dofs_matrix
            coeffs[1][0] = 0.5
            coeffs[1][1] = 0.5
            coeffs[0][0] = 0.1
            coeffs[0][1] = 0.5
            coeffs[0][2] = 0.5
            curve2.set_dofs(np.concatenate(coeffs))
            curves2 = [curve1, curve2]
            curves3 = [curve2, curve1]
            objective1 = LinkingNumber(curves1, downsample)
            objective2 = LinkingNumber(curves2, downsample)
            objective3 = LinkingNumber(curves3, downsample)

            print(
                "Linking number testing (should be 0, 1, 1):",
                objective1.J(),
                objective2.J(),
                objective3.J(),
            )
            np.testing.assert_allclose(objective1.J(), 0, atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(objective2.J(), 1, atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(objective3.J(), 1, atol=1e-14, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
