from functools import partial
import unittest
from simsopt.geo import FrameRotation, ZeroRotation, FramedCurveCentroid, FramedCurveFrenet
from simsopt.configs.zoo import get_data
from simsopt.geo.strain_optimization import LPBinormalCurvatureStrainPenalty, LPTorsionalStrainPenalty
import numpy as np
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from scipy.optimize import minimize


def _make_reference_curve():
    base_curves, _, _, _, _ = get_data(
        "ncsx",
        coil_order=6,
        points_per_period=120,
    )
    return base_curves[0]


def _make_rotation(curve, order, *, allow_none):
    if order == 1:
        rotation = FrameRotation(curve.quadpoints, order)
        rotation.x = np.array([0, 0.1, 0.3])
        rotation_shared = FrameRotation(curve.quadpoints, order, dofs=rotation.dofs)
        assert np.allclose(rotation.x, rotation_shared.x)
        assert np.allclose(
            rotation.alpha(curve.quadpoints),
            rotation_shared.alpha(curve.quadpoints),
        )
        return rotation
    if allow_none:
        return None
    return ZeroRotation(curve.quadpoints)


def _make_framedcurve(order, centroid, *, allow_none):
    curve = _make_reference_curve()
    rotation = _make_rotation(curve, order, allow_none=allow_none)
    if centroid:
        return FramedCurveCentroid(curve, rotation)
    return FramedCurveFrenet(curve, rotation)


def _make_strain_objective(
    objective_cls,
    order,
    centroid,
    *,
    allow_none,
    threshold,
):
    return objective_cls(
        _make_framedcurve(order, centroid, allow_none=allow_none),
        width=1e-3,
        p=2,
        threshold=threshold,
    )


def _make_binormal_objective(order, centroid):
    return _make_strain_objective(
        LPBinormalCurvatureStrainPenalty,
        order,
        centroid,
        allow_none=True,
        threshold=1e-4,
    )


def _make_torsion_objective(order, centroid):
    return _make_strain_objective(
        LPTorsionalStrainPenalty,
        order,
        centroid,
        allow_none=False,
        threshold=1e-8,
    )


def _evaluate_objective(objective_builder, dofs):
    objective = objective_builder()
    objective.x = dofs
    return objective.J()


def _assert_central_difference_contraction(objective_builder):
    objective = objective_builder()
    dofs = np.asarray(objective.x, dtype=float).copy()

    rng = np.random.default_rng(1)
    h = rng.standard_normal(size=dofs.shape)
    df = np.sum(np.asarray(objective.dJ(), dtype=float) * h)

    errf_old = 1e10
    for i in range(9, 14):
        eps = 0.5**i
        f1 = _evaluate_objective(objective_builder, dofs + eps * h)
        f2 = _evaluate_objective(objective_builder, dofs - eps * h)
        errf = np.abs((f1 - f2) / (2 * eps) - df)
        assert errf < 0.3 * errf_old
        errf_old = errf


class CoilStrainTesting(unittest.TestCase):

    def test_strain_opt(self):
        """ 
        Check that for a circular coil, strains 
        can be optimized to vanish using rotation 
        dofs. 
        """
        for centroid in [True, False]:
            quadpoints = np.linspace(0, 1, 10, endpoint=False)
            curve = CurveXYZFourier(quadpoints, order=1)
            curve.set('xc(1)', 1e-4)
            curve.set('ys(1)', 1e-4)
            curve.fix_all()
            order = 2
            rng = np.random.default_rng(1)
            rotation = FrameRotation(quadpoints, order)
            rotation.x = rng.standard_normal(size=(2*order+1,))
            framedcurve_cls = FramedCurveCentroid if centroid else FramedCurveFrenet
            framedcurve = framedcurve_cls(curve, rotation)
            Jt = LPTorsionalStrainPenalty(framedcurve, width=1e-3, p=2, threshold=0)
            Jb = LPBinormalCurvatureStrainPenalty(framedcurve, width=1e-3, p=2, threshold=0)
            J = Jt+Jb

            def fun(dofs):
                J.x = dofs
                grad = J.dJ()
                return J.J(), grad
            minimize(fun, J.x, jac=True, method='L-BFGS-B',
                     options={'maxiter': 100, 'maxcor': 10, 'gtol': 1e-20, 'ftol': 1e-20}, tol=1e-20)
            assert Jt.J() < 1e-12
            assert Jb.J() < 1e-12

    def test_torsion(self):
        for centroid in [True, False]:
            for order in [None, 1]:
                with self.subTest(order=order):
                    self.subtest_torsion(order, centroid)

    def test_binormal_curvature(self):
        for centroid in [True, False]:
            for order in [None, 1]:
                with self.subTest(order=order):
                    self.subtest_binormal_curvature(order, centroid)

    def subtest_binormal_curvature(self, order, centroid):
        assert order in [1, None]
        build_objective = partial(_make_binormal_objective, order, centroid)

        if not centroid and order is None:
            # Binormal curvature vanishes in Frenet frame
            assert build_objective().J() < 1e-12
            return

        _assert_central_difference_contraction(build_objective)

    def subtest_torsion(self, order, centroid):
        assert order in [1, None]
        _assert_central_difference_contraction(
            partial(_make_torsion_objective, order, centroid)
        )
