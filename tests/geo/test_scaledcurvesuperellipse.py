import unittest
import numpy as np

from simsopt.geo import (
    CurveLength,
    CurvePlanarOnSurface,
    CurveSuperEllipse,
    LpCurveCurvature,
    MeanSquaredCurvature,
    ScaledCurveSuperEllipse,
    SurfaceRZFourier,
)

from .test_curve import taylor_test


class SuperEllipseTesting(unittest.TestCase):

    def test_shared_dofs(self):
        # a and n live on the parent, b on each instance.
        shared = CurveSuperEllipse(32, 0.030, 0.075, 4.0)
        rows = [ScaledCurveSuperEllipse(shared, shared_dofs=("a", "n"), b=b) for b in (0.075, 0.068, 0.060)]
        self.assertEqual(list(shared.local_dof_names), ['a', 'n'])
        self.assertEqual(list(rows[0].local_dof_names), ['b'])
        names = set()
        for row in rows:
            names.update(row.dof_names)
        self.assertEqual(len(names), 5)     # a, n, and one b per row

        gammas = [row.gamma().copy() for row in rows]
        shared.set('n', 5.0)
        for row, gamma in zip(rows, gammas):
            self.assertGreater(np.abs(row.gamma() - gamma).max(), 1e-6)

        gammas = [row.gamma().copy() for row in rows]
        rows[1].set('b', 0.080)
        moved = [np.abs(r.gamma() - g).max() > 1e-6 for r, g in zip(rows, gammas)]
        self.assertEqual(moved, [False, True, False])

    def test_matches_curvesuperellipse(self):
        shared = CurveSuperEllipse(32, 0.030, 0.075, 4.0)
        row = ScaledCurveSuperEllipse(shared, shared_dofs=("a", "n"), b=0.068)
        reference = CurveSuperEllipse(32, 0.030, 0.068, 4.0)
        np.testing.assert_allclose(row.gamma(), reference.gamma())

    def test_derivative_reaches_both_owners(self):
        # The vjp split is the part that fails silently if it is wrong, so
        # check every objective that routes through a different vjp.
        for make_J in [CurveLength,
                       lambda c: LpCurveCurvature(c, 2),
                       MeanSquaredCurvature]:
            with self.subTest(objective=make_J):
                shared = CurveSuperEllipse(32, 0.030, 0.075, 4.0)
                row = ScaledCurveSuperEllipse(shared, shared_dofs=("a", "n"), b=0.068)
                J = make_J(row)
                self.assertEqual(len(J.dof_names), 3)
                self.assertTrue(np.all(np.abs(J.dJ()) > 1e-9))

                def f(dofs):
                    J.x = dofs
                    return J.J()

                def df(dofs):
                    J.x = dofs
                    return J.dJ()
                taylor_test(f, df, J.x.copy())

    def test_survives_placement_on_a_surface(self):
        # CurvePlanarOnSurface must fix this class's placement without fixing
        # the shape dofs it inherits from its parent.
        surface = SurfaceRZFourier(
            mpol=1, ntor=0, nfp=1, stellsym=True,
            quadpoints_phi=(np.arange(8) + 0.5)/8,
            quadpoints_theta=(np.arange(6) + 0.5)/6)
        surface.set_rc(0, 0, 0.92)
        surface.set_rc(1, 0, 0.10)
        surface.set_zs(1, 0, 0.15)
        shared = CurveSuperEllipse(32, 0.030, 0.075, 4.0)
        row = ScaledCurveSuperEllipse(shared, shared_dofs=("a", "n"), b=0.068)
        placed = CurvePlanarOnSurface(row, surface, 0, 2)
        self.assertEqual(list(shared.local_dof_names), ['a', 'n'])
        self.assertEqual(list(row.local_dof_names), ['b'])
        self.assertEqual(len(placed.dof_names), 3 + surface.dof_size)


if __name__ == "__main__":
    unittest.main()
