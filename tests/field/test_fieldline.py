import unittest
import logging
import numpy as np
import simsoptpp as sopp

from simsopt.field.magneticfieldclasses import ToroidalField, PoloidalField, InterpolatedField, UniformInterpolationRule
from simsopt.field.tracing import compute_fieldlines, particles_to_vtk, plot_poincare_data, \
    LevelsetStoppingCriterion, MinRStoppingCriterion, MinZStoppingCriterion, MaxRStoppingCriterion, MaxZStoppingCriterion
from simsopt.field.biotsavart import BiotSavart
from simsopt.configs.zoo import get_ncsx_data
from simsopt.field.coil import coils_via_symmetries, Coil, Current
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curvexyzfourier import CurveXYZFourier

logging.basicConfig()

try:
    import pyevtk
except ImportError:
    pyevtk = None


def validate_phi_hits(phi_hits, nphis):
    """
    Assert that we are hitting the phi planes in the correct order.
    For the toroidal field, we should always keep increasing in phi.
    """
    for i in range(len(phi_hits)-1):
        this_idx = int(phi_hits[i][1])
        next_idx = int(phi_hits[i+1][1])
        if not next_idx == (this_idx + 1) % nphis:
            return False
    return True


def _build_levelset(phi_points):
    return sopp.RegularGridInterpolant3D(
        sopp.UniformInterpolationRule(1),
        (0.8, 1.2, 32),
        (0.0, 2 * np.pi, phi_points),
        (-0.1, 0.1, 8),
        1,
        True,
    )


def _signed_wedge(phi_min, phi_max):
    def evaluator(r, phi, z, flatten=True):
        r = np.asarray(r)
        phi = np.mod(np.asarray(phi), 2 * np.pi)
        z = np.asarray(z)
        values = np.ones((r.size, 1))
        mask = (
            (np.abs(r - 1.0) <= 0.05)
            & (np.abs(z) <= 0.02)
            & (phi >= phi_min)
            & (phi <= phi_max)
        )
        values[mask, 0] = -1.0
        if flatten:
            return np.ascontiguousarray(values).flatten()
        return values

    return evaluator


def _trace_single_fieldline(field, *, tmax=2.0, tol=1e-2, stopping_criteria=None):
    return compute_fieldlines(
        field,
        [1.0],
        [0.0],
        tmax=tmax,
        tol=tol,
        phis=[],
        stopping_criteria=[] if stopping_criteria is None else stopping_criteria,
    )


class FieldlineTesting(unittest.TestCase):

    def test_poincare_toroidal(self):
        logger = logging.getLogger('simsopt.field.tracing')
        logger.setLevel(1)
        # Test a toroidal magnetic field with no rotational transform
        R0test = 1.3
        B0test = 0.8
        Bfield = ToroidalField(R0test, B0test)
        nlines = 10
        R0 = [1.1 + i*0.1 for i in range(nlines)]
        Z0 = [0 for i in range(nlines)]
        nphis = 10
        phis = np.linspace(0, 2*np.pi, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            Bfield, R0, Z0, tmax=100, phis=phis, stopping_criteria=[])
        for i in range(nlines):
            assert np.allclose(res_tys[i][:, 3], 0.)
            assert np.allclose(np.linalg.norm(res_tys[i][:, 1:3], axis=1), R0[i])
            assert validate_phi_hits(res_phi_hits[i], nphis)
        if pyevtk is not None:
            particles_to_vtk(res_tys, '/tmp/fieldlines')

    def test_poincare_tokamak(self):
        # Test a simple circular tokamak geometry that
        # consists of a superposition of a purely toroidal
        # and a purely poloidal magnetic field
        R0test = 1.0
        B0test = 1.0
        qtest = 3.2
        Bfield = ToroidalField(R0test, B0test)+PoloidalField(R0test, B0test, qtest)
        nlines = 4
        R0 = [1.05 + i*0.02 for i in range(nlines)]
        Z0 = [0 for i in range(nlines)]
        nphis = 4
        phis = np.linspace(0, 2*np.pi, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            Bfield, R0, Z0, tmax=10, phis=phis, stopping_criteria=[])
        # Check that Poincare plot is a circle in the R,Z plane with R centered at R0
        rtest = [[np.sqrt((np.sqrt(res_tys[i][j][1]**2+res_tys[i][j][2]**2)-R0test)**2+res_tys[i][j][3]**2)-R0[i]+R0test for j in range(len(res_tys[i]))] for i in range(len(res_tys))]
        assert [np.allclose(rtest[i], 0., rtol=1e-5, atol=1e-5) for i in range(nlines)]

    def test_levelset_stopping_detects_within_step_surface_exit(self):
        field = ToroidalField(1.0, 1.0)
        levelset = _build_levelset(256)
        levelset.interpolate_batch(_signed_wedge(1.0, 1.2))
        res_tys, res_phi_hits = _trace_single_fieldline(
            field,
            stopping_criteria=[LevelsetStoppingCriterion(levelset)],
        )

        assert len(res_phi_hits[0]) == 1
        assert res_phi_hits[0][0, 1] == -1
        assert 0.9 < res_phi_hits[0][0, 0] < 1.3
        assert res_phi_hits[0][0, 2] > 0.0
        assert len(res_tys[0]) > 0

    def test_levelset_stopping_detects_subsample_width_surface_exit(self):
        field = ToroidalField(1.0, 1.0)
        levelset = _build_levelset(512)
        levelset.interpolate_batch(_signed_wedge(1.135, 1.150))
        _, res_phi_hits = _trace_single_fieldline(
            field,
            stopping_criteria=[LevelsetStoppingCriterion(levelset)],
        )

        assert len(res_phi_hits[0]) == 1
        assert res_phi_hits[0][0, 1] == -1
        assert 1.05 < res_phi_hits[0][0, 0] < 1.16

    def test_levelset_stopping_detects_leave_and_reenter_within_single_step(self):
        field = ToroidalField(1.0, 1.0)
        baseline_tys, _ = _trace_single_fieldline(field)
        final_step_start = baseline_tys[0][-2, 0]
        final_step_end = baseline_tys[0][-1, 0]
        target_center = final_step_start + 0.30 * (final_step_end - final_step_start)
        half_width = 8.0e-4
        lower_phi = target_center - half_width
        upper_phi = target_center + half_width

        levelset = _build_levelset(8192)
        levelset.interpolate_batch(_signed_wedge(lower_phi, upper_phi))
        _, res_phi_hits = _trace_single_fieldline(
            field,
            stopping_criteria=[LevelsetStoppingCriterion(levelset)],
        )

        assert final_step_start < lower_phi < upper_phi < final_step_end
        assert len(res_phi_hits[0]) == 1
        assert res_phi_hits[0][0, 1] == -1
        assert final_step_start < res_phi_hits[0][0, 0] < final_step_end
        hit_x = res_phi_hits[0][0, 2]
        hit_y = res_phi_hits[0][0, 3]
        hit_z = res_phi_hits[0][0, 4]
        hit_r = np.sqrt(hit_x**2 + hit_y**2)
        hit_phi = np.mod(np.arctan2(hit_y, hit_x), 2 * np.pi)
        assert (lower_phi - 1.5e-3) <= hit_phi <= (upper_phi + 1.5e-3)
        assert levelset.evaluate(hit_r, hit_phi, hit_z)[0] < 0.0
        assert abs(hit_r - 1.0) <= 0.05
        assert abs(hit_z) <= 0.02

    def test_levelset_stopping_refines_to_interpolant_resolution(self):
        field = ToroidalField(1.0, 1.0)
        baseline_tys, _ = _trace_single_fieldline(field)
        final_step_start = baseline_tys[0][-2, 0]
        final_step_end = baseline_tys[0][-1, 0]
        target_fraction = 2561.0 / 8192.0
        target_phi = final_step_start + target_fraction * (final_step_end - final_step_start)

        levelset = _build_levelset(8192)

        def signed_wedge(r, phi, z, flatten=True):
            r = np.asarray(r)
            phi = np.mod(np.asarray(phi), 2 * np.pi)
            z = np.asarray(z)
            values = np.ones((r.size, 1))
            mask = (
                (np.abs(r - 1.0) <= 0.05)
                & (np.abs(z) <= 0.02)
                & (phi >= target_phi - 9.0e-4)
                & (phi <= target_phi + 9.0e-4)
            )
            values[mask, 0] = -1.0
            if flatten:
                return np.ascontiguousarray(values).flatten()
            return values

        levelset.interpolate_batch(signed_wedge)

        _, res_phi_hits = _trace_single_fieldline(
            field,
            stopping_criteria=[LevelsetStoppingCriterion(levelset)],
        )

        assert len(res_phi_hits[0]) == 1
        assert res_phi_hits[0][0, 1] == -1
        assert abs(res_phi_hits[0][0, 0] - target_phi) < 0.02

    def test_poincare_plot(self):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        n = 10
        rrange = (1.0, 1.9, n)
        phirange = (0, 2*np.pi/nfp, n*2)
        zrange = (0, 0.4, n)
        bsh = InterpolatedField(
            bs, UniformInterpolationRule(2),
            rrange, phirange, zrange, True, nfp=3, stellsym=True
        )
        nlines = 4
        r0 = np.linalg.norm(ma.gamma()[0, :2])
        z0 = ma.gamma()[0, 2]
        R0 = [r0 + i*0.01 for i in range(nlines)]
        Z0 = [z0 for i in range(nlines)]
        nphis = 4
        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        try:
            import matplotlib  # noqa
            plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')
        except ImportError:
            pass

    def test_poincare_ncsx_known(self):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        R0 = [np.linalg.norm(ma.gamma()[0, :2])]
        Z0 = [ma.gamma()[0, 2]]
        phis = np.arctan2(ma.gamma()[:, 1], ma.gamma()[:, 0])
        res_tys, res_phi_hits = compute_fieldlines(
            bs, R0, Z0, tmax=10, phis=phis, stopping_criteria=[])
        for i in range(len(phis)-1):
            assert np.linalg.norm(ma.gamma()[i+1, :] - res_phi_hits[0][i, 2:5]) < 1e-4

    def test_poincare_caryhanson(self):
        # Test with a known magnetic field - optimized Cary & Hanson configuration
        # with a magnetic axis at R=0.9413. Field created using the Biot-Savart
        # solver given a set of two helical coils created using the CurveHelical
        # class. The total magnetic field is a superposition of a helical and
        # a toroidal magnetic field.
        curves = [CurveHelical(200, 1, 5, 2, 1., 0.3) for i in range(2)]
        curves[0].x = [np.pi/ 2, 0.2841, 0]
        curves[1].x = [0, 0, 0.2933]
        currents = [3.07e5, -3.07e5]
        Btoroidal = ToroidalField(1.0, 1.0)
        Bhelical = BiotSavart([
            Coil(curves[0], Current(currents[0])),
            Coil(curves[1], Current(currents[1]))])
        bs = Bhelical + Btoroidal
        ma = CurveXYZFourier(300, 1)
        magnetic_axis_radius = 0.9413
        ma.set_dofs([0, 0, magnetic_axis_radius, 0, magnetic_axis_radius, 0, 0, 0, 0])
        R0 = [np.linalg.norm(ma.gamma()[0, :2])]
        Z0 = [ma.gamma()[0, 2]]
        phis = np.arctan2(ma.gamma()[:, 1], ma.gamma()[:, 0])
        res_tys, res_phi_hits = compute_fieldlines(
            bs, R0, Z0, tmax=2, phis=phis)
        for i in range(len(res_phi_hits[0])):
            assert np.linalg.norm(ma.gamma()[i+1, :] - res_phi_hits[0][i, 2:5]) < 2e-3

        # Text StoppingCriterion in R and Z
        # For each case, check that stopping criterion was met.
        # Check that R/Z is less than/greater than the maximum/minimum value.
        Rmax = 1
        res_tys, res_phi_hits = compute_fieldlines(
            bs, [Rmax-0.02], [1], tmax=2000, stopping_criteria=[MaxRStoppingCriterion(Rmax)])
        assert res_phi_hits[0][0, 1] == -1
        assert np.all(np.sqrt(res_tys[0][:, 1]**2 + res_tys[0][:, 2]**2) < Rmax)

        Rmin = 0.3
        res_tys, res_phi_hits = compute_fieldlines(
            bs, [Rmin+0.02], [0.3], tmax=500, stopping_criteria=[MinRStoppingCriterion(Rmin)])
        assert res_phi_hits[0][0, 1] == -1
        assert np.all(np.sqrt(res_tys[0][:, 1]**2 + res_tys[0][:, 2]**2) > Rmin)

        Zmin = -0.1
        res_tys, res_phi_hits = compute_fieldlines(
            bs, [0.97], [Zmin+0.02], tmax=2000, stopping_criteria=[MinZStoppingCriterion(Zmin)]
        )
        assert res_phi_hits[0][0, 1] == -1
        assert np.all(res_tys[0][:, 3] > Zmin)

        Zmax = 0.5
        res_tys, res_phi_hits = compute_fieldlines(
            bs, [0.5], [Zmax-0.02], tmax=2000, stopping_criteria=[MaxZStoppingCriterion(Zmax)]
        )
        assert res_phi_hits[0][0, 1] == -1
        assert np.all(res_tys[0][:, 3] < Zmax)
