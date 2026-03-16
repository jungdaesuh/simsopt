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


if __name__ == "__main__":
    unittest.main()
