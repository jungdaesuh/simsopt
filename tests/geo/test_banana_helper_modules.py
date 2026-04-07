import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
REFERENCE_SURFACES_PATH = EXAMPLES_ROOT / "banana_opt" / "reference_surfaces.py"
STAGE2_GEOMETRY_PATH = EXAMPLES_ROOT / "banana_opt" / "stage2_geometry.py"


def _load_module(module_path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(EXAMPLES_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[0]
    return module


class _FakeSurfaceRZFourier:
    def __init__(self, nfp, stellsym):
        self.nfp = nfp
        self.stellsym = stellsym
        self.rc = {}
        self.zs = {}

    def set_rc(self, m, n, value):
        self.rc[(m, n)] = float(value)

    def set_zs(self, m, n, value):
        self.zs[(m, n)] = float(value)


class BananaReferenceSurfaceTests(unittest.TestCase):
    def test_build_banana_reference_surfaces_sets_expected_modes(self):
        module = _load_module(REFERENCE_SURFACES_PATH, "banana_reference_surfaces")

        with mock.patch.object(module, "SurfaceRZFourier", _FakeSurfaceRZFourier):
            surfaces = module.build_banana_reference_surfaces(
                nfp=5,
                banana_surf_radius=0.23,
            )

        self.assertEqual(surfaces.vessel.nfp, 5)
        self.assertTrue(surfaces.vessel.stellsym)
        self.assertEqual(surfaces.vessel.rc, {(0, 0): 0.976, (1, 0): 0.222})
        self.assertEqual(surfaces.vessel.zs, {(1, 0): 0.222})

        self.assertEqual(surfaces.hbt.rc, {(0, 0): 0.9115, (1, 0): 0.1605})
        self.assertEqual(surfaces.hbt.zs, {(1, 0): 0.152})

        self.assertEqual(
            surfaces.coil_winding_surface.rc,
            {(0, 0): 0.976, (1, 0): 0.23},
        )
        self.assertEqual(
            surfaces.coil_winding_surface.zs,
            {(1, 0): 0.23},
        )


class Stage2GeometryHelperTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module(STAGE2_GEOMETRY_PATH, "banana_stage2_geometry")

    def test_check_all_pairs_skips_adjacent_segments(self):
        segments = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        )

        self.assertFalse(
            self.module.check_all_pairs(segments, tol=1e-6, neighbor_skip=1)
        )

    def test_check_all_pairs_detects_non_neighbor_intersection(self):
        segments = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, -0.2, 0.0], [0.5, 0.2, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ],
            dtype=float,
        )

        self.assertTrue(
            self.module.check_all_pairs(segments, tol=1e-3, neighbor_skip=1)
        )

    def test_is_self_intersecting_detects_crossing_polygon(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        with mock.patch.object(self.module, "gamma_at_t", return_value=points):
            self.assertTrue(
                self.module.is_self_intersecting(
                    curve=object(),
                    npts=4,
                    tol_factor=0.01,
                    neighbor_skip=1,
                )
            )

    def test_is_self_intersecting_rejects_simple_loop(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        with mock.patch.object(self.module, "gamma_at_t", return_value=points):
            self.assertFalse(
                self.module.is_self_intersecting(
                    curve=object(),
                    npts=4,
                    tol_factor=0.01,
                    neighbor_skip=1,
                )
            )
