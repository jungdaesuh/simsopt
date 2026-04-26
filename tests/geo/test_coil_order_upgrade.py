import importlib.util
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simsopt._core.optimizable import load
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import CurveCWSFourierCPP, CurveXYZFourier, SurfaceRZFourier


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
COIL_ORDER_UPGRADE_PATH = EXAMPLES_ROOT / "banana_opt" / "coil_order_upgrade.py"
COIL_GROUPS_PATH = EXAMPLES_ROOT / "banana_opt" / "coil_groups.py"
SINGLE_STAGE_MODULE_PATH = (
    EXAMPLES_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"
)
STAGE2_MODULE_PATH = EXAMPLES_ROOT / "STAGE_2" / "banana_coil_solver.py"


def _load_module(module_path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    sys.path.insert(0, str(EXAMPLES_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[0]
    return module


def _make_circle_curve(*, center, radius, normal):
    curve = CurveXYZFourier(96, 1)
    center_x, center_y, center_z = center
    if normal == "z":
        curve.set_dofs(
            [
                center_x,
                radius,
                0.0,
                center_y,
                radius,
                0.0,
                center_z,
                0.0,
                0.0,
            ]
        )
    elif normal == "x":
        curve.set_dofs(
            [
                center_x,
                0.0,
                0.0,
                center_y,
                radius,
                0.0,
                center_z,
                0.0,
                radius,
            ]
        )
    else:
        raise ValueError(f"Unsupported normal {normal!r}.")
    curve.fix_all()
    return curve


def _build_cws_seed(seed_dir: Path):
    coil_groups = _load_module(COIL_GROUPS_PATH, "coil_groups")
    quadpoints = np.linspace(0.0, 1.0, 32, endpoint=False)
    seed_surface = SurfaceRZFourier(nfp=2, stellsym=True)
    seed_surface.set_rc(0, 0, 0.976)
    seed_surface.set_rc(1, 0, 0.21)
    seed_surface.set_zs(1, 0, 0.21)

    banana_curve = CurveCWSFourierCPP(
        quadpoints,
        order=2,
        surf=seed_surface,
        G=3,
        H=5,
    )
    banana_curve.set("phic(0)", 0.23)
    banana_curve.set("thetac(0)", 0.41)
    banana_curve.set("phic(1)", 0.08)
    banana_curve.set("phis(1)", -0.03)
    banana_curve.set("thetac(1)", -0.06)
    banana_curve.set("thetas(1)", 0.09)
    banana_curve.fix("phic(1)")
    base_banana_current = Current(1.1e4)
    base_banana_current.fix_all()
    banana_coils = coils_via_symmetries(
        [banana_curve],
        [base_banana_current],
        seed_surface.nfp,
        seed_surface.stellsym,
    )

    tf_coils = [
        Coil(
            _make_circle_curve(
                center=(0.86 + 0.03 * index, 0.0, 0.0),
                radius=0.12,
                normal="z",
            ),
            Current(-8.0e4),
        )
        for index in range(2)
    ]
    proxy_coils = [
        Coil(
            _make_circle_curve(center=(0.72, 0.0, 0.0), radius=0.04, normal="z"),
            Current(9.0e3),
        )
    ]
    vf_coils = [
        Coil(
            _make_circle_curve(center=(1.18, 0.0, 0.0), radius=0.18, normal="x"),
            Current(-5.0e2),
        )
    ]
    for coil in [*tf_coils, *proxy_coils, *vf_coils]:
        coil.current.fix_all()

    all_coils = [*tf_coils, *banana_coils, *proxy_coils, *vf_coils]
    bs = BiotSavart(all_coils)
    points = np.array(
        [
            [0.25, 0.10, -0.15],
            [0.35, -0.05, 0.20],
            [0.55, 0.15, 0.05],
            [0.70, -0.10, -0.25],
        ],
        dtype=float,
    )
    bs.set_points(points)
    expected_field = bs.B().copy()
    seed_bs_path = seed_dir / "biot_savart_opt.json"
    bs.save(str(seed_bs_path))

    manifest = coil_groups.build_contiguous_manifest(
        num_tf_coils=len(tf_coils),
        num_banana_coils=len(banana_coils),
        num_proxy_coils=len(proxy_coils),
        num_vf_coils=len(vf_coils),
    )
    stage2_results = {
        "NUM_TF_COILS": len(tf_coils),
        "NUM_BANANA_COILS": len(banana_coils),
        "NUM_PROXY_COILS": len(proxy_coils),
        "NUM_VF_COILS": len(vf_coils),
        "COIL_GROUPS": manifest.to_json_payload(),
        "FINITE_CURRENT_MODE": "wataru_proxy_field",
        "order": 2,
    }
    return {
        "seed_bs_path": seed_bs_path,
        "stage2_results": stage2_results,
        "seed_surface": seed_surface,
        "banana_curve": banana_curve,
        "banana_coils": banana_coils,
        "tf_coils": tf_coils,
        "proxy_coils": proxy_coils,
        "vf_coils": vf_coils,
        "expected_field": expected_field,
    }


class _FakeLoadSurface:
    def __init__(self):
        self._gamma = np.zeros((2, 2, 3), dtype=float)

    def gamma(self):
        return self._gamma.copy()

    def unitnormal(self):
        normals = np.zeros_like(self._gamma)
        normals[..., 2] = 1.0
        return normals

    def to_vtk(self, *_args, **_kwargs):
        return None


class CoilOrderUpgradeTests(unittest.TestCase):
    def test_upgrade_cws_order_preserves_existing_modes_and_free_mask(self):
        module = _load_module(COIL_ORDER_UPGRADE_PATH, "coil_order_upgrade")

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = _build_cws_seed(Path(tmpdir))
            upgraded_curve = module.upgrade_cws_order(seed["banana_curve"], 4)

        self.assertEqual(upgraded_curve.order, 4)
        self.assertEqual(upgraded_curve.G, seed["banana_curve"].G)
        self.assertEqual(upgraded_curve.H, seed["banana_curve"].H)
        np.testing.assert_array_equal(
            upgraded_curve.modes[0][: seed["banana_curve"].modes[0].size],
            seed["banana_curve"].modes[0],
        )
        np.testing.assert_array_equal(
            upgraded_curve.modes[1][: seed["banana_curve"].modes[1].size],
            seed["banana_curve"].modes[1],
        )
        np.testing.assert_array_equal(
            upgraded_curve.modes[2][: seed["banana_curve"].modes[2].size],
            seed["banana_curve"].modes[2],
        )
        np.testing.assert_array_equal(
            upgraded_curve.modes[3][: seed["banana_curve"].modes[3].size],
            seed["banana_curve"].modes[3],
        )
        np.testing.assert_array_equal(
            upgraded_curve.modes[0][seed["banana_curve"].modes[0].size :],
            np.zeros(len(upgraded_curve.modes[0]) - seed["banana_curve"].modes[0].size),
        )
        np.testing.assert_array_equal(
            upgraded_curve.modes[1][seed["banana_curve"].modes[1].size :],
            np.zeros(len(upgraded_curve.modes[1]) - seed["banana_curve"].modes[1].size),
        )
        np.testing.assert_array_equal(
            upgraded_curve.modes[2][seed["banana_curve"].modes[2].size :],
            np.zeros(len(upgraded_curve.modes[2]) - seed["banana_curve"].modes[2].size),
        )
        np.testing.assert_array_equal(
            upgraded_curve.modes[3][seed["banana_curve"].modes[3].size :],
            np.zeros(len(upgraded_curve.modes[3]) - seed["banana_curve"].modes[3].size),
        )
        fixed_index = next(
            index
            for index, dof_name in enumerate(upgraded_curve.full_dof_names)
            if dof_name.endswith("phic(1)")
        )
        new_mode_index = next(
            index
            for index, dof_name in enumerate(upgraded_curve.full_dof_names)
            if dof_name.endswith("phic(4)")
        )
        self.assertFalse(
            upgraded_curve.dofs.free_status[fixed_index]
        )
        self.assertTrue(upgraded_curve.dofs.free_status[new_mode_index])

    def test_upgrade_cws_order_preserves_fix_status_by_name_across_blocks(self):
        # Guards against the positional-copy regression: because DOFs are laid
        # out as [phic, phis, thetac, thetas] with block sizes
        # [O+1, O, O+1, O], growing the order shifts the starting index of
        # every block after phic. Fix status must follow the DOF *name*, not
        # its flat-vector index.
        module = _load_module(COIL_ORDER_UPGRADE_PATH, "coil_order_upgrade")

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = _build_cws_seed(Path(tmpdir))
            source_curve = seed["banana_curve"]
            source_curve.fix("phis(2)")
            source_curve.fix("thetac(2)")
            source_curve.fix("thetas(1)")
            upgraded_curve = module.upgrade_cws_order(source_curve, 4)

        for name in ("phic(1)", "phis(2)", "thetac(2)", "thetas(1)"):
            self.assertFalse(
                upgraded_curve.dofs.is_free(name),
                msg=f"{name} should remain fixed after upgrade",
            )
        for name in (
            "phic(0)", "phic(2)", "phic(3)", "phic(4)",
            "phis(1)", "phis(3)", "phis(4)",
            "thetac(0)", "thetac(1)", "thetac(3)", "thetac(4)",
            "thetas(2)", "thetas(3)", "thetas(4)",
        ):
            self.assertTrue(
                upgraded_curve.dofs.is_free(name),
                msg=f"{name} should be free after upgrade",
            )

    def test_upgrade_cws_order_is_idempotent_when_order_matches(self):
        module = _load_module(COIL_ORDER_UPGRADE_PATH, "coil_order_upgrade")

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = _build_cws_seed(Path(tmpdir))
            source_curve = seed["banana_curve"]
            source_curve.fix("thetas(1)")
            upgraded_curve = module.upgrade_cws_order(
                source_curve, int(source_curve.order)
            )

        self.assertEqual(upgraded_curve.order, int(source_curve.order))
        np.testing.assert_array_equal(
            upgraded_curve.get_dofs(), source_curve.get_dofs()
        )
        np.testing.assert_array_equal(
            upgraded_curve.dofs.free_status,
            source_curve.dofs.free_status,
        )
        # The source curve must not have been mutated.
        self.assertIsNot(upgraded_curve, source_curve)
        self.assertIsNot(upgraded_curve.modes, source_curve.modes)

    def test_upgrade_cws_order_warns_when_truncating_modes(self):
        module = _load_module(COIL_ORDER_UPGRADE_PATH, "coil_order_upgrade")

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = _build_cws_seed(Path(tmpdir))
            with self.assertWarnsRegex(RuntimeWarning, "Truncating CurveCWSFourierCPP"):
                downgraded_curve = module.upgrade_cws_order(seed["banana_curve"], 1)

        self.assertEqual(downgraded_curve.order, 1)
        np.testing.assert_array_equal(
            downgraded_curve.modes[0],
            seed["banana_curve"].modes[0][: len(downgraded_curve.modes[0])],
        )
        np.testing.assert_array_equal(
            downgraded_curve.modes[1],
            seed["banana_curve"].modes[1][: len(downgraded_curve.modes[1])],
        )
        np.testing.assert_array_equal(
            downgraded_curve.modes[2],
            seed["banana_curve"].modes[2][: len(downgraded_curve.modes[2])],
        )
        np.testing.assert_array_equal(
            downgraded_curve.modes[3],
            seed["banana_curve"].modes[3][: len(downgraded_curve.modes[3])],
        )

    def test_upgrade_loaded_seed_biot_savart_order_preserves_field_bitwise(self):
        module = _load_module(COIL_ORDER_UPGRADE_PATH, "coil_order_upgrade")
        stage2_module = _load_module(STAGE2_MODULE_PATH, "stage2_partition_upgrade")

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = _build_cws_seed(Path(tmpdir))
            loaded_bs = load(str(seed["seed_bs_path"]))
            loaded_partitions = stage2_module.partition_loaded_stage2_coils(
                loaded_bs.coils,
                stage2_results=seed["stage2_results"],
                requested_num_tf_coils=len(seed["tf_coils"]),
            )
            upgraded_bs, upgraded_curve, upgraded_banana_coils = (
                module.upgrade_loaded_seed_biot_savart_order(
                    loaded_bs,
                    banana_coils=loaded_partitions.banana_coils,
                    tf_coils=loaded_partitions.tf_coils,
                    proxy_coils=loaded_partitions.proxy_coils,
                    vf_coils=loaded_partitions.vf_coils,
                    new_order=4,
                )
            )

        np.testing.assert_array_equal(upgraded_bs.B(), seed["expected_field"])
        self.assertEqual(upgraded_curve.order, 4)
        self.assertEqual(len(upgraded_banana_coils), len(seed["banana_coils"]))


class SeedOrderUpgradeEntrypointTests(unittest.TestCase):
    def test_stage2_parse_args_accepts_seed_order_upgrade(self):
        module = _load_module(STAGE2_MODULE_PATH, "stage2_seed_order_upgrade")

        with patch.object(
            sys,
            "argv",
            ["banana_coil_solver.py", "--seed-order-upgrade", "4"],
        ):
            args = module.parse_args()

        self.assertEqual(args.seed_order_upgrade, 4)

    def test_single_stage_parse_args_accepts_seed_order_upgrade(self):
        module = _load_module(SINGLE_STAGE_MODULE_PATH, "single_stage_seed_order_upgrade")

        with patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py", "--seed-order-upgrade", "4"],
        ):
            args = module.parse_args()

        self.assertEqual(args.seed_order_upgrade, 4)

    def test_stage2_seed_loader_upgrades_loaded_banana_curve_order(self):
        module = _load_module(STAGE2_MODULE_PATH, "stage2_seed_loader_upgrade")

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = _build_cws_seed(Path(tmpdir))
            out_dir = str(Path(tmpdir) / "outputs") + "/"
            with patch.object(module, "curves_to_vtk", lambda *_args, **_kwargs: None):
                loaded = module.load_stage2_seed_configuration(
                    str(seed["seed_bs_path"]),
                    _FakeLoadSurface(),
                    len(seed["tf_coils"]),
                    out_dir,
                    stage2_results=seed["stage2_results"],
                    seed_order_upgrade=4,
                )

        _, _, banana_curve, banana_coils, *_ = loaded
        self.assertEqual(banana_curve.order, 4)
        self.assertEqual(len(banana_curve.x), 17)
        self.assertEqual(len(banana_coils), len(seed["banana_coils"]))

    def test_single_stage_seed_loader_upgrades_banana_dofs_before_optimizer_state(self):
        module = _load_module(SINGLE_STAGE_MODULE_PATH, "single_stage_seed_loader_upgrade")

        with tempfile.TemporaryDirectory() as tmpdir:
            seed = _build_cws_seed(Path(tmpdir))
            bs, coil_partitions = module.load_stage2_seed_biot_savart(
                str(seed["seed_bs_path"]),
                stage2_results=seed["stage2_results"],
                num_tf_coils=len(seed["tf_coils"]),
                seed_order_upgrade=4,
            )

        banana_curve = next(
            coil.curve
            for coil in coil_partitions.banana_coils
            if hasattr(coil.curve, "order")
        )
        np.testing.assert_array_equal(bs.B(), seed["expected_field"])
        self.assertEqual(banana_curve.order, 4)
        self.assertEqual(len(banana_curve.x), 17)
        self.assertEqual(len(coil_partitions.banana_coils), len(seed["banana_coils"]))


if __name__ == "__main__":
    unittest.main()
