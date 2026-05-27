import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
SIGNED_CW_WOUT_PATH = (
    Path(__file__).resolve().parents[1] / "test_files" / "wout_10x10.nc"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt import current_contracts  # noqa: E402
from banana_opt import jhalpern30_compat as compat  # noqa: E402
from banana_opt import stage2_single_stage_handoff as handoff  # noqa: E402
from banana_opt.coil_groups import build_contiguous_manifest  # noqa: E402


class _SurfaceWithMajorRadius:
    def __init__(self, major_radius: float):
        self._major_radius = float(major_radius)

    def major_radius(self) -> float:
        return self._major_radius


def _make_curve(index: int) -> CurveXYZFourier:
    curve = CurveXYZFourier(16, 1)
    curve.set("xc(0)", 0.01 * float(index))
    curve.set("xc(1)", 0.10)
    curve.set("ys(1)", 0.10)
    curve.fix_all()
    return curve


def _make_coil(index: int, current_A: float) -> Coil:
    current = Current(float(current_A))
    current.fix_all()
    return Coil(_make_curve(index), current)


def _make_biot_savart_with_proxy(proxy_current_A: float) -> BiotSavart:
    vf_current_abs_A = abs(float(proxy_current_A)) / 6.5
    vf_currents_A = [vf_current_abs_A] * 12 + [-vf_current_abs_A] * 8
    currents = (
        [-8.0e4] * compat.JHALPERN30_NUM_TF_COILS
        + [-1.0e4] * compat.JHALPERN30_NUM_BANANA_COILS
        + [float(proxy_current_A)]
        + vf_currents_A
    )
    return BiotSavart(
        [_make_coil(index, current_A) for index, current_A in enumerate(currents)]
    )


def _stage_state() -> dict[str, object]:
    return {
        "iota": 0.191,
        "G": -0.101,
        "volume": 0.13,
        "iota_target": 0.2,
        "stage_idx": 0,
        "stage_mpol": 8,
        "stage_ntor": 6,
        "stage_order": 2,
        "stage_qp": 64,
    }


class Jhalpern30CompatibilityTests(unittest.TestCase):
    def test_bundled_vf_template_matches_recorded_hash_count_and_signs(self):
        template_path = compat.resolve_jhalpern30_vf_template_path(None)

        self.assertEqual(
            compat.sha256_file(template_path),
            compat.JHALPERN30_VF_TEMPLATE_SHA256,
        )
        signs = compat.validate_jhalpern30_vf_template(template_path)

        self.assertEqual(len(signs), 20)
        self.assertEqual(signs, (1.0,) * 12 + (-1.0,) * 8)

    def test_signed_proxy_vf_validation_is_mode_scoped(self):
        proxy_current_A = -6.5e3
        vf_current_A = -1.0e3

        self.assertEqual(
            current_contracts.validate_jhalpern30_proxy_vf_current_convention(
                proxy_plasma_current_A=proxy_current_A,
                vf_current_A=vf_current_A,
            ),
            (proxy_current_A, vf_current_A),
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            current_contracts.validate_hbt_proxy_vf_current_convention(
                proxy_plasma_current_A=proxy_current_A,
                vf_current_A=vf_current_A,
            )
        with self.assertRaisesRegex(ValueError, "jhalpern30 proxy/VF convention"):
            current_contracts.validate_jhalpern30_proxy_vf_current_convention(
                proxy_plasma_current_A=proxy_current_A,
                vf_current_A=1.0e3,
            )

    def test_boozer_current_and_single_surface_mode_use_mu0_signed_proxy_current(self):
        proxy_current_A = -6.5e3

        self.assertEqual(
            current_contracts.resolve_finite_current_mode(
                "jhalpern30_proxy_field",
            ),
            compat.JHALPERN30_FINITE_CURRENT_MODE,
        )
        self.assertEqual(
            current_contracts.resolve_boozer_current_convention(
                compat.JHALPERN30_FINITE_CURRENT_MODE,
            ),
            "mu0",
        )
        self.assertAlmostEqual(
            compat.jhalpern30_proxy_boozer_I(proxy_current_A),
            current_contracts.MU0 * proxy_current_A,
        )

        settings = current_contracts.resolve_plasma_current_settings_for_num_surfaces(
            raw_boozer_I=None,
            plasma_current_A=None,
            finite_current_mode=compat.JHALPERN30_FINITE_CURRENT_MODE,
            default_plasma_current_A=proxy_current_A,
            num_surfaces=1,
            requested_finite_current_mode=compat.JHALPERN30_FINITE_CURRENT_MODE,
        )

        self.assertEqual(settings.mode, compat.JHALPERN30_FINITE_CURRENT_MODE)
        self.assertEqual(settings.effective_mode, compat.JHALPERN30_FINITE_CURRENT_MODE)
        self.assertAlmostEqual(settings.boozer_I, current_contracts.MU0 * proxy_current_A)

    def test_proxy_builder_uses_surface_major_radius_at_z0(self):
        proxy_coil = compat.build_jhalpern30_proxy_plasma_current_coils(
            _SurfaceWithMajorRadius(0.976),
            -6.5e3,
        )[0]

        gamma = proxy_coil.curve.gamma()
        radii = np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)
        np.testing.assert_allclose(radii, 0.976, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(gamma[:, 2], 0.0, rtol=0.0, atol=1.0e-12)
        self.assertFalse(np.any(proxy_coil.curve.dofs_free_status))
        self.assertFalse(np.any(proxy_coil.current.dofs_free_status))
        self.assertAlmostEqual(proxy_coil.current.get_value(), -6.5e3)

    def test_vf_builder_shares_mutable_leaf_current_with_template_signs(self):
        proxy_current_A = -6.5e3
        vf_coils = compat.build_jhalpern30_vf_coils(
            proxy_current_A,
            compat.resolve_jhalpern30_vf_template_path(None),
        )

        self.assertEqual(len(vf_coils), compat.JHALPERN30_NUM_VF_COILS)
        template_signs = compat.validate_jhalpern30_vf_template(
            compat.resolve_jhalpern30_vf_template_path(None),
        )
        first_leaf_current, first_scale = current_contracts.unwrap_current_optimizable(
            vf_coils[0].current,
        )
        self.assertTrue(np.all(first_leaf_current.dofs_free_status))
        self.assertTrue(np.all(vf_coils[0].current.dofs_free_status))
        self.assertAlmostEqual(first_scale, abs(proxy_current_A) / 6.5)

        for coil, template_sign in zip(vf_coils, template_signs, strict=True):
            leaf_current, scale = current_contracts.unwrap_current_optimizable(
                coil.current,
            )
            self.assertIs(leaf_current, first_leaf_current)
            self.assertTrue(np.all(coil.current.dofs_free_status))
            self.assertFalse(np.any(coil.curve.dofs_free_status))
            self.assertAlmostEqual(
                scale,
                abs(proxy_current_A) / 6.5 * template_sign,
            )
            self.assertAlmostEqual(coil.current.get_value(), scale)

        first_leaf_current.set_dofs(np.asarray([1.25], dtype=float))
        for coil, template_sign in zip(vf_coils, template_signs, strict=True):
            self.assertAlmostEqual(
                coil.current.get_value(),
                1.25 * abs(proxy_current_A) / 6.5 * template_sign,
            )

    def test_banana_sign_and_pin_replay_contract(self):
        free_replay = compat.resolve_jhalpern30_banana_current_replay(
            flip_banana=False,
            banana_i_fixed_s2=None,
        )
        flipped_replay = compat.resolve_jhalpern30_banana_current_replay(
            flip_banana=True,
            banana_i_fixed_s2="",
        )
        pinned_replay = compat.resolve_jhalpern30_banana_current_replay(
            flip_banana=True,
            banana_i_fixed_s2="7.5",
        )

        self.assertEqual(free_replay.banana_current_sign, 1)
        self.assertFalse(free_replay.banana_current_pinned)
        self.assertEqual(free_replay.current_scale_A, -1.0e4)
        self.assertEqual(flipped_replay.banana_current_sign, -1)
        self.assertEqual(flipped_replay.current_scale_A, 1.0e4)
        self.assertTrue(pinned_replay.banana_current_pinned)
        self.assertEqual(pinned_replay.current_scale_A, -7.5e3)
        self.assertEqual(pinned_replay.banana_i_fixed_s2_kA, 7.5)
        self.assertTrue(compat.jhalpern30_flip_from_stage_parent("/tmp/I-6.5_flip/stage00"))
        self.assertEqual(compat.jhalpern30_iota_target_sign(flip_banana=True), -1)

    def test_contiguous_manifest_records_historical_51_coil_layout(self):
        manifest = build_contiguous_manifest(
            num_tf_coils=20,
            num_banana_coils=10,
            num_proxy_coils=compat.JHALPERN30_NUM_PROXY_COILS,
            num_vf_coils=compat.JHALPERN30_NUM_VF_COILS,
        )

        self.assertEqual(manifest.total(), 51)
        self.assertEqual(manifest.to_json_payload()[-1]["role"], "vf")
        self.assertEqual(manifest.to_json_payload()[-1]["count"], 20)

    def test_stage_bundle_importer_emits_canonical_results_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_root = temp_path / "I-6.5_flip"
            stage_dir = input_root / "stage00"
            output_dir = temp_path / "output"
            stage_dir.mkdir(parents=True)
            (stage_dir / compat.JHALPERN30_STAGE_BSURF_FILENAME).write_text(
                "{}",
                encoding="utf-8",
            )
            (stage_dir / compat.JHALPERN30_STAGE_STATE_FILENAME).write_text(
                json.dumps(_stage_state()),
                encoding="utf-8",
            )
            biot_savart = _make_biot_savart_with_proxy(-6.5e3)

            def load_boozer_surface(_path: str) -> SimpleNamespace:
                return SimpleNamespace(biotsavart=biot_savart)

            bs_path, results_path = compat.import_jhalpern30_stage_bundle(
                input_root,
                output_dir,
                plasma_surf_path=SIGNED_CW_WOUT_PATH,
                load_fn=load_boozer_surface,
            )

            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertTrue(bs_path.is_file())

        self.assertEqual(
            results["FINITE_CURRENT_MODE"],
            compat.JHALPERN30_FINITE_CURRENT_MODE,
        )
        self.assertEqual(results["BOOZER_CURRENT_CONVENTION"], "mu0")
        self.assertEqual(results["PROXY_PLACEMENT_MODE"], "surface_major_radius_z0")
        self.assertEqual(results["PLASMA_SURF_PATH"], str(SIGNED_CW_WOUT_PATH))
        self.assertEqual(results["WOUT_CONVENTION"], "signed_cw")
        self.assertFalse(results["WOUT_OFF_SPEC"])
        self.assertEqual(results["TF_CURRENT_A"], -8.0e4)
        self.assertEqual(results["PROXY_PLASMA_CURRENT_A"], -6.5e3)
        self.assertEqual(results["BOOZER_I"], compat.jhalpern30_proxy_boozer_I(-6.5e3))
        self.assertEqual(results["VF_CURRENT_A"], -1.0e3)
        self.assertEqual(
            [coil.current.get_value() for coil in biot_savart.coils[31:]],
            [1.0e3] * 12 + [-1.0e3] * 8,
        )
        self.assertEqual(results["NUM_TF_COILS"], 20)
        self.assertEqual(results["NUM_BANANA_COILS"], 10)
        self.assertEqual(results["NUM_PROXY_COILS"], 1)
        self.assertEqual(results["NUM_VF_COILS"], 20)
        self.assertEqual(results["TOTAL_COILS"], 51)
        self.assertTrue(results["FLIP_BANANA"])
        self.assertEqual(results["BANANA_CURRENT_SIGN"], -1)
        self.assertFalse(results["BANANA_CURRENT_PINNED"])
        self.assertIsNone(results["BANANA_I_FIXED_S2_KA"])
        self.assertEqual(results["IOTA_TARGET_SIGN"], -1)
        self.assertEqual(results["JHALPERN30_STAGE_NAME"], "stage00")
        self.assertEqual(results["G"], -0.101)
        handoff.validate_stage2_seed_contract(results)


if __name__ == "__main__":
    unittest.main()
