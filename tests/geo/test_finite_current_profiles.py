import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
WRAPPER_PATH = EXAMPLE_ROOT / "run_stage2_to_single_stage.py"
SIGNED_CW_WOUT_PATH = (
    Path(__file__).resolve().parents[1] / "test_files" / "wout_10x10.nc"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import workflow_helpers  # noqa: E402
from banana_opt import jhalpern30_compat as compat  # noqa: E402
from banana_opt.finite_current_profiles import (  # noqa: E402
    JHALPERN30_PROFILE,
    VACUUM_PROFILE,
    WATARU_PROFILE,
    format_proxy_current_sign_convention_help,
    get_finite_current_profile,
)
from banana_opt.json_compat import load_boozer_finite_i  # noqa: E402


def _load_stage2_wrapper_module():
    module_name = f"run_stage2_to_single_stage_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _jhalpern_biot_savart(proxy_current_A: float) -> BiotSavart:
    vf_current_abs_A = abs(float(proxy_current_A)) * JHALPERN30_PROFILE.vf_current_ratio
    vf_currents_A = [vf_current_abs_A] * 12 + [-vf_current_abs_A] * 8
    currents = (
        [-8.0e4] * JHALPERN30_PROFILE.default_num_tf_coils
        + [-1.0e4] * JHALPERN30_PROFILE.default_num_banana_coils
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


def _pre_boozer_repair_fields(finite_current_mode: str) -> dict[str, object]:
    return {
        "MAJOR_RADIUS": 0.976,
        "TOROIDAL_FLUX": 0.24,
        "LENGTH_WEIGHT": 5.0e-4,
        "CC_WEIGHT": 100.0,
        "CC_THRESHOLD": 0.05,
        "CURVATURE_WEIGHT": 1.0e-4,
        "CURVATURE_THRESHOLD": 40.0,
        "banana_surf_radius": 0.142,
        "order": 2,
        "TF_CURRENT_A": -8.0e4,
        "BANANA_INIT_CURRENT_A": -1.0e4,
        "BANANA_CURRENT_MAX_A": 1.6e4,
        "LENGTH_TARGET": 1.9,
        "FINITE_CURRENT_MODE": finite_current_mode,
        "PROXY_PLASMA_CURRENT_A": -6.5e3,
        "VF_CURRENT_A": -1.0e3,
        "VF_TEMPLATE_PATH": str(JHALPERN30_PROFILE.default_vf_template_path),
    }


def _workflow_stage2_spec(finite_current_mode: str) -> workflow_helpers.Stage2SeedSpec:
    return workflow_helpers.Stage2SeedSpec(
        plasma_surf_filename="demo.nc",
        major_radius=0.976,
        toroidal_flux=0.24,
        length_weight=5.0e-4,
        cc_weight=100.0,
        cc_threshold=0.05,
        curvature_weight=1.0e-4,
        curvature_threshold=40.0,
        banana_surf_radius=0.142,
        tf_current_A=-8.0e4,
        order=2,
        finite_current_mode=finite_current_mode,
    )


class FiniteCurrentProfileTests(unittest.TestCase):
    def test_vacuum_profile_records_tf_banana_only_layout(self):
        profile = get_finite_current_profile("vacuum")

        self.assertIs(profile, VACUUM_PROFILE)
        self.assertEqual(profile.mode, "vacuum")
        self.assertEqual(profile.default_num_tf_coils, 20)
        self.assertEqual(profile.default_num_banana_coils, 10)
        self.assertEqual(profile.default_num_proxy_coils, 0)
        self.assertEqual(profile.default_num_vf_coils, 0)
        self.assertEqual(profile.default_total_coils, 30)
        self.assertEqual(profile.build_default_coil_groups_manifest().total(), 30)
        self.assertEqual(profile.g0_policy, "signed_explicit_tf_current")
        self.assertEqual(profile.proxy_placement_policy, "none")
        self.assertEqual(profile.proxy_vf_current_scalar_policy, "none")
        self.assertEqual(profile.vf_current_sign_policy, "none")
        self.assertEqual(profile.vf_current_mutability, "none")
        self.assertIsNone(profile.default_vf_template_path)
        self.assertIsNone(profile.vf_template_sha256)
        self.assertEqual(profile.proxy_current_sign_convention.key, "none")
        self.assertEqual(
            profile.proxy_current_sign_summary_fields()["proxy_current_scalar_policy"],
            "none",
        )

    def test_wataru_profile_preserves_default_template_and_counts(self):
        profile = get_finite_current_profile("wataru_proxy_field")

        self.assertIs(profile, WATARU_PROFILE)
        self.assertEqual(profile.mode, "wataru_proxy_field")
        self.assertEqual(profile.default_num_tf_coils, 20)
        self.assertEqual(profile.default_num_banana_coils, 10)
        self.assertEqual(profile.default_num_proxy_coils, 1)
        self.assertEqual(profile.default_num_vf_coils, 20)
        self.assertEqual(profile.default_total_coils, 51)
        self.assertEqual(profile.build_default_coil_groups_manifest().total(), 51)
        self.assertEqual(profile.g0_policy, "signed_explicit_tf_current")
        self.assertEqual(
            profile.proxy_placement_policy, "vmec_axis_zeroth_coefficients"
        )
        self.assertEqual(
            profile.proxy_vf_current_scalar_policy, "nonnegative_magnitude"
        )
        self.assertEqual(
            profile.vf_current_sign_policy, "template_sign_vf_current_scalar"
        )
        self.assertEqual(profile.vf_current_mutability, "independent_fixed_current")
        self.assertEqual(
            profile.proxy_current_sign_convention.key,
            "wataru_nonnegative_proxy_vf_magnitude",
        )
        self.assertEqual(
            profile.proxy_current_sign_convention.signedness,
            "nonnegative_magnitude",
        )
        self.assertIn(
            "nonnegative proxy/VF current magnitudes",
            " ".join(profile.proxy_current_sign_help_line().split()),
        )
        self.assertEqual(
            profile.proxy_current_sign_summary_fields()["proxy_current_scalar_policy"],
            "nonnegative_magnitude",
        )
        self.assertIsNotNone(profile.default_vf_template_path)
        self.assertTrue(profile.default_vf_template_path.is_file())
        self.assertEqual(
            len(load_boozer_finite_i(str(profile.default_vf_template_path)).coils),
            20,
        )
        self.assertEqual(
            hashlib.sha256(profile.default_vf_template_path.read_bytes()).hexdigest(),
            profile.vf_template_sha256,
        )

    def test_jhalpern_profile_records_51_coil_layout_and_rejection_boundary(self):
        profile = get_finite_current_profile("jhalpern30_proxy_field")

        self.assertIs(profile, JHALPERN30_PROFILE)
        self.assertEqual(profile.default_num_tf_coils, 20)
        self.assertEqual(profile.default_num_banana_coils, 10)
        self.assertEqual(profile.default_num_proxy_coils, 1)
        self.assertEqual(profile.default_num_vf_coils, 20)
        self.assertEqual(profile.default_total_coils, 51)
        self.assertEqual(profile.build_default_coil_groups_manifest().total(), 51)
        self.assertEqual(profile.g0_policy, "signed_explicit_tf_current")
        self.assertEqual(profile.proxy_placement_policy, "surface_major_radius_z0")
        self.assertEqual(
            profile.proxy_vf_current_scalar_policy,
            "signed_physical_scalar",
        )
        self.assertEqual(
            profile.vf_current_sign_policy, "template_sign_abs_proxy_current"
        )
        self.assertEqual(profile.vf_current_mutability, "shared_unfixed_scaled_current")
        self.assertEqual(
            profile.proxy_current_sign_convention.key,
            "jhalpern30_signed_upstream_proxy_loop",
        )
        self.assertEqual(
            profile.proxy_current_sign_convention.frame,
            "upstream_jhalpern30_proxy_loop",
        )
        self.assertEqual(
            profile.proxy_current_sign_convention.signedness,
            "signed_physical_scalar",
        )
        self.assertIn("do not infer", profile.proxy_current_sign_help_line())
        self.assertIn(
            "flip the local proxy winding",
            profile.proxy_current_sign_help_line(),
        )
        self.assertIn("positive raises iota", profile.proxy_current_sign_help_line())
        self.assertIn(
            "negative is counter-current/collapse",
            profile.proxy_current_sign_help_line(),
        )
        self.assertNotIn(
            "positive raises iota",
            profile.proxy_current_sign_metadata_fields()[
                "PROXY_CURRENT_OPERATOR_WARNING"
            ],
        )
        self.assertIn(
            "run_stage2_to_single_stage.py:pre_boozer_repair",
            profile.rejected_entrypoints,
        )

    def test_proxy_current_sign_help_lists_mode_specific_contracts(self):
        help_text = format_proxy_current_sign_convention_help(
            ("wataru_proxy_field", "jhalpern30_proxy_field"),
        )

        self.assertIn("wataru_nonnegative_proxy_vf_magnitude", help_text)
        self.assertIn("jhalpern30_signed_upstream_proxy_loop", help_text)
        self.assertIn("scalar_policy=nonnegative_magnitude", help_text)
        self.assertIn("scalar_policy=signed_physical_scalar", help_text)

    def test_unknown_profile_mode_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported finite-current profile"):
            get_finite_current_profile("boozer_surrogate")

    def test_workflow_helpers_fail_loudly_for_unsupported_profile_modes(self):
        with self.assertRaisesRegex(ValueError, "Unsupported finite-current profile"):
            workflow_helpers.resolve_finite_current_vf_template_path(
                "boozer_surrogate",
                None,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported finite-current profile"):
            workflow_helpers.format_stage2_finite_current_suffix(
                _workflow_stage2_spec("typo_proxy_field"),
            )

    def test_vacuum_workflow_uses_no_vf_template_and_unique_suffix(self):
        self.assertIsNone(
            workflow_helpers.resolve_finite_current_vf_template_path("vacuum", None)
        )
        with self.assertRaisesRegex(ValueError, "does not use a VF template"):
            workflow_helpers.resolve_finite_current_vf_template_path(
                "vacuum",
                "/tmp/vf_template.json",
            )
        self.assertEqual(
            workflow_helpers.format_stage2_finite_current_suffix(
                _workflow_stage2_spec("vacuum"),
            ),
            "-FCM=vacuum-PPC=0-VFC=0",
        )

    def test_stage2_to_single_stage_rejects_jhalpern_pre_boozer_repair(self):
        wrapper = _load_stage2_wrapper_module()
        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "FINITE_CURRENT_MODE='wataru_proxy_field'",
        ):
            wrapper.build_recovery_command(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                recovery_output_root=Path("/tmp/recovery"),
                original_stage2_results=_pre_boozer_repair_fields(
                    JHALPERN30_PROFILE.mode,
                ),
            )

    def test_jhalpern_importer_metadata_matches_profile(self):
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
            biot_savart = _jhalpern_biot_savart(-6.5e3)

            def load_boozer_surface(_path: str) -> SimpleNamespace:
                return SimpleNamespace(biotsavart=biot_savart)

            _bs_path, results_path = compat.import_jhalpern30_stage_bundle(
                input_root,
                output_dir,
                plasma_surf_path=SIGNED_CW_WOUT_PATH,
                load_fn=load_boozer_surface,
            )

            results = json.loads(results_path.read_text(encoding="utf-8"))

        self.assertEqual(results["FINITE_CURRENT_MODE"], JHALPERN30_PROFILE.mode)
        self.assertEqual(
            results["BOOZER_CURRENT_CONVENTION"],
            JHALPERN30_PROFILE.boozer_current_convention,
        )
        self.assertEqual(results["G0_POLICY"], JHALPERN30_PROFILE.g0_policy)
        self.assertEqual(
            results["PROXY_PLACEMENT_MODE"],
            JHALPERN30_PROFILE.proxy_placement_policy,
        )
        self.assertEqual(
            results["PROXY_VF_CURRENT_SCALAR_POLICY"],
            JHALPERN30_PROFILE.proxy_vf_current_scalar_policy,
        )
        self.assertEqual(
            results["VF_TEMPLATE_SHA256"],
            JHALPERN30_PROFILE.vf_template_sha256,
        )
        self.assertEqual(
            results["VF_CURRENT_SIGN_POLICY"],
            JHALPERN30_PROFILE.vf_current_sign_policy,
        )
        self.assertEqual(
            results["VF_CURRENT_MUTABILITY"],
            JHALPERN30_PROFILE.vf_current_mutability,
        )
        self.assertEqual(
            results["NUM_TF_COILS"], JHALPERN30_PROFILE.default_num_tf_coils
        )
        self.assertEqual(
            results["NUM_BANANA_COILS"],
            JHALPERN30_PROFILE.default_num_banana_coils,
        )
        self.assertEqual(
            results["NUM_PROXY_COILS"],
            JHALPERN30_PROFILE.default_num_proxy_coils,
        )
        self.assertEqual(
            results["NUM_VF_COILS"], JHALPERN30_PROFILE.default_num_vf_coils
        )
        self.assertEqual(results["TOTAL_COILS"], JHALPERN30_PROFILE.default_total_coils)
        missing_keys = [
            key
            for key in JHALPERN30_PROFILE.required_artifact_metadata_keys
            if key not in results
        ]
        self.assertEqual(missing_keys, [])


if __name__ == "__main__":
    unittest.main()
