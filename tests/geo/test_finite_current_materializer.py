import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveXYZFourier


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
SIGNED_CW_WOUT_PATH = (
    Path(__file__).resolve().parents[1] / "test_files" / "wout_10x10.nc"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.artifact_contracts import compute_stage2_bs_sha256  # noqa: E402
from banana_opt.coil_groups import build_contiguous_manifest  # noqa: E402
from banana_opt.current_contracts import FiniteCurrentMode, MU0  # noqa: E402
from banana_opt.finite_current_materializer import (  # noqa: E402
    MATERIALIZED_IOTA_TRUST_STATUS_UNRUN,
    MaterializedFiniteCurrentSeedRequest,
    _validate_derived_boozer_I,
    materialize_finite_current_seed,
)
from banana_opt.finite_current_profiles import get_finite_current_profile  # noqa: E402
from banana_opt.json_compat import load_boozer_finite_i  # noqa: E402
from banana_opt.stage2_single_stage_handoff import (  # noqa: E402
    partition_loaded_stage2_coils,
    validate_loaded_seed_current_source_contract,
    validate_stage2_seed_bootability_contract,
    validate_stage2_seed_contract,
)
from workflow_runner_common import load_stage2_artifact_results  # noqa: E402


def _make_circle_curve(*, center: tuple[float, float, float], radius: float):
    curve = CurveXYZFourier(32, 1)
    center_x, center_y, center_z = center
    curve.set_dofs(
        [
            center_x,
            radius,
            0.0,
            center_y,
            0.0,
            radius,
            center_z,
            0.0,
            0.0,
        ]
    )
    curve.fix_all()
    return curve


def _make_coil(index: int, current_A: float) -> Coil:
    current = Current(float(current_A))
    current.fix_all()
    return Coil(
        _make_circle_curve(
            center=(0.82 + 0.004 * float(index), 0.03 * float(index % 5), 0.0),
            radius=0.045 + 0.002 * float(index % 3),
        ),
        current,
    )


def _stage2_contract_fields() -> dict[str, object]:
    from banana_opt import hardware_contracts

    return {
        "MAJOR_RADIUS": 0.976,
        "TOROIDAL_FLUX": 0.24,
        "banana_surf_radius": hardware_contracts.BANANA_WINDING_MINOR_RADIUS_M,
        "COIL_LENGTH": 2.0,
        "CURVE_CURVE_MIN_DIST": 0.05,
        "CURVE_SURFACE_MIN_DIST": 0.015,
        "SURFACE_VESSEL_MIN_DIST": 0.04,
        "MAX_CURVATURE": 100.0,
        "CURVATURE_THRESHOLD": 100.0,
        "POLOIDAL_EXTENT_RAD": 45.0 * np.pi / 180.0,
        "POLOIDAL_EXTENT_THRESHOLD_RAD": 45.0 * np.pi / 180.0,
        "COIL_WIDTH": 0.10,
        "WIDTH_MIN_THRESHOLD": 0.1,
        "WIDTH_MAX_THRESHOLD": 0.17,
        "SELF_INTERSECT_PENALTY": 0.0,
        "SELF_INTERSECT_THRESHOLD": 0.0,
        "SHORTEST_SELF_DISTANCE": 0.01,
        "SELF_INTERSECT_MIN_DISTANCE": 0.01,
        "BANANA_CURRENT_A": 1.1e4,
        "BANANA_INIT_CURRENT_A": -1.0e4,
        "BANANA_CURRENT_MAX_A": 1.6e4,
        "TF_CURRENT_A": -8.0e4,
        "FINAL_LCFS_MAJOR_RADIUS_M": hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
        "FINAL_LCFS_MINOR_RADIUS_M": hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M,
        "CC_THRESHOLD": 0.05,
        "CC_WEIGHT": 100.0,
        "CURVATURE_WEIGHT": 1.0e-4,
        "LENGTH_TARGET": 1.9,
        "LENGTH_MIN_TARGET": 0.95,
        "LENGTH_WEIGHT": 5.0e-4,
        "order": 2,
        "PLASMA_SURF_PATH": str(SIGNED_CW_WOUT_PATH),
        "WOUT_CONVENTION": "signed_cw",
        "WOUT_OFF_SPEC": False,
        "FINITE_CURRENT_MODE": "vacuum",
        "BOOZER_BOOTABLE": True,
        "IOTA_NEAR_TARGET": True,
        "IOTA_FEASIBLE": True,
        "BOOTABILITY_REASON": "ok",
        "BOOTABILITY_STAGE": "probe",
        "BOOTABILITY_TARGET_IOTA": 0.2,
        "BOOTABILITY_SOLVED_IOTA": 0.2,
        "BOOTABILITY_SELF_INTERSECTING": False,
        "BOOTABILITY_SOLVE_SUCCESS": True,
        "BOOTABILITY_ABS_IOTA_ERROR": 0.0,
        "BOOTABILITY_ERROR_TYPE": None,
        "BOOTABILITY_ERROR_MESSAGE": None,
        "BOOZER_SOLVE_SUCCESS": True,
        "BOOZER_SELF_INTERSECTING": False,
        "BOOZER_CONSTRAINED_RESIDUAL_NORM": 1.0e-14,
        "BOOZER_TRUSTED": True,
        "IOTA_OBJECTIVE_ACTIVE": True,
        "BOOZER_TRUST_REASON": "trusted",
        "BOOZER_TRUST_TOL": 1.0e-11,
        "SEED_ROLE": "coil_seed_handoff",
        "DIAGNOSTIC_ONLY": False,
        "PRODUCTION_HANDOFF_READY": True,
        "HANDOFF_BLOCKING_GATE": None,
        "PROMOTION_READY": True,
    }


def _jhalpern_replay_fields() -> dict[str, object]:
    return {
        "FLIP_BANANA": True,
        "BANANA_CURRENT_SIGN": -1,
        "BANANA_CURRENT_PINNED": False,
        "BANANA_I_FIXED_S2_KA": None,
        "IOTA_TARGET_SIGN": -1,
        "JHALPERN30_STAGE_NAME": "stage00",
        "JHALPERN30_STAGE_STATE": {
            "iota": 0.29789,
            "G": -2.0106,
            "volume": 0.03993,
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_source_seed(
    seed_dir: Path,
    *,
    extra_results: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object], BiotSavart]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    tf_coils = [_make_coil(index, -8.0e4) for index in range(20)]
    banana_coils = [_make_coil(20 + index, 1.1e4) for index in range(10)]
    source_bs = BiotSavart([*tf_coils, *banana_coils])
    source_bs_path = seed_dir / "biot_savart_opt.json"
    source_bs.save(str(source_bs_path))
    manifest = build_contiguous_manifest(
        num_tf_coils=20,
        num_banana_coils=10,
        num_proxy_coils=0,
        num_vf_coils=0,
    )
    source_results = {
        **_stage2_contract_fields(),
        "NUM_TF_COILS": 20,
        "NUM_BANANA_COILS": 10,
        "NUM_PROXY_COILS": 0,
        "NUM_VF_COILS": 0,
        "TOTAL_COILS": manifest.total(),
        "COIL_GROUPS": manifest.to_json_payload(),
        "STAGE2_BS_PATH": str(source_bs_path),
        "STAGE2_BS_SHA256": compute_stage2_bs_sha256(source_bs_path),
    }
    if extra_results is not None:
        source_results.update(extra_results)
    _write_json(seed_dir / "results.json", source_results)
    return source_bs_path, source_results, source_bs


def _materialize(
    source_bs_path: Path,
    output_root: Path,
    *,
    finite_current_mode: FiniteCurrentMode = "wataru_proxy_field",
    proxy_current_A: float = 6.5e3,
    vf_current_A: float | None = None,
):
    return materialize_finite_current_seed(
        MaterializedFiniteCurrentSeedRequest(
            source_biot_savart=source_bs_path,
            output_root=output_root,
            finite_current_mode=finite_current_mode,
            proxy_current_A=proxy_current_A,
            vf_current_A=vf_current_A,
        )
    )


def _sample_field(bs_path: Path, points: np.ndarray) -> np.ndarray:
    bs = load_boozer_finite_i(str(bs_path))
    bs.set_points(points)
    return bs.B().copy()


def _assert_sign_metadata(
    test_case: unittest.TestCase,
    results: dict[str, object],
    finite_current_mode: FiniteCurrentMode,
) -> None:
    profile = get_finite_current_profile(finite_current_mode)
    for key, expected_value in profile.proxy_current_sign_metadata_fields().items():
        test_case.assertEqual(results[key], expected_value)


class FiniteCurrentMaterializerTests(unittest.TestCase):
    def test_wataru_materialization_writes_replay_eligible_stage2_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")

            result = _materialize(source_bs_path, root / "mat_wataru")

            self.assertTrue(result.single_stage_replay_eligible)
            results_path, results = load_stage2_artifact_results(
                result.output_biot_savart
            )
            self.assertEqual(results_path, result.output_results)
            validate_stage2_seed_contract(results)
            output_bs = load_boozer_finite_i(str(result.output_biot_savart))
            partitions = partition_loaded_stage2_coils(
                output_bs.coils,
                stage2_results=results,
                requested_num_tf_coils=20,
            )

            self.assertEqual(partitions.num_tf_coils, 20)
            self.assertEqual(partitions.num_banana_coils, 10)
            self.assertEqual(partitions.num_proxy_coils, 1)
            self.assertEqual(partitions.num_vf_coils, 20)
            self.assertAlmostEqual(results["BOOZER_I"], MU0 * 6.5e3)
            self.assertEqual(
                results["MATERIALIZED_IOTA_TRUST_STATUS"],
                MATERIALIZED_IOTA_TRUST_STATUS_UNRUN,
            )
            self.assertEqual(results["SEED_ROLE"], "materialized_finite_current_seed")
            _assert_sign_metadata(self, results, "wataru_proxy_field")
            result_payload = result.to_json_payload()
            self.assertEqual(
                result_payload["proxy_current_sign_convention"],
                "wataru_nonnegative_proxy_vf_magnitude",
            )
            self.assertEqual(
                result_payload["proxy_current_scalar_policy"],
                "nonnegative_magnitude",
            )
            self.assertTrue(results["DIAGNOSTIC_ONLY"])
            self.assertFalse(results["PRODUCTION_HANDOFF_READY"])
            self.assertFalse(results["BOOZER_TRUSTED"])
            self.assertFalse(results["IOTA_FEASIBLE"])
            self.assertIsNone(results["BOOTABILITY_SOLVED_IOTA"])
            self.assertIsNone(results["FINAL_IOTA"])
            self.assertIsNone(results["FIELD_ERROR"])
            self.assertEqual(
                results["STAGE2_BS_SHA256"],
                compute_stage2_bs_sha256(result.output_biot_savart),
            )
            with self.assertRaisesRegex(ValueError, "not single-stage bootable"):
                validate_stage2_seed_bootability_contract(results)
            validate_loaded_seed_current_source_contract(
                finite_current_mode="wataru_proxy_field",
                effective_current_mode="wataru_proxy_field",
                plasma_current_A=6.5e3,
                plasma_current_input_source="physical_A",
                stage2_results=results,
                coil_partitions=partitions,
            )

    def test_proxy_current_changes_sampled_field_without_mutating_donor(self):
        points = np.array(
            [[0.25, 0.10, -0.15], [0.35, -0.05, 0.20], [0.55, 0.15, 0.05]],
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, source_bs = _build_source_seed(root / "source")
            source_hash_before = compute_stage2_bs_sha256(source_bs_path)
            source_currents_before = [
                float(coil.current.get_value()) for coil in source_bs.coils
            ]
            source_bs.set_points(points)
            source_field_before = source_bs.B().copy()

            result_1 = _materialize(
                source_bs_path,
                root / "mat_6500",
                proxy_current_A=6.5e3,
            )
            result_2 = _materialize(
                source_bs_path,
                root / "mat_13000",
                proxy_current_A=1.3e4,
            )

            field_1 = _sample_field(result_1.output_biot_savart, points)
            field_2 = _sample_field(result_2.output_biot_savart, points)
            self.assertFalse(np.allclose(field_1, field_2, rtol=0.0, atol=1.0e-12))
            self.assertEqual(
                source_hash_before, compute_stage2_bs_sha256(source_bs_path)
            )
            self.assertEqual(
                source_currents_before,
                [float(coil.current.get_value()) for coil in source_bs.coils],
            )
            source_bs.set_points(points)
            np.testing.assert_allclose(source_bs.B(), source_field_before)

    def test_jhalpern_materialization_derives_signed_boozer_i(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(
                root / "source",
                extra_results=_jhalpern_replay_fields(),
            )

            result = _materialize(
                source_bs_path,
                root / "mat_jhalpern",
                finite_current_mode="jhalpern30_proxy_field",
                proxy_current_A=-6.5e3,
            )
            _, results = load_stage2_artifact_results(result.output_biot_savart)
            output_bs = load_boozer_finite_i(str(result.output_biot_savart))
            partitions = partition_loaded_stage2_coils(
                output_bs.coils,
                stage2_results=results,
                requested_num_tf_coils=20,
            )

            validate_stage2_seed_contract(results)
            self.assertEqual(partitions.num_proxy_coils, 1)
            self.assertEqual(partitions.num_vf_coils, 20)
            self.assertEqual(results["FINITE_CURRENT_MODE"], "jhalpern30_proxy_field")
            _assert_sign_metadata(self, results, "jhalpern30_proxy_field")
            result_payload = result.to_json_payload()
            self.assertEqual(
                result_payload["proxy_current_sign_convention"],
                "jhalpern30_signed_upstream_proxy_loop",
            )
            self.assertEqual(
                result_payload["proxy_current_scalar_policy"],
                "signed_physical_scalar",
            )
            self.assertAlmostEqual(results["PROXY_PLASMA_CURRENT_A"], -6.5e3)
            self.assertAlmostEqual(results["VF_CURRENT_A"], -1.0e3)
            self.assertAlmostEqual(results["BOOZER_I"], MU0 * -6.5e3)

    def test_source_boozer_i_is_recomputed_from_realized_proxy_current(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(
                root / "source",
                extra_results={"BOOZER_I": 123.0},
            )

            result = _materialize(source_bs_path, root / "mat_wataru")
            _, results = load_stage2_artifact_results(result.output_biot_savart)

            self.assertAlmostEqual(results["BOOZER_I"], MU0 * 6.5e3)

    def test_derived_boozer_i_validator_fails_on_inconsistent_metadata(self):
        with self.assertRaisesRegex(ValueError, "Materialized BOOZER_I must match"):
            _validate_derived_boozer_I(
                realized_proxy_current_A=6.5e3,
                boozer_I=MU0 * 6.5e3,
                recorded_boozer_I=0.0,
                convention="mu0",
            )

    def test_wataru_negative_vf_override_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            output_root = root / "mat_wataru_negative_vf"

            with self.assertRaisesRegex(
                ValueError, "HBT VF current must be non-negative"
            ):
                _materialize(
                    source_bs_path,
                    output_root,
                    vf_current_A=-1.0e3,
                )
            self.assertFalse((output_root / "biot_savart_opt.json").exists())
            self.assertFalse((output_root / "results.json").exists())

    def test_wataru_zero_proxy_current_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            output_root = root / "mat_wataru_zero"

            with self.assertRaisesRegex(
                ValueError, "requires non-zero proxy_current_A"
            ):
                _materialize(
                    source_bs_path,
                    output_root,
                    proxy_current_A=0.0,
                )
            self.assertFalse((output_root / "biot_savart_opt.json").exists())
            self.assertFalse((output_root / "results.json").exists())

    def test_missing_source_results_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs = BiotSavart([_make_coil(0, -8.0e4)])
            source_bs_path = root / "source" / "biot_savart_opt.json"
            source_bs_path.parent.mkdir(parents=True)
            source_bs.save(str(source_bs_path))

            with self.assertRaisesRegex(FileNotFoundError, "requires a Stage-2"):
                _materialize(source_bs_path, root / "mat_wataru")

    def test_missing_source_checksum_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, source_results, _ = _build_source_seed(root / "source")
            source_results.pop("STAGE2_BS_SHA256")
            _write_json(root / "source" / "results.json", source_results)

            with self.assertRaisesRegex(ValueError, "missing STAGE2_BS_SHA256"):
                _materialize(source_bs_path, root / "mat_wataru")

    def test_source_checksum_mismatch_fails_before_materialization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, source_results, _ = _build_source_seed(root / "source")
            source_results["STAGE2_BS_SHA256"] = "not-the-source-hash"
            _write_json(root / "source" / "results.json", source_results)

            with self.assertRaisesRegex(ValueError, "Source Stage-2 checksum mismatch"):
                _materialize(source_bs_path, root / "mat_wataru")

    def test_jhalpern_missing_replay_metadata_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            output_root = root / "mat_jhalpern_missing"

            with self.assertRaisesRegex(ValueError, "donor replay metadata"):
                _materialize(
                    source_bs_path,
                    output_root,
                    finite_current_mode="jhalpern30_proxy_field",
                    proxy_current_A=-6.5e3,
                )
            self.assertFalse((output_root / "biot_savart_opt.json").exists())
            self.assertFalse((output_root / "results.json").exists())

    def test_jhalpern_null_replay_metadata_fails_without_partial_output(self):
        for null_key in ("FLIP_BANANA", "JHALPERN30_STAGE_STATE"):
            with self.subTest(null_key=null_key):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    root = Path(tmp_dir)
                    replay_fields = _jhalpern_replay_fields()
                    replay_fields[null_key] = None
                    source_bs_path, _, _ = _build_source_seed(
                        root / "source",
                        extra_results=replay_fields,
                    )
                    output_root = root / "mat_jhalpern_null"

                    with self.assertRaisesRegex(
                        ValueError, "non-null donor replay metadata"
                    ):
                        _materialize(
                            source_bs_path,
                            output_root,
                            finite_current_mode="jhalpern30_proxy_field",
                            proxy_current_A=-6.5e3,
                        )
                    self.assertFalse((output_root / "biot_savart_opt.json").exists())
                    self.assertFalse((output_root / "results.json").exists())

    def test_tampered_output_fails_stage2_checksum_binding(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            result = _materialize(source_bs_path, root / "mat_wataru")
            result.output_biot_savart.write_text(
                result.output_biot_savart.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_stage2_artifact_results(result.output_biot_savart)

    def test_cli_smoke_writes_reloadable_artifact(self):
        from materialize_finite_current_seed import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            output_root = root / "cli_out"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--source-biot-savart",
                        str(source_bs_path),
                        "--output-root",
                        str(output_root),
                        "--finite-current-mode",
                        "wataru_proxy_field",
                        "--proxy-current-A",
                        "6500",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                payload["proxy_current_sign_convention"],
                "wataru_nonnegative_proxy_vf_magnitude",
            )
            _, results = load_stage2_artifact_results(
                output_root / "biot_savart_opt.json"
            )
            self.assertEqual(results["FINITE_CURRENT_MODE"], "wataru_proxy_field")

    def test_cli_help_exposes_mode_specific_sign_contract(self):
        from materialize_finite_current_seed import parse_args

        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as caught:
            with contextlib.redirect_stdout(stdout):
                parse_args(["--help"])

        self.assertEqual(caught.exception.code, 0)
        help_text = stdout.getvalue()
        normalized_help = " ".join(help_text.split())
        self.assertIn(
            "sign semantics are selected by --finite-current-mode",
            normalized_help,
        )
        self.assertIn("wataru_proxy_field accepts nonnegative", help_text)
        self.assertIn("jhalpern30_proxy_field accepts a signed upstream", help_text)

    def test_sweep_wrapper_records_untrusted_iota_gate_summary(self):
        from run_materialized_current_sweep import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            output_root = root / "sweep"

            exit_code = main(
                [
                    "--source-biot-savart",
                    str(source_bs_path),
                    "--output-root",
                    str(output_root),
                    "--finite-current-mode",
                    "wataru_proxy_field",
                    "--current-grid-A",
                    "6500,13000",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads(
                (output_root / "materialized_current_sweep_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(summary["materialized_iota_trusted"])
            self.assertEqual(
                summary["materialized_iota_trust_status"],
                MATERIALIZED_IOTA_TRUST_STATUS_UNRUN,
            )
            self.assertEqual(
                summary["proxy_current_sign_convention"],
                "wataru_nonnegative_proxy_vf_magnitude",
            )
            self.assertEqual(len(summary["rows"]), 2)
            self.assertTrue(summary["rows"][0]["field_values_finite"])
            self.assertEqual(
                summary["rows"][0]["proxy_current_sign_convention"],
                "wataru_nonnegative_proxy_vf_magnitude",
            )
            self.assertIn(
                "proxy_current_sign_convention",
                (output_root / "materialized_current_sweep_summary.csv").read_text(
                    encoding="utf-8"
                ),
            )
            with self.assertRaisesRegex(FileExistsError, "Sweep output already exists"):
                main(
                    [
                        "--source-biot-savart",
                        str(source_bs_path),
                        "--output-root",
                        str(output_root),
                        "--finite-current-mode",
                        "wataru_proxy_field",
                        "--current-grid-A",
                        "6500,13000",
                    ]
                )

    def test_sweep_preflights_per_current_outputs_before_writing_any_row(self):
        from run_materialized_current_sweep import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            output_root = root / "sweep"
            stale_later_output = output_root / "current_001_p13000A"
            stale_later_output.mkdir(parents=True)
            (stale_later_output / "biot_savart_opt.json").write_text(
                "{}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileExistsError, "current_001_p13000A"):
                main(
                    [
                        "--source-biot-savart",
                        str(source_bs_path),
                        "--output-root",
                        str(output_root),
                        "--finite-current-mode",
                        "wataru_proxy_field",
                        "--current-grid-A",
                        "6500,13000",
                    ]
                )
            self.assertFalse(
                (output_root / "current_000_p6500A" / "biot_savart_opt.json").exists()
            )

    def test_sweep_prevalidates_all_currents_before_writing_any_row(self):
        from run_materialized_current_sweep import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_bs_path, _, _ = _build_source_seed(root / "source")
            output_root = root / "sweep"

            with self.assertRaisesRegex(ValueError, "entry 1 is zero"):
                main(
                    [
                        "--source-biot-savart",
                        str(source_bs_path),
                        "--output-root",
                        str(output_root),
                        "--finite-current-mode",
                        "wataru_proxy_field",
                        "--current-grid-A",
                        "6500,0",
                    ]
                )
            self.assertFalse(
                (output_root / "current_000_p6500A" / "biot_savart_opt.json").exists()
            )


class FiniteCurrentMaterializerProfileTests(unittest.TestCase):
    def test_profiles_expose_field_source_modes_used_by_materializer(self):
        wataru = get_finite_current_profile("wataru_proxy_field")
        jhalpern = get_finite_current_profile("jhalpern30_proxy_field")

        self.assertEqual(wataru.default_num_proxy_coils, 1)
        self.assertEqual(wataru.default_num_vf_coils, 20)
        self.assertEqual(jhalpern.default_num_proxy_coils, 1)
        self.assertEqual(jhalpern.default_num_vf_coils, 20)


if __name__ == "__main__":
    unittest.main()
