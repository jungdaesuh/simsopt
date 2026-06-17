import json
import tempfile
import unittest
from pathlib import Path

from examples.single_stage_optimization.banana_opt.vacuum_topology_campaign import (
    STRICT_VACUUM_CURRENT_LINEAGE,
    STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
    STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE,
    build_strict_vacuum_seed_manifest,
    failed_strict_vacuum_checks,
    format_captured_command,
    strict_vacuum_boozer_interchange_manifest,
    strict_vacuum_metadata_status,
    strict_vacuum_seed_input_status,
    validate_strict_vacuum_command,
    write_strict_vacuum_seed_manifest,
)


class VacuumTopologyCampaignTests(unittest.TestCase):
    def test_validate_strict_vacuum_command_rejects_current_flags(self):
        checks = validate_strict_vacuum_command(
            [
                "--strict-vacuum-current",
                "--finite-current-mode=wataru_proxy_field",
                "--boozer-I",
                "0.0",
                "BoozerSurfaceFiniteI",
            ]
        )

        self.assertFalse(checks["passed"])
        self.assertEqual(
            checks["forbidden_flags"],
            ["--finite-current-mode=wataru_proxy_field", "--boozer-I"],
        )
        self.assertEqual(checks["forbidden_substrings"], ["BoozerSurfaceFiniteI"])
        self.assertEqual(
            failed_strict_vacuum_checks(checks),
            ["forbidden_flags", "forbidden_substrings"],
        )

    def test_strict_vacuum_metadata_status_accepts_plain_vacuum_lineage(self):
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": (
                    STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE
                ),
                "STAGE1_CANDIDATE_ID": "s01_3240f0",
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENTS_A": [-15910.0],
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
                "BOOZER_SURFACE_CLASSES": ["BoozerSurface"],
                "BOOZER_SURFACE_MODULES": ["simsopt.geo.boozersurface"],
            }
        )

        self.assertTrue(checks["passed"])

    def test_strict_vacuum_metadata_status_accepts_shared_cws_signed_currents(self):
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": (
                    STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE
                ),
                "STAGE1_CANDIDATE_ID": "s01_88d5ae",
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENT_MODE": "shared",
                "BANANA_INIT_CURRENT_A": -15910.0,
                "BANANA_CURRENT_A": -15910.0,
                "BANANA_CURRENTS_A": [-15910.0, -15910.0, -15910.0, -15910.0],
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
                "BOOZER_SURFACE_CLASSES": ["BoozerSurface"],
                "BOOZER_SURFACE_MODULES": ["simsopt.geo.boozersurface"],
            }
        )

        self.assertTrue(checks["passed"])

    def test_strict_vacuum_metadata_status_rejects_finite_i_lineage(self):
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENTS_A": [-15910.0],
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASSES": ["BoozerSurfaceFiniteI"],
                "BOOZER_SURFACE_MODULES": ["banana_opt.boozer_finite_current"],
            }
        )

        self.assertFalse(checks["passed"])
        self.assertIn(
            "finite_current_mode_absent",
            failed_strict_vacuum_checks(checks),
        )
        self.assertIn(
            "boozer_surface_class_plain",
            failed_strict_vacuum_checks(checks),
        )

    def test_strict_vacuum_metadata_status_rejects_positive_embedded_currents(self):
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
                "TF_CURRENT_A": 80000.0,
                "BANANA_CURRENT_A": 15910.0,
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
            }
        )

        self.assertFalse(checks["passed"])
        self.assertIn(
            "signed_tf_current_negative",
            failed_strict_vacuum_checks(checks),
        )
        self.assertIn(
            "signed_banana_current_negative",
            failed_strict_vacuum_checks(checks),
        )

    def test_strict_vacuum_metadata_status_accepts_independent_symmetry_currents(self):
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENT_MODE": "independent",
                "BANANA_INIT_CURRENT_A": 15910.0,
                "BANANA_CURRENT_A": 15910.0,
                "BANANA_CURRENTS_A": [-15910.0, 15910.0],
                "BANANA_CURRENT_MAX_ABS_A": 15910.0,
                "BANANA_CURRENT_CONTROL_METRIC": "max_abs",
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
            }
        )

        self.assertTrue(checks["passed"])

    def test_strict_vacuum_metadata_status_rejects_uncontracted_mixed_currents(self):
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENT_MODE": "independent",
                "BANANA_INIT_CURRENT_A": -15910.0,
                "BANANA_CURRENTS_A": [-15910.0, 15910.0],
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
            }
        )

        self.assertFalse(checks["passed"])
        self.assertIn(
            "signed_banana_current_negative",
            failed_strict_vacuum_checks(checks),
        )

    def test_strict_vacuum_metadata_status_accepts_independent_negative_embedded_currents(
        self,
    ):
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENT_MODE": "independent",
                "BANANA_INIT_CURRENT_A": 15910.0,
                "BANANA_CURRENT_A": 15910.0,
                "BANANA_CURRENTS_A": [-15910.0, -12000.0],
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
            }
        )

        self.assertTrue(checks["passed"])

    def test_strict_vacuum_metadata_status_rejects_mixed_shared_currents(self):
        # Positive BASE current = CW-lane sign violation. Note: [-I, +I] with a
        # negative base is the contract-CORRECT resumed shared-mode shape per
        # program Hard Invariant 12 (exact alternation after stellarator
        # symmetry, "negated, not flattened") and is now accepted; the rejected
        # shapes are a positive base or ragged magnitudes.
        checks = strict_vacuum_metadata_status(
            {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": STRICT_VACUUM_SEED_LINEAGE_LEGACY_CONTROL,
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENT_MODE": "shared",
                "BANANA_INIT_CURRENT_A": -15910.0,
                "BANANA_CURRENT_A": -15910.0,
                "BANANA_CURRENTS_A": [15910.0, -15910.0],
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
            }
        )

        self.assertFalse(checks["passed"])
        self.assertIn(
            "signed_banana_current_negative",
            failed_strict_vacuum_checks(checks),
        )

    def test_strict_vacuum_seed_input_status_allows_metadata_caveat_only(self):
        checks = strict_vacuum_seed_input_status(
            {
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
            },
            num_proxy_coils=0,
            num_vf_coils=0,
        )

        self.assertTrue(checks["passed"])

    def test_strict_vacuum_seed_input_status_accepts_projected_source_groups(self):
        checks = strict_vacuum_seed_input_status(
            {
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
                "EFFECTIVE_CURRENT_MODE": "finite_current",
                "PLASMA_CURRENT_A": 1500.0,
                "BOOZER_I": 0.001,
                "PROXY_PLASMA_CURRENT_A": 1500.0,
                "VF_CURRENT_A": 230.769,
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 20,
            },
            num_proxy_coils=1,
            num_vf_coils=20,
            projected_source_current_groups=True,
        )

        self.assertTrue(checks["passed"])

    def test_strict_vacuum_seed_input_status_accepts_finite_current_metadata_without_source_groups(
        self,
    ):
        checks = strict_vacuum_seed_input_status(
            {
                "FINITE_CURRENT_MODE": "boozer_surrogate",
                "EFFECTIVE_CURRENT_MODE": "boozer_surrogate",
                "PLASMA_CURRENT_A": 2500.0,
                "BOOZER_I": 0.00314159,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
            },
            num_proxy_coils=0,
            num_vf_coils=0,
            projected_source_current_groups=False,
        )

        self.assertTrue(checks["passed"])

    def test_strict_vacuum_boozer_interchange_manifest_marks_baseline_contract(self):
        manifest = strict_vacuum_boozer_interchange_manifest()

        self.assertEqual(manifest["interchange_mode"], STRICT_VACUUM_CURRENT_LINEAGE)
        self.assertEqual(manifest["current_lineage"], STRICT_VACUUM_CURRENT_LINEAGE)
        self.assertTrue(manifest["baseline_replayable"])
        self.assertEqual(
            manifest["requires_boozer_surface_module"],
            "simsopt.geo.boozersurface",
        )
        self.assertEqual(manifest["requires_boozer_surface_class"], "BoozerSurface")
        self.assertTrue(manifest["requires_no_boozer_surface_i_field"])
        self.assertTrue(manifest["requires_no_boozer_surface_finite_i"])
        self.assertEqual(manifest["boozer_state_sidecar_fields"], ["iota", "G"])

    def test_strict_vacuum_manifest_records_hashes_and_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_biot_savart_path = root / "biot_savart_opt.json"
            seed_results_path = root / "results.json"
            warm_start_surface_path = root / "surf_opt_outer_boozer_surface.json"
            plasma_target_path = root / "wout.nc"
            output_results_path = root / "output_results.json"
            for path, payload in (
                (seed_biot_savart_path, "{}"),
                (seed_results_path, "{}"),
                (warm_start_surface_path, "{}"),
                (plasma_target_path, "wout"),
                (output_results_path, "{}"),
            ):
                path.write_text(payload, encoding="utf-8")

            results = {
                "STRICT_VACUUM_CURRENT": True,
                "CURRENT_LINEAGE": STRICT_VACUUM_CURRENT_LINEAGE,
                "STRICT_VACUUM_SEED_LINEAGE": (
                    STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE
                ),
                "STAGE1_CANDIDATE_ID": "s01_3240f0",
                "TF_CURRENT_A": -80000.0,
                "BANANA_CURRENTS_A": [-15910.0],
                "EFFECTIVE_CURRENT_MODE": "vacuum",
                "FINITE_CURRENT_MODE": None,
                "PLASMA_CURRENT_A": 0.0,
                "BOOZER_I": 0.0,
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
                "BOOZER_SURFACE_CLASS": "BoozerSurface",
                "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
                "STAGE2_NUM_PROXY_COILS": 0,
                "STAGE2_NUM_VF_COILS": 0,
                "STAGE2_SOURCE_NUM_PROXY_COILS": 1,
                "STAGE2_SOURCE_NUM_VF_COILS": 20,
            }
            manifest = build_strict_vacuum_seed_manifest(
                command_args=[
                    "--strict-vacuum-current",
                    "--strict-vacuum-lineage=recent_stage1_candidate",
                    "--stage1-candidate-id=s01_3240f0",
                ],
                strict_vacuum_seed_lineage=(
                    STRICT_VACUUM_SEED_LINEAGE_RECENT_STAGE1_CANDIDATE
                ),
                stage1_candidate_id="s01_3240f0",
                seed_biot_savart_path=seed_biot_savart_path,
                seed_results_path=seed_results_path,
                warm_start_surface_path=warm_start_surface_path,
                plasma_target_path=plasma_target_path,
                output_results_path=output_results_path,
                results=results,
                seed_results={
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                    "EFFECTIVE_CURRENT_MODE": "finite_current",
                    "PLASMA_CURRENT_A": 1500.0,
                    "BOOZER_I": 0.001,
                    "PROXY_PLASMA_CURRENT_A": 1500.0,
                    "VF_CURRENT_A": 230.769,
                    "NUM_PROXY_COILS": 1,
                    "NUM_VF_COILS": 20,
                },
                seed_artifact_role="single_stage_resume",
                projected_source_current_groups=True,
            )
            manifest_path = root / "seed_manifest.json"
            write_strict_vacuum_seed_manifest(manifest_path, manifest)

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["command"]["validation"]["passed"])
        self.assertEqual(payload["lineage"], "recent_stage1_candidate")
        self.assertFalse(payload["control_only"])
        self.assertTrue(payload["production_candidate"])
        self.assertTrue(payload["baseline_replayable"])
        self.assertTrue(payload["boozer_interchange_manifest"]["baseline_replayable"])
        self.assertEqual(
            payload["boozer_interchange_manifest"]["current_lineage"],
            STRICT_VACUUM_CURRENT_LINEAGE,
        )
        self.assertEqual(payload["stage1_candidate_id"], "s01_3240f0")
        self.assertEqual(
            payload["source_current_group_projection"],
            "tf_banana_only",
        )
        self.assertTrue(payload["result_metadata_validation"]["passed"])
        self.assertIsNotNone(payload["inherited_seed_caveat"])
        self.assertTrue(payload["source_files"]["seed_biot_savart"]["exists"])
        self.assertEqual(len(payload["source_files"]["output_results"]["sha256"]), 64)
        self.assertEqual(
            format_captured_command("python", ["--strict-vacuum-current"]),
            "python --strict-vacuum-current",
        )


if __name__ == "__main__":
    unittest.main()
