import json
import tempfile
import unittest
from pathlib import Path

from examples.single_stage_optimization.banana_opt.vacuum_topology_campaign import (
    STRICT_VACUUM_CURRENT_LINEAGE,
    build_strict_vacuum_seed_manifest,
    failed_strict_vacuum_checks,
    format_captured_command,
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
            }
            manifest = build_strict_vacuum_seed_manifest(
                command_args=["--strict-vacuum-current"],
                seed_biot_savart_path=seed_biot_savart_path,
                seed_results_path=seed_results_path,
                warm_start_surface_path=warm_start_surface_path,
                plasma_target_path=plasma_target_path,
                output_results_path=output_results_path,
                results=results,
                seed_results={"FINITE_CURRENT_MODE": "wataru_proxy_field"},
                seed_artifact_role="single_stage_resume",
            )
            manifest_path = root / "seed_manifest.json"
            write_strict_vacuum_seed_manifest(manifest_path, manifest)

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["command"]["validation"]["passed"])
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
