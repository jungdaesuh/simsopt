"""Regression tests for Wataru-faithful seeded-restart VF metadata consistency.

Fix #4: a seeded Stage 2 restart must trust the donor artifact's recorded
VF_TEMPLATE_PATH verbatim. Silently promoting a legacy zero-VF donor's
``None`` path to the bundled default would desync artifact metadata from
the actual ``bs.coils`` layout (which partition_loaded_stage2_coils slices
from the saved BiotSavart, not from the resolved config).
"""

import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
STAGE_2_PATH = EXAMPLES_ROOT / "STAGE_2" / "banana_coil_solver.py"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))


def _load_stage2_solver_module():
    module_name = f"banana_coil_solver_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, STAGE_2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage2_solver = _load_stage2_solver_module()


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        finite_current_mode=None,
        proxy_plasma_current_A=None,
        vf_current_A=None,
        vf_template_path=None,
        stage2_bs_path=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _legacy_zero_vf_donor_results():
    return {
        "FINITE_CURRENT_MODE": "wataru_proxy_field",
        "PROXY_PLASMA_CURRENT_A": 0.0,
        "VF_CURRENT_A": 0.0,
        "VF_TEMPLATE_PATH": None,
        "NUM_VF_COILS": 0,
    }


def _full_vf_donor_results():
    return {
        "FINITE_CURRENT_MODE": "wataru_proxy_field",
        "PROXY_PLASMA_CURRENT_A": -1.0e3,
        "VF_CURRENT_A": 3.0e3,
        "VF_TEMPLATE_PATH": "/recorded/donor_vf_template.json",
        "NUM_VF_COILS": 4,
    }


class SeededRestartTrustsDonorMetadataTests(unittest.TestCase):
    def test_legacy_zero_vf_donor_keeps_null_vf_template_path(self):
        # Legacy donor: no VF template recorded, no VF coils in the artifact.
        # Restart must NOT silently upgrade to the bundled default.
        donor_results = _legacy_zero_vf_donor_results()

        config = stage2_solver._resolve_stage2_finite_current_config(
            _args(stage2_bs_path="/some/donor.json"),
            stage2_results=donor_results,
        )

        self.assertIsNone(config.vf_template_path)
        self.assertEqual(config.vf_current_A, 0.0)
        self.assertEqual(config.proxy_plasma_current_A, 0.0)

    def test_full_vf_donor_roundtrips_its_recorded_template(self):
        donor_results = _full_vf_donor_results()

        config = stage2_solver._resolve_stage2_finite_current_config(
            _args(stage2_bs_path="/some/donor.json"),
            stage2_results=donor_results,
        )

        self.assertEqual(config.vf_template_path, donor_results["VF_TEMPLATE_PATH"])

    def test_legacy_zero_vf_donor_rejects_cli_vf_template_override(self):
        donor_results = _legacy_zero_vf_donor_results()

        with self.assertRaisesRegex(
            ValueError,
            "Legacy zero-VF Stage 2 donors cannot override --vf-template-path",
        ):
            stage2_solver._resolve_stage2_finite_current_config(
                _args(
                    stage2_bs_path="/some/donor.json",
                    vf_template_path="/cli/override.json",
                ),
                stage2_results=donor_results,
            )

    def test_seeded_restart_rejects_cli_path_mismatch_against_recorded_template(self):
        donor_results = _full_vf_donor_results()

        with self.assertRaisesRegex(
            ValueError,
            r"--vf-template-path=.*does not match the loaded Stage 2 artifact metadata value",
        ):
            stage2_solver._resolve_stage2_finite_current_config(
                _args(
                    stage2_bs_path="/some/donor.json",
                    vf_template_path="/cli/override.json",
                ),
                stage2_results=donor_results,
            )

    def test_seeded_restart_rejects_cli_numeric_mismatch_against_recorded_current(self):
        donor_results = _full_vf_donor_results()

        with self.assertRaisesRegex(
            ValueError,
            r"--vf-current-A=.*does not match the loaded Stage 2 artifact metadata value",
        ):
            stage2_solver._resolve_stage2_finite_current_config(
                _args(
                    stage2_bs_path="/some/donor.json",
                    vf_current_A=2.5e3,
                ),
                stage2_results=donor_results,
            )


class FreshRunAutoResolvesBundledTemplateTests(unittest.TestCase):
    def test_fresh_run_fills_bundled_default_when_none_given(self):
        config = stage2_solver._resolve_stage2_finite_current_config(
            _args(),  # no stage2_bs_path, no explicit vf_template_path
            stage2_results=None,
        )

        self.assertIsNotNone(config.vf_template_path)
        self.assertTrue(Path(config.vf_template_path).is_file())

    def test_fresh_run_preserves_explicit_template_path(self):
        config = stage2_solver._resolve_stage2_finite_current_config(
            _args(vf_template_path="/explicit/fresh.json"),
            stage2_results=None,
        )

        self.assertEqual(config.vf_template_path, "/explicit/fresh.json")

    def test_fresh_run_rejects_nonzero_vf_without_any_template_source(self):
        # Guard: if a caller has managed to clear both the explicit flag and the
        # bundled default path (e.g. the repo was shipped without the JSON),
        # a non-zero VF current must raise rather than silently dropping VF.
        from unittest import mock

        with mock.patch.object(
            stage2_solver,
            "resolve_wataru_vf_template_path",
            return_value=None,
        ):
            with self.assertRaises(ValueError):
                stage2_solver._resolve_stage2_finite_current_config(
                    _args(vf_current_A=3.0e3),
                    stage2_results=None,
                )


if __name__ == "__main__":
    unittest.main()
