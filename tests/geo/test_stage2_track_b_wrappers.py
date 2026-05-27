import importlib.util
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
UNIFIED_RUNNER_PATH = EXAMPLE_ROOT / "run_stage2_to_single_stage.py"


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_unified_runner_module():
    return load_module(UNIFIED_RUNNER_PATH, "run_stage2_to_single_stage")


class UnifiedRunnerStage2InputTests(unittest.TestCase):
    def test_generated_stage2_input_threads_constraint_metadata(self):
        module = load_unified_runner_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / "stage2" / "biot_savart_opt.json"
            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                python_executable="python",
                dry_run=False,
                stage2_output_root=None,
                stage2_profile="standard_80ka",
                stage2_spec_json=None,
                equilibria_dir=None,
                stage2_timeout_seconds=0.0,
                stage2_cc_threshold=None,
                stage2_curvature_threshold=None,
                stage2_order=None,
                stage2_tf_current_A=None,
                stage2_toroidal_flux=None,
                stage2_basin_seed=None,
                iota_target=0.2,
                vol_target=0.10,
                alm_fix_signal_mismatch_guard=False,
            )

            with patch.object(
                module,
                "ensure_stage2_artifact_result",
                return_value=SimpleNamespace(
                    artifact_path=artifact_path,
                    artifact_reused=True,
                ),
            ) as ensure_mock, patch.object(
                module.stage2_alm_runner,
                "load_validated_stage2_artifact",
                return_value=(
                    artifact_path.with_name("results.json"),
                    {"OPTIMIZER_SUCCESS": True},
                ),
            ) as load_mock:
                stage2_input = module.resolve_stage2_input(
                    args,
                    output_root=root / "outputs",
                )

            ensure_kwargs = ensure_mock.call_args.kwargs
            self.assertEqual(
                ensure_kwargs["constraint_profile_label"],
                "profile:standard_80ka",
            )
            self.assertIn("constraint_metadata", load_mock.call_args.kwargs)
            self.assertIn(
                "--constraint-profile-label",
                stage2_input["command"],
            )
            self.assertTrue(stage2_input["artifact_reused"])

if __name__ == "__main__":
    unittest.main()
