import importlib.util
import sys
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
WORKFLOW_HELPERS_PATH = EXAMPLES_ROOT / "workflow_helpers.py"


def _load_workflow_helpers_module():
    module_name = f"workflow_helpers_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, WORKFLOW_HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workflow_helpers = _load_workflow_helpers_module()


class ResolveWataruVfTemplatePathTests(unittest.TestCase):
    def test_returns_explicit_path_when_provided(self):
        resolved = workflow_helpers.resolve_wataru_vf_template_path(
            finite_current_mode="wataru_proxy_field",
            vf_current_A=0.0,
            vf_template_path="/custom/path.json",
        )
        self.assertEqual(resolved, "/custom/path.json")

    def test_returns_bundled_default_when_path_missing_and_zero_current(self):
        resolved = workflow_helpers.resolve_wataru_vf_template_path(
            finite_current_mode="wataru_proxy_field",
            vf_current_A=0.0,
            vf_template_path=None,
        )
        self.assertIsNotNone(resolved)
        self.assertTrue(Path(resolved).is_file())
        self.assertEqual(
            Path(resolved),
            workflow_helpers.DEFAULT_WATARU_VF_TEMPLATE_PATH,
        )

    def test_returns_bundled_default_when_path_empty_and_zero_current(self):
        resolved = workflow_helpers.resolve_wataru_vf_template_path(
            finite_current_mode="wataru_proxy_field",
            vf_current_A=0.0,
            vf_template_path="",
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(
            Path(resolved),
            workflow_helpers.DEFAULT_WATARU_VF_TEMPLATE_PATH,
        )

    def test_returns_bundled_default_when_path_missing_and_nonzero_current(self):
        resolved = workflow_helpers.resolve_wataru_vf_template_path(
            finite_current_mode="wataru_proxy_field",
            vf_current_A=3.0e3,
            vf_template_path=None,
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(
            Path(resolved),
            workflow_helpers.DEFAULT_WATARU_VF_TEMPLATE_PATH,
        )

    def test_mode_label_does_not_influence_resolution(self):
        # Legacy artifact-provenance label stays accepted; it no longer gates
        # template selection after the Wataru-faithful refactor.
        resolved_wataru = workflow_helpers.resolve_wataru_vf_template_path(
            finite_current_mode="wataru_proxy_field",
            vf_current_A=0.0,
            vf_template_path=None,
        )
        resolved_legacy = workflow_helpers.resolve_wataru_vf_template_path(
            finite_current_mode="boozer_surrogate",
            vf_current_A=0.0,
            vf_template_path=None,
        )
        self.assertEqual(resolved_wataru, resolved_legacy)


class Stage2SeedSpecVfTemplateResolutionTests(unittest.TestCase):
    def _minimal_spec(self, **overrides):
        kwargs = dict(
            plasma_surf_filename="demo.nc",
            major_radius=0.92,
            toroidal_flux=0.5,
            length_weight=1.0,
            cc_weight=1.0,
            cc_threshold=0.05,
            curvature_weight=1.0,
            curvature_threshold=100.0,
            banana_surf_radius=0.21,
            tf_current_A=8.0e4,
            order=2,
        )
        kwargs.update(overrides)
        return workflow_helpers.Stage2SeedSpec(**kwargs)

    def test_auto_resolves_bundled_template_at_zero_current(self):
        spec = self._minimal_spec(
            proxy_plasma_current_A=0.0,
            vf_current_A=0.0,
            vf_template_path=None,
        )
        self.assertIsNotNone(spec.vf_template_path)
        self.assertEqual(
            Path(spec.vf_template_path),
            workflow_helpers.DEFAULT_WATARU_VF_TEMPLATE_PATH,
        )

    def test_preserves_explicit_template_path(self):
        spec = self._minimal_spec(
            vf_template_path="/explicit/vf_template.json",
        )
        self.assertEqual(spec.vf_template_path, "/explicit/vf_template.json")


class FormatStage2FiniteCurrentSuffixTests(unittest.TestCase):
    def _spec(
        self,
        *,
        proxy_plasma_current_A=0.0,
        vf_current_A=0.0,
        vf_template_path=None,
    ):
        return workflow_helpers.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.92,
            toroidal_flux=0.5,
            length_weight=1.0,
            cc_weight=1.0,
            cc_threshold=0.05,
            curvature_weight=1.0,
            curvature_threshold=100.0,
            banana_surf_radius=0.21,
            tf_current_A=8.0e4,
            order=2,
            proxy_plasma_current_A=proxy_plasma_current_A,
            vf_current_A=vf_current_A,
            vf_template_path=vf_template_path,
        )

    def test_baseline_stays_unsuffixed_even_with_resolved_template(self):
        spec = self._spec(
            proxy_plasma_current_A=0.0,
            vf_current_A=0.0,
            vf_template_path=None,  # will auto-resolve to bundled path
        )
        # Sanity-check that the template actually auto-resolved, so the test
        # exercises the "populated template + zero currents" branch.
        self.assertIsNotNone(spec.vf_template_path)

        suffix = workflow_helpers.format_stage2_finite_current_suffix(spec)
        self.assertEqual(suffix, "")

    def test_nonzero_plasma_current_emits_suffix(self):
        spec = self._spec(proxy_plasma_current_A=-1.0e3, vf_current_A=0.0)
        suffix = workflow_helpers.format_stage2_finite_current_suffix(spec)
        self.assertIn("-FCM=wataru_proxy_field", suffix)
        self.assertIn("-PPC=-1000", suffix)

    def test_nonzero_vf_current_emits_suffix(self):
        spec = self._spec(proxy_plasma_current_A=0.0, vf_current_A=3.0e3)
        suffix = workflow_helpers.format_stage2_finite_current_suffix(spec)
        self.assertIn("-VFC=3000", suffix)


if __name__ == "__main__":
    unittest.main()
