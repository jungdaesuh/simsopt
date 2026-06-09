import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt import hardware_contracts as hc  # noqa: E402
from banana_opt import viewer_export as module  # noqa: E402


class _FakeCurve:
    def __init__(self, z_offset):
        self._points = np.asarray(
            [
                [0.90, 0.00, z_offset],
                [0.90, 0.10, z_offset],
                [0.80, 0.10, z_offset],
                [0.80, 0.00, z_offset],
            ],
            dtype=float,
        )

    def gamma(self):
        return self._points


class _FakeCurrent:
    def __init__(self, value):
        self._value = float(value)

    def get_value(self):
        return self._value


class _FakeCoil:
    def __init__(self, index, current):
        self.curve = _FakeCurve(float(index) * 0.01)
        self.current = _FakeCurrent(current)


def _coils():
    return tuple(
        [_FakeCoil(index, -80000.0) for index in range(20)]
        + [_FakeCoil(index + 20, -10000.0) for index in range(10)]
    )


def _count_equivalent_51_coils():
    return tuple(
        [_FakeCoil(index, -80000.0) for index in range(20)]
        + [_FakeCoil(index + 20, -10000.0) for index in range(10)]
        + [_FakeCoil(30, 0.0)]
        + [_FakeCoil(index + 31, 0.0) for index in range(20)]
    )


def _baseline_payload():
    return {
        "@class": "SIMSON",
        "simsopt_objs": {
            "BoozerSurface1": {
                "@module": "simsopt.geo.boozersurface",
                "@class": "BoozerSurface",
            }
        },
    }


class ViewerExportTests(unittest.TestCase):
    def test_defaults_match_type_kk_hardware_contract(self):
        self.assertEqual(
            module.DEFAULT_WINDING_R0_M,
            hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )
        self.assertEqual(module.DEFAULT_WINDING_A_M, hc.BANANA_WINDING_MINOR_RADIUS_M)
        self.assertEqual(
            module.DEFAULT_BRACKET_WIDTH_MM,
            hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM,
        )
        self.assertEqual(
            module.DEFAULT_BRACKET_DEPTH_MM,
            hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM,
        )

        config = module.ViewerExportConfig(
            input_path=Path("input.json"),
            output_path=Path("output.viewer.json"),
        )
        self.assertEqual(config.winding_r0_m, hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M)
        self.assertEqual(config.winding_a_m, hc.BANANA_WINDING_MINOR_RADIUS_M)
        self.assertEqual(config.bracket_width_mm, hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM)
        self.assertEqual(config.bracket_depth_mm, hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM)

    def test_baseline_vacuum_lineage_uses_plain_boozer_no_i_contract(self):
        checks = module.baseline_vacuum_lineage_payload(_baseline_payload())
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["finite_i_boozer_surface_count"], 0)
        self.assertEqual(checks["i_field_count"], 0)

        failed = module.baseline_vacuum_lineage_payload(
            {
                "@class": "SIMSON",
                "simsopt_objs": {
                    "BoozerSurfaceFiniteI1": {
                        "@module": "banana_opt.boozer_finite_current",
                        "@class": "BoozerSurfaceFiniteI",
                        "I": 1.0e-6,
                    }
                },
            }
        )
        self.assertFalse(failed["passed"])

    def test_partition_source_coils_selects_banana_scope_for_vacuum_profile(self):
        partition = module.partition_source_coils(
            _coils(),
            contract_family="baseline_original_vacuum",
            coil_scope="banana",
            finite_current_mode=None,
        )

        self.assertEqual(len(partition.tf_coils), 20)
        self.assertEqual(len(partition.render_coils), 10)
        self.assertEqual(partition.finite_current_mode, "vacuum")
        self.assertEqual(partition.render_coils[0].current.get_value(), -10000.0)

    def test_partition_source_coils_accepts_count_equivalent_51_coil_layout(self):
        partition = module.partition_source_coils(
            _count_equivalent_51_coils(),
            contract_family="baseline_original_vacuum",
            coil_scope="banana",
            finite_current_mode=None,
        )

        self.assertEqual(len(partition.tf_coils), 20)
        self.assertEqual(len(partition.render_coils), 10)
        self.assertEqual(
            partition.finite_current_mode,
            "metadata_free_20tf_10banana_1proxy_20vf",
        )
        self.assertEqual(partition.render_coils[0].current.get_value(), -10000.0)

    def test_generated_solid_payload_matches_viewer_artifact_shape(self):
        partition = module.SourcePartition(
            tf_coils=_coils()[:20],
            render_coils=_coils()[20:],
            finite_current_mode="vacuum",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            source_path.write_text(json.dumps(_baseline_payload()), encoding="utf-8")
            config = module.ViewerExportConfig(
                input_path=source_path,
                output_path=Path(tmpdir) / "viewer_artifact.json",
                contract_family="baseline_original_vacuum",
                conversion_mode="generated_solid",
            )

            payload = module.build_artifact_payload(
                config,
                source_path=source_path,
                contract_family="baseline_original_vacuum",
                lineage={"passed": True},
                loaded=module.LoadedSource(
                    biot_savart=SimpleNamespace(coils=_coils()),
                    source_object_class="BoozerSurface",
                ),
                partition=partition,
            )

        self.assertEqual(payload["artifact_type"], module.ARTIFACT_TYPE)
        self.assertEqual(payload["schema_version"], module.SCHEMA_VERSION)
        self.assertEqual(payload["source"]["contract_family"], "baseline_original_vacuum")
        self.assertTrue(payload["source"]["vacuum_lineage_ok"])
        self.assertFalse(payload["source"]["finite_current_lineage_ok"])
        self.assertEqual(payload["tf_current_kA"], -80.0)
        self.assertEqual(len(payload["coils"]), 10)
        self.assertEqual(len(payload["generated_meshes"]), 10)
        self.assertEqual(
            payload["hardware_contract"]["contract_id"],
            hc.TYPE_KK_HARDWARE_CONTRACT_ID,
        )
        self.assertEqual(
            payload["hardware_contract"]["bracket_width_mm"],
            hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM,
        )
        self.assertEqual(
            payload["hardware_contract"]["bracket_depth_mm"],
            hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM,
        )

        first_mesh = payload["generated_meshes"][0]
        self.assertGreater(len(first_mesh["positions"]), 0)
        self.assertEqual(len(first_mesh["indices"]) % 3, 0)
        self.assertGreater(payload["validation"]["generated_mesh_triangles"], 0)

    def test_finite_i_lineage_requires_nonzero_i_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "finite_i.json"
            path.write_text(
                json.dumps(
                    {
                        "@class": "SIMSON",
                        "simsopt_objs": {
                            "BoozerSurfaceFiniteI1": {
                                "@module": "banana_opt.boozer_finite_current",
                                "@class": "BoozerSurfaceFiniteI",
                                "I": 1.0e-6,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            lineage = module.lineage_for_family(path, "simsopt_surrogate_finite_i")
            self.assertTrue(lineage["passed"])
            self.assertEqual(lineage["finite_i_field_count"], 1)


if __name__ == "__main__":
    unittest.main()
