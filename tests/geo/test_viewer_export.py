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
        # Winding radii default to None: the audit uses the realized geometry from
        # the sibling results.json (falling back to the contract defaults), rather
        # than pinning the on-spec contract values regardless of the artifact.
        self.assertIsNone(config.winding_r0_m)
        self.assertIsNone(config.winding_a_m)
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

    def test_rendered_coil_payloads_labels_tf_and_banana_by_role(self):
        coils = _coils()
        partition = module.SourcePartition(
            tf_coils=coils[:20],
            render_coils=coils,
            finite_current_mode="vacuum",
        )

        payloads = module.rendered_coil_payloads(partition)

        self.assertEqual(
            [p["id"] for p in payloads[:20]],
            [f"tf_{index + 1:03d}" for index in range(20)],
        )
        self.assertEqual(
            [p["id"] for p in payloads[20:]],
            [f"banana_{index + 1:03d}" for index in range(10)],
        )
        # Role comes from tf_coils membership, not the render order or current sign.
        self.assertEqual(payloads[0]["current_A"], -80000.0)
        self.assertEqual(payloads[20]["current_A"], -10000.0)

    def test_rendered_coil_payloads_banana_scope_has_no_tf_ids(self):
        coils = _coils()
        partition = module.SourcePartition(
            tf_coils=coils[:20],
            render_coils=coils[20:],
            finite_current_mode="vacuum",
        )

        payloads = module.rendered_coil_payloads(partition)

        self.assertEqual(
            [p["id"] for p in payloads],
            [f"banana_{index + 1:03d}" for index in range(10)],
        )
        self.assertFalse(any(str(p["id"]).startswith("tf_") for p in payloads))

    def test_rendered_coil_payloads_counters_are_per_role_and_order_independent(self):
        coils = _coils()
        # Interleave TF and banana coils so a single global counter would mislabel:
        # role is decided by tf_coils membership, each role numbered independently.
        partition = module.SourcePartition(
            tf_coils=(coils[0], coils[1]),
            render_coils=(coils[0], coils[20], coils[1], coils[21]),
            finite_current_mode="vacuum",
        )

        payloads = module.rendered_coil_payloads(partition)

        self.assertEqual(
            [p["id"] for p in payloads],
            ["tf_001", "banana_001", "tf_002", "banana_002"],
        )

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

            winding = module.resolve_realized_winding_surface(
                source_path,
                override_r0_m=config.winding_r0_m,
                override_a_m=config.winding_a_m,
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
                winding=winding,
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

    def _write_results_sibling(self, source_path, *, r0_m, a_m):
        source_path.write_text(json.dumps(_baseline_payload()), encoding="utf-8")
        results = {
            module.RESULTS_WINDING_R0_KEY: r0_m,
            module.RESULTS_WINDING_A_KEY: a_m,
        }
        (source_path.parent / module.STAGE2_RESULTS_FILENAME).write_text(
            json.dumps(results), encoding="utf-8"
        )

    def test_resolve_winding_falls_back_to_contract_when_no_results_sibling(self):
        # Legacy/metadata-free artifact: no sibling results.json, so the audit
        # uses the on-spec Type KK contract defaults (byte-identical to before).
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            source_path.write_text(json.dumps(_baseline_payload()), encoding="utf-8")
            winding = module.resolve_realized_winding_surface(
                source_path, override_r0_m=None, override_a_m=None
            )
        self.assertEqual(winding.r0_m, hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M)
        self.assertEqual(winding.a_m, hc.BANANA_WINDING_MINOR_RADIUS_M)

    def test_resolve_winding_uses_realized_offspec_minor_radius_from_results(self):
        # Off-spec remap (a=0.120, R0 unchanged at 0.903): the realized radii are
        # read from the sibling results.json, not the fixed contract.
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            self._write_results_sibling(source_path, r0_m=0.903, a_m=0.120)
            winding = module.resolve_realized_winding_surface(
                source_path, override_r0_m=None, override_a_m=None
            )
        self.assertEqual(winding.r0_m, 0.903)
        self.assertEqual(winding.a_m, 0.120)

    def test_resolve_winding_onspec_results_matches_contract_default(self):
        # On-spec a=0.142 artifact whose results.json records the spec values:
        # resolution returns exactly the contract defaults (no drift).
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            self._write_results_sibling(
                source_path,
                r0_m=hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
                a_m=hc.BANANA_WINDING_MINOR_RADIUS_M,
            )
            winding = module.resolve_realized_winding_surface(
                source_path, override_r0_m=None, override_a_m=None
            )
        self.assertEqual(winding.r0_m, hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M)
        self.assertEqual(winding.a_m, hc.BANANA_WINDING_MINOR_RADIUS_M)

    def test_explicit_winding_override_wins_over_realized_results(self):
        # An explicit override (investigation run) takes precedence over the
        # realized results, per radius.
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            self._write_results_sibling(source_path, r0_m=0.903, a_m=0.120)
            winding = module.resolve_realized_winding_surface(
                source_path, override_r0_m=0.95, override_a_m=None
            )
        self.assertEqual(winding.r0_m, 0.95)
        self.assertEqual(winding.a_m, 0.120)

    def test_payload_winding_surface_reflects_offspec_minor_radius(self):
        # The emitted artifact's winding_surface (what the oracle reports as
        # winding_a_m) reflects the realized off-spec minor radius.
        partition = module.SourcePartition(
            tf_coils=_coils()[:20],
            render_coils=_coils()[20:],
            finite_current_mode="vacuum",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "surf_opt_boozer_surface.json"
            self._write_results_sibling(source_path, r0_m=0.903, a_m=0.120)
            config = module.ViewerExportConfig(
                input_path=source_path,
                output_path=Path(tmpdir) / "viewer_artifact.json",
                contract_family="baseline_original_vacuum",
            )
            winding = module.resolve_realized_winding_surface(
                source_path,
                override_r0_m=config.winding_r0_m,
                override_a_m=config.winding_a_m,
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
                winding=winding,
            )
        self.assertEqual(payload["winding_surface"], {"R0_m": 0.903, "a_m": 0.120})

    def test_generated_mesh_geometry_tracks_realized_major_radius(self):
        # The swept U-channel frame's radial direction is built from the realized
        # winding major radius; a different R0 produces different mesh positions,
        # proving R0 is wired into geometry (not just metadata).
        partition = module.SourcePartition(
            tf_coils=(),
            render_coils=_coils()[20:21],
            finite_current_mode="vacuum",
        )
        meshes_spec = module.generated_mesh_payloads(
            partition.render_coils,
            winding_r0_m=hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            bracket_width_mm=hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM,
            bracket_depth_mm=hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM,
        )
        meshes_other = module.generated_mesh_payloads(
            partition.render_coils,
            winding_r0_m=0.5,
            bracket_width_mm=hc.TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM,
            bracket_depth_mm=hc.TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM,
        )
        self.assertNotEqual(
            meshes_spec[0]["positions"], meshes_other[0]["positions"]
        )

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
