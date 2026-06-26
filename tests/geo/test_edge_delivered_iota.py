import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.edge_delivered_iota import (  # noqa: E402
    EDGE_HELICITY_STATUS_CO,
    EDGE_HELICITY_STATUS_COUNTER,
    EDGE_HELICITY_STATUS_UNKNOWN,
    EDGE_IOTA_MODE_OFF,
    EDGE_IOTA_MODE_REPORT,
    EDGE_IOTA_STATUS_FAIL,
    EDGE_IOTA_STATUS_MISSING_INPUTS,
    EDGE_IOTA_STATUS_PASS,
    EdgeIotaConfig,
    EdgeIotaSample,
    LcfsBoundary,
    TraceIotaResult,
    build_edge_iota_profile,
    edge_iota_missing_payload,
    edge_iota_report_payload,
    edge_radial_labels,
    evaluate_edge_iota_profile,
    load_lcfs_boundary,
    profile_json_payload,
    read_eqdsk,
    validate_trace_samples_against_q,
    write_profile_json,
)


def _eqdsk_float_block(values):
    lines = []
    for offset in range(0, len(values), 5):
        lines.append("".join(f"{value:16.9E}" for value in values[offset : offset + 5]))
    return lines


def _write_synthetic_eqdsk(path: Path) -> None:
    nw = 5
    nh = 5
    rleft = 0.70
    rdim = 0.40
    zmid = 0.0
    zdim = 0.40
    raxis = 0.90
    zaxis = 0.0
    r_grid = rleft + np.linspace(0.0, rdim, nw)
    z_grid = zmid - zdim / 2.0 + np.linspace(0.0, zdim, nh)
    psirz = np.empty((nh, nw), dtype=float)
    for iz, z_value in enumerate(z_grid):
        for ir, r_value in enumerate(r_grid):
            psirz[iz, ir] = (r_value - raxis) ** 2 + (z_value - zaxis) ** 2
    header = f"TEST EQDSK    0 {nw:3d} {nh:3d}"
    scalars = [
        rdim,
        zdim,
        raxis,
        rleft,
        zmid,
        raxis,
        zaxis,
        0.0,
        0.04,
        0.35,
        14265.0,
        0.0,
        0.0,
        raxis,
        0.0,
        zaxis,
        0.0,
        0.04,
        0.0,
        0.0,
    ]
    fpol = [0.315] * nw
    zeros = [0.0] * nw
    qpsi = [1.8, 2.0, 2.4, 2.8, 3.2]
    boundary = [
        (1.05, 0.0),
        (0.90, 0.15),
        (0.75, 0.0),
        (0.90, -0.15),
    ]
    limiter = [(0.70, -0.20), (1.10, -0.20), (1.10, 0.20), (0.70, 0.20)]
    payload = [header]
    payload.extend(_eqdsk_float_block(scalars))
    profile_values = (
        fpol
        + zeros
        + zeros
        + zeros
        + list(psirz.reshape(nw * nh))
        + qpsi
    )
    payload.extend(_eqdsk_float_block(profile_values))
    payload.append(f"{len(boundary):5d}{len(limiter):5d}")
    rz_values = []
    for r_value, z_value in boundary:
        rz_values.extend([r_value, z_value])
    for r_value, z_value in limiter:
        rz_values.extend([r_value, z_value])
    payload.extend(_eqdsk_float_block(rz_values))
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


class EdgeDeliveredIotaContractTests(unittest.TestCase):
    def test_active_mode_requires_explicit_eqdsk_and_lcfs_inputs(self):
        config = EdgeIotaConfig(eqdsk_path=None, lcfs_path=None)

        config.validate(EDGE_IOTA_MODE_OFF)
        with self.assertRaisesRegex(
            ValueError,
            "report edge-iota mode requires --stage2-edge-iota-eqdsk, "
            "--stage2-edge-iota-lcfs",
        ):
            config.validate(EDGE_IOTA_MODE_REPORT)

    def test_edge_radial_labels_are_ordered_edge_band_samples(self):
        np.testing.assert_allclose(
            edge_radial_labels((0.70, 1.0), 4),
            (0.70, 0.80, 0.90, 1.0),
        )

    def test_active_mode_rejects_invalid_gate_thresholds(self):
        invalid_configs = [
            (
                EdgeIotaConfig(
                    eqdsk_path="shot.eqdsk",
                    lcfs_path="lcfs.json",
                    edge_delta_abs_iota_target_min=0.0,
                ),
                "target minimum",
            ),
            (
                EdgeIotaConfig(
                    eqdsk_path="shot.eqdsk",
                    lcfs_path="lcfs.json",
                    edge_survival_fraction_min=1.1,
                ),
                "survival fraction minimum",
            ),
            (
                EdgeIotaConfig(
                    eqdsk_path="shot.eqdsk",
                    lcfs_path="lcfs.json",
                    edge_width_max=0.0,
                ),
                "width maximum",
            ),
        ]

        for config, message in invalid_configs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    config.validate(EDGE_IOTA_MODE_REPORT)

    def test_read_eqdsk_parses_profiles_boundary_and_q_interpolator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eqdsk_path = Path(tmpdir) / "synthetic.eqdsk"
            _write_synthetic_eqdsk(eqdsk_path)

            eqdsk = read_eqdsk(eqdsk_path)
            tokamak = eqdsk.build_axisymmetric_field()

        self.assertEqual(eqdsk.nw, 5)
        self.assertEqual(eqdsk.nh, 5)
        self.assertEqual(eqdsk.rbbbs.size, 4)
        self.assertAlmostEqual(tokamak.q_at(0.90, 0.0), 1.8)
        self.assertAlmostEqual(tokamak.q_at(1.10, 0.0), 3.2)
        metadata = eqdsk.to_metadata()
        self.assertEqual(metadata["q_axis"], 1.8)
        self.assertEqual(metadata["q_edge"], 3.2)

    def test_validate_trace_samples_against_q_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eqdsk_path = Path(tmpdir) / "synthetic.eqdsk"
            _write_synthetic_eqdsk(eqdsk_path)
            tokamak = read_eqdsk(eqdsk_path).build_axisymmetric_field()

            good = validate_trace_samples_against_q(
                tokamak,
                [(0.0, 0.90, 0.0, 1.0 / 1.8)],
                relative_tolerance=1.0e-9,
            )
            bad = validate_trace_samples_against_q(
                tokamak,
                [(0.0, 0.90, 0.0, 0.1)],
                relative_tolerance=1.0e-3,
            )

        self.assertTrue(good.passed)
        self.assertEqual(good.max_relative_error(), 0.0)
        self.assertFalse(bad.passed)
        self.assertGreater(bad.max_relative_error(), 0.1)

    def test_profile_summaries_and_report_payload_use_edge_specific_fields(self):
        samples = [
            EdgeIotaSample.from_iotas(
                radius_label="s075",
                r_over_a=0.75,
                seed_point=(1.0, 0.0),
                iota_tokamak=0.30,
                iota_hybrid=0.43,
                edge_width=0.01,
            ),
            EdgeIotaSample.from_iotas(
                radius_label="s100",
                r_over_a=1.0,
                seed_point=(1.1, 0.0),
                iota_tokamak=0.28,
                iota_hybrid=0.40,
                edge_width=0.02,
            ),
        ]

        profile = build_edge_iota_profile(samples, helicity_sign=1)
        payload = edge_iota_report_payload(
            profile,
            profile_json_path="edge_iota_profile.json",
        )

        self.assertEqual(profile.edge_iota_status, EDGE_IOTA_STATUS_PASS)
        self.assertEqual(profile.edge_helicity_status, EDGE_HELICITY_STATUS_CO)
        self.assertAlmostEqual(profile.edge_delta_abs_iota_min, 0.12)
        self.assertAlmostEqual(profile.edge_delta_signed_iota_mean, 0.125)
        self.assertEqual(payload["EDGE_IOTA_PROFILE_JSON"], "edge_iota_profile.json")
        self.assertEqual(payload["EDGE_IOTA_STATUS"], EDGE_IOTA_STATUS_PASS)
        self.assertNotIn("STAGE2_IOTA_VALUE", payload)

    def test_counter_helicity_fails_even_when_absolute_delta_is_positive(self):
        profile = build_edge_iota_profile(
            [
                EdgeIotaSample.from_iotas(
                    radius_label="counter",
                    r_over_a=0.90,
                    seed_point=(1.0, 0.0),
                    iota_tokamak=-0.30,
                    iota_hybrid=-0.45,
                )
            ],
            helicity_sign=1,
        )

        self.assertEqual(profile.edge_helicity_status, EDGE_HELICITY_STATUS_COUNTER)
        self.assertEqual(profile.edge_iota_status, EDGE_IOTA_STATUS_FAIL)
        self.assertAlmostEqual(profile.edge_delta_abs_iota_min, 0.15)

    def test_current_champion_fixture_is_edge_delivered_iota_failing(self):
        champion_like_samples = [
            EdgeIotaSample.from_iotas(
                radius_label="r/a=0.77",
                r_over_a=0.77,
                seed_point=(1.036203819, -0.000206926028),
                iota_tokamak=-0.414470773278691,
                iota_hybrid=-0.556000342038297,
                converged=True,
                survived=True,
                edge_width=0.12628456548821188,
            ),
            EdgeIotaSample.from_iotas(
                radius_label="r/a=0.92",
                r_over_a=0.92,
                seed_point=(1.056203819, -0.000206926028),
                iota_tokamak=-0.3476048376804627,
                iota_hybrid=-0.045626055519005504,
                converged=False,
                survived=False,
                edge_width=0.4425447313383,
                reason="saved hybrid_signs_2026-06-25 edge trace is broad/non-surviving",
            ),
        ]

        profile = build_edge_iota_profile(champion_like_samples, helicity_sign=1)

        self.assertEqual(profile.edge_iota_status, EDGE_IOTA_STATUS_FAIL)
        self.assertLess(profile.edge_surface_survival_fraction, 1.0)

    def test_profile_json_payload_can_persist_samples_and_config(self):
        config = EdgeIotaConfig(
            eqdsk_path="shot.eqdsk",
            lcfs_path="lcfs.json",
            edge_band=(0.75, 1.0),
            sample_count=2,
            helicity_sign=1,
        )
        profile = build_edge_iota_profile(
            [
                EdgeIotaSample.from_iotas(
                    radius_label="s075",
                    r_over_a=0.75,
                    seed_point=(1.0, 0.0),
                    iota_tokamak=0.30,
                    iota_hybrid=0.41,
                )
            ],
            helicity_sign=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profile.json"
            write_profile_json(profile, output_path, config=config)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["EDGE_IOTA_STATUS"], EDGE_IOTA_STATUS_PASS)
        self.assertEqual(payload["config"]["edge_band"], [0.75, 1.0])
        self.assertEqual(payload["samples"][0]["seed_point_m"], [1.0, 0.0])
        self.assertEqual(
            profile_json_payload(profile, config=config)["config"]["sample_count"],
            2,
        )

    def test_missing_payload_is_schema_complete(self):
        payload = edge_iota_missing_payload()

        self.assertEqual(payload["EDGE_IOTA_STATUS"], EDGE_IOTA_STATUS_MISSING_INPUTS)
        self.assertIn("EDGE_IOTA_PROFILE_JSON", payload)
        self.assertIn("EDGE_HELICITY_STATUS", payload)

    def test_load_lcfs_boundary_accepts_hbt_sidecar_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lcfs_path = Path(tmpdir) / "lcfs.json"
            lcfs_path.write_text(
                json.dumps({"R_m": [1.05, 0.90, 0.75], "Z_m": [0.0, 0.15, 0.0]}),
                encoding="utf-8",
            )

            boundary = load_lcfs_boundary(lcfs_path)

        self.assertAlmostEqual(boundary.minor_radius_from_axis(0.90), 0.15)

    def test_lcfs_boundary_contains_inside_and_boundary_points(self):
        boundary = LcfsBoundary(
            r_m=(1.10, 0.90, 0.70, 0.90),
            z_m=(0.00, 0.20, 0.00, -0.20),
        )

        self.assertTrue(boundary.contains_point(0.90, 0.00))
        self.assertTrue(boundary.contains_point(1.10, 0.00))
        self.assertFalse(boundary.contains_point(1.18, 0.00))

    def test_profile_evaluator_marks_lcfs_exiting_trace_non_surviving(self):
        config = EdgeIotaConfig(
            eqdsk_path="shot.eqdsk",
            lcfs_path="lcfs.json",
            edge_band=(0.75, 1.0),
            sample_count=2,
            helicity_sign=1,
        )
        boundary = LcfsBoundary(
            r_m=(1.10, 0.90, 0.70, 0.90),
            z_m=(0.00, 0.20, 0.00, -0.20),
        )
        traces = [
            TraceIotaResult(iota=0.30, hits_rz_m=((1.00, 0.00), (0.96, 0.02))),
            TraceIotaResult(iota=0.43, hits_rz_m=((1.00, 0.00), (0.98, 0.02))),
            TraceIotaResult(iota=0.30, hits_rz_m=((1.10, 0.00), (1.08, 0.01))),
            TraceIotaResult(iota=0.45, hits_rz_m=((1.10, 0.00), (1.18, 0.00))),
        ]

        with patch(
            "banana_opt.edge_delivered_iota.trace_iota",
            side_effect=traces,
        ):
            profile = evaluate_edge_iota_profile(
                tokamak_field=lambda r, phi, z: (0.0, 1.0, 0.0),
                hybrid_field=lambda r, phi, z: (0.0, 1.0, 0.0),
                axis_r_m=0.90,
                axis_z_m=0.0,
                minor_radius_m=0.20,
                config=config,
                lcfs_boundary=boundary,
            )

        self.assertEqual(profile.edge_iota_status, EDGE_IOTA_STATUS_FAIL)
        self.assertAlmostEqual(profile.edge_surface_survival_fraction, 0.5)
        self.assertFalse(profile.samples[-1].survived)

    def test_profile_evaluator_zero_banana_current_reports_zero_edge_iota_delta(self):
        config = EdgeIotaConfig(
            eqdsk_path="shot.eqdsk",
            lcfs_path="lcfs.json",
            edge_band=(0.75, 1.0),
            sample_count=2,
            helicity_sign=1,
        )
        traces = [
            TraceIotaResult(iota=0.31, hits_rz_m=((1.05, 0.00), (1.04, 0.01))),
            TraceIotaResult(iota=0.31, hits_rz_m=((1.05, 0.00), (1.04, 0.01))),
            TraceIotaResult(iota=0.28, hits_rz_m=((1.10, 0.00), (1.08, 0.01))),
            TraceIotaResult(iota=0.28, hits_rz_m=((1.10, 0.00), (1.08, 0.01))),
        ]

        with patch(
            "banana_opt.edge_delivered_iota.trace_iota",
            side_effect=traces,
        ):
            profile = evaluate_edge_iota_profile(
                tokamak_field=lambda r, phi, z: (0.0, 1.0, 0.0),
                hybrid_field=lambda r, phi, z: (0.0, 1.0, 0.0),
                axis_r_m=0.90,
                axis_z_m=0.0,
                minor_radius_m=0.20,
                config=config,
            )

        self.assertEqual(profile.edge_iota_status, EDGE_IOTA_STATUS_FAIL)
        self.assertEqual(profile.edge_helicity_status, EDGE_HELICITY_STATUS_UNKNOWN)
        self.assertAlmostEqual(profile.edge_delta_abs_iota_min, 0.0)
        self.assertAlmostEqual(profile.edge_delta_abs_iota_p10, 0.0)
        self.assertAlmostEqual(profile.edge_delta_abs_iota_mean, 0.0)
        self.assertAlmostEqual(profile.edge_delta_signed_iota_min, 0.0)
        self.assertAlmostEqual(profile.edge_delta_signed_iota_mean, 0.0)
        for sample in profile.samples:
            self.assertTrue(sample.converged)
            self.assertTrue(sample.survived)
            self.assertAlmostEqual(sample.delta_iota_signed, 0.0)
            self.assertAlmostEqual(sample.delta_abs_iota, 0.0)

    def test_profile_evaluator_records_hybrid_trace_failure_as_failed_sample(self):
        config = EdgeIotaConfig(
            eqdsk_path="shot.eqdsk",
            lcfs_path="lcfs.json",
            edge_band=(0.75, 1.0),
            sample_count=2,
            helicity_sign=1,
        )
        traces = [
            TraceIotaResult(iota=0.30, hits_rz_m=((1.00, 0.00),)),
            RuntimeError("field-line trace did not reach the requested turn count."),
            TraceIotaResult(iota=0.30, hits_rz_m=((1.10, 0.00),)),
            TraceIotaResult(iota=0.45, hits_rz_m=((1.10, 0.00),)),
        ]

        with patch(
            "banana_opt.edge_delivered_iota.trace_iota",
            side_effect=traces,
        ):
            profile = evaluate_edge_iota_profile(
                tokamak_field=lambda r, phi, z: (0.0, 1.0, 0.0),
                hybrid_field=lambda r, phi, z: (0.0, 1.0, 0.0),
                axis_r_m=0.90,
                axis_z_m=0.0,
                minor_radius_m=0.20,
                config=config,
            )

        self.assertEqual(profile.edge_iota_status, EDGE_IOTA_STATUS_FAIL)
        self.assertAlmostEqual(profile.edge_surface_survival_fraction, 0.5)
        self.assertFalse(profile.samples[0].converged)
        self.assertEqual(profile.samples[0].reason, "hybrid_trace_failed")


if __name__ == "__main__":
    unittest.main()
