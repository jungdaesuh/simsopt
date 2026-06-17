"""Finite-build pack partitioning in banana_opt.viewer_export.

Covers the 2026-06-11 fix: viewer_export must partition finite-build PACK
payloads (TF coils + banana filament packs) either from the Stage-2
``results.json`` sibling metadata or from explicit
``--num-tf-coils``/``--filaments-per-banana`` flags, reducing each pack to its
centerline curve. Before the fix every such payload raised ``ValueError:
Metadata-free input cannot be partitioned uniquely`` (and the pack CLI flags
did not exist), so these tests fail wholesale against the old behavior.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt import viewer_export as module  # noqa: E402


class _FakeCurve:
    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)

    def gamma(self):
        return self._points


class _FakeCurrent:
    def __init__(self, value):
        self._value = float(value)

    def get_value(self):
        return self._value


class _FakeCoil:
    def __init__(self, points, current):
        self.curve = _FakeCurve(points)
        self.current = _FakeCurrent(current)


_CENTERLINE = np.asarray(
    [
        [0.90, 0.00, 0.00],
        [0.90, 0.10, 0.00],
        [0.80, 0.10, 0.05],
        [0.80, 0.00, 0.05],
    ],
    dtype=float,
)

# Constant filament offsets that sum to exactly zero in floating point, so the
# pack centerline equals the offset-free curve exactly.
_PACK_OFFSETS = (
    np.asarray([0.001, 0.0, 0.0]),
    np.asarray([-0.001, 0.0, 0.0]),
    np.asarray([0.0, 0.002, 0.0]),
    np.asarray([0.0, -0.002, 0.0]),
)
_FILAMENTS_PER_BANANA = len(_PACK_OFFSETS)


def _tf_coil(index):
    return _FakeCoil(_CENTERLINE + np.asarray([0.0, 0.0, 1.0 + 0.01 * index]), -80000.0)


def _filament_pack(banana_index, filament_current_A=-250.0):
    pack_centerline = _CENTERLINE + np.asarray([0.0, 0.0, 0.10 * banana_index])
    return [
        _FakeCoil(pack_centerline + offset, filament_current_A)
        for offset in _PACK_OFFSETS
    ]


def _pack_payload_coils(num_tf=2, num_bananas=2):
    coils = [_tf_coil(index) for index in range(num_tf)]
    for banana_index in range(num_bananas):
        coils.extend(_filament_pack(banana_index))
    return tuple(coils)


def _stage2_pack_results(num_tf=2, num_bananas=2, num_proxy=0, num_vf=0):
    filament_coils = num_bananas * _FILAMENTS_PER_BANANA
    cursor_proxy = num_tf + filament_coils
    return {
        "COIL_GROUPS": [
            {"role": "tf", "start": 0, "count": num_tf},
            {"role": "banana", "start": num_tf, "count": filament_coils},
            {"role": "proxy", "start": cursor_proxy, "count": num_proxy},
            {"role": "vf", "start": cursor_proxy + num_proxy, "count": num_vf},
        ],
        "FINITE_BUILD_ENABLED": True,
        "FINITEBUILD_FILAMENTS_PER_BANANA": _FILAMENTS_PER_BANANA,
        "FINITEBUILD_NUMFILAMENTS_N": 2,
        "FINITEBUILD_NUMFILAMENTS_B": 2,
    }


def _write_sibling_results(tmpdir, results):
    results_path = Path(tmpdir) / module.STAGE2_RESULTS_FILENAME
    results_path.write_text(json.dumps(results), encoding="utf-8")
    return Path(tmpdir) / "surf_opt_boozer_surface.json"


class PackFlagsPartitionTests(unittest.TestCase):
    def test_explicit_pack_flags_reduce_packs_to_exact_centerlines(self):
        coils = _pack_payload_coils(num_tf=2, num_bananas=2)

        partition = module.partition_source_coils(
            coils,
            contract_family="simsopt_surrogate_vacuum",
            coil_scope="banana",
            finite_current_mode=None,
            num_tf_coils=2,
            filaments_per_banana=_FILAMENTS_PER_BANANA,
        )

        self.assertEqual(len(partition.tf_coils), 2)
        self.assertEqual(len(partition.render_coils), 2)
        self.assertEqual(
            partition.finite_current_mode,
            f"pack_flags_2tf_8banana_{_FILAMENTS_PER_BANANA}fil",
        )
        for banana_index, reduced in enumerate(partition.render_coils):
            expected = _CENTERLINE + np.asarray([0.0, 0.0, 0.10 * banana_index])
            np.testing.assert_array_equal(
                reduced.curve.gamma(),
                expected,
                err_msg="pack mean must recover the exact centerline",
            )
            self.assertEqual(reduced.current.get_value(), -1000.0)

    def test_pack_flags_reject_non_divisible_filament_count(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            module.resolved_partition_spec(
                total_coils=2 + 9,
                contract_family="simsopt_surrogate_vacuum",
                finite_current_mode=None,
                num_tf_coils=2,
                filaments_per_banana=_FILAMENTS_PER_BANANA,
            )

    def test_pack_flags_reject_tf_count_consuming_all_coils(self):
        with self.assertRaisesRegex(ValueError, "leaves no banana coils"):
            module.resolved_partition_spec(
                total_coils=10,
                contract_family="simsopt_surrogate_vacuum",
                finite_current_mode=None,
                num_tf_coils=10,
                filaments_per_banana=2,
            )

    def test_lone_pack_flag_rejected_at_partition(self):
        with self.assertRaisesRegex(ValueError, "must be passed together"):
            module.resolved_partition_spec(
                total_coils=10,
                contract_family="simsopt_surrogate_vacuum",
                finite_current_mode=None,
                filaments_per_banana=2,
            )


class SiblingResultsMetadataTests(unittest.TestCase):
    def test_sibling_results_metadata_partitions_pack_payload(self):
        coils = _pack_payload_coils(num_tf=2, num_bananas=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = _write_sibling_results(tmpdir, _stage2_pack_results())

            partition = module.partition_source_coils(
                coils,
                contract_family="simsopt_surrogate_vacuum",
                coil_scope="banana",
                finite_current_mode=None,
                source_path=source_path,
            )

        self.assertEqual(len(partition.tf_coils), 2)
        self.assertEqual(len(partition.render_coils), 2)
        self.assertEqual(
            partition.finite_current_mode,
            f"stage2_results_metadata_2tf_8banana_0proxy_0vf_{_FILAMENTS_PER_BANANA}fil",
        )
        np.testing.assert_array_equal(
            partition.render_coils[0].curve.gamma(), _CENTERLINE
        )
        self.assertEqual(partition.render_coils[0].current.get_value(), -1000.0)

    def test_all_scope_renders_tf_plus_reduced_bananas_plus_trailing_groups(self):
        proxy_coil = _FakeCoil(_CENTERLINE + np.asarray([0.0, 0.0, 2.0]), 100.0)
        vf_coils = [
            _FakeCoil(_CENTERLINE + np.asarray([0.0, 0.0, 3.0 + index]), 50.0)
            for index in range(2)
        ]
        coils = _pack_payload_coils(num_tf=2, num_bananas=2) + (proxy_coil, *vf_coils)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = _write_sibling_results(
                tmpdir, _stage2_pack_results(num_proxy=1, num_vf=2)
            )

            partition = module.partition_source_coils(
                coils,
                contract_family="simsopt_surrogate_vacuum",
                coil_scope="all",
                finite_current_mode=None,
                source_path=source_path,
            )

        # 2 TF + 2 reduced bananas + 1 proxy + 2 VF.
        self.assertEqual(len(partition.render_coils), 7)
        self.assertIs(partition.render_coils[0], coils[0])
        self.assertIsInstance(partition.render_coils[2], module.PackCenterlineCoil)
        self.assertIs(partition.render_coils[4], proxy_coil)

    def test_missing_sibling_preserves_metadata_free_error(self):
        coils = _pack_payload_coils(num_tf=2, num_bananas=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                ValueError, "cannot be partitioned uniquely"
            ):
                module.partition_source_coils(
                    coils,
                    contract_family="simsopt_surrogate_vacuum",
                    coil_scope="banana",
                    finite_current_mode=None,
                    source_path=Path(tmpdir) / "surf_opt_boozer_surface.json",
                )

    def test_sibling_manifest_contradicting_coil_count_raises(self):
        coils = _pack_payload_coils(num_tf=2, num_bananas=2)[:-1]
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = _write_sibling_results(tmpdir, _stage2_pack_results())
            with self.assertRaisesRegex(ValueError, "manifest expects"):
                module.partition_source_coils(
                    coils,
                    contract_family="simsopt_surrogate_vacuum",
                    coil_scope="banana",
                    finite_current_mode=None,
                    source_path=source_path,
                )

    def test_banana_count_not_divisible_by_recorded_filaments_raises(self):
        results = _stage2_pack_results()
        results["FINITEBUILD_FILAMENTS_PER_BANANA"] = 3
        results["FINITEBUILD_NUMFILAMENTS_N"] = 1
        results["FINITEBUILD_NUMFILAMENTS_B"] = 3
        coils = _pack_payload_coils(num_tf=2, num_bananas=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = _write_sibling_results(tmpdir, results)
            with self.assertRaisesRegex(ValueError, "not divisible"):
                module.partition_source_coils(
                    coils,
                    contract_family="simsopt_surrogate_vacuum",
                    coil_scope="banana",
                    finite_current_mode=None,
                    source_path=source_path,
                )


class Stage2FilamentsPerBananaTests(unittest.TestCase):
    def test_thin_artifact_defaults_to_one_filament(self):
        self.assertEqual(module.stage2_filaments_per_banana({}), 1)
        self.assertEqual(
            module.stage2_filaments_per_banana({"FINITE_BUILD_ENABLED": False}), 1
        )

    def test_recorded_count_used_and_cross_checked(self):
        self.assertEqual(
            module.stage2_filaments_per_banana(
                {
                    "FINITE_BUILD_ENABLED": True,
                    "FINITEBUILD_FILAMENTS_PER_BANANA": 14,
                    "FINITEBUILD_NUMFILAMENTS_N": 2,
                    "FINITEBUILD_NUMFILAMENTS_B": 7,
                }
            ),
            14,
        )
        self.assertEqual(
            module.stage2_filaments_per_banana(
                {
                    "FINITE_BUILD_ENABLED": True,
                    "FINITEBUILD_NUMFILAMENTS_N": 2,
                    "FINITEBUILD_NUMFILAMENTS_B": 7,
                }
            ),
            14,
        )

    def test_inconsistent_filament_metadata_raises(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            module.stage2_filaments_per_banana(
                {
                    "FINITE_BUILD_ENABLED": True,
                    "FINITEBUILD_FILAMENTS_PER_BANANA": 14,
                    "FINITEBUILD_NUMFILAMENTS_N": 2,
                    "FINITEBUILD_NUMFILAMENTS_B": 6,
                }
            )

    def test_enabled_without_filament_counts_raises(self):
        with self.assertRaisesRegex(ValueError, "lack"):
            module.stage2_filaments_per_banana({"FINITE_BUILD_ENABLED": True})


class StockProfileRegressionTests(unittest.TestCase):
    """The pre-fix resolution order for registered profiles must be untouched."""

    def _stock_coils(self, total):
        return tuple(
            _FakeCoil(_CENTERLINE + np.asarray([0.0, 0.0, 0.01 * index]), -10000.0)
            for index in range(total)
        )

    def test_vacuum_30_coil_resolution_unchanged(self):
        spec = module.resolved_partition_spec(
            total_coils=30,
            contract_family="baseline_original_vacuum",
            finite_current_mode=None,
        )
        self.assertEqual(spec.finite_current_mode, "vacuum")
        self.assertEqual(spec.filaments_per_banana, 1)

    def test_count_equivalent_51_coil_resolution_unchanged(self):
        spec = module.resolved_partition_spec(
            total_coils=51,
            contract_family="simsopt_surrogate_finite_i",
            finite_current_mode=None,
        )
        self.assertEqual(
            spec.finite_current_mode, "metadata_free_20tf_10banana_1proxy_20vf"
        )
        self.assertEqual(spec.filaments_per_banana, 1)

    def test_stock_profile_never_reads_sibling_results(self):
        coils = self._stock_coils(30)
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / module.STAGE2_RESULTS_FILENAME
            results_path.write_text("NOT VALID JSON{", encoding="utf-8")

            partition = module.partition_source_coils(
                coils,
                contract_family="baseline_original_vacuum",
                coil_scope="banana",
                finite_current_mode=None,
                source_path=Path(tmpdir) / "surf_opt_boozer_surface.json",
            )

        self.assertEqual(partition.finite_current_mode, "vacuum")
        self.assertEqual(len(partition.render_coils), 10)


class RealSimsoptPackReductionTests(unittest.TestCase):
    """Mean-of-filaments equals the master centerline for real simsopt packs."""

    def test_multifilament_symmetry_pack_reduces_to_master_centerline(self):
        from simsopt.field import Current, ScaledCurrent, coils_via_symmetries
        from simsopt.geo import CurveXYZFourier, create_multifilament_grid

        master = CurveXYZFourier(np.linspace(0, 1, 32, endpoint=False), order=2)
        master.set("xc(0)", 0.9)
        master.set("xc(1)", 0.05)
        master.set("ys(1)", 0.05)
        master.set("zs(1)", 0.03)
        master.set("zc(2)", 0.01)

        numfilaments_n, numfilaments_b = 2, 7
        filaments = create_multifilament_grid(
            master,
            numfilaments_n,
            numfilaments_b,
            gapsize_n=0.0099,
            gapsize_b=0.0066,
            rotation_order=1,
            frame="centroid",
        )
        nfilaments = numfilaments_n * numfilaments_b
        net_current = ScaledCurrent(Current(1.0), -15837.2861029015)
        filament_current = ScaledCurrent(net_current, 1.0 / nfilaments)
        nfp, stellsym = 5, True
        coils = coils_via_symmetries(
            filaments, [filament_current] * nfilaments, nfp, stellsym
        )
        self.assertEqual(len(coils), nfilaments * nfp * 2)

        reduced = module.reduce_filament_packs_to_centerlines(coils, nfilaments)

        self.assertEqual(len(reduced), nfp * 2)
        # Identity symmetry image: mean of the pack is the master centerline.
        np.testing.assert_allclose(
            reduced[0].curve.gamma(),
            master.gamma(),
            rtol=0.0,
            atol=1.0e-12,
            err_msg="pack mean must recover the master centerline",
        )
        # Every image carries the net pack current with the symmetry flip sign.
        self.assertAlmostEqual(
            reduced[0].current.get_value(), -15837.2861029015, places=6
        )
        self.assertAlmostEqual(
            reduced[1].current.get_value(), 15837.2861029015, places=6
        )
        # Rotated images: mean equals the rotated master centerline.
        rotated_pack_mean = reduced[2].curve.gamma()
        angle = 2.0 * np.pi / nfp
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(
            rotated_pack_mean,
            master.gamma() @ rotation.T,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_reduce_rejects_non_divisible_pack(self):
        coils = _filament_pack(0)[:3]
        with self.assertRaisesRegex(ValueError, "Cannot reduce 3 filament coils"):
            module.reduce_filament_packs_to_centerlines(coils, 4)


class PackFlagConfigValidationTests(unittest.TestCase):
    def _config(self, **overrides):
        return module.ViewerExportConfig(
            input_path=Path("input.json"),
            output_path=Path("output.viewer.json"),
            **overrides,
        )

    def test_pack_flags_conflict_with_finite_current_mode(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            module.validate_config(
                self._config(
                    finite_current_mode="vacuum",
                    num_tf_coils=20,
                    filaments_per_banana=14,
                )
            )

    def test_pack_flags_must_be_passed_together(self):
        with self.assertRaisesRegex(ValueError, "together"):
            module.validate_config(self._config(num_tf_coils=20))
        with self.assertRaisesRegex(ValueError, "together"):
            module.validate_config(self._config(filaments_per_banana=14))

    def test_pack_flag_bounds(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            module.validate_config(
                self._config(num_tf_coils=20, filaments_per_banana=0)
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            module.validate_config(
                self._config(num_tf_coils=-1, filaments_per_banana=14)
            )

    def test_parser_wires_pack_flags_into_config(self):
        parser = module.build_parser()
        args = parser.parse_args(
            [
                "--input",
                "in.json",
                "--output",
                "out.viewer.json",
                "--num-tf-coils",
                "20",
                "--filaments-per-banana",
                "14",
            ]
        )
        config = module.config_from_args(args)
        self.assertEqual(config.num_tf_coils, 20)
        self.assertEqual(config.filaments_per_banana, 14)

        default_args = parser.parse_args(
            ["--input", "in.json", "--output", "out.viewer.json"]
        )
        default_config = module.config_from_args(default_args)
        self.assertIsNone(default_config.num_tf_coils)
        self.assertIsNone(default_config.filaments_per_banana)


if __name__ == "__main__":
    unittest.main()
