import importlib.util
import sys
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
COIL_GROUPS_PATH = EXAMPLES_ROOT / "banana_opt" / "coil_groups.py"


def _load_coil_groups_module():
    module_name = f"coil_groups_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, COIL_GROUPS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


coil_groups = _load_coil_groups_module()


def _make_sentinel_coils(count):
    return [f"coil_{index}" for index in range(count)]


class BuildContiguousManifestTests(unittest.TestCase):
    def test_sets_cumulative_starts(self):
        manifest = coil_groups.build_contiguous_manifest(
            num_tf_coils=20,
            num_banana_coils=10,
            num_proxy_coils=1,
            num_vf_coils=4,
        )
        roles = [group.role for group in manifest.groups]
        starts = [group.start for group in manifest.groups]
        counts = [group.count for group in manifest.groups]
        self.assertEqual(roles, ["tf", "banana", "proxy", "vf"])
        self.assertEqual(starts, [0, 20, 30, 31])
        self.assertEqual(counts, [20, 10, 1, 4])
        self.assertEqual(manifest.total(), 35)

    def test_allows_zero_proxy_and_vf(self):
        manifest = coil_groups.build_contiguous_manifest(
            num_tf_coils=20,
            num_banana_coils=10,
            num_proxy_coils=0,
            num_vf_coils=0,
        )
        self.assertEqual(manifest.total(), 30)
        self.assertEqual(manifest.count_for_role("proxy"), 0)
        self.assertEqual(manifest.count_for_role("vf"), 0)

    def test_rejects_negative_count(self):
        with self.assertRaises(ValueError):
            coil_groups.build_contiguous_manifest(
                num_tf_coils=-1,
                num_banana_coils=10,
                num_proxy_coils=0,
                num_vf_coils=0,
            )


class JsonRoundTripTests(unittest.TestCase):
    def test_round_trip(self):
        original = coil_groups.build_contiguous_manifest(
            num_tf_coils=20,
            num_banana_coils=10,
            num_proxy_coils=1,
            num_vf_coils=4,
        )
        payload = original.to_json_payload()
        reparsed = coil_groups.CoilGroupsManifest.from_json_payload(payload)
        self.assertEqual(original, reparsed)

    def test_from_json_rejects_non_list(self):
        with self.assertRaises(TypeError):
            coil_groups.CoilGroupsManifest.from_json_payload({"role": "tf"})

    def test_from_json_rejects_missing_key(self):
        with self.assertRaises(KeyError):
            coil_groups.CoilGroupsManifest.from_json_payload(
                [{"role": "tf", "start": 0}]
            )

    def test_from_json_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            coil_groups.CoilGroupsManifest.from_json_payload(
                [{"role": "tf", "start": -1, "count": 3}]
            )


class PartitionAndValidateTests(unittest.TestCase):
    def test_partition_returns_role_scoped_slices(self):
        manifest = coil_groups.build_contiguous_manifest(
            num_tf_coils=3,
            num_banana_coils=2,
            num_proxy_coils=1,
            num_vf_coils=2,
        )
        coils = _make_sentinel_coils(8)
        partitions = coil_groups.partition_coils_by_manifest(coils, manifest)
        self.assertEqual(partitions["tf"], ("coil_0", "coil_1", "coil_2"))
        self.assertEqual(partitions["banana"], ("coil_3", "coil_4"))
        self.assertEqual(partitions["proxy"], ("coil_5",))
        self.assertEqual(partitions["vf"], ("coil_6", "coil_7"))

    def test_validate_size_mismatch_raises(self):
        manifest = coil_groups.build_contiguous_manifest(
            num_tf_coils=3,
            num_banana_coils=2,
            num_proxy_coils=0,
            num_vf_coils=0,
        )
        with self.assertRaises(ValueError):
            coil_groups.validate_manifest_against_coils(
                manifest, total_loaded_coils=4
            )

    def test_validate_non_contiguous_manifest_raises(self):
        non_contiguous = coil_groups.CoilGroupsManifest(
            groups=(
                coil_groups.CoilGroup(role="tf", start=0, count=3),
                coil_groups.CoilGroup(role="banana", start=5, count=2),
            )
        )
        with self.assertRaises(ValueError):
            coil_groups.validate_manifest_against_coils(
                non_contiguous, total_loaded_coils=7
            )


class ReadManifestFromResultsTests(unittest.TestCase):
    def test_returns_none_when_key_missing(self):
        self.assertIsNone(coil_groups.read_manifest_from_results({}))

    def test_returns_none_when_value_empty(self):
        self.assertIsNone(
            coil_groups.read_manifest_from_results({"COIL_GROUPS": None})
        )
        self.assertIsNone(
            coil_groups.read_manifest_from_results({"COIL_GROUPS": ""})
        )

    def test_parses_payload(self):
        payload = [
            {"role": "tf", "start": 0, "count": 2},
            {"role": "banana", "start": 2, "count": 1},
        ]
        manifest = coil_groups.read_manifest_from_results(
            {"COIL_GROUPS": payload}
        )
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.total(), 3)


class InferManifestFromLegacyTests(unittest.TestCase):
    def test_uses_recorded_counts(self):
        manifest = coil_groups.infer_manifest_from_legacy_counts(
            {
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 10,
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 4,
            },
            total_loaded_coils=35,
        )
        self.assertEqual(manifest.count_for_role("tf"), 20)
        self.assertEqual(manifest.count_for_role("banana"), 10)
        self.assertEqual(manifest.count_for_role("proxy"), 1)
        self.assertEqual(manifest.count_for_role("vf"), 4)

    def test_defaults_proxy_and_vf_to_zero(self):
        manifest = coil_groups.infer_manifest_from_legacy_counts(
            {"NUM_TF_COILS": 20, "NUM_BANANA_COILS": 10},
            total_loaded_coils=30,
        )
        self.assertEqual(manifest.count_for_role("proxy"), 0)
        self.assertEqual(manifest.count_for_role("vf"), 0)

    def test_uses_requested_tf_when_absent(self):
        manifest = coil_groups.infer_manifest_from_legacy_counts(
            {"NUM_BANANA_COILS": 10},
            total_loaded_coils=30,
            requested_num_tf_coils=20,
        )
        self.assertEqual(manifest.count_for_role("tf"), 20)
        self.assertEqual(manifest.count_for_role("banana"), 10)

    def test_infers_banana_from_total_when_absent(self):
        manifest = coil_groups.infer_manifest_from_legacy_counts(
            {"NUM_TF_COILS": 20, "NUM_PROXY_COILS": 1, "NUM_VF_COILS": 4},
            total_loaded_coils=35,
        )
        self.assertEqual(manifest.count_for_role("banana"), 10)

    def test_raises_without_tf_signal(self):
        with self.assertRaises(ValueError):
            coil_groups.infer_manifest_from_legacy_counts(
                {"NUM_BANANA_COILS": 10}, total_loaded_coils=10
            )

    def test_raises_on_negative_banana(self):
        with self.assertRaises(ValueError):
            coil_groups.infer_manifest_from_legacy_counts(
                {"NUM_TF_COILS": 30, "NUM_PROXY_COILS": 0, "NUM_VF_COILS": 0},
                total_loaded_coils=10,
            )


class ResolveManifestTests(unittest.TestCase):
    def test_prefers_payload_over_legacy(self):
        payload = [
            {"role": "tf", "start": 0, "count": 20},
            {"role": "banana", "start": 20, "count": 10},
            {"role": "proxy", "start": 30, "count": 1},
            {"role": "vf", "start": 31, "count": 4},
        ]
        resolution = coil_groups.resolve_manifest(
            {
                "COIL_GROUPS": payload,
                # Legacy counts are deliberately inconsistent; the payload wins.
                "NUM_TF_COILS": 99,
                "NUM_BANANA_COILS": 99,
            },
            total_loaded_coils=35,
        )
        self.assertFalse(resolution.is_legacy_inferred)
        self.assertEqual(resolution.manifest.count_for_role("tf"), 20)

    def test_falls_back_to_legacy(self):
        resolution = coil_groups.resolve_manifest(
            {
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 10,
                "NUM_PROXY_COILS": 0,
                "NUM_VF_COILS": 0,
            },
            total_loaded_coils=30,
        )
        self.assertTrue(resolution.is_legacy_inferred)
        self.assertEqual(resolution.manifest.count_for_role("banana"), 10)

    def test_raises_on_tf_count_mismatch_in_payload(self):
        payload = [
            {"role": "tf", "start": 0, "count": 20},
            {"role": "banana", "start": 20, "count": 10},
        ]
        with self.assertRaises(ValueError):
            coil_groups.resolve_manifest(
                {"COIL_GROUPS": payload},
                total_loaded_coils=30,
                requested_num_tf_coils=22,
            )

    def test_raises_on_tf_count_mismatch_in_legacy(self):
        with self.assertRaises(ValueError):
            coil_groups.resolve_manifest(
                {"NUM_TF_COILS": 20, "NUM_BANANA_COILS": 10},
                total_loaded_coils=30,
                requested_num_tf_coils=22,
            )


if __name__ == "__main__":
    unittest.main()
