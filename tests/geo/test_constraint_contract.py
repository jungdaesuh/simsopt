import importlib
import sys
import unittest
from pathlib import Path


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_constraint_contract_module():
    return importlib.import_module("banana_opt.constraint_contract")


def load_hardware_contracts_module():
    return importlib.import_module("banana_opt.hardware_contracts")


def load_artifact_contracts_module():
    return importlib.import_module("banana_opt.artifact_contracts")


class ConstraintContractResolverTests(unittest.TestCase):
    def test_hardware_default_contract_matches_ssot_values(self):
        module = load_constraint_contract_module()
        hc = load_hardware_contracts_module()

        contract = module.hardware_default_contract()

        self.assertEqual(
            contract["VACUUM_VESSEL_MAJOR_RADIUS_M"],
            hc.VACUUM_VESSEL_MAJOR_RADIUS_M,
        )
        self.assertEqual(
            contract["VACUUM_VESSEL_MINOR_RADIUS_M"],
            hc.VACUUM_VESSEL_MINOR_RADIUS_M,
        )
        self.assertEqual(
            contract["BANANA_WINDING_SURFACE_MAJOR_RADIUS_M"],
            hc.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )
        self.assertEqual(contract["TF_CURRENT_A"], hc.TF_CURRENT_CW_DEFAULT_A)
        self.assertEqual(
            contract["BANANA_CURRENT_MAX_A"],
            hc.BANANA_CURRENT_HARD_LIMIT_A,
        )
        self.assertEqual(contract["COIL_LENGTH_TARGET_M"], hc.COIL_LENGTH_TARGET_M)
        self.assertEqual(contract["CC_THRESHOLD"], hc.COIL_COIL_MIN_DIST_M)
        self.assertEqual(
            contract["COIL_PLASMA_MIN_DIST_M"],
            hc.COIL_PLASMA_MIN_DIST_M,
        )
        self.assertEqual(
            contract["PLASMA_VESSEL_MIN_DIST_M"],
            hc.PLASMA_VESSEL_MIN_DIST_M,
        )
        self.assertEqual(contract["CURVATURE_THRESHOLD"], hc.MAX_CURVATURE_INV_M)
        self.assertEqual(
            contract["banana_surf_radius"],
            hc.BANANA_WINDING_MINOR_RADIUS_M,
        )
        self.assertEqual(contract["TARGET_LCFS_MAX_MAJOR_RADIUS_M"], 0.92)
        self.assertEqual(contract["TARGET_LCFS_MAX_MINOR_RADIUS_M"], 0.15)

    def test_resolve_with_no_inputs_returns_hardware_defaults(self):
        module = load_constraint_contract_module()

        contract, trace = module.resolve_constraint_contract()

        self.assertEqual(dict(contract), module.hardware_default_contract())
        for source in trace.values():
            self.assertEqual(source, module.CONSTRAINT_SOURCE_HARDWARE)

    def test_ladder_precedence_cli_beats_spec_json_beats_profile(self):
        module = load_constraint_contract_module()

        contract, trace = module.resolve_constraint_contract(
            profile={"CC_THRESHOLD": 0.07},
            spec_json={"CC_THRESHOLD": 0.08},
            cli_overrides={"CC_THRESHOLD": 0.09},
        )

        self.assertEqual(contract["CC_THRESHOLD"], 0.09)
        self.assertEqual(trace["CC_THRESHOLD"], module.CONSTRAINT_SOURCE_CLI)

    def test_spec_json_wins_over_profile_when_cli_silent(self):
        module = load_constraint_contract_module()

        contract, trace = module.resolve_constraint_contract(
            profile={"CC_THRESHOLD": 0.07},
            spec_json={"CC_THRESHOLD": 0.08},
        )

        self.assertEqual(contract["CC_THRESHOLD"], 0.08)
        self.assertEqual(trace["CC_THRESHOLD"], module.CONSTRAINT_SOURCE_SPEC_JSON)

    def test_fixed_geometry_cannot_be_overridden_via_ladder(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "Fixed-geometry field"):
            module.resolve_constraint_contract(
                profile={"VACUUM_VESSEL_MAJOR_RADIUS_M": 0.5},
            )
        with self.assertRaisesRegex(ValueError, "Fixed-geometry field"):
            module.resolve_constraint_contract(
                cli_overrides={"BANANA_WINDING_SURFACE_MAJOR_RADIUS_M": 1.0},
            )

    def test_offspec_major_radius_requires_explicit_acceptance(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "vacuum-vessel major radius"):
            module.resolve_constraint_contract(
                offspec_major_radius_m=0.5,
                accept_offspec_major_radius=False,
            )

    def test_offspec_major_radius_is_accepted_with_flag(self):
        module = load_constraint_contract_module()

        contract, trace = module.resolve_constraint_contract(
            offspec_major_radius_m=0.85,
            accept_offspec_major_radius=True,
        )

        self.assertEqual(contract["VACUUM_VESSEL_MAJOR_RADIUS_M"], 0.85)
        self.assertEqual(contract["BANANA_WINDING_SURFACE_MAJOR_RADIUS_M"], 0.85)
        self.assertEqual(
            trace["VACUUM_VESSEL_MAJOR_RADIUS_M"],
            module.CONSTRAINT_SOURCE_OFFSPEC_MAJOR_RADIUS,
        )

    def test_target_lcfs_ceiling_rejects_oversize_major_radius(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "target LCFS major radius"):
            module.resolve_constraint_contract(
                cli_overrides={"TARGET_LCFS_MAX_MAJOR_RADIUS_M": 1.5},
            )

    def test_target_lcfs_ceiling_rejects_oversize_minor_radius(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "target LCFS minor radius"):
            module.resolve_constraint_contract(
                cli_overrides={"TARGET_LCFS_MAX_MINOR_RADIUS_M": 0.5},
            )

    def test_target_lcfs_ceiling_defaults_are_distinct_from_vessel(self):
        module = load_constraint_contract_module()

        contract, _trace = module.resolve_constraint_contract()

        self.assertNotEqual(
            contract["TARGET_LCFS_MAX_MAJOR_RADIUS_M"],
            contract["VACUUM_VESSEL_MAJOR_RADIUS_M"],
        )
        self.assertNotEqual(
            contract["TARGET_LCFS_MAX_MINOR_RADIUS_M"],
            contract["VACUUM_VESSEL_MINOR_RADIUS_M"],
        )

    def test_length_target_allows_values_between_target_and_hard_limit(self):
        module = load_constraint_contract_module()

        contract, _trace = module.resolve_constraint_contract(
            cli_overrides={"COIL_LENGTH_TARGET_M": 1.95},
        )

        self.assertEqual(contract["COIL_LENGTH_TARGET_M"], 1.95)

    def test_length_target_rejects_values_above_hardware_limit(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "COIL_LENGTH_TARGET_M exceeds the hardware limit"):
            module.resolve_constraint_contract(
                cli_overrides={"COIL_LENGTH_TARGET_M": 2.0001},
            )

    def test_allow_offspec_engineering_accepts_raised_length_and_banana_limits(self):
        module = load_constraint_contract_module()

        contract, _trace = module.resolve_constraint_contract(
            cli_overrides={
                "COIL_LENGTH_TARGET_M": 3.0,
                "BANANA_CURRENT_MAX_A": 20000.0,
                "CURVATURE_THRESHOLD": 150.0,
            },
            allow_offspec_engineering=True,
        )

        self.assertEqual(contract["COIL_LENGTH_TARGET_M"], 3.0)
        self.assertEqual(contract["BANANA_CURRENT_MAX_A"], 20000.0)
        self.assertEqual(contract["CURVATURE_THRESHOLD"], 150.0)

    def test_curvature_threshold_above_hardware_limit_requires_offspec_flag(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(
            ValueError,
            "CURVATURE_THRESHOLD exceeds the hardware limit",
        ):
            module.resolve_constraint_contract(
                cli_overrides={"CURVATURE_THRESHOLD": 150.0},
            )

    def test_engineering_offspec_fields_reports_current_length_and_curvature(self):
        module = load_constraint_contract_module()

        offspec = module.engineering_offspec_fields(
            {
                "banana_current_max_A": 20000.0,
                "length_target": 3.0,
                "curvature_threshold": 150.0,
            }
        )

        self.assertEqual(
            offspec,
            (
                "BANANA_CURRENT_MAX_A",
                "COIL_LENGTH_TARGET_M",
                "CURVATURE_THRESHOLD",
            ),
        )

    def test_tf_current_limit_rejects_zero_positive_and_over_limit_current(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "TF coil current"):
            module.resolve_constraint_contract(
                cli_overrides={"TF_CURRENT_A": 0.0},
            )
        with self.assertRaisesRegex(ValueError, "TF coil current"):
            module.resolve_constraint_contract(
                cli_overrides={"TF_CURRENT_A": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "TF coil current"):
            module.resolve_constraint_contract(
                cli_overrides={"TF_CURRENT_A": -8.1e4},
            )

    def test_unknown_field_in_any_layer_raises(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            module.resolve_constraint_contract(
                profile={"definitely_not_a_field": 1.0},
            )


class ConstraintContractHashTests(unittest.TestCase):
    def test_hash_is_deterministic_for_equal_contracts(self):
        module = load_constraint_contract_module()

        contract_a, _ = module.resolve_constraint_contract()
        contract_b, _ = module.resolve_constraint_contract()

        self.assertEqual(
            module.compute_constraint_contract_hash(contract_a),
            module.compute_constraint_contract_hash(contract_b),
        )

    def test_hash_changes_when_engineering_field_changes(self):
        module = load_constraint_contract_module()

        baseline_contract, _ = module.resolve_constraint_contract()
        altered_contract, _ = module.resolve_constraint_contract(
            cli_overrides={"CC_THRESHOLD": 0.06},
        )

        self.assertNotEqual(
            module.compute_constraint_contract_hash(baseline_contract),
            module.compute_constraint_contract_hash(altered_contract),
        )

    def test_hash_independent_of_dict_insertion_order(self):
        module = load_constraint_contract_module()

        contract_ordered, _ = module.resolve_constraint_contract()
        contract_shuffled = dict(reversed(list(contract_ordered.items())))

        self.assertEqual(
            module.compute_constraint_contract_hash(contract_ordered),
            module.compute_constraint_contract_hash(contract_shuffled),
        )

    def test_hash_refuses_partial_contract(self):
        module = load_constraint_contract_module()
        partial = {"TF_CURRENT_A": -80000.0}

        with self.assertRaisesRegex(ValueError, "partial constraint contract"):
            module.compute_constraint_contract_hash(partial)


class ConstraintContractMetadataTests(unittest.TestCase):
    def test_build_metadata_includes_all_required_keys(self):
        module = load_constraint_contract_module()

        contract, trace = module.resolve_constraint_contract()
        metadata = module.build_constraint_metadata(
            contract,
            profile_name="standard_80ka",
            override_reason="cli:cc_threshold",
            trace=trace,
        )

        self.assertEqual(metadata["CONSTRAINT_PROFILE"], "standard_80ka")
        self.assertEqual(metadata["OVERRIDE_REASON"], "cli:cc_threshold")
        self.assertEqual(
            metadata["CONTRACT_SCHEMA_VERSION"],
            module.CONSTRAINT_SCHEMA_VERSION,
        )
        self.assertEqual(
            metadata["CONTRACT_HASH"],
            module.compute_constraint_contract_hash(contract),
        )
        self.assertEqual(
            dict(metadata["EFFECTIVE_VALUES"]),
            {key: float(contract[key]) for key in contract},
        )
        self.assertIn("CONSTRAINT_PROVENANCE", metadata)

    def test_merge_override_reason_deduplicates_and_uses_semicolon_separator(self):
        module = load_constraint_contract_module()

        self.assertEqual(
            module.merge_override_reason(
                "cli:cc_threshold",
                "allow_offspec_engineering_constraints",
            ),
            "cli:cc_threshold;allow_offspec_engineering_constraints",
        )
        self.assertEqual(
            module.merge_override_reason(
                "cli:cc_threshold;allow_offspec_engineering_constraints",
                "allow_offspec_engineering_constraints",
            ),
            "cli:cc_threshold;allow_offspec_engineering_constraints",
        )


class ConstraintContractWireNamesTests(unittest.TestCase):
    def test_wire_name_resolver_accepts_lowercase_and_uppercase(self):
        module = load_constraint_contract_module()

        contract_lower, _ = module.resolve_constraint_contract_from_wire_names(
            cli_overrides={"tf_current_A": -70000.0},
        )
        contract_upper, _ = module.resolve_constraint_contract_from_wire_names(
            cli_overrides={"TF_CURRENT_A": -70000.0},
        )

        self.assertEqual(contract_lower["TF_CURRENT_A"], -70000.0)
        self.assertEqual(contract_upper["TF_CURRENT_A"], -70000.0)

    def test_wire_name_resolver_silently_drops_fixed_geometry_aliases(self):
        module = load_constraint_contract_module()
        hc = load_hardware_contracts_module()

        contract, _ = module.resolve_constraint_contract_from_wire_names(
            profile={
                "major_radius": hc.VACUUM_VESSEL_MAJOR_RADIUS_M,
                "tf_current_A": -70000.0,
            },
        )

        self.assertEqual(
            contract["VACUUM_VESSEL_MAJOR_RADIUS_M"],
            hc.VACUUM_VESSEL_MAJOR_RADIUS_M,
        )
        self.assertEqual(contract["TF_CURRENT_A"], -70000.0)

    def test_wire_name_resolver_rejects_canonical_fixed_geometry_override_keys(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "Fixed-geometry field"):
            module.resolve_constraint_contract_from_wire_names(
                cli_overrides={"VACUUM_VESSEL_MAJOR_RADIUS_M": 0.976},
            )

    def test_wire_name_resolver_rejects_unknown_typos(self):
        module = load_constraint_contract_module()

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            module.resolve_constraint_contract_from_wire_names(
                cli_overrides={"cc_threhsold": 0.09},
            )


class ArtifactContractsSchemaVersionTests(unittest.TestCase):
    def test_current_schema_version_accepted_silently(self):
        import warnings
        module = load_artifact_contracts_module()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            module.validate_constraint_contract_schema_version(
                Path("/tmp/fake_results.json"),
                {
                    "CONTRACT_SCHEMA_VERSION": (
                        module.CURRENT_CONSTRAINT_CONTRACT_SCHEMA_VERSION
                    ),
                    "CONTRACT_HASH": "deadbeef",
                },
                owner_label="unit-test",
            )

    def test_legacy_schema_version_rejected(self):
        module = load_artifact_contracts_module()

        with self.assertRaisesRegex(ValueError, "legacy constraint contract schema"):
            module.validate_constraint_contract_schema_version(
                Path("/tmp/fake_results.json"),
                {
                    "CONTRACT_SCHEMA_VERSION": (
                        module.LEGACY_CONSTRAINT_CONTRACT_SCHEMA_VERSION
                    ),
                },
                owner_label="unit-test",
            )

    def test_missing_schema_version_rejected(self):
        module = load_artifact_contracts_module()

        with self.assertRaisesRegex(ValueError, "legacy constraint contract schema"):
            module.validate_constraint_contract_schema_version(
                Path("/tmp/fake_results.json"),
                {},
                owner_label="unit-test",
            )

    def test_future_schema_version_rejected(self):
        module = load_artifact_contracts_module()

        future_version = module.CURRENT_CONSTRAINT_CONTRACT_SCHEMA_VERSION + 1
        with self.assertRaisesRegex(ValueError, "incompatible with the current schema"):
            module.validate_constraint_contract_schema_version(
                Path("/tmp/fake_results.json"),
                {"CONTRACT_SCHEMA_VERSION": future_version},
                owner_label="unit-test",
            )

    def test_validate_stage2_artifact_metadata_rejects_future_schema(self):
        module = load_artifact_contracts_module()

        with self.assertRaisesRegex(ValueError, "incompatible with the current schema"):
            module.validate_stage2_artifact_metadata(
                Path("/tmp/fake_results.json"),
                {"CONTRACT_SCHEMA_VERSION": 999},
                expected_metadata={},
                owner_label="unit-test",
                experiment_family="unit-test",
            )


class ArtifactContractsLegacyUpgradeTests(unittest.TestCase):
    def test_legacy_upgrade_injects_schema_version_zero_and_no_synthetic_hash(self):
        module = load_artifact_contracts_module()

        upgraded = module.upgrade_legacy_stage2_artifact_results({})

        self.assertEqual(
            upgraded["CONTRACT_SCHEMA_VERSION"],
            module.LEGACY_CONSTRAINT_CONTRACT_SCHEMA_VERSION,
        )
        self.assertNotIn("CONTRACT_HASH", upgraded)
        self.assertIsNone(upgraded["CONSTRAINT_PROFILE"])
        self.assertIsNone(upgraded["EFFECTIVE_VALUES"])
        self.assertIsNone(upgraded["OVERRIDE_REASON"])

    def test_legacy_upgrade_preserves_prior_constraint_metadata(self):
        module = load_artifact_contracts_module()

        upgraded = module.upgrade_legacy_stage2_artifact_results({
            "CONTRACT_SCHEMA_VERSION": 1,
            "CONTRACT_HASH": "abc123",
            "CONSTRAINT_PROFILE": "standard_80ka",
            "EFFECTIVE_VALUES": {"TF_CURRENT_A": -80000.0},
            "OVERRIDE_REASON": None,
        })

        self.assertEqual(upgraded["CONTRACT_SCHEMA_VERSION"], 1)
        self.assertEqual(upgraded["CONTRACT_HASH"], "abc123")
        self.assertEqual(upgraded["CONSTRAINT_PROFILE"], "standard_80ka")
        self.assertEqual(
            upgraded["EFFECTIVE_VALUES"],
            {"TF_CURRENT_A": -80000.0},
        )
        self.assertIsNone(upgraded["OVERRIDE_REASON"])


if __name__ == "__main__":
    unittest.main()
