import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geo._frontier_test_helpers import (
    load_frontier_archive_module,
    load_frontier_campaign_module,
    load_frontier_conditioning_module,
    load_frontier_contracts_module,
    load_frontier_reporting_module,
    load_frontier_runtime_calibration_module,
    load_frontier_scalarization_module,
    load_frontier_solver_checkpoint_module,
    load_incumbents_module,
    make_frontier_archive_member,
)


class FrontierContractTests(unittest.TestCase):
    STAGE2_RESULTS = {
        "PLASMA_SURF_FILENAME": "demo.nc",
        "init_only": False,
        "FINAL_IOTA": 0.15,
        "FINAL_VOLUME": 0.10,
        "NONQS_RATIO": 0.012,
        "BOOZER_RESIDUAL": 0.008,
    }
    HYPERVOLUME_REFERENCE = {
        "iota": 0.15,
        "volume": 0.10,
        "qa_error": 0.012,
        "boozer_residual": 0.008,
    }

    def _runtime_defaults(self, runtime_calibration, **overrides):
        values = {
            "profile_name": "reduced_fixture_v1",
            "requested_num_lanes": None,
            "requested_lane_budget": None,
            "requested_total_budget": None,
            "requested_checkpoint_every": None,
            "requested_early_stop_patience_lanes": None,
            "requested_early_stop_min_certified": None,
            "requested_early_stop_min_hypervolume_gain": None,
        }
        values.update(overrides)
        return runtime_calibration.resolve_frontier_runtime_defaults(**values)

    def test_manifest_and_summary_validate_frozen_v4_contract(self):
        campaign_module = load_frontier_campaign_module()
        contracts = load_frontier_contracts_module()
        reporting = load_frontier_reporting_module()
        scalarization = load_frontier_scalarization_module()
        runtime_calibration = load_frontier_runtime_calibration_module()

        args = campaign_module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
            ]
        )
        runtime_defaults = runtime_calibration.resolve_frontier_runtime_defaults_from_args(args)
        lane_specs = scalarization.generate_multilane_local_specs(
            num_lanes=1,
            iotas_weight=args.iotas_weight,
            frontier_volume_weight=args.frontier_volume_weight,
            res_weight=args.res_weight,
            lane_budget=runtime_defaults.lane_budget,
        )
        stage2_bs_path = Path("/tmp/demo/biot_savart_opt.json")
        manifest = reporting.build_frontier_campaign_manifest(
            args,
            campaign_id="campaign123",
            stage2_bs_path=stage2_bs_path,
            stage2_results_path=stage2_bs_path.with_name("results.json"),
            stage2_results=self.STAGE2_RESULTS,
            lane_specs=lane_specs,
            runtime_defaults=runtime_defaults,
        )
        contracts.validate_frontier_campaign_manifest_payload(manifest)
        self.assertIn("FRONTIER_RUNTIME_CALIBRATION", manifest)
        self.assertIn("FRONTIER_EARLY_STOP_POLICY", manifest)
        self.assertEqual(
            manifest["PARETO_OBJECTIVE_VECTOR"],
            contracts.pareto_objective_vector_contract(),
        )
        self.assertEqual(
            manifest["FRONTIER_EARLY_STOP_POLICY"],
            runtime_calibration.build_frontier_early_stop_policy(runtime_defaults),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "outputs"
            paths = reporting.resolve_frontier_campaign_paths(
                output_root,
                summary_path=output_root / "summary.json",
            )
            summary = reporting.build_frontier_campaign_summary(
                args,
                campaign_id="campaign123",
                stage2_bs_path=stage2_bs_path,
                stage2_results_path=stage2_bs_path.with_name("results.json"),
                stage2_results=self.STAGE2_RESULTS,
                paths=paths,
                lane_specs=lane_specs,
                target_payload=None,
                lane_records=[],
                archive_members=[],
                recommendation_payload=None,
                delta_fn=lambda left, right: None,
                runtime_defaults=runtime_defaults,
                early_stop_status=runtime_calibration.build_initial_frontier_early_stop_status(
                    runtime_defaults=runtime_defaults,
                    archive_members=[],
                ),
            )
        contracts.validate_frontier_campaign_summary_payload(summary)
        self.assertEqual(summary["recommended_member"]["frontier_archive_size"], 0)
        for payload, validator, field in (
            (dict(manifest), contracts.validate_frontier_campaign_manifest_payload, "FRONTIER_ENGINE"),
            (dict(summary), contracts.validate_frontier_campaign_summary_payload, "frontier_engine"),
        ):
            payload[field] = "unsupported_frontier_engine_v0"
            with self.assertRaisesRegex(ValueError, "Unsupported frontier engine"):
                validator(payload)

        for deleted_field in contracts.DELETED_FRONTIER_SUMMARY_FIELDS:
            deleted_field_summary = {**summary, deleted_field: {}}
            with self.assertRaisesRegex(ValueError, "Deleted frontier summary fields"):
                contracts.validate_frontier_campaign_summary_payload(deleted_field_summary)

    def test_lane_contract_validators_reject_deleted_engine(self):
        contracts = load_frontier_contracts_module()
        lane_payload = {
            "schema_version": "frontier_lane_contract_v1",
            "lane_id": "lane_01",
            "campaign_id": "campaign123",
            "engine": "unsupported_frontier_engine_v0",
            "scalarization_type": "weight_schedule_v1",
            "scalarization_params": {},
            "constraint_mode": "frontier_v2_single_lane_contract",
            "optimizer_budget": 10,
            "rng_seed": 0,
            "rerun_contract": {},
        }

        with self.assertRaisesRegex(ValueError, "Unsupported frontier engine"):
            contracts.validate_frontier_lane_contract_payload(lane_payload)
        with self.assertRaisesRegex(ValueError, "Unsupported frontier engine"):
            contracts.validate_frontier_lane_record_payload(
                {
                    **lane_payload,
                    "schema_version": "frontier_lane_record_v1",
                    "status": "completed",
                    "command": ["python"],
                    "weights": {},
                    "lane_budget": 10,
                    "provisional_member_ids": [],
                    "certified_member_ids": [],
                    "final_certified": False,
                }
            )

    def test_initial_early_stop_status_uses_runtime_calibration_ssot(self):
        runtime_calibration = load_frontier_runtime_calibration_module()

        runtime_defaults = self._runtime_defaults(runtime_calibration)

        status = runtime_calibration.build_initial_frontier_early_stop_status(
            runtime_defaults=runtime_defaults,
            archive_members=[],
        )

        self.assertEqual(
            status["policy"],
            runtime_calibration.build_frontier_early_stop_policy(runtime_defaults),
        )
        self.assertEqual(status["best_archive_size"], 0)
        self.assertFalse(status["triggered"])

    def test_update_frontier_early_stop_status_handles_flat_hv_sequence(self):
        archive_module = load_frontier_archive_module()
        runtime_calibration = load_frontier_runtime_calibration_module()
        runtime_defaults = self._runtime_defaults(
            runtime_calibration,
            requested_num_lanes=1,
            requested_lane_budget=10,
            requested_early_stop_patience_lanes=2,
            requested_early_stop_min_certified=1,
            requested_early_stop_min_hypervolume_gain=1.0e-6,
        )
        member = make_frontier_archive_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.0,
            distance_from_seed=0.1,
            results_path="/tmp/results.json",
        )
        status = None
        for _ in range(3):
            status = runtime_calibration.update_frontier_early_stop_status(
                status=status,
                certified_archive_members_list=[member],
                hypervolume_reference=self.HYPERVOLUME_REFERENCE,
                runtime_defaults=runtime_defaults,
            )

        assert status is not None
        self.assertTrue(status["triggered"])
        self.assertEqual(status["reason"], "archive_stagnation")
        self.assertEqual(status["no_improvement_streak"], 2)

    def test_update_frontier_early_stop_status_treats_archive_growth_as_progress(self):
        archive_module = load_frontier_archive_module()
        runtime_calibration = load_frontier_runtime_calibration_module()
        runtime_defaults = self._runtime_defaults(
            runtime_calibration,
            requested_num_lanes=2,
            requested_lane_budget=10,
            requested_early_stop_patience_lanes=1,
            requested_early_stop_min_certified=1,
            requested_early_stop_min_hypervolume_gain=1.0,
        )
        iota_member = make_frontier_archive_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.10,
            qa_error=0.012,
            boozer_residual=0.008,
            soft_search_score=-1.0,
            distance_from_seed=0.1,
            results_path="/tmp/lane_01/results.json",
        )
        volume_member = make_frontier_archive_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.15,
            volume=0.11,
            qa_error=0.012,
            boozer_residual=0.008,
            soft_search_score=-1.1,
            distance_from_seed=0.1,
            results_path="/tmp/lane_02/results.json",
        )

        status = runtime_calibration.update_frontier_early_stop_status(
            status=None,
            certified_archive_members_list=[iota_member],
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
            runtime_defaults=runtime_defaults,
        )
        status = runtime_calibration.update_frontier_early_stop_status(
            status=status,
            certified_archive_members_list=[iota_member, volume_member],
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
            runtime_defaults=runtime_defaults,
        )

        self.assertFalse(status["triggered"])
        self.assertEqual(status["no_improvement_streak"], 0)
        self.assertEqual(status["best_archive_size"], 2)
        self.assertEqual(status["best_hypervolume"], 0.0)

    def test_epsilon_thresholds_round_trip(self):
        contracts = load_frontier_contracts_module()

        thresholds = contracts.EpsilonThresholds(qa_max=0.011, boozer_max=0.007)
        payload = thresholds.as_payload()
        from_contract = contracts.EpsilonThresholds.from_rerun_contract(
            {"scalarization_params": payload},
            require_all=True,
        )
        from_goal_config = contracts.EpsilonThresholds.from_goal_config(
            SimpleNamespace(
                epsilon_constraint_qa_max=0.011,
                epsilon_constraint_boozer_max=0.007,
            )
        )

        self.assertEqual(
            payload,
            {
                "epsilon_constraint_qa_max": 0.011,
                "epsilon_constraint_boozer_max": 0.007,
            },
        )
        self.assertEqual(from_contract, thresholds)
        self.assertEqual(from_goal_config, thresholds)

    def test_archive_validator_rejects_provisional_member_in_final_archive(self):
        archive_module = load_frontier_archive_module()
        contracts = load_frontier_contracts_module()

        member = make_frontier_archive_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.17,
            volume=0.105,
            qa_error=0.011,
            boozer_residual=0.007,
            soft_search_score=-1.0,
            distance_from_seed=0.1,
            results_path="/tmp/results.json",
        )
        payload = archive_module.serialize_frontier_archive(
            [member],
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )
        payload["members"][0]["archive_state"] = "provisional"

        with self.assertRaisesRegex(ValueError, "Expected frontier archive state"):
            contracts.validate_frontier_archive_payload(payload)

    def test_frontier_conditioning_gate_requires_seed_and_first_accepted(self):
        conditioning = load_frontier_conditioning_module()

        seed_report = conditioning.build_frontier_conditioning_report(
            {
                "J_QS_objective": 0.02,
                "J_Boozer_objective": 0.01,
                "J_iota": 0.03,
                "J_volume": 0.025,
                "frontier_trust_penalty": 0.005,
                "frontier_epsilon_penalty": 0.004,
                "dJ_QS_objective": np.array([0.2, 0.1]),
                "dJ_Boozer_objective": np.array([0.1, 0.1]),
                "dJ_iota": np.array([0.25, 0.15]),
                "dJ_volume": np.array([0.2, 0.2]),
            },
            sample_label="seed",
        )
        first_accepted_report = conditioning.build_frontier_conditioning_report(
            {
                "J_QS_objective": 0.015,
                "J_Boozer_objective": 0.012,
                "J_iota": 0.025,
                "J_volume": 0.020,
                "frontier_trust_penalty": 0.006,
                "frontier_epsilon_penalty": 0.003,
                "dJ_QS_objective": np.array([0.18, 0.12]),
                "dJ_Boozer_objective": np.array([0.11, 0.09]),
                "dJ_iota": np.array([0.22, 0.14]),
                "dJ_volume": np.array([0.18, 0.16]),
            },
            sample_label="first_accepted",
        )

        gate = conditioning.build_frontier_conditioning_gate(
            seed_report=seed_report,
            first_accepted_report=first_accepted_report,
        )

        self.assertTrue(gate["usable_scale_ok"])
        self.assertEqual(gate["sample_ok"]["seed"], True)
        self.assertEqual(gate["sample_ok"]["first_accepted"], True)

    def test_runtime_defaults_use_explicit_calibration_profile(self):
        runtime_calibration = load_frontier_runtime_calibration_module()

        resolved = self._runtime_defaults(
            runtime_calibration,
            requested_checkpoint_every=0,
        )

        self.assertEqual(resolved.calibration_profile.profile_name, "reduced_fixture_v1")
        self.assertEqual(resolved.lane_budget, 300)
        self.assertEqual(resolved.total_budget, 900)
        self.assertEqual(resolved.checkpoint_every, 5)
        self.assertEqual(resolved.early_stop_patience_lanes, 2)

    def test_runtime_defaults_reject_negative_early_stop_inputs(self):
        runtime_calibration = load_frontier_runtime_calibration_module()
        for kwarg in (
            {"requested_early_stop_patience_lanes": -1},
            {"requested_early_stop_min_certified": -1},
            {"requested_early_stop_min_hypervolume_gain": -1.0e-9},
            {"requested_early_stop_min_hypervolume_gain": float("nan")},
            {"requested_early_stop_min_hypervolume_gain": float("inf")},
        ):
            with self.assertRaises(ValueError):
                self._runtime_defaults(runtime_calibration, **kwarg)

    def test_solver_checkpoint_round_trip_preserves_conditioning_reports(self):
        checkpoint_module = load_frontier_solver_checkpoint_module()
        incumbents = load_incumbents_module()

        incumbent = incumbents.SingleStageIncumbentState(
            x=np.array([1.0, 2.0]),
            surface_state={"surface": [1.0]},
            objective_total=1.25,
            objective_grad=np.array([0.1, 0.2]),
            search_eval={"total": 1.25, "grad": [0.1, 0.2]},
            surface_status={"ok": True},
            search_surface_status={"ok": True},
            accepted_hardware_status={"success": True},
            topology_gate_status={"success": True},
        )
        seed_report = {
            "schema_version": "frontier_conditioning_v1",
            "sample_label": "seed",
            "usable_scale_ok": True,
        }
        first_accepted_report = {
            "schema_version": "frontier_conditioning_v1",
            "sample_label": "first_accepted",
            "usable_scale_ok": True,
        }

        payload = checkpoint_module.build_solver_checkpoint_payload(
            goal_mode="frontier",
            constraint_method="penalty",
            stage2_bs_path="/tmp/seed.json",
            requested_maxiter=300,
            runtime_maxiter=120,
            accepted_iterations=5,
            accepted_boozer_stage="initial",
            accepted_incumbent=incumbent,
            best_accepted_incumbent=incumbent,
            best_accepted_stage="initial",
            best_accepted_metric=-1.0,
            best_feasible_incumbent=incumbent,
            best_feasible_stage="initial",
            best_feasible_metric=-1.0,
            out_dir_iter="/tmp/out",
            run_counters={"it": 6},
            conditioning_seed_report=seed_report,
            conditioning_first_accepted_report=first_accepted_report,
        )

        self.assertEqual(payload["conditioning_seed_report"], seed_report)
        self.assertEqual(
            payload["conditioning_first_accepted_report"],
            first_accepted_report,
        )
