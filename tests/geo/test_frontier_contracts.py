import importlib
import importlib.util
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
FRONTIER_CAMPAIGN_PATH = EXAMPLE_ROOT / "run_single_stage_frontier_campaign.py"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_frontier_campaign_module():
    return load_module(FRONTIER_CAMPAIGN_PATH, "run_single_stage_frontier_campaign")


def load_frontier_archive_module():
    return importlib.import_module("banana_opt.frontier_archive")


def load_frontier_conditioning_module():
    return importlib.import_module("banana_opt.frontier_conditioning")


def load_frontier_contracts_module():
    return importlib.import_module("banana_opt.frontier_contracts")


def load_frontier_reporting_module():
    return importlib.import_module("banana_opt.frontier_campaign_reporting")


def load_frontier_runtime_calibration_module():
    return importlib.import_module("banana_opt.frontier_runtime_calibration")


def load_frontier_solver_checkpoint_module():
    return importlib.import_module("banana_opt.frontier_solver_checkpoint")


def load_incumbents_module():
    return importlib.import_module("banana_opt.incumbents")


class FrontierContractTests(unittest.TestCase):
    def test_manifest_and_summary_validate_frozen_v4_contract(self):
        campaign_module = load_frontier_campaign_module()
        contracts = load_frontier_contracts_module()
        reporting = load_frontier_reporting_module()
        runtime_calibration = load_frontier_runtime_calibration_module()

        args = campaign_module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
            ]
        )
        runtime_defaults = runtime_calibration.resolve_frontier_runtime_defaults(
            profile_name=args.frontier_runtime_calibration_profile,
            requested_num_lanes=args.frontier_num_lanes,
            requested_lane_budget=args.frontier_lane_budget,
            requested_total_budget=args.frontier_total_budget,
            requested_checkpoint_every=args.checkpoint_every,
            requested_early_stop_patience_lanes=args.frontier_early_stop_patience_lanes,
            requested_early_stop_min_certified=args.frontier_early_stop_min_certified,
            requested_early_stop_min_hypervolume_gain=args.frontier_early_stop_min_hypervolume_gain,
        )
        lane_specs = campaign_module.generate_multilane_local_specs(
            num_lanes=1,
            iotas_weight=args.iotas_weight,
            frontier_volume_weight=args.frontier_volume_weight,
            res_weight=args.res_weight,
            lane_budget=runtime_defaults.lane_budget,
        )
        stage2_bs_path = Path("/tmp/demo/biot_savart_opt.json")
        stage2_results = {
            "PLASMA_SURF_FILENAME": "demo.nc",
            "init_only": False,
            "FINAL_IOTA": 0.15,
            "FINAL_VOLUME": 0.10,
            "NONQS_RATIO": 0.012,
            "BOOZER_RESIDUAL": 0.008,
        }
        manifest = reporting.build_frontier_campaign_manifest(
            args,
            campaign_id="campaign123",
            stage2_bs_path=stage2_bs_path,
            stage2_results_path=stage2_bs_path.with_name("results.json"),
            stage2_results=stage2_results,
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
                stage2_results=stage2_results,
                paths=paths,
                lane_specs=lane_specs,
                target_payload=None,
                lane_records=[],
                archive_members=[],
                recommendation_payload=None,
                delta_fn=lambda left, right: None,
                runtime_defaults=runtime_defaults,
                early_stop_status={
                    "policy": runtime_calibration.build_frontier_early_stop_policy(
                        runtime_defaults
                    ),
                    "triggered": False,
                    "reason": None,
                    "no_improvement_streak": 0,
                    "best_hypervolume": None,
                    "best_archive_size": 0,
                    "stopped_after_lane_id": None,
                },
            )
        contracts.validate_frontier_campaign_summary_payload(summary)
        self.assertEqual(
            summary["recommended_member"]["frontier_archive_size"],
            0,
        )

    def test_initial_early_stop_status_uses_runtime_calibration_ssot(self):
        runtime_calibration = load_frontier_runtime_calibration_module()

        runtime_defaults = runtime_calibration.resolve_frontier_runtime_defaults(
            profile_name="reduced_fixture_v1",
            requested_num_lanes=None,
            requested_lane_budget=None,
            requested_total_budget=None,
            requested_checkpoint_every=None,
            requested_early_stop_patience_lanes=None,
            requested_early_stop_min_certified=None,
            requested_early_stop_min_hypervolume_gain=None,
        )

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

    def test_summary_validator_accepts_optional_nsga3_fields(self):
        campaign_module = load_frontier_campaign_module()
        contracts = load_frontier_contracts_module()
        reporting = load_frontier_reporting_module()
        runtime_calibration = load_frontier_runtime_calibration_module()

        args = campaign_module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-engine",
                "nsga3",
                "--frontier-reference-mode",
                "achievement_chebyshev_full_simplex_v1",
            ]
        )
        runtime_defaults = runtime_calibration.resolve_frontier_runtime_defaults(
            profile_name=args.frontier_runtime_calibration_profile,
            requested_num_lanes=args.frontier_num_lanes,
            requested_lane_budget=args.frontier_lane_budget,
            requested_total_budget=args.frontier_total_budget,
            requested_checkpoint_every=args.checkpoint_every,
            requested_early_stop_patience_lanes=args.frontier_early_stop_patience_lanes,
            requested_early_stop_min_certified=args.frontier_early_stop_min_certified,
            requested_early_stop_min_hypervolume_gain=args.frontier_early_stop_min_hypervolume_gain,
        )
        lane_specs = campaign_module.generate_frontier_lane_specs(
            reference_mode=args.frontier_reference_mode,
            num_lanes=1,
            iotas_weight=args.iotas_weight,
            frontier_volume_weight=args.frontier_volume_weight,
            res_weight=args.res_weight,
            lane_budget=runtime_defaults.lane_budget,
            stage2_results={
                "FINAL_IOTA": 0.15,
                "FINAL_VOLUME": 0.10,
                "NONQS_RATIO": 0.012,
                "BOOZER_RESIDUAL": 0.008,
            },
            reference_points_file=None,
            epsilon_spec_file=None,
            full_simplex_partitions=1,
        )
        stage2_bs_path = Path("/tmp/demo/biot_savart_opt.json")
        stage2_results = {
            "PLASMA_SURF_FILENAME": "demo.nc",
            "init_only": False,
            "FINAL_IOTA": 0.15,
            "FINAL_VOLUME": 0.10,
            "NONQS_RATIO": 0.012,
            "BOOZER_RESIDUAL": 0.008,
        }
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
                stage2_results=stage2_results,
                paths=paths,
                lane_specs=lane_specs,
                target_payload=None,
                lane_records=[],
                archive_members=[],
                recommendation_payload=None,
                delta_fn=lambda left, right: None,
                runtime_defaults=runtime_defaults,
                early_stop_status={
                    "policy": runtime_calibration.build_frontier_early_stop_policy(
                        runtime_defaults
                    ),
                    "triggered": False,
                    "reason": None,
                    "no_improvement_streak": 0,
                    "best_hypervolume": None,
                    "best_archive_size": 0,
                    "stopped_after_lane_id": None,
                },
            )
        summary["frontier_generation_history"] = [
            {
                "generation": 1,
                "population_size": 3,
                "feasible_count": 2,
                "archive_size": 1,
                "archive_growth": 1,
                "cv_min": 0.0,
                "cv_mean": 0.1,
                "cv_max": 0.2,
                "failure_histogram": {"evaluator_candidate_valid": 2},
                "cache_hits": 3,
                "cache_misses": 3,
                "hypervolume": 1.0e-4,
            }
        ]
        summary["frontier_hypervolume_history"] = [
            {
                "lane_id": "generation_0001",
                "status": "completed",
                "archive_size": 1,
                "hypervolume": 1.0e-4,
            }
        ]
        summary["frontier_engine_stats"] = {
            "population_size": 3,
            "generations": 1,
            "archive_size": 1,
            "cache_hits": 3,
            "cache_misses": 3,
        }
        summary["frontier_evaluator_spec"] = {
            "schema_version": "single_stage_frontier_evaluator_spec_v1",
            "run_identity": "nsga3-test",
        }
        summary["frontier_evaluator_spec_path"] = "/tmp/demo/evaluator_spec.json"
        summary["frontier_population_checkpoint_path"] = "/tmp/demo/population_checkpoint.json"
        summary["frontier_generation_history_path"] = "/tmp/demo/generation_history.json"

        contracts.validate_frontier_campaign_summary_payload(summary)

    def test_archive_validator_rejects_provisional_member_in_final_archive(self):
        archive_module = load_frontier_archive_module()
        contracts = load_frontier_contracts_module()

        member = archive_module.FrontierArchiveMember(
            member_id="campaign:lane_01",
            lane_id="lane_01",
            campaign_id="campaign",
            archive_state="certified",
            dominance_signature={},
            objective_metrics={
                "iota": 0.17,
                "volume": 0.105,
                "qa_error": 0.011,
                "boozer_residual": 0.007,
            },
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
            constraint_metrics={},
            hard_certification_ok=True,
            soft_search_score=-1.0,
            distance_from_seed=0.1,
            hypervolume_contribution=None,
            recommendation_flags={},
            rerun_contract={},
            result_source="final",
            results_path="/tmp/results.json",
            termination_reason="ok",
            success=True,
        )
        payload = archive_module.serialize_frontier_archive(
            [member],
            hypervolume_reference={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
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

        resolved = runtime_calibration.resolve_frontier_runtime_defaults(
            profile_name="reduced_fixture_v1",
            requested_num_lanes=None,
            requested_lane_budget=None,
            requested_total_budget=None,
            requested_checkpoint_every=0,
            requested_early_stop_patience_lanes=None,
            requested_early_stop_min_certified=None,
            requested_early_stop_min_hypervolume_gain=None,
        )

        self.assertEqual(resolved.calibration_profile.profile_name, "reduced_fixture_v1")
        self.assertEqual(resolved.lane_budget, 300)
        self.assertEqual(resolved.total_budget, 900)
        self.assertEqual(resolved.checkpoint_every, 5)
        self.assertEqual(resolved.early_stop_patience_lanes, 2)

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
