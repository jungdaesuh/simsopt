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


def load_frontier_archive_module():
    return importlib.import_module("banana_opt.frontier_archive")


def load_frontier_dominance_module():
    return importlib.import_module("banana_opt.frontier_dominance")


def load_frontier_engine_base_module():
    return importlib.import_module("banana_opt.frontier_engine_base")


def load_frontier_recommendation_module():
    return importlib.import_module("banana_opt.frontier_recommendation")


class FrontierArchiveTests(unittest.TestCase):
    def _make_member(
        self,
        archive_module,
        *,
        member_id: str,
        iota: float,
        volume: float,
        qa_error: float,
        boozer_residual: float,
        soft_search_score: float,
        reference_metrics: dict[str, float] | None = None,
    ):
        return archive_module.FrontierArchiveMember(
            member_id=member_id,
            lane_id=member_id.split(":")[-1],
            campaign_id="campaign",
            archive_state="certified",
            dominance_signature={},
            objective_metrics={
                "iota": iota,
                "volume": volume,
                "qa_error": qa_error,
                "boozer_residual": boozer_residual,
            },
            reference_metrics=(
                {
                    "iota": 0.15,
                    "volume": 0.10,
                    "qa_error": 0.012,
                    "boozer_residual": 0.008,
                }
                if reference_metrics is None
                else dict(reference_metrics)
            ),
            constraint_metrics={},
            hard_certification_ok=True,
            soft_search_score=soft_search_score,
            distance_from_seed=0.0,
            hypervolume_contribution=None,
            recommendation_flags={},
            rerun_contract={},
            result_source="final",
            results_path=f"/tmp/{member_id}.json",
            termination_reason="ok",
            success=True,
        )

    def test_dominates_respects_objective_directions(self):
        module = load_frontier_dominance_module()

        better = {
            "iota": 0.18,
            "volume": 0.11,
            "qa_error": 0.010,
            "boozer_residual": 0.007,
        }
        worse = {
            "iota": 0.17,
            "volume": 0.10,
            "qa_error": 0.011,
            "boozer_residual": 0.008,
        }

        self.assertTrue(module.dominates(better, worse))
        self.assertFalse(module.dominates(worse, better))

    def test_update_frontier_archive_removes_dominated_member(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.17,
            volume=0.10,
            qa_error=0.011,
            boozer_residual=0.008,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.5,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "inserted")
        self.assertEqual(update["dominated_members"], ["campaign:lane_01"])
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_02"])

    def test_update_frontier_archive_skips_dominated_candidate(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.17,
            volume=0.10,
            qa_error=0.011,
            boozer_residual=0.008,
            soft_search_score=-1.5,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "dominated")
        self.assertEqual(update["dominated_by"], ["campaign:lane_01"])
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_01"])

    def test_update_frontier_archive_prefers_lower_search_score_for_duplicate(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.1800001,
            volume=0.1100001,
            qa_error=0.0100001,
            boozer_residual=0.0070000001,
            soft_search_score=-1.2,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "duplicate_replaced")
        self.assertEqual(update["replaced_member_id"], "campaign:lane_01")
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_02"])

    def test_update_frontier_archive_keeps_dominating_candidate_even_if_duplicate_score_is_worse(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.0100,
            boozer_residual=0.0070,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.1800002,
            volume=0.1100002,
            qa_error=0.0099999,
            boozer_residual=0.0069999,
            soft_search_score=-0.9,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "duplicate_replaced")
        self.assertEqual(update["replaced_member_id"], "campaign:lane_01")
        self.assertEqual(update["dominated_members"], ["campaign:lane_01"])
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_02"])

    def test_update_frontier_archive_uses_fixed_ideal_nadir_normalization_for_duplicates(self):
        archive_module = load_frontier_archive_module()
        dominance_module = load_frontier_dominance_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.15,
            volume=0.10,
            qa_error=0.012,
            boozer_residual=0.008,
            soft_search_score=-1.0,
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.156,
            volume=0.09995,
            qa_error=0.012,
            boozer_residual=0.008,
            soft_search_score=-1.5,
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
        )
        fixed_ideal_nadir_normalization = {
            "schema_version": dominance_module.PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
            "kind": dominance_module.PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
            "distance_metric": "euclidean",
            "reference_metrics": {
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
            "ideal_metrics": {
                "iota": 0.30,
                "volume": 0.12,
                "qa_error": 0.008,
                "boozer_residual": 0.004,
            },
            "nadir_metrics": {
                "iota": 0.10,
                "volume": 0.08,
                "qa_error": 0.020,
                "boozer_residual": 0.012,
            },
            "metric_rules": dict(
                dominance_module.PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES
            ),
        }

        default_members, default_update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )
        normalized_members, normalized_update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
            pareto_objective_normalization=fixed_ideal_nadir_normalization,
        )

        self.assertEqual(default_update["action"], "inserted")
        self.assertEqual(len(default_members), 2)
        self.assertEqual(normalized_update["action"], "duplicate_replaced")
        self.assertEqual(normalized_update["replaced_member_id"], incumbent.member_id)
        self.assertEqual(len(normalized_members), 1)

    def test_replay_archive_from_lane_records_reapplies_dominance_updates(self):
        archive_module = load_frontier_archive_module()
        engine_base_module = load_frontier_engine_base_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.17,
            volume=0.10,
            qa_error=0.011,
            boozer_residual=0.008,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.5,
        )

        archive_members, incumbent_update = archive_module.update_frontier_archive(
            [],
            incumbent,
        )
        archive_members, candidate_update = archive_module.update_frontier_archive(
            archive_members,
            candidate,
        )

        lane_records = []
        for lane_id, member, archive_update in (
            ("lane_01", incumbent, incumbent_update),
            ("lane_02", candidate, candidate_update),
        ):
            lane_contract = engine_base_module.build_frontier_lane_contract(
                campaign_id="campaign",
                lane_id=lane_id,
                engine="multilane_local",
                scalarization_type="weight_schedule_v1",
                scalarization_params={"iota_share": 0.5, "volume_share": 0.5},
                constraint_mode="frontier_v2_single_lane_contract",
                warm_start_source="seed.json",
                optimizer_budget=10,
                rng_seed=0,
                rerun_contract={},
            )
            lane_records.append(
                engine_base_module.build_frontier_lane_record(
                    lane_contract,
                    command=["python"],
                    weights={
                        "iotas_weight": 1.0,
                        "frontier_volume_weight": 1.0,
                        "res_weight": 1.0,
                    },
                    lane_budget=10,
                    status="completed",
                    result_source="final",
                    success=True,
                    archive_state=member.archive_state,
                    archive_member=member,
                    archive_update=archive_update,
                    results_path=f"/tmp/{lane_id}.json",
                    results={},
                )
            )

        replayed_members = engine_base_module.replay_archive_from_lane_records(
            lane_records
        )

        self.assertEqual(
            [member.member_id for member in replayed_members],
            ["campaign:lane_02"],
        )

    def test_finalize_archive_member_converts_provisional_member_to_canonical_final_member(self):
        archive_module = load_frontier_archive_module()
        payload = {
            "result_source": "final",
            "results_path": "/tmp/lane_01/results.json",
            "results": {
                "FINAL_IOTA": 0.18,
                "FINAL_VOLUME": 0.11,
                "NONQS_RATIO": 0.01,
                "BOOZER_RESIDUAL": 0.007,
                "FINAL_FEASIBILITY_OK": True,
                "HARDWARE_CONSTRAINTS_OK": True,
                "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                "FRONTIER_TRUST_OK": True,
                "FRONTIER_REFERENCE_IOTA": 0.15,
                "FRONTIER_REFERENCE_VOLUME": 0.10,
                "FRONTIER_REFERENCE_QA": 0.012,
                "FRONTIER_REFERENCE_BOOZER": 0.008,
                "FRONTIER_RANK_OBJECTIVE_J": -1.0,
                "OPTIMIZER_SUCCESS": True,
                "TERMINATION_MESSAGE": "ok",
            },
        }

        provisional_member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_01",
            payload=payload,
            rerun_contract={},
            archive_state=archive_module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
        )
        final_member = archive_module.finalize_archive_member(provisional_member)

        self.assertEqual(
            provisional_member.member_id,
            "campaign:lane_01:provisional",
        )
        self.assertEqual(
            provisional_member.archive_state,
            archive_module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
        )
        self.assertEqual(final_member.member_id, "campaign:lane_01")
        self.assertEqual(
            final_member.archive_state,
            archive_module.FRONTIER_ARCHIVE_STATE_CERTIFIED,
        )

    def test_build_frontier_lane_record_tracks_provisional_and_certified_ids(self):
        archive_module = load_frontier_archive_module()
        engine_base_module = load_frontier_engine_base_module()
        payload = {
            "result_source": "final",
            "results_path": "/tmp/lane_01/results.json",
            "results": {
                "FINAL_IOTA": 0.18,
                "FINAL_VOLUME": 0.11,
                "NONQS_RATIO": 0.01,
                "BOOZER_RESIDUAL": 0.007,
                "FINAL_FEASIBILITY_OK": True,
                "HARDWARE_CONSTRAINTS_OK": True,
                "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                "FRONTIER_TRUST_OK": True,
                "FRONTIER_REFERENCE_IOTA": 0.15,
                "FRONTIER_REFERENCE_VOLUME": 0.10,
                "FRONTIER_REFERENCE_QA": 0.012,
                "FRONTIER_REFERENCE_BOOZER": 0.008,
                "FRONTIER_RANK_OBJECTIVE_J": -1.0,
                "OPTIMIZER_SUCCESS": True,
                "TERMINATION_MESSAGE": "ok",
            },
        }
        provisional_member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_01",
            payload=payload,
            rerun_contract={},
            archive_state=archive_module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
        )
        certified_member = archive_module.finalize_archive_member(provisional_member)
        lane_contract = engine_base_module.build_frontier_lane_contract(
            campaign_id="campaign",
            lane_id="lane_01",
            engine="multilane_local",
            scalarization_type="weight_schedule_v1",
            scalarization_params={"iota_share": 0.5, "volume_share": 0.5},
            constraint_mode="frontier_v2_single_lane_contract",
            warm_start_source="seed.json",
            optimizer_budget=10,
            rng_seed=0,
            rerun_contract={},
        )

        lane_record = engine_base_module.build_frontier_lane_record(
            lane_contract,
            command=["python"],
            weights={
                "iotas_weight": 1.0,
                "frontier_volume_weight": 1.0,
                "res_weight": 1.0,
            },
            lane_budget=10,
            status="completed",
            result_source="final",
            success=True,
            provisional_archive_member=provisional_member,
            archive_state=certified_member.archive_state,
            archive_member=certified_member,
            archive_update={"action": "inserted", "member_id": certified_member.member_id},
            results_path="/tmp/lane_01.json",
            results={},
        )

        self.assertEqual(
            lane_record.provisional_member_ids,
            ["campaign:lane_01:provisional"],
        )
        self.assertEqual(
            lane_record.certified_member_ids,
            ["campaign:lane_01"],
        )
        self.assertTrue(lane_record.final_certified)

    def test_recommend_frontier_member_balanced_prefers_better_overall_tradeoff(self):
        archive_module = load_frontier_archive_module()
        recommendation_module = load_frontier_recommendation_module()

        balanced = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.165,
            volume=0.108,
            qa_error=0.011,
            boozer_residual=0.0075,
            soft_search_score=-1.1,
        )
        iota_only = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.19,
            volume=0.095,
            qa_error=0.015,
            boozer_residual=0.011,
            soft_search_score=-1.0,
        )

        recommendation = recommendation_module.recommend_frontier_member(
            [iota_only, balanced],
            policy_name="balanced",
        )

        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(
            recommendation["recommended_member"].member_id,
            "campaign:lane_01",
        )
        self.assertEqual(recommendation["policy_name"], "balanced")

    def test_hypervolume_contributions_are_computed_from_reference_point(self):
        archive_module = load_frontier_archive_module()

        member_a = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.20,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.006,
            soft_search_score=-1.0,
        )
        member_b = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.17,
            volume=0.13,
            qa_error=0.010,
            boozer_residual=0.006,
            soft_search_score=-1.1,
        )
        hypervolume_reference = {
            "iota": 0.15,
            "volume": 0.10,
            "qa_error": 0.012,
            "boozer_residual": 0.008,
        }

        annotated_members = archive_module.annotate_hypervolume_contributions(
            [member_a, member_b],
            hypervolume_reference=hypervolume_reference,
        )
        hypervolume_total = archive_module.frontier_archive_hypervolume(
            annotated_members,
            hypervolume_reference=hypervolume_reference,
        )

        self.assertIsNotNone(hypervolume_total)
        assert hypervolume_total is not None
        self.assertAlmostEqual(hypervolume_total, 3.6e-9)
        contributions = {
            member.member_id: member.hypervolume_contribution
            for member in annotated_members
        }
        self.assertAlmostEqual(contributions["campaign:lane_01"], 1.2e-9)
        self.assertAlmostEqual(contributions["campaign:lane_02"], 1.6e-9)

        serialized = archive_module.serialize_frontier_archive(
            [member_a, member_b],
            hypervolume_reference=hypervolume_reference,
        )
        self.assertAlmostEqual(serialized["hypervolume_total"], 3.6e-9)
        self.assertEqual(
            serialized["hypervolume_reference"],
            hypervolume_reference,
        )

    def test_build_archive_member_from_results_applies_epsilon_certification_contract(self):
        archive_module = load_frontier_archive_module()
        payload = {
            "result_source": "final",
            "results_path": "/tmp/lane_safe_iota/results.json",
            "results": {
                "FINAL_IOTA": 0.18,
                "FINAL_VOLUME": 0.11,
                "NONQS_RATIO": 0.013,
                "BOOZER_RESIDUAL": 0.007,
                "FINAL_FEASIBILITY_OK": True,
                "HARDWARE_CONSTRAINTS_OK": True,
                "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                "FRONTIER_TRUST_OK": True,
                "FRONTIER_REFERENCE_IOTA": 0.15,
                "FRONTIER_REFERENCE_VOLUME": 0.10,
                "FRONTIER_REFERENCE_QA": 0.012,
                "FRONTIER_REFERENCE_BOOZER": 0.008,
                "FRONTIER_RANK_OBJECTIVE_J": -1.0,
                "OPTIMIZER_SUCCESS": True,
                "TERMINATION_MESSAGE": "ok",
            },
        }

        member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_safe_iota",
            payload=payload,
            rerun_contract={
                "scalarization_type": "epsilon_constraint_sweep_v1",
                "scalarization_params": {
                    "epsilon_constraint_qa_max": 0.012,
                    "epsilon_constraint_boozer_max": 0.008,
                },
            },
        )

        self.assertFalse(member.hard_certification_ok)
        self.assertFalse(member.constraint_metrics["epsilon_constraints_ok"])
        self.assertAlmostEqual(
            member.constraint_metrics["epsilon_constraint_violations"]["qa_error"],
            0.001,
        )
